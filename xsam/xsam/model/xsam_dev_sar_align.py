from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmengine import print_log
from xtuner.utils.device import get_device

from .modules import (
    ConnectorConfig,
    ConnectorModel,
    DynamicProjectorConfig,
    DynamicProjectorModel,
    SarCondAdapter,
)
from .modules.modality_router import ModalityRouter
from .utils import prepare_inputs_labels_for_multimodal
from .xsam_base import XSamModel as BaseXSamModel


class XSamSarAlignModel(BaseXSamModel):
    """Evaluation-oriented X-SAM variant with SAR dual-path adapters."""

    def __init__(
        self,
        *args,
        use_sar_adapters=False,
        freeze_optical_adapters=False,
        freeze_sar_projectors=False,
        use_sar_cond_adapter=False,
        sar_cond_adapter_hidden_dim=256,
        sar_cond_adapter_residual_scale=1.0,
        sar_router_hidden_dim=256,
        sar_gate_loss_weight=0.1,
        sar_infer_threshold=0.5,
        force_route_mode="auto",
        projector_depth=2,
        downsample_ratio=0.5,
        connector_type=None,
        connector_hidden_dim=256,
        connector_scale_factor=(4, 2, 1, 0.5),
        use_activation_checkpointing=True,
        **kwargs,
    ):
        self.use_sar_adapters = use_sar_adapters
        self.freeze_optical_adapters = freeze_optical_adapters
        self.freeze_sar_projectors = freeze_sar_projectors
        self.use_sar_cond_adapter = use_sar_cond_adapter
        self.sar_gate_loss_weight = sar_gate_loss_weight
        self.sar_infer_threshold = float(sar_infer_threshold)
        self.force_route_mode = force_route_mode
        self._modality_gate_cache = None
        self._modality_label_cache = None
        self._route_weight_cache = None
        self._use_activation_checkpointing = use_activation_checkpointing

        super().__init__(
            *args,
            projector_depth=projector_depth,
            downsample_ratio=downsample_ratio,
            connector_type=connector_type,
            connector_hidden_dim=connector_hidden_dim,
            connector_scale_factor=connector_scale_factor,
            use_activation_checkpointing=use_activation_checkpointing,
            **kwargs,
        )

        if self.use_sar_adapters:
            self._build_sar_adapters(
                projector_depth=projector_depth,
                downsample_ratio=downsample_ratio,
                connector_type=connector_type,
                connector_hidden_dim=connector_hidden_dim,
                connector_scale_factor=connector_scale_factor,
                use_sar_cond_adapter=use_sar_cond_adapter,
                sar_cond_adapter_hidden_dim=sar_cond_adapter_hidden_dim,
                sar_cond_adapter_residual_scale=sar_cond_adapter_residual_scale,
                sar_router_hidden_dim=sar_router_hidden_dim,
            )
            self._copy_optical_to_sar()
            if self.freeze_optical_adapters:
                self._freeze_optical_adapters()
            if self.freeze_sar_projectors:
                self._freeze_sar_projectors()
            if self._use_activation_checkpointing:
                self._enable_sar_input_require_grads()
                self._set_sar_gradient_checkpointing(enabled=True)
            print_log("Initialized SAR adapters for eval/dev branch", logger="current")
            if self.force_route_mode != "auto":
                print_log(f"Force route mode enabled: {self.force_route_mode}", logger="current")

    def load_state_dict(self, state_dict, strict=True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        if self.use_sar_adapters:
            has_sar = any(
                ("sar_visual_projector." in k)
                or ("sar_seg_projector." in k)
                or ("sar_seg_connector." in k)
                or ("sar_cond_adapter." in k)
                or ("modality_router." in k)
                for k in state_dict.keys()
            )
            if not has_sar:
                self._copy_optical_to_sar()
                print_log(
                    "Checkpoint has no SAR keys; copied optical weights into SAR adapters",
                    logger="current",
                )
        return incompatible

    def _build_sar_adapters(
        self,
        projector_depth,
        downsample_ratio,
        connector_type,
        connector_hidden_dim,
        connector_scale_factor,
        use_sar_cond_adapter,
        sar_cond_adapter_hidden_dim,
        sar_cond_adapter_residual_scale,
        sar_router_hidden_dim,
    ):
        if hasattr(self, "visual_projector"):
            sar_vis_cfg = DynamicProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth,
            )
            self.sar_visual_projector = DynamicProjectorModel(sar_vis_cfg).to(self.visual_encoder.dtype)

        if hasattr(self, "seg_projector"):
            sar_seg_proj_cfg = DynamicProjectorConfig(
                visual_hidden_size=self.segmentor.enc_config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                downsample_ratio=downsample_ratio,
                depth=projector_depth,
            )
            self.sar_seg_projector = DynamicProjectorModel(sar_seg_proj_cfg).to(self.segmentor.dtype)

        if hasattr(self, "seg_connector"):
            n_levels = self.segmentor.dec_config.num_feature_levels
            sar_conn_cfg = ConnectorConfig(
                segmentor_encoder_channels=[self.segmentor.enc_config.hidden_size] * n_levels,
                hidden_channels=connector_hidden_dim,
                scale_factor=connector_scale_factor[-n_levels:],
                connector_type=connector_type,
            )
            self.sar_seg_connector = ConnectorModel(sar_conn_cfg).to(self.segmentor.dtype)

        if use_sar_cond_adapter and self.segmentor is not None and self.segmentor.decoder is not None:
            self.sar_cond_adapter = SarCondAdapter(
                embed_dim=self.segmentor.dec_config.hidden_size,
                hidden_dim=sar_cond_adapter_hidden_dim,
                residual_scale=sar_cond_adapter_residual_scale,
            ).to(self.segmentor.dtype)

        if self.visual_encoder is not None:
            self.modality_router = ModalityRouter(
                in_dim=self.visual_encoder.config.hidden_size,
                hidden_dim=sar_router_hidden_dim,
            ).to(self.visual_encoder.dtype)

        print_log(
            "Built SAR adapters: "
            f"visual={hasattr(self, 'sar_visual_projector')}, "
            f"seg_proj={hasattr(self, 'sar_seg_projector')}, "
            f"connector={hasattr(self, 'sar_seg_connector')}, "
            f"cond_adapter={hasattr(self, 'sar_cond_adapter')}, "
            f"router={hasattr(self, 'modality_router')}",
            logger="current",
        )

    def _copy_optical_to_sar(self):
        if hasattr(self, "sar_visual_projector") and hasattr(self, "visual_projector"):
            self.sar_visual_projector.load_state_dict(self.visual_projector.state_dict())
        if hasattr(self, "sar_seg_projector") and hasattr(self, "seg_projector"):
            self.sar_seg_projector.load_state_dict(self.seg_projector.state_dict())
        if hasattr(self, "sar_seg_connector") and hasattr(self, "seg_connector"):
            self.sar_seg_connector.load_state_dict(self.seg_connector.state_dict())

    def _freeze_optical_adapters(self):
        if hasattr(self, "visual_projector"):
            self.visual_projector.requires_grad_(False)
        if hasattr(self, "seg_projector"):
            self.seg_projector.requires_grad_(False)
        if hasattr(self, "seg_connector"):
            self.seg_connector.requires_grad_(False)
        if hasattr(self, "llm_projector"):
            self.llm_projector.requires_grad_(False)

    def _freeze_sar_projectors(self):
        if hasattr(self, "sar_visual_projector"):
            self.sar_visual_projector.requires_grad_(False)
        if hasattr(self, "sar_seg_projector"):
            self.sar_seg_projector.requires_grad_(False)

    def _enable_sar_input_require_grads(self):
        for name in ("sar_visual_projector", "sar_seg_projector", "sar_seg_connector"):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "enable_input_require_grads"):
                module.enable_input_require_grads()

    def _set_sar_gradient_checkpointing(self, enabled):
        action = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        for name in ("sar_visual_projector", "sar_seg_projector", "sar_seg_connector"):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, action):
                getattr(module, action)()

    @staticmethod
    def _mix_by_weight(opt_out, sar_out, weight):
        w = weight
        while w.dim() < opt_out.dim():
            w = w.unsqueeze(-1)
        w = w.to(device=opt_out.device, dtype=opt_out.dtype)
        return (1.0 - w) * opt_out + w * sar_out

    def _route_module_out(self, opt_out, sar_out, route_weight):
        if sar_out is None:
            return opt_out
        if isinstance(opt_out, (list, tuple)):
            return [self._mix_by_weight(o, s, route_weight) for o, s in zip(opt_out, sar_out)]
        return self._mix_by_weight(opt_out, sar_out, route_weight)

    def _get_route_weight(self, vis_tokens, modality, batch_size, device, dtype):
        if self.force_route_mode == "optical":
            self._modality_label_cache = None
            self._modality_gate_cache = None
            route_weight = torch.zeros(batch_size, device=device, dtype=dtype)
            self._route_weight_cache = route_weight
            return route_weight
        if self.force_route_mode == "sar":
            self._modality_label_cache = None
            self._modality_gate_cache = None
            route_weight = torch.ones(batch_size, device=device, dtype=dtype)
            self._route_weight_cache = route_weight
            return route_weight

        gate = None
        if hasattr(self, "modality_router") and vis_tokens is not None:
            gate = self.modality_router(vis_tokens.detach())

        if modality is not None:
            if not torch.is_tensor(modality):
                modality = torch.tensor(modality, device=device)
            modality = modality.to(device=device).long().view(-1)
            if modality.numel() == 1 and batch_size > 1:
                modality = modality.expand(batch_size)
            self._modality_label_cache = modality
            self._modality_gate_cache = gate
            route_weight = modality.float()
            self._route_weight_cache = route_weight
            return route_weight

        self._modality_label_cache = None
        self._modality_gate_cache = gate
        if gate is not None:
            route_weight = (gate > self.sar_infer_threshold).to(dtype=dtype)
            self._route_weight_cache = route_weight
            return route_weight
        route_weight = torch.zeros(batch_size, device=device, dtype=dtype)
        self._route_weight_cache = route_weight
        return route_weight

    def _apply_sar_cond_adapter(self, cond_embeds, local_cond_lens=None):
        if cond_embeds is None or not hasattr(self, "sar_cond_adapter"):
            return cond_embeds
        route_weight = self._route_weight_cache
        if route_weight is None:
            return cond_embeds

        adapted_cond_embeds = self.sar_cond_adapter(cond_embeds)
        if not torch.is_tensor(cond_embeds):
            return adapted_cond_embeds

        if cond_embeds.shape[0] == route_weight.numel():
            return self._mix_by_weight(cond_embeds, adapted_cond_embeds, route_weight)

        if local_cond_lens is not None and sum(local_cond_lens) == cond_embeds.shape[0]:
            repeat_counts = torch.tensor(local_cond_lens, device=route_weight.device, dtype=torch.long)
            expanded_route_weight = torch.repeat_interleave(route_weight.view(-1), repeat_counts, dim=0)
            return self._mix_by_weight(cond_embeds, adapted_cond_embeds, expanded_route_weight)

        return adapted_cond_embeds

    def _process_embeds(self, cond_embeds, seg_embeds, task_name="genseg"):
        cond_embeds, seg_embeds, embed_masks, local_cond_lens, global_cond_lens = super()._process_embeds(
            cond_embeds, seg_embeds, task_name
        )
        cond_embeds = self._apply_sar_cond_adapter(cond_embeds, local_cond_lens)
        return cond_embeds, seg_embeds, embed_masks, local_cond_lens, global_cond_lens

    def _forward_visual_encoder(self, pixel_values):
        pixel_values = pixel_values.to(self.visual_encoder.dtype)
        if self.freeze_visual_encoder:
            with torch.no_grad():
                return self.visual_encoder(pixel_values, output_hidden_states=True)
        return self.visual_encoder(pixel_values, output_hidden_states=True)

    def _forward_segmentor_encoder(self, seg_pixel_values):
        seg_pixel_values = seg_pixel_values.to(self.segmentor.dtype)
        if self.freeze_segmentor_encoder:
            with torch.no_grad():
                return self.segmentor.encoder(
                    seg_pixel_values,
                    output_hidden_states=True,
                    output_attentions=False,
                )
        return self.segmentor.encoder(
            seg_pixel_values,
            output_hidden_states=True,
            output_attentions=False,
        )

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        if not self.use_sar_adapters:
            return super().forward(data_dict, data_samples=data_samples, mode=mode, **kwargs)

        if data_samples is not None:
            data_samples = self._move_data_samples(data_samples)

        self._modality_gate_cache = None
        self._modality_label_cache = None
        self._route_weight_cache = None
        modality = data_dict.pop("modality", None)

        extra_data_dict = {}
        route_weight = None
        vis_tokens_for_router = None

        if "pixel_values" in data_dict and self.visual_encoder is not None:
            visual_outputs = self._forward_visual_encoder(data_dict["pixel_values"])
            vis_tokens = visual_outputs.hidden_states[self.visual_select_layer][:, self.visual_select_indx :]
            vis_tokens_for_router = vis_tokens
            opt_pixel_values = self.visual_projector(vis_tokens)

            if hasattr(self, "sar_visual_projector"):
                route_weight = self._get_route_weight(
                    vis_tokens,
                    modality,
                    batch_size=vis_tokens.shape[0],
                    device=vis_tokens.device,
                    dtype=opt_pixel_values.dtype,
                )
                sar_pixel_values = self.sar_visual_projector(vis_tokens)
                pixel_values = self._route_module_out(opt_pixel_values, sar_pixel_values, route_weight)
            else:
                pixel_values = opt_pixel_values

            data_dict["pixel_values"] = pixel_values.to(self.llm.dtype)
            del visual_outputs

        if "seg_pixel_values" in data_dict and self.segmentor is not None:
            if self.extract_seg_embeds:
                seg_visual_outputs = self._forward_segmentor_encoder(data_dict["seg_pixel_values"])
                seg_hidden_states = seg_visual_outputs.hidden_states
                seg_image_embeddings = (
                    seg_visual_outputs.last_hidden_state
                    if hasattr(seg_visual_outputs, "last_hidden_state")
                    else seg_hidden_states[-1].transpose(1, 2)
                )
                seg_pixel_values = None

                if hasattr(self, "seg_projector"):
                    opt_seg_pixel_values = self.seg_projector(seg_hidden_states[self.visual_select_layer])
                    if hasattr(self, "sar_seg_projector"):
                        if route_weight is None:
                            route_weight = self._get_route_weight(
                                vis_tokens_for_router,
                                modality,
                                batch_size=opt_seg_pixel_values.shape[0],
                                device=opt_seg_pixel_values.device,
                                dtype=opt_seg_pixel_values.dtype,
                            )
                        sar_seg_pixel_values = self.sar_seg_projector(
                            seg_hidden_states[self.visual_select_layer]
                        )
                        seg_pixel_values = self._route_module_out(
                            opt_seg_pixel_values,
                            sar_seg_pixel_values,
                            route_weight,
                        )
                    else:
                        seg_pixel_values = opt_seg_pixel_values
                    seg_pixel_values = seg_pixel_values.to(self.llm.dtype)

                if hasattr(self, "seg_connector"):
                    selected = [seg_hidden_states[i] for i in self.seg_select_layers]
                    opt_seg_image_embeddings = self.seg_connector(selected)
                    if hasattr(self, "sar_seg_connector"):
                        if route_weight is None:
                            route_weight = self._get_route_weight(
                                vis_tokens_for_router,
                                modality,
                                batch_size=selected[0].shape[0],
                                device=selected[0].device,
                                dtype=opt_seg_image_embeddings[0].dtype,
                            )
                        sar_seg_image_embeddings = self.sar_seg_connector(selected)
                        seg_image_embeddings = self._route_module_out(
                            opt_seg_image_embeddings,
                            sar_seg_image_embeddings,
                            route_weight,
                        )
                    else:
                        seg_image_embeddings = opt_seg_image_embeddings
                elif self.segmentor.pixel_decoder is not None and hasattr(seg_visual_outputs, "feature_maps"):
                    seg_image_embeddings = seg_visual_outputs.feature_maps

                data_dict["seg_pixel_values"] = seg_pixel_values
                extra_data_dict = {
                    "seg_pixel_values": None,
                    "seg_image_embeddings": seg_image_embeddings,
                }
                del seg_visual_outputs, seg_hidden_states
            else:
                extra_data_dict = {
                    "seg_pixel_values": data_dict["seg_pixel_values"].to(self.segmentor.dtype),
                    "seg_image_embeddings": None,
                }
                data_dict["seg_pixel_values"] = None
        else:
            data_dict["seg_pixel_values"] = None

        if modality is not None and self._modality_label_cache is None:
            if not torch.is_tensor(modality):
                modality = torch.tensor(modality)
            self._modality_label_cache = modality.long().view(-1)

        if data_dict.get("vprompt_masks", None) is not None and hasattr(self, "vision_sampler"):
            vprompt_masks = data_dict.pop("vprompt_masks")
            class_labels, contiguous_labels = self._get_vgd_labels(data_samples)
            sampled_labels = self._get_attrs_from_data_samples(data_samples, ["sampled_labels"])[0]
            sampled_feats = self.vision_sampler(data_dict[self.sampler_input_feat], vprompt_masks)
            assert all(
                sampled_feat is not None for sampled_feat in sampled_feats
            ), f"{data_dict[self.sampler_input_feat]}, {vprompt_masks}"
            vprompt_feats, vprompt_masks, new_sampled_labels = self._get_vprompt_feats_and_masks(
                sampled_feats,
                vprompt_masks,
                class_labels,
                contiguous_labels,
                sampled_labels,
            )
            data_dict["vprompt_feats"] = vprompt_feats
            kwargs["vprompt_masks"] = vprompt_masks
            kwargs["sampled_labels"] = new_sampled_labels

        if self.llm is not None:
            data_dict = prepare_inputs_labels_for_multimodal(llm=self.llm, **data_dict)

        data_dict.update(extra_data_dict)

        if mode == "loss":
            return self.compute_loss(data_dict, data_samples, **kwargs)
        if mode == "predict":
            return self.predict(data_dict, data_samples, **kwargs)
        if mode == "tensor":
            return self._forward(data_dict, data_samples, **kwargs)
        raise NotImplementedError

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        llm_outputs, seg_outputs = self._forward(data_dict, data_samples, **kwargs)
        device = next(self.parameters()).device
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)

        if llm_outputs is not None and seg_outputs is None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss = loss_llm
            loss_dict = {"loss": loss, "loss_llm": loss_llm, "loss_seg": zero}
        elif llm_outputs is None and seg_outputs is not None:
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_seg
            loss_dict = {"loss": loss, "loss_llm": zero, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        elif llm_outputs is not None and seg_outputs is not None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_llm + loss_seg
            loss_dict = {"loss": loss, "loss_llm": loss_llm, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        else:
            raise ValueError("llm_outputs and seg_outputs are both None")

        if (
            self.use_sar_adapters
            and self._modality_gate_cache is not None
            and self._modality_label_cache is not None
            and self.sar_gate_loss_weight > 0
        ):
            gate = self._modality_gate_cache.float().view(-1)
            label = self._modality_label_cache.float().view(-1).to(device=gate.device)
            if gate.numel() == label.numel():
                loss_gate = F.binary_cross_entropy(gate, label) * self.sar_gate_loss_weight
                loss = loss + loss_gate
                loss_dict["loss"] = loss
                loss_dict["loss_gate"] = loss_gate

        return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if not self.use_sar_adapters:
            return state_dict

        to_return = OrderedDict(state_dict)
        to_return.update({k: v for k, v in state_dict.items() if "sar_visual_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "sar_seg_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "sar_seg_connector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "sar_cond_adapter." in k})
        to_return.update({k: v for k, v in state_dict.items() if "modality_router." in k})
        return to_return

    def activation_checkpointing_enable(self):
        super().activation_checkpointing_enable()
        self._set_sar_gradient_checkpointing(enabled=True)
        if hasattr(self, "sar_cond_adapter"):
            enable_fn = getattr(self.sar_cond_adapter, "gradient_checkpointing_enable", None)
            if callable(enable_fn):
                enable_fn()

    def activation_checkpointing_disable(self):
        super().activation_checkpointing_disable()
        self._set_sar_gradient_checkpointing(enabled=False)
        if hasattr(self, "sar_cond_adapter"):
            disable_fn = getattr(self.sar_cond_adapter, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                disable_fn()

    @staticmethod
    def _move_data_samples(data_samples):
        from ..utils.misc import data_sample_to_device

        return data_sample_to_device(data_samples, device=get_device())
