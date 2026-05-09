import os.path as osp
import torch

from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint

from xsam.utils.logging import print_log


def load_checkpoint(model, pth_model: str) -> None:
    """Load model checkpoint.
    
    This function is primarily used for evaluation/inference scenarios (eval.py, visualize.py, demo.py).
    Training uses mmengine/DeepSpeed's own checkpoint loading mechanisms.
    
    Supports both standard PyTorch checkpoints and DeepSpeed checkpoints.
    For DeepSpeed checkpoints (mp_rank_00_model_states.pt), extracts model weights from 'module' key.
    
    Note: This function does NOT affect training. Training uses:
    - mmengine's CheckpointHook for saving/loading
    - DeepSpeed's own checkpoint mechanism
    - guess_load_checkpoint() directly for pretrained weights (s1_pretrained_pth, s2_pretrained_pth)
    
    Args:
        model: The model to load weights into
        pth_model: Path to checkpoint file or directory
    """
    if not osp.exists(pth_model):
        print_log(f"Checkpoint file not found: {pth_model}", logger="current")
        return

    # Check if it's a DeepSpeed checkpoint file
    # Only trigger special handling for DeepSpeed checkpoint files (used in evaluation)
    # Training checkpoints are handled by mmengine/DeepSpeed, not this function
    is_deepspeed_checkpoint = "mp_rank_00_model_states.pt" in pth_model or "model_states.pt" in pth_model
    
    backend = get_file_backend(pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio

        with patch_fileio():
            if is_deepspeed_checkpoint:
                # Load DeepSpeed checkpoint directly
                checkpoint = torch.load(pth_model, map_location="cpu")
                # Extract model weights from 'module' key
                if "module" in checkpoint:
                    state_dict = checkpoint["module"]
                    # Remove 'module.' prefix if present in keys
                    if any(k.startswith("module.") for k in state_dict.keys()):
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    print_log(f"Loaded DeepSpeed checkpoint, extracted model weights from 'module' key", logger="current")
                else:
                    # Try to use the checkpoint directly (might already be state_dict)
                    state_dict = checkpoint
                    print_log(f"Loaded DeepSpeed checkpoint, using checkpoint directly", logger="current")
            else:
                state_dict = guess_load_checkpoint(pth_model)
    else:
        if is_deepspeed_checkpoint:
            # Load DeepSpeed checkpoint directly
            checkpoint = torch.load(pth_model, map_location="cpu")
            # Extract model weights from 'module' key
            if "module" in checkpoint:
                state_dict = checkpoint["module"]
                # Remove 'module.' prefix if present in keys
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                print_log(f"Loaded DeepSpeed checkpoint, extracted model weights from 'module' key", logger="current")
            else:
                # Try to use the checkpoint directly (might already be state_dict)
                state_dict = checkpoint
                print_log(f"Loaded DeepSpeed checkpoint, using checkpoint directly", logger="current")
        else:
            state_dict = guess_load_checkpoint(pth_model)

    # Filter out non-model keys (optimizer, scheduler, etc.)
    # This is especially important for DeepSpeed checkpoints which contain training state
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict:
            # Direct match - use it
            filtered_state_dict[key] = value
        elif not key.startswith(("optimizer", "lr_scheduler", "buffer_names", "param_shapes", 
                                  "frozen_param", "shared_params", "random_ltd", "sparse_tensor",
                                  "skipped_steps", "global_steps", "global_samples", "dp_world_size",
                                  "mp_world_size", "ds_config", "ds_version", "meta", "message_hub",
                                  "param_schedulers", "optim_wrapper", "data_sampler")):
            # Not a training-related key, might be a model key with different prefix
            # Keep it for now, strict=False will handle mismatches
            filtered_state_dict[key] = value

    # Load with strict=False to handle any remaining mismatches
    # This is safe because:
    # 1. For standard checkpoints, guess_load_checkpoint already handles format conversion
    # 2. For DeepSpeed checkpoints, we've extracted the model weights from 'module' key
    # 3. Training doesn't use this function, so this won't affect training behavior
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    matched_keys = [k for k in filtered_state_dict.keys() if k in model_state_dict.keys()]
    print_log(f"Load checkpoint from {pth_model}", logger="current")
    print_log(f"Matched keys: {len(matched_keys)} / {len(filtered_state_dict.keys())}", logger="current")
    if missing_keys:
        print_log(f"Missing keys: {len(missing_keys)} (first 10: {missing_keys[:10]})", logger="current")
    if unexpected_keys:
        print_log(f"Unexpected keys: {len(unexpected_keys)} (first 10: {unexpected_keys[:10]})", logger="current")

    for key in missing_keys:
        if "embed_tokens" in key:
            print_log(f"Warning: Missing embed_tokens key: {key}", logger="current")