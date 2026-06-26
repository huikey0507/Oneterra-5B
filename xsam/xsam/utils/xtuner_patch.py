"""Monkey-patch xtuner LLaMA attention for single-GPU / non-distributed inference."""

import torch.distributed as dist


def patch_xtuner_llama_attn_for_single_gpu() -> None:
    """Guard ``dist.get_rank()`` when the default process group is not initialized.

    Xtuner's dispatched ``llama_attn_forward`` calls ``dist.get_rank()`` unconditionally.
    Demo and other single-process entry points (``launcher=none``) do not call
    ``init_process_group``, which triggers:
    ``ValueError: Default process group has not been initialized``.
    """
    try:
        import xtuner.model.modules.dispatch.llama as llama_mod
    except ImportError:
        return

    if getattr(llama_mod, "_xsam_dist_guard_patched", False):
        return

    _original_forward = llama_mod.llama_attn_forward

    def llama_attn_forward_patched(self, *args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            return _original_forward(self, *args, **kwargs)

        _orig_get_rank = dist.get_rank
        dist.get_rank = lambda: 0  # type: ignore[method-assign]
        try:
            return _original_forward(self, *args, **kwargs)
        finally:
            dist.get_rank = _orig_get_rank

    llama_mod.llama_attn_forward = llama_attn_forward_patched
    llama_mod._xsam_dist_guard_patched = True
