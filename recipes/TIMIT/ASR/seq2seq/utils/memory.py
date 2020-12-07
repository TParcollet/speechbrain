import torch


def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append(
            {
                "layer_idx": idx,
                "call_idx": call_idx,
                "layer_type": type(self).__name__,
                "exp": exp,
                "hook_type": hook_type,
                "mem_all": mem_all,
                "mem_cached": mem_cached,
            }
        )

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(
        _generate_mem_hook(hr, mem_log, idx, "pre", exp)
    )
    hr.append(h)

    h = mod.register_forward_hook(
        _generate_mem_hook(hr, mem_log, idx, "fwd", exp)
    )
    hr.append(h)

    h = mod.register_backward_hook(
        _generate_mem_hook(hr, mem_log, idx, "bwd", exp)
    )
    hr.append(h)


def log_mem(model, predictions, targets, stage, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules):
        for m in model.modules[module].modules():
            _add_memory_hooks(idx, m, mem_log, exp, hr)

    try:
        model.compute_objectives(predictions, targets, stage)
    finally:
        [h.remove() for h in hr]

        return mem_log
