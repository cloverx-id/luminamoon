from __future__ import annotations

from typing import Any, Callable, Dict

from .config import (
    bound_mode_code,
    is_dual_buffer,
    master_weight_code,
    validate_and_normalize_group,
)
from .constants import TEMP_STATE_DICT_KEYS
from .standalone import ParamStateDict
from .validation import HAS_NUMPY, HAS_TORCH, _numel, np, torch


def convert_torch_state_buffer(
    src_buf: Any,
    device: Any,
    target_dtype: Any,
    fp16_is_bf16_bits: bool = True,
) -> Any:
    if not (HAS_TORCH and torch is not None and isinstance(src_buf, torch.Tensor)):
        return src_buf
    if src_buf.dtype == target_dtype:
        if src_buf.dtype == torch.float16 and not fp16_is_bf16_bits:
            from .torch_ops import _encode_state_from_f32

            buf_dev = src_buf.to(device=device).contiguous()
            out = torch.empty_like(buf_dev, dtype=torch.float16, memory_format=torch.contiguous_format)
            _encode_state_from_f32(out, buf_dev.float(), use_sr=False)
            return out
        return src_buf.to(device=device, dtype=target_dtype).contiguous()
    if src_buf.dtype == torch.float16 or target_dtype == torch.float16:
        from .torch_ops import _decode_state_f32, _encode_state_from_f32

        buf_dev = src_buf.to(device=device).contiguous()
        val_f32 = (
            _decode_state_f32(buf_dev)
            if (src_buf.dtype != torch.float16 or fp16_is_bf16_bits)
            else buf_dev.float()
        )
        if target_dtype == torch.float16:
            out = torch.empty_like(buf_dev, dtype=torch.float16, memory_format=torch.contiguous_format)
            _encode_state_from_f32(out, val_f32, use_sr=False)
            return out
        return val_f32.to(dtype=target_dtype).contiguous()
    return src_buf.to(device=device, dtype=target_dtype).contiguous()


def _clean_torch_scratch(sv: Any) -> Any:
    sv_clean = sv.detach().clone()
    sv_clean[:3].zero_()
    sv_clean[5].zero_()
    return sv_clean


def _clean_numpy_scratch(sv: Any) -> Any:
    sv_clean = sv.copy()
    sv_clean[:3] = 0.0
    sv_clean[5] = 0.0
    return sv_clean


def export_state_dict(opt: Any, super_state_dict_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    param_ids = {id(p) for group in opt.param_groups for p in group.get("params", [])}

    if not getattr(opt, "_is_torch_mode", False):
        param_mappings: Dict[Any, int] = {}
        start_index = 0
        packed_groups = []
        for group in opt.param_groups:
            packed = {k: v for k, v in group.items() if k not in ("params", "_validated_sig")}
            p_indices = []
            for p in group.get("params", []):
                p_key = opt.state._key(p)
                if p_key not in param_mappings:
                    param_mappings[p_key] = start_index
                    start_index += 1
                p_indices.append(param_mappings[p_key])
            packed["params"] = p_indices
            packed_groups.append(packed)

        serialized_state = {}
        for k, v in opt.state.items():
            out_key = param_mappings[k] if k in param_mappings else k
            if isinstance(v, dict):
                s_dict = {}
                for sk, sv in v.items():
                    if sk in TEMP_STATE_DICT_KEYS or (sk == "master_param" and id(sv) in param_ids):
                        continue
                    if sk == "scratch" and HAS_NUMPY and np is not None and isinstance(sv, np.ndarray) and sv.size >= 6:
                        s_dict[sk] = _clean_numpy_scratch(sv)
                    elif HAS_NUMPY and np is not None and isinstance(sv, np.ndarray):
                        s_dict[sk] = sv.copy()
                    else:
                        s_dict[sk] = sv
                serialized_state[out_key] = s_dict
            else:
                serialized_state[out_key] = v
        return {"state": serialized_state, "param_groups": packed_groups}

    raw_dict = super_state_dict_fn()
    clean_state = {}
    for k, p_state in raw_dict.get("state", {}).items():
        if not isinstance(p_state, dict):
            clean_state[k] = p_state
            continue
        entry: Dict[str, Any] = {}
        has_fp16_buf = False
        for sk, sv in p_state.items():
            if sk in TEMP_STATE_DICT_KEYS or (sk == "master_param" and id(sv) in param_ids):
                continue
            if sk == "scratch" and HAS_TORCH and torch is not None and isinstance(sv, torch.Tensor) and sv.numel() >= 6:
                entry[sk] = _clean_torch_scratch(sv)
            elif HAS_TORCH and torch is not None and isinstance(sv, torch.Tensor):
                if sk in ("exp_avg", "exp_avg_sq") and sv.dtype == torch.float16:
                    has_fp16_buf = True
                entry[sk] = sv.detach().clone()
            else:
                entry[sk] = sv
        if has_fp16_buf:
            entry["_fp16_state_format"] = "bf16_bits"
        clean_state[k] = entry

    clean_groups = [
        {gk: gv for gk, gv in grp.items() if gk != "_validated_sig"}
        for grp in raw_dict.get("param_groups", [])
    ]
    return {"state": clean_state, "param_groups": clean_groups}


def import_state_dict(
    opt: Any,
    state_dict: Dict[str, Any],
    super_load_fn: Callable[[Dict[str, Any]], None],
) -> None:
    if not getattr(opt, "_is_torch_mode", False):
        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        if len(opt.param_groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        for g, sg in zip(opt.param_groups, saved_groups):
            if len(g.get("params", [])) != len(sg.get("params", [])):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
                )

        new_state = ParamStateDict()
        for group, saved_group in zip(opt.param_groups, saved_groups):
            for k, v in saved_group.items():
                if k != "params":
                    group[k] = v
            for k, v in opt.defaults.items():
                group.setdefault(k, v)
            validate_and_normalize_group(group)
            cautious_g = bool(group["cautious"])
            radial_g = bound_mode_code(bool(group["bound"]), group["bound_type"], float(group["bound_ratio"])) == 1
            for p, saved_id in zip(group.get("params", []), saved_group.get("params", [])):
                p_key = new_state._key(p)
                s_key = new_state._key(saved_id)
                st_entry = (
                    saved_state[saved_id]
                    if saved_id in saved_state
                    else (saved_state[s_key] if s_key in saved_state else saved_state.get(p_key))
                )
                if isinstance(st_entry, dict):
                    p_st = {
                        sk: (
                            np.ascontiguousarray(sv.copy())
                            if (HAS_NUMPY and np is not None and isinstance(sv, np.ndarray))
                            else sv
                        )
                        for sk, sv in st_entry.items()
                    }
                    if HAS_NUMPY and np is not None:
                        sc = p_st.get("scratch")
                        if isinstance(sc, np.ndarray) and sc.size >= 6:
                            sc_clean = np.ascontiguousarray(sc.astype(np.float32, copy=True))
                            sc_clean[:3] = 0.0
                            sc_clean[5] = 0.0
                        else:
                            sc_clean = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
                            if isinstance(sc, np.ndarray) and sc.size > 0:
                                sc_clean[: min(sc.size, 6)] = sc.astype(np.float32)[: min(sc.size, 6)]
                                sc_clean[:3] = 0.0
                                sc_clean[5] = 0.0
                        if not cautious_g:
                            sc_clean[3] = 1.0
                        if not radial_g:
                            sc_clean[4] = 1.0
                        p_st["scratch"] = sc_clean
                    new_state[p] = p_st
        opt.state = new_state
        return

    raw_state = state_dict.get("state", {})
    saved_groups = state_dict.get("param_groups", [])
    raw_param_state: Dict[Any, dict] = {}
    if len(opt.param_groups) == len(saved_groups):
        for group, saved_group in zip(opt.param_groups, saved_groups):
            for p, p_id in zip(group.get("params", []), saved_group.get("params", [])):
                st_entry = raw_state.get(p_id) if p_id in raw_state else raw_state.get(p)
                if isinstance(st_entry, dict):
                    raw_param_state[p] = {
                        k: (
                            v.detach().clone()
                            if (HAS_TORCH and torch is not None and isinstance(v, torch.Tensor))
                            else v
                        )
                        for k, v in st_entry.items()
                    }

    super_load_fn(state_dict)
    if not isinstance(opt.state, ParamStateDict):
        opt.state = ParamStateDict(opt.state)

    assert torch is not None
    for group in opt.param_groups:
        for k, v in opt.defaults.items():
            group.setdefault(k, v)
        validate_and_normalize_group(group)
        master_mode = master_weight_code(group["master_weights"])
        buf_count = group["buffer"]
        cautious_g = bool(group["cautious"])
        radial_g = bound_mode_code(bool(group["bound"]), group["bound_type"], float(group["bound_ratio"])) == 1

        for p in group["params"]:
            if p not in opt.state:
                continue
            st = opt.state[p]
            raw_st = raw_param_state.get(p, {})
            for temp_key in TEMP_STATE_DICT_KEYS:
                st.pop(temp_key, None)
            if "step" in st and isinstance(st["step"], torch.Tensor):
                st["step"] = int(st["step"].item())

            if "scratch" in st and isinstance(st["scratch"], torch.Tensor):
                src_sc = raw_st["scratch"] if isinstance(raw_st.get("scratch"), torch.Tensor) else st["scratch"]
                old_sc = src_sc.to(device=p.device, dtype=torch.float32).contiguous()
                if old_sc.numel() < 6:
                    new_sc = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], device=p.device, dtype=torch.float32)
                    new_sc[: old_sc.numel()].copy_(old_sc)
                    new_sc[:3].zero_()
                    new_sc[5].zero_()
                    st["scratch"] = new_sc
                else:
                    old_sc[:3].zero_()
                    old_sc[5].zero_()
                    st["scratch"] = old_sc
            else:
                st["scratch"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], device=p.device, dtype=torch.float32)

            if not cautious_g:
                st["scratch"][3] = 1.0
            if not radial_g:
                st["scratch"][4] = 1.0

            if master_mode == 0 or p.dtype == torch.float32:
                st.pop("master_param", None)
            elif "master_param" in st and isinstance(st["master_param"], torch.Tensor):
                src_mp = (
                    raw_st["master_param"]
                    if isinstance(raw_st.get("master_param"), torch.Tensor)
                    else st["master_param"]
                )
                st["master_param"] = src_mp.to(device=p.device, dtype=torch.float32).contiguous()
            else:
                st["master_param"] = p.detach().clone().float().contiguous()

            target_state_dtype = torch.float32 if (master_mode == 2 or p.dtype == torch.float32) else p.dtype
            fp16_is_bf16_bits = (
                raw_st.get("_fp16_state_format") == "bf16_bits"
                or st.get("_fp16_state_format") == "bf16_bits"
                or "scratch" in raw_st
                or "scratch" in st
                or "buf_count" in raw_st
                or "buf_count" in st
            )
            st.pop("_fp16_state_format", None)
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in st and isinstance(st[buf_key], torch.Tensor):
                    src_buf = raw_st[buf_key] if isinstance(raw_st.get(buf_key), torch.Tensor) else st[buf_key]
                    st[buf_key] = convert_torch_state_buffer(
                        src_buf, p.device, target_state_dtype, fp16_is_bf16_bits=fp16_is_bf16_bits
                    )
            if not isinstance(st.get("exp_avg"), torch.Tensor):
                st["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
            if not is_dual_buffer(buf_count):
                st.pop("exp_avg_sq", None)
            elif not isinstance(st.get("exp_avg_sq"), torch.Tensor):
                st["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
            st["buf_count"] = buf_count


def ensure_param_state(
    opt: Any,
    p: Any,
    p_for_step: Any,
    group: Dict[str, Any],
    is_torch_tensor: bool,
) -> Dict[str, Any]:
    buf_count = group["buffer"]
    master_mode = master_weight_code(group["master_weights"])
    cautious = bool(group["cautious"])
    radial_active = bound_mode_code(bool(group["bound"]), group["bound_type"], float(group["bound_ratio"])) == 1
    dual_state = is_dual_buffer(buf_count)

    state = opt.state[p]
    if is_torch_tensor and torch is not None:
        state_dtype: Any = torch.float32 if (master_mode == 2 or p.dtype == torch.float32) else p.dtype
    elif HAS_NUMPY and np is not None and hasattr(p, "dtype"):
        state_dtype = np.float32 if (master_mode == 2 or p.dtype == np.float32) else p.dtype
    else:
        state_dtype = "float32"

    if len(state) == 0:
        state["step"] = 0
        if is_torch_tensor and torch is not None:
            state["scratch"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format)
            if dual_state:
                state["exp_avg_sq"] = torch.zeros_like(
                    p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                )
            if master_mode > 0 and p.dtype in (torch.float16, torch.bfloat16):
                state["master_param"] = p_for_step.detach().clone().float().contiguous()
        elif HAS_NUMPY and np is not None:
            state["scratch"] = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
            state["exp_avg"] = np.zeros_like(p_for_step, dtype=state_dtype)
            if dual_state:
                state["exp_avg_sq"] = np.zeros_like(p_for_step, dtype=state_dtype)
            if master_mode > 0 and getattr(p, "dtype", None) != np.float32:
                state["master_param"] = np.array(p_for_step, dtype=np.float32, copy=True)
        else:
            p_numel = _numel(p)
            state["scratch"] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            state["exp_avg"] = [0.0] * p_numel
            if dual_state:
                state["exp_avg_sq"] = [0.0] * p_numel
        state["buf_count"] = buf_count
        return state

    if is_torch_tensor and torch is not None:
        if master_mode > 0 and p.dtype != torch.float32:
            if state.get("master_param") is None:
                state["master_param"] = p_for_step.detach().clone().float().contiguous()
            elif state["master_param"].device != p.device or state["master_param"].dtype != torch.float32:
                state["master_param"] = state["master_param"].to(device=p.device, dtype=torch.float32, non_blocking=True)
        else:
            state.pop("master_param", None)

        if "scratch" not in state or state["scratch"].numel() < 6:
            scratch = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], device=p.device, dtype=torch.float32)
            if isinstance(state.get("scratch"), torch.Tensor):
                old_sc = state["scratch"].to(device=p.device, dtype=torch.float32)
                scratch[: old_sc.numel()].copy_(old_sc)
            state["scratch"] = scratch
            state["buf_count"] = buf_count
        elif state["scratch"].device != p.device or state["scratch"].dtype != torch.float32:
            state["scratch"] = state["scratch"].to(device=p.device, dtype=torch.float32, non_blocking=True)
            state["buf_count"] = buf_count
        elif state.get("buf_count", buf_count) != buf_count:
            state["scratch"].zero_()
            state["scratch"][3] = 1.0
            state["scratch"][4] = 1.0
            state["buf_count"] = buf_count

        if not cautious:
            state["scratch"][3] = 1.0
        if not radial_active:
            state["scratch"][4] = 1.0

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format)
        elif state["exp_avg"].device != p.device or state["exp_avg"].dtype != state_dtype:
            state["exp_avg"] = convert_torch_state_buffer(state["exp_avg"], p.device, state_dtype)

        if dual_state:
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(
                    p_for_step, dtype=state_dtype, memory_format=torch.contiguous_format
                )
            elif state["exp_avg_sq"].device != p.device or state["exp_avg_sq"].dtype != state_dtype:
                state["exp_avg_sq"] = convert_torch_state_buffer(state["exp_avg_sq"], p.device, state_dtype)
        else:
            state.pop("exp_avg_sq", None)
    else:
        if state.get("buf_count", buf_count) != buf_count:
            if HAS_NUMPY and np is not None and hasattr(state["scratch"], "__setitem__"):
                state["scratch"][:] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            state["buf_count"] = buf_count
        if not cautious and hasattr(state.get("scratch"), "__setitem__"):
            state["scratch"][3] = 1.0
        if not radial_active and hasattr(state.get("scratch"), "__setitem__"):
            state["scratch"][4] = 1.0
        if dual_state and "exp_avg_sq" not in state and HAS_NUMPY and np is not None:
            state["exp_avg_sq"] = np.zeros_like(p_for_step, dtype=state_dtype)
        elif not dual_state:
            state.pop("exp_avg_sq", None)
        if master_mode == 0 or (HAS_NUMPY and np is not None and getattr(p, "dtype", None) == np.float32):
            state.pop("master_param", None)
        elif master_mode > 0 and "master_param" not in state and HAS_NUMPY and np is not None:
            state["master_param"] = np.array(p_for_step, dtype=np.float32, copy=True)

    return state
