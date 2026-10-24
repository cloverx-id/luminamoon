from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .validation import LuminaVValidationMixin


class LuminaVStateMixin(LuminaVValidationMixin):
    def _get_generator(
        self,
        device: torch.device,
        seed: Optional[int] = None,
        slot: int = 0,
        reseed: bool = False,
    ) -> torch.Generator:
        effective_seed = int(self.seed if seed is None else seed) & 0x7FFFFFFF
        dev_type = device.type
        if device.index is not None:
            dev_idx = device.index
        elif dev_type == "cuda" and torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
        elif dev_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            dev_idx = torch.xpu.current_device()
        elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_device_index") and torch.accelerator.is_available():
            dev_idx = torch.accelerator.current_device_index()
        else:
            dev_idx = 0
        key = (dev_type, dev_idx, slot if reseed else effective_seed)
        if key not in self._generators:
            try:
                gen = torch.Generator(device=device)
            except (RuntimeError, ValueError, TypeError):
                gen = torch.Generator(device="cpu")
            gen.manual_seed(effective_seed)
            self._generators[key] = gen
        elif reseed:
            self._generators[key].manual_seed(effective_seed)
        return self._generators[key]

    def _get_step_generators(
        self,
        device: torch.device,
        seed: int,
        step: int,
        param_index: int,
        need_m: bool,
        need_p: bool,
    ) -> Tuple[Optional[torch.Generator], Optional[torch.Generator]]:
        base_seed = (seed + step * 0x9E3779B9 + param_index * 10007) & 0x7FFFFFFF
        gen_m = self._get_generator(device, seed=base_seed, slot=0, reseed=True) if need_m else None
        seed_p = (base_seed + 0x55555555) & 0x7FFFFFFF
        gen_p = self._get_generator(device, seed=seed_p, slot=1, reseed=True) if need_p else None
        return gen_m, gen_p

    def state_dict(self) -> dict:
        raw_dict = super().state_dict()
        temp_keys = {"p_contig", "exp_avg_contig", "exp_avg_sq_contig", "mp_contig", "_plan", "_contig_step"}
        param_ids = {id(p) for group in self.param_groups for p in group.get("params", [])}
        clean_state = {}
        for k, p_state in raw_dict.get("state", {}).items():
            if isinstance(p_state, dict):
                entry = {}
                for sk, sv in p_state.items():
                    if sk in temp_keys or (sk == "master_param" and id(sv) in param_ids):
                        continue
                    if sk == "scratch" and isinstance(sv, Tensor) and sv.numel() >= 6:
                        sv_clean = sv.detach().clone()
                        sv_clean[:3].zero_()
                        sv_clean[5].zero_()
                        entry[sk] = sv_clean
                    else:
                        entry[sk] = sv
                clean_state[k] = entry
            else:
                clean_state[k] = p_state
        return {
            "state": clean_state,
            "param_groups": raw_dict.get("param_groups", []),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        raw_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        raw_param_state: Dict[Tensor, dict] = {}
        if len(self.param_groups) == len(saved_groups):
            for group, saved_group in zip(self.param_groups, saved_groups):
                for p, p_id in zip(group.get("params", []), saved_group.get("params", [])):
                    st_entry = raw_state.get(p_id) if p_id in raw_state else raw_state.get(p)
                    if isinstance(st_entry, dict):
                        raw_param_state[p] = {
                            k: (v.detach().clone() if k == "scratch" else v.detach()) if isinstance(v, Tensor) else v
                            for k, v in st_entry.items()
                        }

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group.setdefault(k, v)
            self._validate_and_normalize_group(group)
            master_mode = group.get("master_weights", "none")
            buf_count = group.get("buffer", 2)
            cautious_g = group.get("cautious", True)
            radial_g = bool(
                group.get("bound", True)
                and group.get("bound_type", "coordinate") == "radial"
                and group.get("bound_ratio", 0.05) > 0.0
            )
            for p in group["params"]:
                if p not in self.state:
                    continue
                st = self.state[p]
                raw_st = raw_param_state.get(p, {})
                for temp_key in ("p_contig", "exp_avg_contig", "exp_avg_sq_contig", "mp_contig", "_plan", "_contig_step"):
                    st.pop(temp_key, None)
                if "step" in st and isinstance(st["step"], Tensor):
                    st["step"] = int(st["step"].item())
                if "scratch" in st and isinstance(st["scratch"], Tensor):
                    src_sc = raw_st["scratch"] if ("scratch" in raw_st and isinstance(raw_st["scratch"], Tensor)) else st["scratch"]
                    old_sc = src_sc.to(device=p.device, dtype=torch.float32).contiguous()
                    if old_sc.numel() < 6:
                        new_sc = torch.zeros((6,), device=p.device, dtype=torch.float32)
                        new_sc[3] = 1.0
                        new_sc[4] = 1.0
                        new_sc[:old_sc.numel()].copy_(old_sc)
                        new_sc[:3].zero_()
                        new_sc[5].zero_()
                        st["scratch"] = new_sc
                    else:
                        old_sc[:3].zero_()
                        old_sc[5].zero_()
                        st["scratch"] = old_sc
                else:
                    new_sc = torch.zeros((6,), device=p.device, dtype=torch.float32)
                    new_sc[3] = 1.0
                    new_sc[4] = 1.0
                    st["scratch"] = new_sc
                if not cautious_g:
                    st["scratch"][3] = 1.0
                if not radial_g:
                    st["scratch"][4] = 1.0
                if master_mode == "none" or p.dtype == torch.float32:
                    st.pop("master_param", None)
                elif "master_param" in st and isinstance(st["master_param"], Tensor):
                    src_mp = (
                        raw_st["master_param"]
                        if ("master_param" in raw_st and isinstance(raw_st["master_param"], Tensor))
                        else st["master_param"]
                    )
                    mp_t = src_mp.to(device=p.device, dtype=torch.float32).contiguous()
                    st["master_param"] = mp_t.clone() if mp_t.data_ptr() == src_mp.data_ptr() else mp_t
                else:
                    st["master_param"] = p.detach().clone().float().contiguous()
                target_state_dtype = torch.float32 if master_mode == "full" else p.dtype
                for buf_key in ("exp_avg", "exp_avg_sq"):
                    if buf_key in st and isinstance(st[buf_key], Tensor):
                        src_buf = raw_st[buf_key] if (buf_key in raw_st and isinstance(raw_st[buf_key], Tensor)) else st[buf_key]
                        buf_t = src_buf.to(device=p.device, dtype=target_state_dtype).contiguous()
                        st[buf_key] = buf_t.clone() if buf_t.data_ptr() == src_buf.data_ptr() else buf_t
                if "exp_avg" not in st or not isinstance(st["exp_avg"], Tensor):
                    st["exp_avg"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                if buf_count == 1:
                    st.pop("exp_avg_sq", None)
                elif "exp_avg_sq" not in st or not isinstance(st["exp_avg_sq"], Tensor):
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=target_state_dtype, memory_format=torch.contiguous_format)
                st["buf_count"] = buf_count
