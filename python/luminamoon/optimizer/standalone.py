from __future__ import annotations

from typing import Any, Dict


class ParamStateDict(dict):
    def _key(self, k: Any) -> Any:
        try:
            hash(k)
            return k
        except TypeError:
            return id(k)

    def __missing__(self, k: Any) -> Any:
        val: dict = {}
        super().__setitem__(k, val)
        return val

    def __getitem__(self, k: Any) -> Any:
        key = self._key(k)
        if not super().__contains__(key):
            return self.__missing__(key)
        return super().__getitem__(key)

    def __setitem__(self, k: Any, v: Any) -> None:
        super().__setitem__(self._key(k), v)

    def __contains__(self, k: Any) -> bool:
        return super().__contains__(self._key(k))

    def get(self, k: Any, default: Any = None) -> Any:
        key = self._key(k)
        if not super().__contains__(key):
            return default
        return super().get(key, default)

    def pop(self, k: Any, *args: Any) -> Any:
        return super().pop(self._key(k), *args)

    def setdefault(self, k: Any, default: Any = None) -> Any:
        key = self._key(k)
        if not super().__contains__(key):
            super().__setitem__(key, default)
            return default
        return super().__getitem__(key)


class StandaloneOptimizer:
    """Standalone lightweight base class when PyTorch is not used or installed."""
    def __init__(self, params: Any, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.state = ParamStateDict()
        self.param_groups: list[Dict[str, Any]] = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    elif hasattr(p.grad, "zero_"):
                        p.grad.zero_()
                    elif hasattr(p.grad, "fill"):
                        p.grad.fill(0)

    def state_dict(self) -> Dict[str, Any]:
        param_mappings: Dict[Any, int] = {}
        start_index = 0

        def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k not in ("params", "_validated_sig")}
            param_indices = []
            for p in group.get("params", []):
                p_key = self.state._key(p)
                if p_key not in param_mappings:
                    param_mappings[p_key] = start_index
                    start_index += 1
                param_indices.append(param_mappings[p_key])
            packed["params"] = param_indices
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        packed_state = {
            (param_mappings[k] if k in param_mappings else k): v
            for k, v in self.state.items()
        }
        return {"state": packed_state, "param_groups": param_groups}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        if len(self.param_groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        for g, sg in zip(self.param_groups, saved_groups):
            if len(g.get("params", [])) != len(sg.get("params", [])):
                raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")

        new_state = ParamStateDict()
        for g, sg in zip(self.param_groups, saved_groups):
            current_params = g.get("params", [])
            for p, saved_id in zip(current_params, sg.get("params", [])):
                p_key = new_state._key(p)
                s_key = new_state._key(saved_id)
                st_entry = (
                    saved_state[saved_id]
                    if saved_id in saved_state
                    else (
                        saved_state[s_key]
                        if s_key in saved_state
                        else saved_state.get(p_key)
                    )
                )
                if isinstance(st_entry, dict):
                    new_state[p] = {
                        k: (v.copy() if hasattr(v, "copy") else v)
                        for k, v in st_entry.items()
                    }
            for k, v in sg.items():
                if k != "params":
                    g[k] = v
        self.state = new_state
