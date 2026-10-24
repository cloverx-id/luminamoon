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
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state = ParamStateDict(state_dict["state"])
        self.param_groups = state_dict["param_groups"]
