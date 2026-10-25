from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

class PIIMasker:
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    _PHONE_RE = re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")

    def __init__(
        self,
        mask_email: bool = True,
        mask_ip: bool = True,
        mask_phone: bool = True,
        custom_patterns: Optional[Sequence[Tuple[str, str]]] = None,
        mask_token: Optional[str] = None,
    ) -> None:
        self.mask_email = mask_email
        self.mask_ip = mask_ip
        self.mask_phone = mask_phone
        self.mask_token = mask_token
        self.custom_compiled: List[Tuple[re.Pattern[str], str]] = []
        if custom_patterns:
            for pat, repl in custom_patterns:
                self.custom_compiled.append((re.compile(pat), repl))

    def sanitize(self, text: str) -> str:
        out = text
        if self.mask_email:
            repl = self.mask_token if self.mask_token is not None else "[EMAIL]"
            out = self._EMAIL_RE.sub(repl, out)
        if self.mask_ip:
            repl = self.mask_token if self.mask_token is not None else "[IP_ADDRESS]"
            out = self._IPV4_RE.sub(repl, out)
        if self.mask_phone:
            repl = self.mask_token if self.mask_token is not None else "[PHONE]"
            out = self._PHONE_RE.sub(repl, out)
        for reg, repl in self.custom_compiled:
            out = reg.sub(repl, out)
        return out

    def sanitize_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.sanitize(t) for t in texts]

    def mask(self, text: str) -> str:
        return self.sanitize(text)

    def mask_batch(self, texts: Sequence[str]) -> List[str]:
        return self.sanitize_batch(texts)

__all__ = ["PIIMasker"]
