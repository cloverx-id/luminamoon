from __future__ import annotations

from collections.abc import Sequence
import re


class PIIMasker:
    """High-performance pattern sanitizer for Personally Identifiable Information (PII).

    Sanitizes:
    - Email addresses (`[EMAIL]`)
    - IPv4 addresses (`[IP]`)
    - Bearer tokens & API Keys (OpenAI, HuggingFace, GitHub, AWS, etc.) (`[API_KEY]`)
    - Credit card numbers (`[CARD]`)
    - Telephone numbers (`[PHONE]`)
    """

    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
    _IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|(?:::ffff:)?(?:\d{1,3}\.){3}\d{1,3}\b")
    _API_KEY_RE = re.compile(r"\b(?:Bearer\s+[A-Za-z0-9_\-\.]{20,}|(?:hf|ghp|gho|sk|glpat|AKIA)[A-Za-z0-9_\-]{16,})\b")
    _CARD_RE = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
    _PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b")

    def __init__(
        self,
        mask_emails: bool = True,
        mask_ips: bool = True,
        mask_api_keys: bool = True,
        mask_cards: bool = True,
        mask_phones: bool = False,  # False by default to avoid masking general numbers
    ) -> None:
        self.mask_emails = mask_emails
        self.mask_ips = mask_ips
        self.mask_api_keys = mask_api_keys
        self.mask_cards = mask_cards
        self.mask_phones = mask_phones

    def mask(self, text: str) -> str:
        """Sanitizes PII from a single text string."""
        if not text:
            return ""

        s = text
        if self.mask_api_keys:
            s = self._API_KEY_RE.sub("[API_KEY]", s)
        if self.mask_emails:
            s = self._EMAIL_RE.sub("[EMAIL]", s)
        if self.mask_cards:
            s = self._CARD_RE.sub("[CARD]", s)
        if self.mask_ips:
            s = self._IPV4_RE.sub("[IP]", s)
            s = self._IPV6_RE.sub("[IP]", s)
        if self.mask_phones:
            s = self._PHONE_RE.sub("[PHONE]", s)
        return s

    def mask_batch(self, texts: Sequence[str]) -> list[str]:
        """Sanitizes PII from a batch of text strings."""
        return [self.mask(t) for t in texts]

    def __call__(self, text: str) -> str:
        return self.mask(text)


def mask_pii(
    text: str,
    mask_emails: bool = True,
    mask_ips: bool = True,
    mask_api_keys: bool = True,
    mask_cards: bool = True,
    mask_phones: bool = False,
) -> str:
    """Convenience helper to sanitize PII from a text."""
    masker = PIIMasker(
        mask_emails=mask_emails,
        mask_ips=mask_ips,
        mask_api_keys=mask_api_keys,
        mask_cards=mask_cards,
        mask_phones=mask_phones,
    )
    return masker.mask(text)


__all__ = ["PIIMasker", "mask_pii"]
