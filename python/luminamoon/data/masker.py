from __future__ import annotations

from collections.abc import Sequence
import re
from typing import overload


def _passes_luhn(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        d = ord(ch) - 48
        if idx & 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (total % 10) == 0


class PIIMasker:
    """Code-safe pattern sanitizer for Personally Identifiable Information (PII).

    Sanitizes:
    - Bearer tokens & API keys (`[API_KEY]`) using strict provider prefixes.
    - Email addresses (`[EMAIL]`).
    - Credit card numbers (`[CARD]`) with Luhn validation (13-19 digits, Visa/MC/Amex).
    - IPv4 & IPv6 addresses (`[IP]`) without corrupting C++/Rust `::` paths or version strings.
    - Optional telephone numbers (`[PHONE]`, disabled by default).
    """

    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _IPV4_RE = re.compile(
        r"(?<![A-Za-z0-9_.=])(?<!\bver\s)(?<!\bversion\s)"
        r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        r"(?![A-Za-z0-9_]|\.\d)"
    )
    # Strict IPv6 boundaries reject identifier chars around `::` (protects `std::abs`, `torch::cuda`, `dead::beef`).
    _IPV6_RE = re.compile(
        r"(?<![A-Za-z0-9_:])(?:"
        r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
        r"|::[fF]{4}:(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        r"|(?:[0-9a-fA-F]{1,4}:){2,7}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,4})?"
        r"|[0-9a-fA-F]{1,4}::(?:[0-9a-fA-F]{1,4}:){1,5}[0-9a-fA-F]{1,4}"
        r"|[fF][eEcCdD89aAbB][0-9a-fA-F]{2}::[0-9a-fA-F]{1,4}"
        r"|::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}"
        r")(?![A-Za-z0-9_:])"
    )
    _API_KEY_RE = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"
        r"|sk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{16,}"
        r"|hf_[A-Za-z0-9]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|glpat-[A-Za-z0-9_\-]{16,}"
        r"|AKIA[0-9A-Z]{16}"
        r"|AIza[0-9A-Za-z_\-]{30,}"
        r"|xox[baprs]-[0-9A-Za-z\-]{16,}"
        r"|pk_(?:live|test)_[0-9A-Za-z]{16,}"
        r")(?![A-Za-z0-9_@])"
    )
    _CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
    _PHONE_RE = re.compile(
        r"(?<!\w)(?:\+\d{1,3}[ -]?)(?:\(\d{2,4}\)|\d{2,4})[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)"
        r"|(?<!\w)\(\d{2,4}\)[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)"
    )

    def __init__(
        self,
        mask_emails: bool = True,
        mask_ips: bool = True,
        mask_api_keys: bool = True,
        mask_cards: bool = True,
        mask_phones: bool = False,
        *,
        validate_luhn: bool = True,
    ) -> None:
        self.mask_emails = mask_emails
        self.mask_ips = mask_ips
        self.mask_api_keys = mask_api_keys
        self.mask_cards = mask_cards
        self.mask_phones = mask_phones
        self.validate_luhn = validate_luhn

    def _replace_card(self, match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not (13 <= len(digits) <= 19):
            return raw
        if self.validate_luhn and not _passes_luhn(digits):
            return raw
        return "[CARD]"

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
            s = self._CARD_CANDIDATE_RE.sub(self._replace_card, s)
        if self.mask_ips:
            s = self._IPV6_RE.sub("[IP]", s)
            s = self._IPV4_RE.sub("[IP]", s)
        if self.mask_phones:
            s = self._PHONE_RE.sub("[PHONE]", s)
        return s

    def mask_batch(self, texts: Sequence[str]) -> list[str]:
        """Sanitizes PII from a batch of text strings."""
        return [self.mask(t) for t in texts]

    @overload
    def __call__(self, target: str) -> str: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__(self, target: Sequence[str]) -> list[str]: ...

    def __call__(self, target: str | Sequence[str]) -> str | list[str]:
        if isinstance(target, str):
            return self.mask(target)
        return self.mask_batch(target)


def mask_pii(
    text: str,
    mask_emails: bool = True,
    mask_ips: bool = True,
    mask_api_keys: bool = True,
    mask_cards: bool = True,
    mask_phones: bool = False,
    *,
    validate_luhn: bool = True,
) -> str:
    """Convenience helper to sanitize PII from a text."""
    return PIIMasker(
        mask_emails=mask_emails,
        mask_ips=mask_ips,
        mask_api_keys=mask_api_keys,
        mask_cards=mask_cards,
        mask_phones=mask_phones,
        validate_luhn=validate_luhn,
    ).mask(text)


__all__ = ["PIIMasker", "mask_pii"]
