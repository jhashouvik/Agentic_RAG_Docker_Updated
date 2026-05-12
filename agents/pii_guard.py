"""
PII masking and prompt-injection guard.

mask_pii()              — redact PAN, Aadhar, email, phone in any output string
check_prompt_injection() — returns True if the text looks like an injection attempt
"""
import re

# ── PII patterns ──────────────────────────────────────────────

# Indian PAN: 5 uppercase + 4 digits + 1 uppercase  (e.g. BAPPJ1346A)
_PAN = re.compile(r'\b([A-Z]{4})[A-Z]\d{4}[A-Z]\b')

# Aadhar: 12 digits, optionally space-separated in groups of 4
_AADHAR = re.compile(r'\b(\d{4})\s?\d{4}\s?\d{4}\b')

# Email: keep first 2 chars + domain
_EMAIL = re.compile(
    r'\b([a-zA-Z0-9]{2})[a-zA-Z0-9._%+\-]*(@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b'
)

# Indian mobile: optional +91/0 prefix, then 10-digit number starting 6-9
_PHONE = re.compile(r'\b(\+91[-\s]?|0)?([6-9]\d{2})\d{7}\b')

# ── Prompt-injection patterns ─────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'ignore\s+(the\s+)?above',
        r'forget\s+(everything|all|previous)',
        r'you\s+are\s+now\b',
        r'act\s+as\s+(a\s+)?(?!pdf|document|analyst)',  # allow "act as a PDF analyst"
        r'new\s+(system\s+)?prompt',
        r'jailbreak',
        r'\bDAN\s+mode\b',
        r'override\s+(your\s+)?(instructions?|rules?|guidelines?)',
        r'pretend\s+(you\s+are|to\s+be)',
        r'system\s*:\s*you\s+are',
        r'<\s*/?system\s*>',
        r'\[INST\]',
        r'###\s*(instruction|system)\b',
        r'disregard\s+(all\s+)?(previous|prior|above)',
        r'your\s+new\s+(role|persona|identity)\s+is',
    ]
]


def mask_pii(text: str) -> str:
    """Redact PAN, Aadhar, email, and phone numbers from display text."""
    # PAN: show first 4 chars, mask remaining 6 → BAPPXXXXX
    text = _PAN.sub(lambda m: m.group(1) + 'XXXXX', text)
    # Aadhar: show first 4 digits, mask remaining 8 → 4378XXXXXX
    text = _AADHAR.sub(lambda m: m.group(1) + 'XXXXXXXX', text)
    # Email: show first 2 chars + domain → sh****@gmail.com
    text = _EMAIL.sub(lambda m: m.group(1) + '****' + m.group(2), text)
    # Phone: show prefix + first 3 digits, mask last 7 → +91 98XXXXXXX
    text = _PHONE.sub(
        lambda m: (m.group(1) or '') + m.group(2) + 'XXXXXXX', text
    )
    return text


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Returns (is_injection, reason).
    Call before passing user input to the LLM.
    """
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Blocked pattern: `{match.group(0)}`"
    return False, ""