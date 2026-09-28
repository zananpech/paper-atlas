import re


PROMPT_INJECTION_PATTERNS = (
    re.compile(r"ignore (all |your |previous )?instructions", re.I),
    re.compile(r"reveal (your |the )?(system prompt|api key|secret)", re.I),
    re.compile(r"send (the |your )?(api key|secret|credentials)", re.I),
)


class IngestionGuardrail():
    def __init__(self):
        self.injection_patterns = PROMPT_INJECTION_PATTERNS

    def has_suspicious_instructions(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.injection_patterns)