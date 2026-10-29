"""Whitespace normalisation that preserves paragraph structure."""

import re

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _CTRL.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKS.sub("\n\n", text).strip()
