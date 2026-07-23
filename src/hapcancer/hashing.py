import hashlib
import unicodedata
import re

def canonicalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def make_id(content: str, hex_chars: int = 32) -> str:
    s = canonicalize(content)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:hex_chars]