import re
import yaml
import pandas as pd
import unicodedata
from hapcancer.etl.transform.process_birads.re_searches import *

def _normalize_val(raw):
    raw = raw.strip().upper()
    suffix = ""
    if raw.endswith(("A","B","C")):
        raw, suffix = raw[:-1], raw[-1]
    return ROMAN_MAP.get(raw, raw) + suffix

def _line_containing(text, idx):
    start = text.rfind('\n', 0, idx)
    end = text.find('\n', idx)
    start = 0 if start == -1 else start + 1
    end = len(text) if end == -1 else end
    line = text[start:end].strip()
    if not line or len(line) < 5:
        # fallback to sentence end if needed
        period_end = text.find('.', idx)
        if period_end != -1:
            line = text[start:period_end+1].strip()
    return line

def extract_birads_v1(text):
    # 1) Direct "BI-RADS ... <val>" style — iterate and skip edition-like matches
    for m in RE_DIRECT.finditer(text):
        clause = m.group(0).strip()

        # Extra safety: if the clause still contains 'edição/edition' near the value, skip
        tail = text[m.end('val'): m.end('val') + 16]  # look a bit ahead
        if re.search(r'(?iu)^\s*[ºª]', tail) or re.search(r'(?iu)\b(ed(?:\.|i[cç][aã]o)|edition)\b', clause):
            continue

        return {"value": _normalize_val(m.group("val")), "clause": clause}

    # 2) Header contains BIRADS; value nearby as "Classe/Categoria/Classificação <val>"
    for hdr in re.finditer(r'(?is)bi[\s-]?rads', text):
        window = text[hdr.end(): hdr.end() + 600]   # look ahead within section
        mv = RE_NEARBY_VALUE.search(window)
        if mv:
            val = _normalize_val(mv.group("val"))
            clause = _line_containing(text, hdr.end() + mv.start())
            return {"value": val, "clause": clause}

    # 3) Relaxed fallback: if the doc contains BIRADS anywhere, accept the first "Classe/Categoria/… <val>"
    if re.search(r'(?is)bi[\s-]?rads', text):
        mv = RE_NEARBY_VALUE.search(text)
        if mv:
            return {"value": _normalize_val(mv.group("val")), "clause": _line_containing(text, mv.start())}

    return None

#def extract_birads_v1(text):
#    # 1) Direct "BI-RADS: <val>" style
#    m = RE_DIRECT.search(text)
#    if m:
#        return {"value": _normalize_val(m.group("val")), "clause": m.group(0).strip()}
#
#    # 2) Header contains BIRADS; value nearby as "Classe/Categoria/Classificação <val>"
#    for hdr in re.finditer(r'(?is)bi[\s-]?rads', text):
#        window = text[hdr.end(): hdr.end()+600]   # look ahead within section
#        mv = RE_NEARBY_VALUE.search(window)
#        if mv:
#            val = _normalize_val(mv.group("val"))
#            clause = _line_containing(text, hdr.end() + mv.start())
#            return {"value": val, "clause": clause}
#
#    # 3) Relaxed fallback: if the doc contains BIRADS anywhere, accept the first "Classe/Categoria/… <val>"
#    if re.search(r'(?is)bi[\s-]?rads', text):
#        mv = RE_NEARBY_VALUE.search(text)
#        if mv:
#            return {"value": _normalize_val(mv.group("val")), "clause": _line_containing(text, mv.start())}
#
#    return None

def _rank(val):
    # for choosing the largest BI-RADS
    base = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6}
    v = val.upper()
    if v.startswith("4") and len(v) == 2 and v[1] in "ABC":
        return 4 + {"A":0.1, "B":0.2, "C":0.3}[v[1]]
    return float(base.get(v, -1))


def _guess_side(text, span_start, span_end):
    window_start = max(0, span_start - 80)
    window_end = min(len(text), span_end + 80)
    window = text[window_start:window_end]
    m = re.search(r'(?iu)\bmama\s+(direita|esquerda)\b', window)
    if not m:
        return None
    return {"direita":"right", "esquerda":"left"}.get(m.group(1).lower())

def extract_birads_v2(text):
    candidates = []

    # Pass 1: richer pattern that also catches "Categoria: Mama Direita ..."
    for m in RE_FLEX.finditer(text):
        val = _normalize_val(m.group("val"))
        clause = _line_containing(text, m.start("val"))
        side = m.group("side")
        if side:
            side = {"direita":"right", "esquerda":"left"}[side.lower()]
        else:
            side = _guess_side(text, m.start(), m.end())
        candidates.append({"value": val, "clause": clause, "side": side, "rank": _rank(val)})

    # Pass 2: bare lines like "BI - RADS 3" (e.g., after "I.D.: ...")
    if not candidates:
        for m in RE_BARE.finditer(text):
            val = _normalize_val(m.group("val"))
            clause = _line_containing(text, m.start("val"))
            side = _guess_side(text, m.start(), m.end())
            candidates.append({"value": val, "clause": clause, "side": side, "rank": _rank(val)})

    if not candidates:
        return None

    # If there are multiple (e.g., right & left), choose the largest
    best = max(candidates, key=lambda d: d["rank"])
    return {"value": best["value"], "clause": best["clause"]}

# ----- check whether the exam is for the breast ------

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _ci(pat):  # case-insensitive on already accent-stripped text
    return re.compile(pat, re.IGNORECASE)

def _line_containing_for_breast(text, idx):
    start = text.rfind('\n', 0, idx)
    end = text.find('\n', idx)
    start = 0 if start == -1 else start + 1
    end = len(text) if end == -1 else end
    return text[start:end].strip()

POS_ANCHORS = [_ci(_strip_accents(p)) for p in POS_ANCHORS]
POS_MILD    = [_ci(_strip_accents(p)) for p in POS_MILD]
NEG_STRONG  = [_ci(_strip_accents(p)) for p in NEG_STRONG]
NEG_MILD    = [_ci(_strip_accents(p)) for p in NEG_MILD]

def _scan(norm_text, patterns, weight):
    hits = []
    for rx in patterns:
        for m in rx.finditer(norm_text):
            hits.append({"idx": m.start(), "weight": weight})
    return hits

def is_breast_exam(text: str):
    """
    Decide if a report is a BREAST exam (for records where BI-RADS wasn't extracted).
    Returns {'is_breast': bool, 'evidence': str, 'score': int}.
    """
    if not text or not text.strip():
        return {"is_breast": False, "evidence": None, "score": 0}

    raw = text
    norm = _strip_accents(raw.lower())

    hits_anchor = _scan(norm, POS_ANCHORS, 3)
    hits_pmild  = _scan(norm, POS_MILD,   1)
    hits_nstrong= _scan(norm, NEG_STRONG, -3)
    hits_nmild  = _scan(norm, NEG_MILD,   -1)

    score = sum(h["weight"] for h in hits_anchor + hits_pmild + hits_nstrong + hits_nmild)

    # Decision rules:
    # 1) Prefer breast if we see a CLEAR anchor and no overwhelming negatives.
    anchor_present = bool(hits_anchor)
    neg_total = sum(h["weight"] for h in hits_nstrong + hits_nmild)  # negative sum is <= 0

    is_breast = (
        (anchor_present and score >= 1)                  # anchor + net positive
        or (anchor_present and neg_total >= -2)          # anchor and not strongly contradicted
        or (not anchor_present and score >= 3)           # many mild positives, no strong breast token
    )

    # Evidence line: best positive if breast, otherwise best negative
    if is_breast:
        best = (hits_anchor or hits_pmild)
    else:
        best = (hits_nstrong or hits_nmild)
    evidence = _line_containing_for_breast(raw, best[0]["idx"]) if best else None

    return {"is_breast": bool(is_breast), "evidence": evidence, "score": score}

def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

