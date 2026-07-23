import re

ROMAN_MAP = {"I":"1","II":"2","III":"3","IV":"4","V":"5","VI":"6"}

# --- Direct pattern, now guarded against editions right after the number ---
RE_DIRECT = re.compile(
    r"""(?ixu)
    (                                   # full clause
      [^.:\n]*
      (?:categoria\s*)?
      bi[\s-]?rads(?:®)?                # BI-RADS / BIRADS / BI RADS
      [^0-9ivxabc\n]*
      [:;\-\s]*
      (?P<val>
         [0-6](?:[ABC])?                # 0–6 (with 4A/B/C)
         | [IVX]{1,3}(?:[ABC])?         # I–VI (with A/B/C)
      )
      (?!\s*[ºª])                       # <-- DO NOT allow ordinal marks (e.g., 5ª)
      (?!\s*(?:ed(?:\.|i[cç][aã]o)|edition)\b)  # <-- DO NOT allow 'ed.', 'edição', 'edition'
      [^.\n]*
    )
    """
)

# -- When "BIRADS" appears as a header and the value is on the next line(s)
RE_NEARBY_VALUE = re.compile(
    r"""(?ixu)
    (?:classe|categoria|classifica[cç][aã]o)\s*[:\-]?\s*
    (?P<val>[0-6](?:[ABC])?|[IVX]{1,3}(?:[ABC])?)
    [^.\n]*
    """
)

# -- flexible BI-RADS token: BI RADS / BI-RADS / BI - RADS / BI–RADS / BIRADS (+ optional ®)
BI_TOKEN = r"""b[i1l]\s*(?:[-–—]?\s*)?ra[h]?d[sz](?:®)?""" # (typos are allowed)

# -- core pattern: optional "Categoria/Classe/Classificação", optional side, BI_TOKEN, then value
RE_FLEX = re.compile(
    rf"""(?ixu)
    (?P<full>
      [^\n.]*?
      (?:
         (?:categoria|classe|classifica[cç][aã]o)\s*[:\-]?\s*
      )?
      (?:mama\s+(?P<side>direita|esquerda)\s*)?
      {BI_TOKEN}
      [^0-9ivxabc\n]*[:;\-–—\s]*
      (?P<val>[0-6](?:[ABC])?|[IVX]{1,3}(?:[ABC])?)
      [^\n.]*
    )
    """
)


# Fallback when the line is just "... BI - RADS 3" without leading keywords
RE_BARE = re.compile(
    rf"""(?ixu)
    (?P<full>
      [^\n.]*?
      {BI_TOKEN}
      [^0-9ivxabc\n]*[:;\-–—\s]*
      (?P<val>[0-6](?:[ABC])?|[IVX]{1,3}(?:[ABC])?)
      [^\n.]*
    )
    """
)

# -------------------------------------------------------------------

# -- strong anchors: must match at least one of these to confidently call "breast"
POS_ANCHORS = [
    r'\bmamas?\b', r'\bmamaria?s?\b', r'\bmamario?s?\b',
    r'\bmamograf(?:ia|ico?s?)\b',                           # mamografia/mamográfico
    r'\bultra(?:s)?sonograf(?:ia|ico)?\s+(das?\s+)?mamas?\b',
    r'\bbi\s*[-–—]?\s*ra[h]?d[sz]\b',                       # BI-RADS token even without a value
    r'\bmama\s+(direita|esquerda)\b',
]
# -- mild positives (helpful but not sufficient alone)
POS_MILD = [
    r'\bretroareolar\b', r'\bareolas?\b', r'\bmamilos?\b',
    r'\bfibroglandular\b', r'\bretromamari[oa]\b',
    r'\baxila(?:s)?\b',
    r'\bq(?:se|sd|ie|id|sl|il)\b',                          # QSE/QSD/QIE/QID/QSL/QIL
    r'\b(supero|infero)(externo|interno)\b',               # quadrants written out
    r'\bginecomastia\b',
    r'\bCC\b\b|\bMLO\b',                                    # mammography projections
]

# ----- Non-breast negatives (broad) -----
NEG_STRONG = [
    # Neck/Salivary/Thyroid
    r'\btireoide\b', r'\btiroide\b', r'\bparatireoide\b',
    r'\bparotidas?\b', r'\bsubmandibular(?:es)?\b',
    r'\bcarotidas?\b', r'\bregi[aã]o\s+cervical\b', r'\bcervical\b',
    # Abdomen & Pelvis
    r'\babdome\b', r'\babdominal\b', r'\bpelve\b', r'\bp[ée]lvic[ao]\b',
    r'\butero\b', r'\bendomet(rio|rial)\b', r'\bov[aá]rios?\b', r'\banexos?\b',
    r'\bprostata\b', r'\bves[ií]cula\s+seminal\b', r'\bbexiga\b',
    r'\brins?\b|\brenal\b|\bureter(?:es)?\b',
    r'\bfigado\b|\bhepatic[oa]\b', r'\bba[cç]o\b|\bespl[ée]n\w+\b', r'\bpancreas\b',
    r'\bves[ií]cula\s+biliar\b|\bcolecisto\b|\bcoledoco\b',
    r'\badrenais?\b|\bsuprarrenais?\b',
    # Obstetric
    r'\bobst[eé]trico?\b', r'\bfetal\b', r'\bgesta[cç][aã]o\b', r'\bgravidez\b',
    # Vascular Doppler
    r'\bdoppler\b', r'\bvenos[oa]\b', r'\barterial\b', r'\bmmii\b', r'\bmmss\b',
    # MSK / Joints / Spine
    r'\bombro\b', r'\bjoelho\b', r'\bquadril\b', r'\bcotovelo\b', r'\bpunho\b',
    r'\bm[aã]o\b', r'\bp[eé]\b', r'\btornozelo\b', r'\barticula[cç][aã]o\b',
    r'\bcoluna\b', r'\blombar\b', r'\btoracica\b',
    # Chest (non-breast contexts)
    r'\btorax\b', r'\bpulm\w+\b', r'\bpleur\w+\b', r'\bmediastin\w+\b',
]
NEG_MILD = [
    r'\btesticular\b|\bescrot\w+\b|\bpeniana\b',
    r'\bcranio\b|\bencef\w+\b|\bcerebr\w+\b',
    r'\bhombro\b',  # occasional Spanish OCR in mixed datasets
]