# Extracts accused, sections, court, year, outcome, and repeat-offender signals from real judgment prose.

import re

# Party title: "State ... vs X", "X vs State ...".
_VS_RE = re.compile(r'([A-Z][A-Za-z.\s@&]{2,50})\s+(?:vs\.?|versus)\s+([A-Z][A-Za-z.\s@&]{2,50})', re.IGNORECASE)
_SECTION_RE = re.compile(r'(?:Section|Sec\.?|u/s|under section)\s*([0-9]{2,4}\s?[A-Z]{0,2})(?![A-Za-z])', re.IGNORECASE)
_COURT_RE = re.compile(r'(Supreme Court|High Court|Sessions Court|District Court|Special Court|Magistrate)', re.IGNORECASE)

# Repeat-offender / prior-conviction signals — the heart of real-data CONNECT.
_REPEAT_SIGNALS = [
    r'376\s?E', r'previously convicted', r'prior conviction', r'earlier conviction',
    r'repeat offender', r'habitual offender', r'previous conviction', r'section 75 of',
]
_REPEAT_RE = re.compile("|".join(_REPEAT_SIGNALS), re.IGNORECASE)

# Outcome signals.
_CONVICT_RE = re.compile(r'\b(convicted|conviction is upheld|sentence(?:d)?|guilty)\b', re.IGNORECASE)
_ACQUIT_RE = re.compile(r'\b(acquitted|acquittal|benefit of doubt)\b', re.IGNORECASE)

_STOPWORDS = {"the", "state", "union", "of", "india", "others", "anr", "ors", "and", "ministry", "home", "affairs"}


def _clean_party(name):
    # Trim generic words so a party like "State of Maharashtra" doesn't become an accused person.
    name = re.sub(r'\s+', ' ', name).strip(" .")
    name = name.split("@")[0].strip()                      # drop aliases after '@' for the primary name
    tokens = [t for t in name.split() if t.lower() not in _STOPWORDS]
    return " ".join(tokens).strip()


def extract_accused(text):
    # Return the non-State party from the FIRST clean "X vs Y" in the title area only.
    head = text[:1500]                                  # title is near the very top
    for a, b in _VS_RE.findall(head):
        a_state = "state" in a.lower() or "union" in a.lower()
        b_state = "state" in b.lower() or "union" in b.lower()
        cand = _clean_party(b) if (a_state and not b_state) else \
               _clean_party(a) if (b_state and not a_state) else _clean_party(a)
        # Reject captures that begin mid-sentence (lowercase) or are too long to be a name.
        if 3 <= len(cand) <= 40 and cand[:1].isupper() and len(cand.split()) <= 5:
            return cand
    return ""


def extract_aliases(text):
    # Capture "@ Nanu" style aliases the accused is also known by.
    aliases = re.findall(r'@\s*([A-Z][A-Za-z]+)', text[:3000])
    return sorted(set(a.strip() for a in aliases))


def detect_outcome(text):
    # Rough conviction/acquittal read from the judgment body.
    convicted = len(_CONVICT_RE.findall(text))
    acquitted = len(_ACQUIT_RE.findall(text))
    if convicted == 0 and acquitted == 0:
        return "unclear"
    return "convicted" if convicted >= acquitted else "acquitted"


def detect_repeat_offender(text):
    # True only when the judgment describes an ACTUAL prior conviction of the accused,
    # not merely quoting or discussing the repeat-offender law (avoids false positives).
    strong = [
        r'previously convicted', r'prior conviction of the accused',
        r'earlier conviction', r'the accused was convicted (?:earlier|previously)',
        r'in view of the (?:prior|previous) conviction', r'habitual offender',
        r'accused.{0,40}previous conviction',
    ]
    strong_re = re.compile("|".join(strong), re.IGNORECASE)
    hits = sorted(set(m.group(0).strip() for m in strong_re.finditer(text)))

    # A judgment that only defines/quotes the law (like a policy judgment) mentions 376E
    # but has none of the "this accused was previously convicted" phrasings.
    return (len(hits) > 0), hits


def parse_judgment(text):
    # Pull all structured fields from one judgment's text.
    accused = extract_accused(text)
    raw_sections = {re.sub(r'\s+', '', s).strip().upper() for s in _SECTION_RE.findall(text[:12000])}
    sections = sorted(s for s in raw_sections if re.match(r'^\d{2,4}[A-Z]{0,2}$', s))
    court_match = _COURT_RE.search(text[:4000])
    court = court_match.group(1).title() if court_match else "Unknown Court"
    years = sorted({int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text[:6000])})
    year = years[-1] if years else None                    # the judgment year is usually the latest near the top
    is_repeat, repeat_hits = detect_repeat_offender(text)
    return {
        "accused": accused,
        "aliases": extract_aliases(text),
        "sections": sections,
        "court": court,
        "year": year,
        "outcome": detect_outcome(text),
        "repeat_offender_signal": is_repeat,
        "repeat_evidence": repeat_hits,
    }


if __name__ == "__main__":
    import sys
    from langchain_community.document_loaders import PyPDFLoader
    text = " ".join(p.page_content for p in PyPDFLoader(sys.argv[1]).load())
    from pprint import pprint
    pprint(parse_judgment(text))