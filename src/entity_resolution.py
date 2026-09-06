# CONNECT engine: normalizes and matches names, phones, and vehicles to resolve the same person across cases.

import re
import json
from rapidfuzz import fuzz
import jellyfish
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from src.config import logger

# Confidence settings. A shared phone/vehicle is strong evidence; name-only needs a high fuzzy score.
NAME_MATCH_THRESHOLD = 82
STRONG_ID_WEIGHT = 1.0
NAME_WEIGHT = 0.6


def normalize_phone(phone):
    # Reduce any phone format to the last 10 digits (drops +91, spaces, dashes, leading 0).
    digits = re.sub(r"\D", "", phone)
    return digits[-10:] if len(digits) >= 10 else ""


def normalize_vehicle(vehicle):
    # Uppercase, strip separators, and fix common OCR confusions so DL0lAB1234 == DL-01-AB-1234.
    v = vehicle.upper().replace(" ", "").replace("-", "")
    v = v.replace("O", "0").replace("L", "1").replace("I", "1").replace("S", "5")
    return v


def _to_latin(name):
    # Convert Devanagari names to Latin so सुरेश and Suresh can be compared.
    if re.search(r"[\u0900-\u097F]", name):
        try:
            return transliterate(name, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
        except Exception:
            return name.lower()
    return name.lower()


def normalize_name(name):
    # Lowercase, drop punctuation, and transliterate Hindi to Latin.
    latin = _to_latin(name)
    latin = re.sub(r"[^a-z\s]", " ", latin)
    return re.sub(r"\s+", " ", latin).strip()


def name_similarity(name_a, name_b):
    # Blend token-set fuzzy match with a phonetic (metaphone) check; return 0-100.
    a, b = normalize_name(name_a), normalize_name(name_b)
    if not a or not b:
        return 0.0
    fuzzy = fuzz.token_set_ratio(a, b)
    phonetic = 100.0 if jellyfish.metaphone(a) == jellyfish.metaphone(b) else 0.0
    return max(fuzzy, phonetic)


def _entity_sets(case_record):
    # Build normalized phone, vehicle, and name sets for one case.
    phones = {normalize_phone(p) for p in case_record.get("phones", [])}
    vehicles = {normalize_vehicle(v) for v in case_record.get("vehicles", [])}
    names = [n for n in case_record.get("names", []) if n.strip()]
    return {p for p in phones if p}, {v for v in vehicles if v}, names


def match_score(case_a, case_b):
    # Score how likely two cases involve the same person, with the reasons that fired.
    phones_a, vehicles_a, names_a = _entity_sets(case_a)
    phones_b, vehicles_b, names_b = _entity_sets(case_b)

    score = 0.0
    reasons = []

    # Shared phone or vehicle is strong, near-certain evidence.
    if phones_a & phones_b:
        score += STRONG_ID_WEIGHT
        reasons.append(f"shared phone {sorted(phones_a & phones_b)}")
    if vehicles_a & vehicles_b:
        score += STRONG_ID_WEIGHT
        reasons.append(f"shared vehicle {sorted(vehicles_a & vehicles_b)}")

    # Name similarity adds weaker, supporting evidence.
    best_name = 0.0
    for na in names_a:
        for nb in names_b:
            best_name = max(best_name, name_similarity(na, nb))
    if best_name >= NAME_MATCH_THRESHOLD:
        score += NAME_WEIGHT * (best_name / 100.0)
        reasons.append(f"name match {best_name:.0f}")

    return score, reasons


def resolve_entities(case_entity_index):
    # Cluster cases that share a person using union-find over high-confidence matches.
    case_ids = list(case_entity_index.keys())
    parent = {cid: cid for cid in case_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    links = []
    # Compare every case pair once; link the pair when evidence clears the confidence bar.
    for i in range(len(case_ids)):
        for j in range(i + 1, len(case_ids)):
            a, b = case_ids[i], case_ids[j]
            score, reasons = match_score(case_entity_index[a], case_entity_index[b])
            if score >= STRONG_ID_WEIGHT:
                union(a, b)
                links.append({"case_a": a, "case_b": b,
                              "confidence": round(min(score, 1.0), 2), "reasons": reasons})

    # Give each cluster a stable resolved entity id.
    clusters = {}
    for cid in case_ids:
        clusters.setdefault(find(cid), []).append(cid)

    resolved = {}
    for n, (_, members) in enumerate(sorted(clusters.items()), 1):
        entity_id = f"RES-{n:05d}"
        for cid in members:
            resolved[cid] = entity_id

    logger.info("Entity resolution: %d cases -> %d distinct persons.", len(case_ids), len(clusters))
    return {"case_to_entity": resolved, "links": links,
            "clusters": {v: [c for c in resolved if resolved[c] == v] for v in set(resolved.values())}}


if __name__ == "__main__":
    import os
    from src.config import ARTIFACTS_DIR
    with open(os.path.join(ARTIFACTS_DIR, "case_entities.json"), encoding="utf-8") as f:
        index = json.load(f)
    result = resolve_entities(index)
    for entity_id, cases in result["clusters"].items():
        print(entity_id, "->", cases)