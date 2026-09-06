# Builds the cross-case view for a resolved person: linked cases, districts, timeline, and BNS 71 flag.

import os
import json
from src.config import ARTIFACTS_DIR, logger

REPEAT_OFFENDER_MIN_CASES = 2        # BNS 71 concern applies once the same person appears in 2+ cases


def _year_from_case_id(case_id):
    # Pull the year out of a case id like FIR/2017/GGN-B/00456.
    for part in case_id.replace("_", "/").split("/"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def build_entity_graph(case_entity_index, resolution):
    # Assemble per-person records (cases, districts, sections, timeline, repeat flag) from resolved clusters.
    graph = {}
    for entity_id, case_ids in resolution["clusters"].items():
        cases, districts, sections, all_names = [], set(), set(), set()
        for cid in case_ids:
            rec = case_entity_index.get(cid, {})
            jm = rec.get("judgment_meta", {})
            cases.append({"case_id": cid, "district": rec.get("district", "unknown"),
                          "year": jm.get("year") or _year_from_case_id(cid),
                          "court": jm.get("court", ""), "outcome": jm.get("outcome", ""),
                          "repeat": jm.get("repeat_offender_signal", False),
                          "sections": rec.get("sections", []),
                          "names_seen": rec.get("names", [])})
            districts.add(rec.get("district", "unknown"))
            sections.update(rec.get("sections", []))
            all_names.update(rec.get("names", []))

        cases.sort(key=lambda c: c["year"] or 0)
        is_repeat = len(case_ids) >= REPEAT_OFFENDER_MIN_CASES
        graph[entity_id] = {
            "entity_id": entity_id,
            "name_variants": sorted(all_names),
            "case_count": len(case_ids),
            "districts": sorted(districts),
            "sections": sorted(sections),
            "timeline": cases,
            "repeat_offender": is_repeat,
            "flag": "REPEAT OFFENDER - BNS 71" if is_repeat else "",
        }
        if is_repeat:
            logger.info("Repeat-offender flag: %s across %d cases in %d districts.",
                        entity_id, len(case_ids), len(districts))

    _save(graph)
    return graph


def _save(graph):
    # Persist the entity graph for the UI and downstream modules.
    path = os.path.join(ARTIFACTS_DIR, "entity_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    logger.info("Entity graph saved: %d persons -> %s", len(graph), path)


def get_offender_view(graph, entity_id):
    # Return one person's full cross-case record (used by the CONNECT screen and the bail brief).
    return graph.get(entity_id)


if __name__ == "__main__":
    from src.entity_resolution import resolve_entities
    with open(os.path.join(ARTIFACTS_DIR, "case_entities.json"), encoding="utf-8") as f:
        index = json.load(f)
    resolution = resolve_entities(index)
    graph = build_entity_graph(index, resolution)
    for entity_id, rec in graph.items():
        tag = f"  [{rec['flag']}]" if rec["repeat_offender"] else ""
        print(f"{entity_id}: {rec['case_count']} case(s) across {rec['districts']}{tag}")
        for c in rec["timeline"]:
            print(f"     {c['year']}  {c['case_id']}  ({c['district']})")