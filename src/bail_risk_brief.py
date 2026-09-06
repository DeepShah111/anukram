# Bail-Risk Brief: turns a resolved offender's cross-case history plus deadline status into a one-screen bail brief.

from src.entity_graph import get_offender_view
from src.statutory_watch import watch_status
from src.config import logger

# Case metadata the brief needs (arrest date + sections) keyed by case_id.
def _case_facts(case_entity_index, case_id):
    # Return the sections and arrest date recorded for a case.
    rec = case_entity_index.get(case_id, {})
    return rec.get("sections", []), rec.get("arrest_date")


def build_bail_brief(graph, case_entity_index, entity_id, current_case_id, today):
    # Assemble the bail-risk brief for one accused as of a given date (e.g. the 2020 hearing).
    view = get_offender_view(graph, entity_id)
    if not view:
        return {"found": False, "reason": f"No resolved record for {entity_id}."}

    prior_cases = [c for c in view["timeline"] if c["case_id"] != current_case_id]
    sections, arrest_date = _case_facts(case_entity_index, current_case_id)
    watch = watch_status(arrest_date, sections, today=today)

    # Risk level: repeat offender across districts is HIGH; a single prior is MEDIUM; none is LOW.
    if view["repeat_offender"] and len(view["districts"]) >= 2:
        risk = "HIGH"
        recommendation = "Oppose bail. Accused shows a repeat pattern across multiple jurisdictions."
    elif len(prior_cases) >= 1:
        risk = "MEDIUM"
        recommendation = "Bail to be considered with caution; prior case(s) on record."
    else:
        risk = "LOW"
        recommendation = "No prior linked cases on record."

    brief = {
        "found": True,
        "as_of": today,
        "entity_id": entity_id,
        "current_case": current_case_id,
        "name_variants": view["name_variants"],
        "prior_case_count": len(prior_cases),
        "districts": view["districts"],
        "sections_seen": view["sections"],
        "repeat_offender": view["repeat_offender"],
        "statute": "BNS 71" if view["repeat_offender"] else "",
        "watch": watch,
        "risk_level": risk,
        "recommendation": recommendation,
        "prior_cases": prior_cases,
    }
    logger.info("Bail brief for %s: risk=%s, priors=%d.", entity_id, risk, len(prior_cases))
    return brief


def format_brief(brief):
    # Render the brief as readable text for the console/UI.
    if not brief.get("found"):
        return brief.get("reason", "No record.")

    lines = []
    lines.append("=" * 60)
    lines.append(f"BAIL-RISK BRIEF  (as of {brief['as_of']})")
    lines.append(f"Accused (resolved): {brief['entity_id']}   Current case: {brief['current_case']}")
    lines.append(f"Known name spellings: {', '.join(brief['name_variants'])}")
    lines.append("-" * 60)
    lines.append(f"Prior linked cases: {brief['prior_case_count']}  across districts: {', '.join(brief['districts'])}")
    for c in brief["prior_cases"]:
        lines.append(f"    {c['year']}  {c['case_id']}  ({c['district']})")
    if brief["repeat_offender"]:
        lines.append(f"REPEAT OFFENDER - {brief['statute']} applies")
    lines.append("-" * 60)
    w = brief["watch"]
    if w.get("applicable"):
        lines.append(f"Chargesheet deadline: {w['due_date']}  ({w['days_left']} days left)  [{w['status']}]  ({w['basis']})")
    lines.append("-" * 60)
    lines.append(f"RISK LEVEL: {brief['risk_level']}")
    lines.append(f"Recommendation: {brief['recommendation']}")
    lines.append("(Victim identities protected under BNS 72. Links are investigative leads, not proof of guilt.)")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import os, json
    from src.config import ARTIFACTS_DIR
    from src.entity_resolution import resolve_entities
    from src.entity_graph import build_entity_graph

    with open(os.path.join(ARTIFACTS_DIR, "case_entities.json"), encoding="utf-8") as f:
        index = json.load(f)
    graph = build_entity_graph(index, resolve_entities(index))

    # Find the resolved id for the 2020 case and build its brief as of the 2020 hearing.
    target_case = "FIR/2020/DEL-D/00234"
    entity_id = next(e for e, r in graph.items()
                     if any(c["case_id"] == target_case for c in r["timeline"]))
    brief = build_bail_brief(index, index, entity_id, target_case, today="2020-07-20") \
        if False else build_bail_brief(graph, index, entity_id, target_case, today="2020-07-20")
    print(format_brief(brief))