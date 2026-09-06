# Generates the synthetic women-safety case corpus with a planted repeat offender for the demo.

import os
import json
from src.config import CORPUS_DIR, setup_environment, logger

# Same real person, shown with name variants and shared phone/vehicle across districts and years.
REPEAT_OFFENDER = {
    "entity_id": "ENT-00291",
    "cases": [
        {"case_id": "FIR/2013/DEL-A/00123", "year": 2013, "district": "South Delhi", "ps": "Saket",
         "name": "Suresh Kumar", "phone": "+91-98111-22219", "vehicle": "DL-01-AB-1234",
         "sections": ["BNS 74"], "status": "Cold - complainant relocated", "victim": "Complainant-A"},
        {"case_id": "FIR/2017/GGN-B/00456", "year": 2017, "district": "Gurugram", "ps": "Sector 29",
         "name": "S. Kumar", "phone": "098111-22219", "vehicle": "DL 01 AB 1234",
         "sections": ["BNS 69"], "status": "Closed - insufficient evidence", "victim": "Complainant-B"},
        {"case_id": "FIR/2019/FBD-C/00789", "year": 2019, "district": "Faridabad", "ps": "Central",
         "name": "सुरेश कुमार", "phone": "+919811122219", "vehicle": "DL-01-AB-1234",
         "sections": ["BNS 74"], "status": "Cold", "victim": "Complainant-C"},
        {"case_id": "FIR/2020/DEL-D/00234", "year": 2020, "district": "New Delhi", "ps": "Connaught Place",
         "name": "Suresh Kmar", "phone": "+91-98111-22219", "vehicle": "DL0lAB1234",
         "sections": ["BNS 64"], "status": "Under investigation", "arrest_date": "2020-06-10",
         "victim": "Complainant-D"},
    ],
}

# Different people who must NOT be linked to the offender (tests CONNECT precision).
DISTRACTORS = [
    {"entity_id": "ENT-00312", "case_id": "FIR/2018/DEL-E/00567", "year": 2018, "district": "West Delhi",
     "ps": "Janakpuri", "name": "Suresh Sharma", "phone": "+91-99999-11111", "vehicle": "DL-08-CD-4321",
     "sections": ["BNS 74"], "status": "Chargesheet filed", "victim": "Complainant-E"},
    {"entity_id": "ENT-00318", "case_id": "FIR/2019/DEL-F/00981", "year": 2019, "district": "East Delhi",
     "ps": "Preet Vihar", "name": "Ramesh Kumar", "phone": "+91-98111-55555", "vehicle": "DL-05-EF-7788",
     "sections": ["BNS 69"], "status": "Cold", "victim": "Complainant-F"},
]

FIR_TEMPLATE = """FIRST INFORMATION REPORT (FIR)
(Synthetic record - generated for the ANUKRAM demo. Not a real case.)

FIR No        : {case_id}
District      : {district}
Police Station: {ps}
Date of FIR   : {date}
Offence Sections (BNS): {sections}

Accused Name   : {name}
Accused Phone  : {phone}
Accused Vehicle: {vehicle}

Complainant / Victim: {victim}
Sensitivity   : Confidential (victim identity protected under BNS 72)

Brief Facts   : The complainant reported an offence under the sections listed above.
Investigation status: {status}.
{arrest_line}"""

def _render(case):
    # Fill the FIR template for a single case.
    sections = ", ".join(case["sections"])
    date = case.get("date", f"{case['year']}-06-12")
    arrest_line = f"Date of Arrest: {case['arrest_date']}" if case.get("arrest_date") else ""
    return FIR_TEMPLATE.format(
        case_id=case["case_id"], district=case["district"], ps=case["ps"], date=date,
        sections=sections, name=case["name"], phone=case["phone"], vehicle=case["vehicle"],
        victim=case["victim"], status=case["status"], arrest_line=arrest_line,
    )

def generate_corpus():
    # Write every synthetic FIR to the corpus directory plus a manifest of ground-truth entity ids.
    setup_environment()
    cases = [(c, REPEAT_OFFENDER["entity_id"]) for c in REPEAT_OFFENDER["cases"]]
    cases += [(d, d["entity_id"]) for d in DISTRACTORS]

    manifest = []
    for case, entity_id in cases:
        file_name = case["case_id"].replace("/", "_") + ".txt"
        with open(os.path.join(CORPUS_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(_render(case))
        manifest.append({
            "case_id": case["case_id"], "file": file_name, "year": case["year"],
            "district": case["district"], "true_entity_id": entity_id,
            "is_women_safety": True, "sections": case["sections"],
            "arrest_date": case.get("arrest_date"),
        })
        logger.info("Wrote synthetic case %s (%s)", case["case_id"], entity_id)

    with open(os.path.join(CORPUS_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Corpus complete: %d cases written to %s", len(manifest), CORPUS_DIR)
    return manifest

if __name__ == "__main__":
    generate_corpus()