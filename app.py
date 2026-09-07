# Gradio demo UI for ANUKRAM (real-data, with guidance panel): search, CONNECT, bail brief, audit integrity.
import spaces
# Minimal GPU function to satisfy ZeroGPU's startup requirement; the app runs on CPU otherwise.
@spaces.GPU(duration=1)
def _warmup():
    return "ok"
import os
import json
import gradio as gr

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.config import ARTIFACTS_DIR
from src.document_ingestion import load_and_chunk_documents, build_case_entity_index
from src.entity_resolution import resolve_entities
from src.entity_graph import build_entity_graph
from src.bail_risk_brief import build_bail_brief, format_brief
from src.retrieval_engine import HybridRetrievalEngine
from src.generation_agent import generate_answer
from src.shield_redaction import mask_text
from src.audit_ledger import log_event, verify_integrity

# Load the whole engine once at startup, on the real judgments in data/raw_docs.
print("Starting ANUKRAM on real judgments...")
CHUNKS = load_and_chunk_documents()
CASE_INDEX = build_case_entity_index(CHUNKS)
GRAPH = build_entity_graph(CASE_INDEX, resolve_entities(CASE_INDEX))
RETRIEVER = HybridRetrievalEngine().build(document_chunks=CHUNKS, rebuild=False)
print("ANUKRAM ready.")

ENTITY_LIST = "\n".join(
    f"{eid}   →   {', '.join(rec['name_variants'][:1]) or 'unknown'}"
    f"{'   [REPEAT OFFENDER]' if any(c.get('repeat') for c in rec['timeline']) else ''}"
    for eid, rec in GRAPH.items()
)

# Guidance shown to judges: exactly what to type and what they will see.
SUGGESTIONS = """### 🔍 Try these (real questions on real judgments)

**Search tab — ask these:**
- *What was Vijay Mohan Jadhav convicted of, and was there any prior conviction?*
  → surfaces a **real repeat offender** (§376E) with cited pages
- *What does the law say about disclosing the identity of a rape victim?*
  → answered from the **real Supreme Court** Nipun Saxena judgment
- *What were the facts and outcome in the Bhupinder Sharma case?*
  → real facts + sentence, every line cited

**CONNECT tab:**
- Enter a person id (see the list) to view the accused's record and repeat-offender flag.

**Audit tab:**
- Click **Verify Integrity** → the tamper-proof log re-checks itself live.

**SHIELD (victim protection):**
- Switch **Role** to *RTI_Public*, re-ask a question → victim identities are masked (BNS 72).

---
*Every answer is grounded in a real, published Indian court judgment and cited to its page. ANUKRAM surfaces documented history — it does not predict.*"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#f4f6fb",
    button_primary_background_fill="#1C2E5E",
    button_primary_background_fill_hover="#16244A",
    button_primary_text_color="#ffffff",
)

HEADER = """
<div style="background:#1C2E5E;padding:18px 24px;border-radius:10px;margin-bottom:6px;">
  <div style="color:#ffffff;font-size:26px;font-weight:700;">🛡️ ANUKRAM</div>
  <div style="color:#C0902A;font-size:15px;font-weight:600;letter-spacing:1px;">
     WOMEN-SAFETY INVESTIGATION INTELLIGENCE &nbsp;·&nbsp; CONNECT · WATCH · SHIELD</div>
  <div style="color:#c9d4ea;font-size:12px;margin-top:4px;">
     SIH 2026 · Team VARIANTS · PS SIH26190 · MHA / NCRB Women Safety Division · Live on real public judgments</div>
</div>
"""

def do_search(question, role):
    # Retrieve, generate a cited answer, apply SHIELD for the role, and log the search.
    if not question.strip():
        return "Please enter a question."
    docs = RETRIEVER.invoke(question)
    result = generate_answer(question, docs)
    answer = mask_text(result["answer"], role, "search")
    log_event("SEARCH", role, "multiple", question)
    sources = "\n".join(f"• {s['case_id']}  (page {s['page']})" for s in result["sources"])
    banner = "" if role in ("IO", "Prosecutor", "Court") else \
        "🔒 Viewing as RTI/Public — victim identities are masked (BNS 72).\n\n"
    return f"{banner}{answer}\n\n─── Sources considered ───\n{sources}"


def do_connect(entity_id, role):
    # Show the record for a resolved person, with court/year/outcome for real judgments.
    rec = GRAPH.get(entity_id.strip())
    if not rec:
        return f"No record for '{entity_id}'. Pick an id from the list above."
    log_event("CONNECT_VIEW", role, entity_id, "viewed record")
    out = []
    is_repeat = any(c.get("repeat") for c in rec["timeline"])
    if is_repeat:
        out.append("🚩  REPEAT OFFENDER  —  prior conviction on record (BNS 71 / §376E)")
        out.append("=" * 50)
    out.append(f"Resolved person : {rec['entity_id']}")
    out.append(f"Accused         : {', '.join(rec['name_variants'])}")
    out.append(f"Cases on record : {rec['case_count']}")
    out.append("")
    out.append("CASE RECORD:")
    for c in rec["timeline"]:
        court = c.get("court", "") or "Court"
        year = c.get("year", "") or "year n/a"
        outcome = f" · {c['outcome']}" if c.get("outcome") else ""
        out.append(f"   {year}   {c['case_id']}   ({court}{outcome})")
    out.append("")
    out.append("(Investigative leads, not proof of guilt. Confidence-scored, human-reviewed.)")
    return "\n".join(out)


def do_bail_brief(case_id, today, role):
    # Build and show the bail-risk brief for the accused in a given case.
    case_id = case_id.strip()
    entity_id = next((e for e, r in GRAPH.items()
                      if any(c["case_id"] == case_id for c in r["timeline"])), None)
    if not entity_id:
        return f"Case '{case_id}' not found. Pick one from the list above."
    brief = build_bail_brief(GRAPH, CASE_INDEX, entity_id, case_id, today=today.strip())
    log_event("BAIL_BRIEF", role, case_id, f"as of {today}")
    return format_brief(brief)


def do_verify_and_show():
    # Recompute the audit chain and return a clear status line plus the raw ledger.
    result = verify_integrity()
    status = result.get("status")
    headline = (f"✅  VERIFIED  —  {result.get('message')}" if status == "VERIFIED"
                else f"❌  TAMPERED  —  {result.get('message')}" if status == "TAMPERED"
                else f"ℹ️  {result.get('message')}")
    path = os.path.join(ARTIFACTS_DIR, "audit", "audit_ledger.jsonl")
    ledger = open(path, encoding="utf-8").read() if os.path.exists(path) else "Ledger is empty."
    return headline, ledger


CASE_LIST = "\n".join(sorted({c["case_id"] for rec in GRAPH.values() for c in rec["timeline"]}))

with gr.Blocks(title="ANUKRAM", theme=THEME) as demo:
    gr.HTML(HEADER)

    with gr.Row():
        # Left: the working tabs. Right: the guidance panel for judges.
        with gr.Column(scale=2):
            gr.Markdown("**Select your role** — controls whether victim identities are visible (SHIELD / BNS 72).")
            role = gr.Radio(choices=["IO", "Prosecutor", "Court", "RTI_Public"], value="IO", label="Role")

            with gr.Tab("🔎  Search"):
                gr.Markdown("Ask in plain English or Hindi. Every answer is cited to its source judgment and page.")
                q = gr.Textbox(label="Question",
                               placeholder="What was Vijay Mohan Jadhav convicted of, and was there any prior conviction?")
                search_btn = gr.Button("Search", variant="primary")
                search_out = gr.Textbox(label="Answer with sources", lines=14)
                search_btn.click(do_search, [q, role], search_out)

            with gr.Tab("🔗  CONNECT"):
                gr.Markdown("The accused's record — and the repeat-offender flag where the court recorded a prior conviction.")
                gr.Textbox(value=ENTITY_LIST, label="Persons in the corpus", lines=4, interactive=False)
                entity = gr.Textbox(label="Enter a person id", value="RES-00004")
                connect_btn = gr.Button("Show record", variant="primary")
                connect_out = gr.Textbox(label="Case record", lines=12)
                connect_btn.click(do_connect, [entity, role], connect_out)

            with gr.Tab("⚖️  Bail-Risk Brief"):
                gr.Markdown("The accused's documented history, assembled for a bail decision.")
                gr.Textbox(value=CASE_LIST, label="Available case ids", lines=4, interactive=False)
                case = gr.Textbox(label="Case", value="judgment_vijay_jadhav.pdf")
                today = gr.Textbox(value="2021-01-01", label="Hearing date (YYYY-MM-DD)")
                bail_btn = gr.Button("Generate brief", variant="primary")
                bail_out = gr.Textbox(label="Bail-risk brief", lines=16)
                bail_btn.click(do_bail_brief, [case, today, role], bail_out)

            with gr.Tab("🔐  Audit — Verify Integrity"):
                gr.Markdown("Every action is written to a hash-chained log. Editing any past entry breaks the chain.")
                verify_btn = gr.Button("Verify Integrity", variant="primary")
                verify_out = gr.Textbox(label="Integrity status", lines=2)
                ledger_out = gr.Textbox(label="Audit ledger (raw)", lines=10)
                verify_btn.click(do_verify_and_show, None, [verify_out, ledger_out])

        with gr.Column(scale=1):
            gr.Markdown(SUGGESTIONS)

    gr.Markdown("<div style='text-align:center;color:#8a93a6;font-size:12px;'>"
                "ANUKRAM runs entirely on Indian government infrastructure (MeghRaj-ready). "
                "Corpus: real public judgments from Indian Kanoon (Supreme Court & High Courts).</div>")


if __name__ == "__main__":
    demo.launch(ssr_mode=False)