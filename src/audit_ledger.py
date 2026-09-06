# Hash-chained audit ledger: append-only events where each entry hashes the previous one; verifiable on demand.

import os
import json
import hashlib
from datetime import datetime
from src.config import AUDIT_DIR, setup_environment, logger

LEDGER_PATH = os.path.join(AUDIT_DIR, "audit_ledger.jsonl")
GENESIS_HASH = "0" * 64


def _entry_hash(prev_hash, event):
    # Hash of the previous entry's hash plus this event's data (the tamper-evidence chain).
    raw = prev_hash + json.dumps(event, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _last_hash():
    # Return the hash of the most recent ledger entry, or the genesis hash if empty.
    if not os.path.exists(LEDGER_PATH):
        return GENESIS_HASH
    last = GENESIS_HASH
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["entry_hash"]
    return last


def log_event(action, user_role, case_id, detail=""):
    # Append one tamper-evident event to the ledger.
    setup_environment()
    event = {"time": datetime.now().isoformat(timespec="seconds"),
             "action": action, "role": user_role, "case_id": case_id, "detail": detail}
    prev_hash = _last_hash()
    entry = {"event": event, "prev_hash": prev_hash,
             "entry_hash": _entry_hash(prev_hash, {**event, "prev_hash": prev_hash})}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Audit event logged: %s on %s by %s", action, case_id, user_role)
    return entry


def verify_integrity():
    # Recompute the whole chain; return VERIFIED if untouched, else the first broken entry.
    if not os.path.exists(LEDGER_PATH):
        return {"status": "EMPTY", "message": "No audit entries yet."}

    prev_hash = GENESIS_HASH
    with open(LEDGER_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for i, entry in enumerate(entries, 1):
        event = entry["event"]
        expected = _entry_hash(prev_hash, {**event, "prev_hash": prev_hash})
        if entry["prev_hash"] != prev_hash or entry["entry_hash"] != expected:
            return {"status": "TAMPERED", "broken_at": i, "total": len(entries),
                    "message": f"Integrity check FAILED at entry {i}."}
        prev_hash = entry["entry_hash"]

    return {"status": "VERIFIED", "total": len(entries),
            "message": f"All {len(entries)} entries verified. Chain intact."}


if __name__ == "__main__":
    log_event("SEARCH", "IO", "FIR/2020/DEL-D/00234", "queried accused history")
    log_event("VIEW", "Prosecutor", "FIR/2013/DEL-A/00123", "opened prior case")
    print(verify_integrity())