# SHIELD: masks victim/minor identities (BNS 72, POCSO 23) for lower roles/exports and logs each masking event.

import os
import re
import json
from datetime import datetime
from src.config import AUDIT_DIR, ROLES, setup_environment, logger

# Roles allowed to see unmasked victim identities.
PRIVILEGED_ROLES = {"IO", "Prosecutor", "Court"}

# Patterns that indicate a victim/complainant field in our records.
_VICTIM_LINE_RE = re.compile(r'(Complainant\s*/?\s*Victim\s*:\s*)(.+)', re.IGNORECASE)
_COMPLAINANT_TOKEN_RE = re.compile(r'Complainant[\s\-\u2010\u2011]*[A-Z0-9]+', re.IGNORECASE)


def _log_masking(case_id, role, count):
    # Append a masking event to the SHIELD log so protection can be proven later.
    setup_environment()
    path = os.path.join(AUDIT_DIR, "shield_log.jsonl")
    event = {"time": datetime.now().isoformat(timespec="seconds"),
             "case_id": case_id, "role": role, "masked_items": count}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def mask_text(text, role, case_id="unknown"):
    # Return text with victim identities masked unless the role is privileged; log the masking.
    if role in PRIVILEGED_ROLES:
        return text

    masked_count = 0

    # Mask any explicit "Complainant/Victim:" field first.
    def _mask_field(match):
        nonlocal masked_count
        masked_count += 1
        return match.group(1) + "[VICTIM IDENTITY PROTECTED - BNS 72]"
    text = _VICTIM_LINE_RE.sub(_mask_field, text)

    # Mask any Complainant-X token appearing anywhere in the text.
    text, n = _COMPLAINANT_TOKEN_RE.subn("[VICTIM IDENTITY PROTECTED - BNS 72]", text)
    masked_count += n

    if masked_count:
        _log_masking(case_id, role, masked_count)
        logger.info("SHIELD masked %d victim item(s) for role %s in %s.", masked_count, role, case_id)
    return text


if __name__ == "__main__":
    sample = "Complainant / Victim: Complainant-A\nBrief Facts: ...(Complainant-A) reported..."
    print("--- IO (full access) ---")
    print(mask_text(sample, "IO", "FIR/2013/DEL-A/00123"))
    print("\n--- RTI_Public (masked) ---")
    print(mask_text(sample, "RTI_Public", "FIR/2013/DEL-A/00123"))