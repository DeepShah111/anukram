# WATCH: computes statutory chargesheet deadlines (BNSS 193/187) for a case and returns a Red/Amber/Green status.

from datetime import datetime, date

# Chargesheet deadline in days by offence seriousness (BNSS 193/187).
SERIOUS_SECTIONS = {"BNS 64", "BNS 65", "BNS 66", "BNS 70", "POCSO 6"}   # rape / aggravated → 90 days
DEADLINE_SERIOUS = 90
DEADLINE_STANDARD = 60


def _parse_date(value):
    # Accept a YYYY-MM-DD string or a date object; return a date or None.
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def deadline_days_for(sections):
    # Return 90 days if any section is a serious offence, else 60.
    for s in sections or []:
        if s.upper() in SERIOUS_SECTIONS:
            return DEADLINE_SERIOUS
    return DEADLINE_STANDARD


def watch_status(arrest_date, sections, today=None):
    # Compute days remaining to the chargesheet deadline and a R/A/G status for one case.
    arrest = _parse_date(arrest_date)
    if arrest is None:
        return {"applicable": False, "reason": "No arrest date on record."}

    today = _parse_date(today) or date.today()
    limit = deadline_days_for(sections)
    due = arrest.toordinal() + limit
    days_left = due - today.toordinal()

    # Red = breached or <=7 days, Amber = <=21 days, else Green.
    if days_left <= 7:
        status = "RED"
    elif days_left <= 21:
        status = "AMBER"
    else:
        status = "GREEN"

    return {
        "applicable": True,
        "arrest_date": arrest.isoformat(),
        "deadline_days": limit,
        "due_date": date.fromordinal(due).isoformat(),
        "days_left": days_left,
        "status": status,
        "basis": "BNSS 193/187",
    }


if __name__ == "__main__":
    # Demo: the 2020 case, evaluated as if today is mid-2020.
    print(watch_status("2020-06-10", ["BNS 64"], today="2020-07-20"))