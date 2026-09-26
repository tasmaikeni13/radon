"""Conservative publication gate while heavy-run evidence is pending."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "phases" / "state.json"


def main() -> None:
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    required = ("6", "7", "8")
    pending = [phase for phase in required if state["phases"][phase]["status"] != "MEASURED"]
    if pending:
        raise SystemExit(
            "Publication evidence is pending measured heavy runs for phases "
            + ", ".join(pending)
            + ". Smoke tests and fixed summary values cannot certify Phase 9."
        )
    raise SystemExit(
        "Phase 9 still requires a provenance-aware audit of raw runs, held-out metrics, "
        "figures, proofs, and paper before any publication certificate can be issued."
    )


if __name__ == "__main__":
    main()
