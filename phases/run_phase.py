#!/usr/bin/env python3
"""Programmatic Phase Runner and Adaptive Dependency Engine for RADON.

Allows AI research agents and human investigators to inspect, verify,
and execute research phases, trigger automated failure diagnostics,
and cascade dependency invalidations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

PHASES_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PHASES_DIR.parent
STATE_FILE = PHASES_DIR / "state.json"


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"State file {STATE_FILE} not found.")
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def print_status_table(state: dict[str, Any]) -> None:
    print("=" * 80)
    print("  RADON Research Phases: Status & Dependency Overview")
    print(f"  Target Pod: {state.get('hardware_target', 'Unknown')}")
    print("=" * 80)
    phases = state.get("phases", {})
    print(f"{'ID':<4} | {'Status':<12} | {'Phase Name':<45} | {'Deps'}")
    print("-" * 80)
    for pid_str, p in sorted(phases.items(), key=lambda x: int(x[0])):
        pid = p["id"]
        status = p["status"]
        name = p["name"]
        if len(name) > 43:
            name = name[:40] + "..."
        deps = ",".join(str(d) for d in p.get("dependencies", [])) or "None"
        print(f"{pid:<4} | {status:<12} | {name:<45} | {deps}")
    print("=" * 80)


def cascade_invalidation(phase_id: int, state: dict[str, Any]) -> list[int]:
    """Recursively mark all downstream dependent phases as INVALIDATED."""
    phases = state.get("phases", {})
    invalidated = []

    def get_direct_dependents(pid: int) -> list[int]:
        deps = []
        for other_id_str, other_phase in phases.items():
            if pid in other_phase.get("dependencies", []):
                deps.append(int(other_id_str))
        return deps

    queue = get_direct_dependents(phase_id)
    while queue:
        current = queue.pop(0)
        if current not in invalidated:
            invalidated.append(current)
            phases[str(current)]["status"] = "INVALIDATED"
            queue.extend(get_direct_dependents(current))

    return invalidated


def run_gate(gate_cmd: str) -> bool:
    if gate_cmd.startswith("python3 ") or gate_cmd.startswith("python "):
        cmd_parts = gate_cmd.split(" ", 1)
        gate_cmd = f"{sys.executable} {cmd_parts[1]}"
    print(f"\n[EXEC] Running verification gate: {gate_cmd}")
    try:
        proc = subprocess.run(
            gate_cmd,
            shell=True,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1200,
        )
        print(proc.stdout)
        return proc.returncode == 0
    except Exception as e:
        print(f"[ERROR] Gate execution failed with exception: {e}")
        return False


def verify_phase(phase_id: int, state: dict[str, Any], verbose: bool = True) -> bool:
    pid_str = str(phase_id)
    phase = state.get("phases", {}).get(pid_str)
    if not phase:
        print(f"[ERROR] Phase {phase_id} not found in state registry.")
        return False

    name = phase["name"]
    gate_cmd = phase.get("gate_cmd")
    deps = phase.get("dependencies", [])

    if verbose:
        print(f"\n>>> Verifying Phase {phase_id}: {name}")
        print(f"    Dependencies: {deps or 'None'}")

    for dep_id in deps:
        dep_phase = state.get("phases", {}).get(str(dep_id))
        if not dep_phase or dep_phase["status"] != "PASSED":
            print(f"[FAIL] Dependency Phase {dep_id} is not PASSED (Current: {dep_phase.get('status')}).")
            phase["status"] = "BLOCKED"
            save_state(state)
            return False

    success = run_gate(gate_cmd) if gate_cmd else True
    if success:
        phase["status"] = "PASSED"
        phase["last_verified"] = datetime.now(timezone.utc).isoformat()
        print(f"[PASS] Phase {phase_id} gate certified successfully.")
    else:
        phase["status"] = "FAILED"
        print(f"[FAIL] Phase {phase_id} gate failed verification.")
        inval = cascade_invalidation(phase_id, state)
        if inval:
            print(f"[CASCADE] Invalidated downstream phases: {inval}")

    save_state(state)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Print phase status table")
    parser.add_argument("--phase", type=int, help="Execute and verify specific phase ID")
    parser.add_argument("--all", action="store_true", help="Execute and verify all phases sequentially")
    parser.add_argument(
        "--invalidate",
        type=int,
        help="Trigger manual invalidation cascade from phase ID",
    )
    args = parser.parse_args()

    state = load_state()

    if args.status or (not args.phase and not args.all and not args.invalidate):
        print_status_table(state)
        return

    if args.invalidate:
        inval = cascade_invalidation(args.invalidate, state)
        save_state(state)
        print(f"Cascaded invalidation from Phase {args.invalidate}: Invalidated {inval}")
        print_status_table(state)
        return

    if args.phase:
        success = verify_phase(args.phase, state)
        sys.exit(0 if success else 1)

    if args.all:
        for pid in sorted(state.get("phases", {}).keys(), key=int):
            if not verify_phase(int(pid), state):
                print(f"[ABORT] Stopped execution due to failure in Phase {pid}.")
                sys.exit(1)
        print("\n[ALL PHASES PASSED] Full research pipeline certified!")
        sys.exit(0)


if __name__ == "__main__":
    main()
