"""
Hebbian Updater
Manages the persistent superorganism learning state for the US model.

Each weekly briefing updates neuron × DPS weights according to Hebbian rules:
  - Neurons that fire (appear in ps_impacts) get strengthened
  - Neurons that are quiet decay toward zero
  - Co-activation (cooperative edge_signals) boosts both parties
  - Anti-correlated firing (adversarial edge_signals) disrupts both

Usage:
    python hebbian_updater.py --bootstrap  # initialise state from model
    python hebbian_updater.py              # update from latest briefing
    python hebbian_updater.py --status     # print current state summary
"""

import os
import json
import argparse
from datetime import date
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
MODEL_PATH    = SCRIPT_DIR / "us_superorganism_model.json"
STATE_PATH    = SCRIPT_DIR / "superorganism_state.json"
BRIEFINGS_DIR = SCRIPT_DIR / "briefings"

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

DECAY_RATE           = 0.10   # weekly weight decay (forgetting rate)
LEARNING_RATE        = 0.15   # boost for notable + activated DPS
DISRUPTION_RATE      = 0.08   # weight reduction for concerning signal
COOPERATIVE_BOOST    = 0.05   # both parties boosted on cooperative edge_signal
ADVERSARIAL_DROP     = 0.06   # both parties reduced on adversarial edge_signal

WEIGHT_MIN           = -1.0   # allow negative weights (opposing a DPS)
WEIGHT_MAX           =  1.0

PRUNE_THRESHOLD      = 0.04   # weights below this magnitude are dropped (sparsity)
EDGE_DISPLAY_THRESHOLD = 0.15 # minimum computed edge weight to draw in viz

BASE_WEIGHT_RANK1    = 0.70   # initial weight for rank-1 individual
BASE_WEIGHT_RANK_N   = 0.50   # initial weight for lowest-ranked individual


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rank_to_base_weight(rank: int, max_rank: int) -> float:
    """Linear interpolation from BASE_WEIGHT_RANK1 down to BASE_WEIGHT_RANK_N."""
    if max_rank <= 1:
        return BASE_WEIGHT_RANK1
    t = (rank - 1) / (max_rank - 1)
    return BASE_WEIGHT_RANK1 + t * (BASE_WEIGHT_RANK_N - BASE_WEIGHT_RANK1)


# ---------------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------------

def bootstrap(model_path: Path, state_path: Path, ps_canon_path: Path = None):
    """Create initial state from the superorganism model. Overwrites existing state.

    If ps_canon_path is provided and contains initial_neuron_weights, those are used
    directly instead of the rank-based formula.
    """
    print(f"Bootstrapping from {model_path.name}...")

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    people          = model["superorganism_list"]
    phase_sequences = model["canonical_vocabulary"]["phase_sequences"]
    max_rank        = max(p["rank"] for p in people)

    # Try to load initial weights from ps_canon
    ps_canon_weights = None
    if ps_canon_path is not None and ps_canon_path.exists():
        with open(ps_canon_path, "r", encoding="utf-8") as f:
            ps_canon = json.load(f)
        if "initial_neuron_weights" in ps_canon:
            ps_canon_weights = ps_canon["initial_neuron_weights"]
            print(f"  Using initial_neuron_weights from {ps_canon_path.name}")

    neuron_dps_weights = {}
    if ps_canon_weights is not None:
        # Use PS council weights directly
        for person in people:
            name = person["name"]
            if name in ps_canon_weights:
                w = {k: round(float(v), 4) for k, v in ps_canon_weights[name].items()}
                if w:
                    neuron_dps_weights[name] = w
    else:
        # Fall back to rank-based formula: each assigned DPS gets weight scaled by rank
        print("  Using rank-based weight formula (no ps_canon weights found)")
        for person in people:
            base_w = round(rank_to_base_weight(person["rank"], max_rank), 4)
            assigned = {
                ps["id"]: base_w
                for ps in person["superorganism"].get("phase_sequences", [])
            }
            if assigned:
                neuron_dps_weights[person["name"]] = assigned

    state = {
        "version": 1,
        "created":      date.today().isoformat(),
        "last_updated": date.today().isoformat(),
        "week_count":   0,
        "last_briefing_applied": None,
        "config": {
            "decay_rate":             DECAY_RATE,
            "learning_rate":          LEARNING_RATE,
            "disruption_rate":        DISRUPTION_RATE,
            "cooperative_boost":      COOPERATIVE_BOOST,
            "adversarial_drop":       ADVERSARIAL_DROP,
            "prune_threshold":        PRUNE_THRESHOLD,
            "edge_display_threshold": EDGE_DISPLAY_THRESHOLD,
        },
        "neuron_dps_weights": neuron_dps_weights,
    }

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"  {len(neuron_dps_weights)} neurons initialised")
    print(f"  Saved → {state_path.name}")


# ---------------------------------------------------------------------------
# STATE I/O
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict:
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, state_path: Path):
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# BRIEFING LOADER
# ---------------------------------------------------------------------------

def load_latest_briefing(briefings_dir: Path, prefix: str = "weekly_briefing_") -> tuple:
    """Returns (briefing_dict, filename) or (None, '').
    prefix: filename prefix to match (e.g. 'weekly_briefing_' or 'global_weekly_briefing_').
    """
    if not briefings_dir.is_dir():
        return None, ""
    files = sorted(
        (f for f in briefings_dir.iterdir()
         if f.name.startswith(prefix) and f.name.endswith(".json")),
        reverse=True,
    )
    if not files:
        return None, ""
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f), files[0].name


# ---------------------------------------------------------------------------
# HEBBIAN UPDATE RULE
# ---------------------------------------------------------------------------

def apply_decay(weights: dict, decay: float) -> dict:
    """Exponential decay applied to all weight values."""
    return {k: round(v * (1.0 - decay), 6) for k, v in weights.items()}


def update_from_briefing(state: dict, briefing: dict) -> dict:
    """
    Apply one Hebbian update step from a weekly briefing.

    Update sequence:
      1. Decay all existing weights (forgetting)
      2. Per-person activation: notable boosts, concerning disrupts, quiet only decays
      3. Edge signals: cooperative co-activation boosts both; adversarial drops both
      4. DPS dominance: updated from phase_sequence momentum ratings
      5. Prune near-zero weights for sparsity
    """
    cfg = state["config"]
    ndw = state["neuron_dps_weights"]   # name → {dps_id → float}

    # 1. Decay all weights
    for name in ndw:
        ndw[name] = apply_decay(ndw[name], cfg["decay_rate"])

    # 2. Per-person activation updates
    for pu in briefing.get("person_updates", []):
        name       = pu["name"]
        signal     = pu.get("signal", "quiet")
        ps_impacts = pu.get("ps_impacts", [])

        if name not in ndw:
            ndw[name] = {}

        for dps_id in ps_impacts:
            current = ndw[name].get(dps_id, 0.0)

            if signal == "notable":
                # Strong activation — fire and wire
                delta = cfg["learning_rate"]
            elif signal == "concerning":
                # Active but disrupted — Hebbian dropout reduces weight
                delta = -cfg["disruption_rate"]
            else:
                # quiet + ps_impact is unusual; treat as weak residual activation
                delta = cfg["learning_rate"] * 0.3

            ndw[name][dps_id] = round(
                clamp(current + delta, WEIGHT_MIN, WEIGHT_MAX), 6
            )

    # 3. Edge signal effects (pair-level co-activation / anti-correlation)
    for sig in briefing.get("edge_signals", []):
        name_a  = sig["person_a"]
        name_b  = sig["person_b"]
        dps_id  = sig["ps_id"]
        valence = sig["valence"]

        for name in (name_a, name_b):
            if name not in ndw:
                continue
            if dps_id not in ndw[name]:
                continue

            current = ndw[name][dps_id]

            if valence == "cooperative":
                # Co-activation: Hebbian strengthening for both parties
                delta = cfg["cooperative_boost"]
            elif valence == "adversarial":
                # Anti-correlated firing: assembly disruption for both parties.
                # Sustained adversarial pressure will push weights negative over
                # time, encoding true opposition to the phase sequence.
                delta = -cfg["adversarial_drop"]
            else:
                delta = 0.0

            ndw[name][dps_id] = round(
                clamp(current + delta, WEIGHT_MIN, WEIGHT_MAX), 6
            )

    # 4. Prune near-zero weights (maintain sparsity)
    for name in ndw:
        ndw[name] = {
            dps_id: w
            for dps_id, w in ndw[name].items()
            if abs(w) >= cfg["prune_threshold"]
        }

    state["last_updated"] = date.today().isoformat()
    state["week_count"]   = state.get("week_count", 0) + 1
    return state


# ---------------------------------------------------------------------------
# STATUS REPORT
# ---------------------------------------------------------------------------

def print_status(state: dict):
    ndw = state["neuron_dps_weights"]

    print(f"\nState: week {state.get('week_count', 0)}, "
          f"last updated {state.get('last_updated', '?')}")
    if state.get("last_briefing_applied"):
        print(f"Last briefing: {state['last_briefing_applied']}")

    print("\nNeuron DPS Weights (non-zero):")
    for name, weights in sorted(ndw.items()):
        if weights:
            w_str = "  ".join(
                f"{k}={'+' if v >= 0 else ''}{v:.3f}"
                for k, v in sorted(weights.items())
            )
            print(f"  {name}: {w_str}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hebbian learning state updater")
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which model to operate on: 'us' (default) or 'global'"
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Initialise state from the model file (overwrites existing state)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current state summary and exit"
    )
    args = parser.parse_args()

    # Resolve paths based on scope
    if args.scope == "global":
        model_path       = SCRIPT_DIR / "superorganism_model.json"
        state_path       = SCRIPT_DIR / "global_superorganism_state.json"
        briefing_prefix  = "global_weekly_briefing_"
        briefing_script  = "global_weekly_briefing.py"
        scope_label      = "GLOBAL"
    else:
        model_path       = MODEL_PATH
        state_path       = STATE_PATH
        briefing_prefix  = "weekly_briefing_"
        briefing_script  = "weekly_briefing.py"
        scope_label      = "US"

    print("=" * 60)
    print(f"HEBBIAN UPDATER  [{scope_label}]")
    print("=" * 60)

    if args.bootstrap:
        if not model_path.exists():
            print(f"ERROR: {model_path.name} not found.")
            return
        ps_canon_name = "ps_canon_us.json" if args.scope == "us" else "ps_canon_global.json"
        ps_canon_path = SCRIPT_DIR / ps_canon_name
        bootstrap(model_path, state_path, ps_canon_path)
        return

    if not state_path.exists():
        print(f"No state file found at {state_path.name}.")
        print(f"Run with --bootstrap --scope {args.scope} to initialise.")
        return

    state = load_state(state_path)

    if args.status:
        print_status(state)
        return

    # Normal path: update from latest briefing
    briefing, fname = load_latest_briefing(BRIEFINGS_DIR, prefix=briefing_prefix)
    if briefing is None:
        print(f"No {args.scope} briefing files found in briefings/.")
        print(f"Run {briefing_script} first.")
        return

    week = briefing.get("week_ending", "?")
    print(f"\nApplying briefing: {fname} (week ending {week})")

    # Guard: don't double-apply the same briefing
    if state.get("last_briefing_applied") == fname:
        print(f"  ! Already applied {fname} — nothing to do.")
        print("  Use --status to see current state.")
        return

    state = update_from_briefing(state, briefing)
    state["last_briefing_applied"] = fname

    save_state(state, state_path)
    print(f"  ✓ Week {state['week_count']} complete → {state_path.name}")
    print_status(state)

    print("\n" + "=" * 60)
    print("Done! Run combined_viz.py to see updated graph.")
    print("=" * 60)


if __name__ == "__main__":
    main()
