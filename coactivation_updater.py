"""
Coactivation Updater

For each PS that fired CAs in the latest briefing:
  - Identifies co-activated conscious neurons (members of fired CAs)
  - Identifies active CAs that fired under that PS
  - Runs pairwise Claude Haiku comparisons for N-N and CA-CA pairs
  - Updates rolling co-activation scores in us_coactivation_state.json

Scoring:
  reinforcing = +1, neutral = 0, adversarial = -1
  Rolling update: score += observation * learning_rate  (then decayed weekly)
  Only conscious neurons with active signal are included in N-N pairs.
  Spontaneous neurons are never included in pairwise comparisons.

Usage:
    python coactivation_updater.py --bootstrap   # initialise empty state
    python coactivation_updater.py               # update from latest briefing
    python coactivation_updater.py --status      # print current state summary
"""

import os
import json
import argparse
from datetime import date
from itertools import combinations
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

SCRIPT_DIR    = Path(__file__).parent
BRIEFINGS_DIR = SCRIPT_DIR / "briefings"
HAIKU_MODEL   = "claude-haiku-4-5-20251001"

SCOPE_CONFIG = {
    "us": {
        "state_file":    "us_coactivation_state.json",
        "model_file":    "us_superorganism_model.json",
        "briefing_prefix": "weekly_briefing_",
        "label":         "US",
    },
    "global": {
        "state_file":    "global_coactivation_state.json",
        "model_file":    "superorganism_model.json",
        "briefing_prefix": "global_weekly_briefing_",
        "label":         "Global",
    },
}

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

DECAY_RATE             = 0.15   # weekly score decay toward zero
LEARNING_RATE          = 0.25   # weight of each new observation
EDGE_DISPLAY_THRESHOLD = 0.15   # min |score| to draw an edge in the viz


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def pair_key(a: str, b: str) -> str:
    """Canonical sorted key so A|||B == B|||A."""
    return "|||".join(sorted([a, b]))


def score_to_label(score: float) -> str:
    if score >  0.05: return "reinforcing"
    if score < -0.05: return "adversarial"
    return "neutral"


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_latest_briefing(briefing_prefix: str) -> dict | None:
    if not BRIEFINGS_DIR.exists():
        return None
    files = sorted(
        (f for f in os.listdir(BRIEFINGS_DIR)
         if f.startswith(briefing_prefix) and f.endswith(".json")
         and "_raw_" not in f),
        reverse=True,
    )
    if not files:
        return None
    with open(BRIEFINGS_DIR / files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: Path) -> dict:
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------------

def bootstrap(path: Path, label: str):
    state = {
        "week_count":   0,
        "last_updated": date.today().isoformat(),
        "scope":        label,
        "config": {
            "decay_rate":             DECAY_RATE,
            "learning_rate":          LEARNING_RATE,
            "edge_display_threshold": EDGE_DISPLAY_THRESHOLD,
        },
        "neuron_coactivation": {},
        "ca_coactivation":     {},
    }
    save_state(state, path)
    print(f"Bootstrapped empty {label} state -> {path.name}")


# ---------------------------------------------------------------------------
# HAIKU COMPARISON
# ---------------------------------------------------------------------------

def compare_pair(
    client: anthropic.Anthropic,
    ps_id: str, ps_name: str, ps_summary: str,
    label_a: str, summary_a: str,
    label_b: str, summary_b: str,
) -> tuple[str, str]:
    """
    Ask Haiku whether two entities are reinforcing, adversarial, or neutral
    with respect to each other within the given phase sequence this week.
    Returns (label, rationale).
    """
    prompt = (
        f"Phase Sequence: {ps_id} — {ps_name}\n"
        f"{ps_summary}\n\n"
        f"{label_a}: {summary_a}\n\n"
        f"{label_b}: {summary_b}\n\n"
        f"Within this phase sequence, are these two entities reinforcing, adversarial, "
        f"or neutral with respect to each other this week? "
        f"Only consider their relationship as it bears on the domain described by this phase sequence. "
        f"If their news this week is not meaningfully related to this phase sequence's domain, "
        f"return neutral regardless of their general relationship to each other.\n"
        f'Respond with JSON only: {{"label": "reinforcing"|"adversarial"|"neutral", '
        f'"rationale": "one sentence"}}'
    )
    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    text  = message.content[0].text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    data  = json.loads(text[start:end])
    return data.get("label", "neutral"), data.get("rationale", "")


# ---------------------------------------------------------------------------
# SCORE MANAGEMENT
# ---------------------------------------------------------------------------

def decay_all(state: dict, decay_rate: float):
    """Decay all existing scores toward zero each week."""
    for bucket in ("neuron_coactivation", "ca_coactivation"):
        for entry in state[bucket].values():
            entry["score"] = entry["score"] * (1 - decay_rate)
            entry["label"] = score_to_label(entry["score"])


def update_pair(
    state: dict, bucket: str, key: str,
    label: str, ps_id: str, week: str, learning_rate: float,
):
    obs = {"reinforcing": 1.0, "neutral": 0.0, "adversarial": -1.0}.get(label, 0.0)

    if key not in state[bucket]:
        state[bucket][key] = {
            "score":        0.0,
            "label":        "neutral",
            "observations": 0,
            "last_ps":      [],
            "last_updated": week,
        }

    entry          = state[bucket][key]
    entry["score"] = max(-1.0, min(1.0, entry["score"] + obs * learning_rate))
    entry["label"] = score_to_label(entry["score"])
    entry["observations"] += 1
    if ps_id not in entry["last_ps"]:
        entry["last_ps"] = [ps_id] + entry["last_ps"][:2]  # keep last 3 PSs
    entry["last_updated"] = week


# ---------------------------------------------------------------------------
# MAIN UPDATE
# ---------------------------------------------------------------------------

def run_update(briefing: dict, model: dict, state: dict, client: anthropic.Anthropic):
    meta                = briefing.get("_metadata", {})
    conscious_names     = set(meta.get("neurons_conscious", []))
    ps_assemblies_fired = meta.get("ps_assemblies_fired", {})
    week                = briefing.get("week_ending", date.today().isoformat())

    cfg           = state.get("config", {})
    decay_rate    = cfg.get("decay_rate",    DECAY_RATE)
    learning_rate = cfg.get("learning_rate", LEARNING_RATE)

    # Build lookups from briefing
    person_updates = {p["name"]: p for p in briefing.get("person_updates", [])}
    ca_updates     = {a["id"]:   a for a in briefing.get("assembly_updates", [])}
    ps_updates     = {p["id"]:   p for p in briefing.get("phase_sequence_updates", [])}

    # Build CA → set of member neuron names from model
    ca_to_members: dict[str, set] = {}
    for person in model["superorganism_list"]:
        for ca in person.get("superorganism", {}).get("cell_assemblies", []):
            ca_id = ca.get("id") or ca.get("name", "")
            if ca_id:
                ca_to_members.setdefault(ca_id, set()).add(person["name"])

    # Decay all existing scores before applying new observations
    decay_all(state, decay_rate)

    total_nn = 0
    total_cc = 0

    for ps_id, fired_ca_ids in ps_assemblies_fired.items():
        ps_info    = ps_updates.get(ps_id, {})
        ps_name    = ps_info.get("name",    ps_id)
        ps_summary = ps_info.get("summary", "")

        print(f"\n[{ps_id}: {ps_name}]")

        # --- N-N pairs ---
        # Conscious neurons that are members of any CA fired under this PS,
        # have active signal, and have a summary this week.
        neurons_in_ps = set()
        for ca_id in fired_ca_ids:
            neurons_in_ps |= ca_to_members.get(ca_id, set()) & conscious_names

        eligible_neurons = [
            n for n in neurons_in_ps
            if person_updates.get(n, {}).get("signal", "active") == "active"
            and person_updates.get(n, {}).get("summary", "")
        ]

        nn_pairs = list(combinations(sorted(eligible_neurons), 2))
        print(f"  N-N: {len(eligible_neurons)} eligible neurons -> {len(nn_pairs)} pairs")

        for name_a, name_b in nn_pairs:
            key = pair_key(name_a, name_b)
            try:
                label, _ = compare_pair(
                    client, ps_id, ps_name, ps_summary,
                    name_a, person_updates[name_a]["summary"],
                    name_b, person_updates[name_b]["summary"],
                )
                update_pair(state, "neuron_coactivation", key, label, ps_id, week, learning_rate)
                print(f"    {name_a} x {name_b} -> {label}")
                total_nn += 1
            except Exception as e:
                print(f"    ! {name_a} × {name_b}: {e}")

        # --- CA-CA pairs ---
        # Active CAs that fired under this PS with a summary this week.
        eligible_cas = [
            ca_id for ca_id in fired_ca_ids
            if ca_updates.get(ca_id, {}).get("signal", "active") == "active"
            and ca_updates.get(ca_id, {}).get("summary", "")
        ]

        cc_pairs = list(combinations(sorted(eligible_cas), 2))
        print(f"  CA-CA: {len(eligible_cas)} eligible CAs -> {len(cc_pairs)} pairs")

        for ca_a, ca_b in cc_pairs:
            key = pair_key(ca_a, ca_b)
            try:
                label, _ = compare_pair(
                    client, ps_id, ps_name, ps_summary,
                    ca_updates[ca_a].get("name", ca_a), ca_updates[ca_a]["summary"],
                    ca_updates[ca_b].get("name", ca_b), ca_updates[ca_b]["summary"],
                )
                update_pair(state, "ca_coactivation", key, label, ps_id, week, learning_rate)
                print(f"    {ca_a} x {ca_b} -> {label}")
                total_cc += 1
            except Exception as e:
                print(f"    ! {ca_a} × {ca_b}: {e}")

    state["week_count"]   = state.get("week_count", 0) + 1
    state["last_updated"] = week

    print(f"\nDone: {total_nn} N-N pairs, {total_cc} CA-CA pairs processed.")
    return state


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

def print_status(state: dict):
    print(f"Week count:   {state.get('week_count', 0)}")
    print(f"Last updated: {state.get('last_updated', '?')}")

    for bucket, label in (("neuron_coactivation", "Neuron pairs"), ("ca_coactivation", "CA pairs")):
        entries     = state.get(bucket, {})
        reinforcing = sum(1 for e in entries.values() if e["label"] == "reinforcing")
        adversarial = sum(1 for e in entries.values() if e["label"] == "adversarial")
        neutral     = len(entries) - reinforcing - adversarial
        print(f"\n{label}: {len(entries)}")
        print(f"  reinforcing: {reinforcing}  adversarial: {adversarial}  neutral: {neutral}")
        threshold = state.get("config", {}).get("edge_display_threshold", EDGE_DISPLAY_THRESHOLD)
        drawable  = sum(1 for e in entries.values() if abs(e["score"]) >= threshold)
        print(f"  above display threshold ({threshold}): {drawable}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Coactivation Updater")
    parser.add_argument("--scope",     choices=["us", "global"], default="us",
                        help="Which model to run (default: us)")
    parser.add_argument("--bootstrap", action="store_true", help="Initialise empty state file")
    parser.add_argument("--status",    action="store_true", help="Print current state summary")
    args = parser.parse_args()

    cfg        = SCOPE_CONFIG[args.scope]
    state_path = SCRIPT_DIR / cfg["state_file"]
    model_path = SCRIPT_DIR / cfg["model_file"]

    if args.bootstrap:
        bootstrap(state_path, cfg["label"])
        return

    state = load_state(state_path)
    if state is None:
        print(f"No state file found. Run with --bootstrap --scope {args.scope} first.")
        return

    if args.status:
        print_status(state)
        return

    print("=" * 60)
    print(f"COACTIVATION UPDATER [{cfg['label'].upper()}]")
    print("=" * 60)

    briefing = load_latest_briefing(cfg["briefing_prefix"])
    if not briefing:
        print(f"No briefing found in briefings/ with prefix '{cfg['briefing_prefix']}'")
        return
    print(f"\nBriefing: week ending {briefing.get('week_ending', '?')}")

    model = load_model(model_path)
    print(f"Model:    {len(model['superorganism_list'])} neurons")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    state  = run_update(briefing, model, state, client)
    save_state(state, state_path)
    print(f"\nState saved -> {state_path.name}")


if __name__ == "__main__":
    main()
