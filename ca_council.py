"""
CA Council — LLM Council for Cell Assembly Generation

Reads ps_canon_{scope}.json (produced by ps_council.py) and generates
canonical cell assemblies for each phase sequence.

A cell assembly = a real organization, coalition, or institutional cluster
that acts as a coordinated unit within one or more phase sequences.

Stages:
    1. Independent proposals — 4 LLMs each propose 3-5 assemblies per PS.
       Top neurons are selected by weight internally, then shuffled and
       stripped of weights before being shown (blinded neuron list).
    2. Peer review           — each model flags gaps and redundancies in peers.
    3. Chairman synthesis    — Claude produces a canonical assembly list with
                               sequential IDs, merging cross-PS duplicates into
                               single entries with multiple ps_memberships.
    4. Neuron assignment     — Claude assigns member neurons to each assembly,
                               one PS at a time for focused, accurate results.

Output: ca_canon_{scope}.json

Usage:
    python ca_council.py --scope us
    python ca_council.py --scope global
    python ca_council.py --scope us --skip-assignment
"""

import os
import json
import random
import argparse
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from google import genai

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

SCOPE_CONFIG = {
    "us": {
        "label":               "US",
        "ps_canon_file":       "ps_canon_us.json",
        "output_file":         "ca_canon_us.json",
        "ca_id_prefix":        "CA",
        "n_assemblies_per_ps": "3-5",
        "scope_description":   (
            "US domestic power dynamics — politics, technology, finance, "
            "media, defense, energy, judiciary"
        ),
    },
    "global": {
        "label":               "Global",
        "ps_canon_file":       "ps_canon_global.json",
        "output_file":         "ca_canon_global.json",
        "ca_id_prefix":        "CA",
        "n_assemblies_per_ps": "3-5",
        "scope_description":   (
            "Global geopolitics — US-China competition, military, finance, "
            "technology, information, energy, regional realignments"
        ),
    },
}

CLAUDE_MODEL = "claude-opus-4-6"
GPT_MODEL    = "gpt-5.2"
GROK_MODEL   = "grok-4-1-fast-reasoning"
GEMINI_MODEL = "gemini-3-pro-preview"

# Top N neurons per PS selected by weight (then shuffled before showing to LLMs)
N_NEURONS_PER_PS = 15
# Top N neurons per PS shown during neuron assignment (Stage 4)
N_NEURONS_ASSIGN = 25


# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------

def get_anthropic_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_xai_client():
    return OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_ps_canon(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_list(text: str) -> list:
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


def extract_json_obj(text: str) -> dict:
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def get_top_neurons_per_ps(
    phase_sequences: list,
    neuron_weights: dict,
    n: int,
) -> dict:
    """
    For each PS, select the top-N neurons by weight.
    Returns dict of {ps_id: [{"name": ..., "title": ...}, ...]} where
    the list is shuffled (weights stripped) to prevent rank-order inference.

    neuron_weights format: {"Name": {"DPS-01": 0.85, ...}, ...}
    We need a reverse lookup: ps_id → sorted list of (name, weight) pairs.
    """
    # Build a lookup of title per neuron name from ps_canon metadata
    # (We don't have titles in ps_canon directly — build from the ranked list if available)
    # Fallback: store name only
    ps_ids = {ps["id"] for ps in phase_sequences}

    # Reverse: ps_id → [(name, weight), ...]
    ps_to_neurons: dict[str, list] = {ps_id: [] for ps_id in ps_ids}
    for name, weights in neuron_weights.items():
        for ps_id, w in weights.items():
            if ps_id in ps_to_neurons and w > 0:
                ps_to_neurons[ps_id].append((name, w))

    # Sort by weight descending, take top N, shuffle to blind
    result = {}
    for ps_id in ps_ids:
        ranked = sorted(ps_to_neurons[ps_id], key=lambda x: -x[1])[:n]
        names  = [{"name": name} for name, _ in ranked]
        random.shuffle(names)
        result[ps_id] = names

    return result


# ---------------------------------------------------------------------------
# STAGE 1: INDEPENDENT PROPOSALS
# ---------------------------------------------------------------------------

def build_proposal_prompt(phase_sequences: list, top_neurons: dict, cfg: dict) -> str:
    label        = cfg["label"]
    scope_desc   = cfg["scope_description"]
    n_target     = cfg["n_assemblies_per_ps"]

    ps_blocks = ""
    for ps in phase_sequences:
        ps_id      = ps["id"]
        neurons    = top_neurons.get(ps_id, [])
        neuron_str = ", ".join(n["name"] for n in neurons) if neurons else "(none identified)"
        ps_blocks += (
            f"\n{ps_id}: {ps['name']}\n"
            f"  Definition: {ps['definition']}\n"
            f"  Key actors (alphabetical, unranked): {neuron_str}\n"
        )

    return f"""You are helping design canonical cell assemblies for a {label} superorganism model.

## CONTEXT

In this Hebbian framework:
- **Phase sequences** = major structural dynamics in {label} power
- **Cell assemblies** = real organizations, coalitions, or institutional clusters
  that fire together within a phase sequence (like neurons in a cell assembly)
- A cell assembly is NOT a person — it is an org or coordinated group
  (e.g. "DOGE Task Force", "OpenAI", "Senate Republican Caucus", "Fed Board of Governors")

## YOUR TASK

For each phase sequence below, propose {n_target} cell assemblies.

Each assembly should:
- Be a real, nameable organization or institutional cluster (not a vague category)
- Act as a coherent unit within that phase sequence
- Be distinct from assemblies you propose for other PS (minimize cross-PS duplication)

## PHASE SEQUENCES

{ps_blocks}

Return only a JSON object mapping each PS ID to its list of assemblies:
{{
  "DPS-01": [
    {{"name": "...", "description": "one sentence: what this org does in this PS"}},
    ...
  ],
  "DPS-02": [...],
  ...
}}"""


def _call_proposal(model_name: str, prompt: str,
                   anthropic_client, openai_client, xai_client, gemini_client) -> dict:
    if model_name == "claude":
        msg = anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4096, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(msg.content[0].text)
    elif model_name == "chatgpt":
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL, temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an expert analyst of organizational power dynamics."},
                {"role": "user",   "content": prompt},
            ],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif model_name == "grok":
        resp = xai_client.chat.completions.create(
            model=GROK_MODEL, temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an expert analyst of organizational power dynamics."},
                {"role": "user",   "content": prompt},
            ],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif model_name == "gemini":
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return extract_json_obj(resp.text)
    raise ValueError(f"Unknown model: {model_name}")


def stage_1_proposals(phase_sequences: list, neuron_weights: dict, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 1: INDEPENDENT PROPOSALS")
    print("=" * 60)
    print(f"\nSelecting top {N_NEURONS_PER_PS} neurons per PS (blinded)...")

    top_neurons = get_top_neurons_per_ps(phase_sequences, neuron_weights, N_NEURONS_PER_PS)
    prompt      = build_proposal_prompt(phase_sequences, top_neurons, cfg)

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    results = {}
    for model_name in ("claude", "chatgpt", "grok", "gemini"):
        print(f"  Querying {model_name}...")
        try:
            proposals = _call_proposal(
                model_name, prompt,
                anthropic_client, openai_client, xai_client, gemini_client,
            )
            results[model_name] = proposals
            total = sum(len(v) for v in proposals.values())
            print(f"  ✓ {model_name}: {total} assemblies across {len(proposals)} PS")
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")
            results[model_name] = {}

    return results


# ---------------------------------------------------------------------------
# STAGE 2: PEER REVIEW
# ---------------------------------------------------------------------------

def build_peer_review_prompt(reviewer_proposals: dict, other_proposals: dict,
                              phase_sequences: list, cfg: dict) -> str:
    label = cfg["label"]

    own_block = ""
    for ps in phase_sequences:
        ps_id = ps["id"]
        assemblies = reviewer_proposals.get(ps_id, [])
        own_block += f"\n{ps_id} ({ps['name']}):\n"
        for a in assemblies:
            own_block += f"  - {a['name']}: {a.get('description', '')}\n"

    other_block = ""
    for lbl, proposals in other_proposals.items():
        other_block += f"\n### {lbl}:\n"
        for ps in phase_sequences:
            ps_id      = ps["id"]
            assemblies = proposals.get(ps_id, [])
            if assemblies:
                other_block += f"  {ps_id}: " + "; ".join(a["name"] for a in assemblies) + "\n"

    return f"""You previously proposed cell assemblies for the {label} superorganism model.

## YOUR PROPOSALS
{own_block}

## OTHER MODELS' PROPOSALS
{other_block}

Review the other models' proposals and identify:
1. **Gaps** — important organizations missing from their lists that you included or think are critical
2. **Redundancies** — assemblies proposed for multiple PS that are really the same organization
3. **Vague entries** — any assembly that is too generic to be meaningfully tracked (e.g. "Tech Executives")

Return only a JSON object:
{{
  "gaps": ["brief description of what's missing and for which PS"],
  "redundancies": ["name A and name B are the same org, appears in DPS-X and DPS-Y"],
  "vague_entries": ["name X in DPS-Y is too broad — suggest: ..."]
}}"""


def _call_peer_review(reviewer: str, prompt: str,
                      anthropic_client, openai_client, xai_client, gemini_client) -> dict:
    if reviewer == "claude":
        msg = anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(msg.content[0].text)
    elif reviewer == "chatgpt":
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL, temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif reviewer == "grok":
        resp = xai_client.chat.completions.create(
            model=GROK_MODEL, temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif reviewer == "gemini":
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return extract_json_obj(resp.text)
    raise ValueError(f"Unknown reviewer: {reviewer}")


def stage_2_peer_review(all_proposals: dict, phase_sequences: list, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 2: PEER REVIEW")
    print("=" * 60)
    print("\nEach model reviews peers' proposals for gaps, redundancies, and vague entries...\n")

    model_names = [m for m in ("claude", "chatgpt", "grok", "gemini") if all_proposals.get(m)]

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    peer_reviews = {}
    for reviewer in model_names:
        others     = {f"Model {chr(65+i)}": all_proposals[m]
                      for i, m in enumerate(m for m in model_names if m != reviewer)}
        prompt     = build_peer_review_prompt(
            all_proposals[reviewer], others, phase_sequences, cfg
        )
        print(f"  Getting review from {reviewer}...")
        try:
            review = _call_peer_review(
                reviewer, prompt,
                anthropic_client, openai_client, xai_client, gemini_client,
            )
            peer_reviews[reviewer] = review
            n_gaps  = len(review.get("gaps", []))
            n_redun = len(review.get("redundancies", []))
            n_vague = len(review.get("vague_entries", []))
            print(f"  ✓ {reviewer}: {n_gaps} gaps, {n_redun} redundancies, {n_vague} vague entries flagged")
        except Exception as e:
            print(f"  ✗ {reviewer}: {e}")
            peer_reviews[reviewer] = None

    return peer_reviews


# ---------------------------------------------------------------------------
# STAGE 3: CHAIRMAN SYNTHESIS
# ---------------------------------------------------------------------------

def build_synthesis_prompt(all_proposals: dict, peer_reviews: dict,
                            phase_sequences: list, cfg: dict) -> str:
    label    = cfg["label"]
    prefix   = cfg["ca_id_prefix"]
    n_target = cfg["n_assemblies_per_ps"]

    proposals_block = ""
    for model_name, proposals in all_proposals.items():
        if not proposals:
            continue
        proposals_block += f"\n### {model_name.upper()}:\n"
        for ps in phase_sequences:
            ps_id      = ps["id"]
            assemblies = proposals.get(ps_id, [])
            if assemblies:
                proposals_block += f"  {ps_id} ({ps['name']}):\n"
                for a in assemblies:
                    proposals_block += f"    - {a['name']}: {a.get('description','')}\n"

    reviews_block = ""
    for reviewer, review in peer_reviews.items():
        if not review:
            continue
        reviews_block += f"\n### {reviewer.upper()} flags:\n"
        for gap in review.get("gaps", []):
            reviews_block += f"  GAP: {gap}\n"
        for r in review.get("redundancies", []):
            reviews_block += f"  REDUNDANCY: {r}\n"
        for v in review.get("vague_entries", []):
            reviews_block += f"  VAGUE: {v}\n"

    ps_list = "\n".join(
        f"  {ps['id']}: {ps['name']} — {ps['definition']}"
        for ps in phase_sequences
    )

    return f"""You are the Chairman of a Cell Assembly Council for the {label} superorganism model.

Four AI models have each proposed cell assemblies (organizations, coalitions, institutional clusters)
for {len(phase_sequences)} phase sequences. You also have peer reviews flagging issues.

## PHASE SEQUENCES
{ps_list}

## ALL PROPOSALS
{proposals_block}

## PEER REVIEW FLAGS
{reviews_block}

## YOUR TASK

Produce a canonical cell assembly list with the following rules:

1. **{n_target} assemblies per PS** — curate the best, drop vague or redundant entries
2. **Cross-PS deduplication** — if the same real organization appears under multiple PS,
   create ONE entry with `ps_memberships` listing all relevant PS IDs
3. **Assign sequential IDs** — {prefix}-01, {prefix}-02, ... (globally, not per-PS)
4. **Name precisely** — use the organization's real name or a specific, trackable label

Return only a JSON array:
[
  {{
    "id": "{prefix}-01",
    "name": "...",
    "description": "one sentence: what this org is and its structural role",
    "ps_memberships": ["{phase_sequences[0]['id']}", ...]
  }},
  ...
]"""


def stage_3_chairman_synthesis(all_proposals: dict, peer_reviews: dict,
                                phase_sequences: list, cfg: dict) -> list:
    print("\n" + "=" * 60)
    print("STAGE 3: CHAIRMAN SYNTHESIS")
    print("=" * 60)
    print("\nClaude (Chairman) synthesizing canonical assembly list...\n")

    client  = get_anthropic_client()
    prompt  = build_synthesis_prompt(all_proposals, peer_reviews, phase_sequences, cfg)
    message = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=6000, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    assemblies = extract_json_list(message.content[0].text)
    print(f"✓ Chairman synthesis: {len(assemblies)} canonical cell assemblies")
    return assemblies


# ---------------------------------------------------------------------------
# STAGE 4: NEURON ASSIGNMENT (per PS)
# ---------------------------------------------------------------------------

def build_assignment_prompt(ps: dict, ps_assemblies: list,
                             top_neurons: list, cfg: dict) -> str:
    label = cfg["label"]

    assemblies_block = "\n".join(
        f"  {a['id']}: {a['name']} — {a.get('description','')}"
        for a in ps_assemblies
    )
    neurons_block = "\n".join(
        f"  - {n['name']}"
        for n in top_neurons
    )

    return f"""You are assigning neurons (individuals) to cell assemblies for the {label} superorganism.

## PHASE SEQUENCE
{ps['id']}: {ps['name']}
Definition: {ps['definition']}

## CELL ASSEMBLIES IN THIS PS
{assemblies_block}

## CANDIDATE NEURONS (key actors in this PS)
{neurons_block}

## YOUR TASK

For each cell assembly above, list which neurons from the candidate list are members.
A neuron belongs to an assembly if they are genuinely part of that organization or coalition.
A neuron may belong to multiple assemblies if accurate.
Not all neurons need to be assigned — only include genuine membership.

Return only a JSON object:
{{
  "{ps_assemblies[0]['id'] if ps_assemblies else 'CA-01'}": ["Name A", "Name B", ...],
  ...
}}"""


def stage_4_neuron_assignment(
    phase_sequences: list,
    cell_assemblies: list,
    neuron_weights: dict,
    cfg: dict,
) -> dict:
    """
    Assign neurons to assemblies, one PS at a time.
    Returns: {assembly_id: [neuron_name, ...]}
    """
    print("\n" + "=" * 60)
    print("STAGE 4: NEURON ASSIGNMENT")
    print("=" * 60)
    print(f"\nAssigning neurons to assemblies ({len(phase_sequences)} PS, one call each)...\n")

    client = get_anthropic_client()

    # Build PS → assemblies lookup
    ps_to_assemblies: dict[str, list] = {ps["id"]: [] for ps in phase_sequences}
    for ca in cell_assemblies:
        for ps_id in ca.get("ps_memberships", []):
            if ps_id in ps_to_assemblies:
                ps_to_assemblies[ps_id].append(ca)

    # Build PS → top neurons (by weight, shuffled to blind rank)
    top_neurons_per_ps = get_top_neurons_per_ps(phase_sequences, neuron_weights, N_NEURONS_ASSIGN)

    all_assignments: dict[str, list] = {ca["id"]: [] for ca in cell_assemblies}

    for ps in phase_sequences:
        ps_id        = ps["id"]
        ps_assemblies = ps_to_assemblies.get(ps_id, [])
        top_neurons   = top_neurons_per_ps.get(ps_id, [])

        if not ps_assemblies:
            print(f"  {ps_id}: no assemblies assigned — skipping")
            continue

        print(f"  {ps_id} ({ps['name']}): {len(ps_assemblies)} assemblies, {len(top_neurons)} neurons...")
        prompt  = build_assignment_prompt(ps, ps_assemblies, top_neurons, cfg)
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=2048, temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            assignment = extract_json_obj(message.content[0].text)
            for ca_id, members in assignment.items():
                if ca_id in all_assignments:
                    # Merge without duplicates (neuron may appear across multiple PS calls)
                    existing = set(all_assignments[ca_id])
                    all_assignments[ca_id].extend(m for m in members if m not in existing)
            print(f"  ✓ {ps_id}: {sum(len(v) for k,v in assignment.items() if k in all_assignments)} neuron assignments")
        except Exception as e:
            print(f"  ✗ {ps_id}: {e}")

    return all_assignments


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def build_neuron_assembly_memberships(
    cell_assemblies: list, assembly_neuron_map: dict
) -> dict:
    """
    Invert assembly_neuron_map to get {neuron_name: [ca_id, ...]} for the assembler.
    """
    memberships: dict[str, list] = {}
    for ca in cell_assemblies:
        ca_id   = ca["id"]
        members = assembly_neuron_map.get(ca_id, [])
        for name in members:
            memberships.setdefault(name, []).append(ca_id)
    return memberships


def save_output(
    all_proposals: dict,
    peer_reviews: dict,
    cell_assemblies: list,
    assembly_neuron_map: dict,
    neuron_assembly_memberships: dict,
    cfg: dict,
    output_path: Path,
):
    output = {
        "metadata": {
            "date":             date.today().isoformat(),
            "scope":            cfg["label"],
            "models_queried":   ["claude", "chatgpt", "grok", "gemini"],
            "n_cell_assemblies": len(cell_assemblies),
            "n_neurons_assigned": len(neuron_assembly_memberships),
        },
        "stage_1_proposals":  all_proposals,
        "stage_2_peer_reviews": {k: v for k, v in peer_reviews.items() if v is not None},
        "cell_assemblies":    cell_assemblies,
        "assembly_neuron_map": assembly_neuron_map,
        "neuron_assembly_memberships": neuron_assembly_memberships,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved → {output_path.name}")


def print_summary(cell_assemblies: list, assembly_neuron_map: dict, cfg: dict):
    print("\n" + "=" * 60)
    print(f"{cfg['label'].upper()} CELL ASSEMBLIES ({len(cell_assemblies)} total)")
    print("=" * 60)
    for ca in cell_assemblies:
        ps_str   = ", ".join(ca.get("ps_memberships", []))
        members  = assembly_neuron_map.get(ca["id"], [])
        mem_str  = ", ".join(members[:4]) + (f" +{len(members)-4} more" if len(members) > 4 else "")
        print(f"  {ca['id']}: {ca['name']}  [{ps_str}]")
        print(f"       {ca.get('description','')[:80]}")
        if members:
            print(f"       Members: {mem_str}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(scope: str, skip_assignment: bool = False):
    cfg = SCOPE_CONFIG[scope]

    canon_path  = SCRIPT_DIR / cfg["ps_canon_file"]
    output_path = SCRIPT_DIR / cfg["output_file"]

    print("=" * 60)
    print(f"CA COUNCIL [{cfg['label'].upper()}]")
    print(f"Date: {date.today().isoformat()}")
    print("=" * 60)

    if not canon_path.exists():
        print(f"ERROR: {canon_path.name} not found.")
        print(f"  Run: python ps_council.py --scope {scope}")
        return

    print(f"\nLoading {canon_path.name}...")
    ps_canon        = load_ps_canon(canon_path)
    phase_sequences = ps_canon.get("phase_sequences", [])
    neuron_weights  = ps_canon.get("initial_neuron_weights", {})

    print(f"  {len(phase_sequences)} phase sequences")
    print(f"  {len(neuron_weights)} neurons with weights")

    if not phase_sequences:
        print("ERROR: No phase sequences in ps_canon.")
        return
    if not neuron_weights:
        print("WARNING: No neuron weights in ps_canon — blinded lists will be empty.")

    # Stage 1
    all_proposals = stage_1_proposals(phase_sequences, neuron_weights, cfg)
    active = {m: p for m, p in all_proposals.items() if p}
    print(f"\n--- Stage 1 Summary ---")
    for m, p in all_proposals.items():
        total  = sum(len(v) for v in p.values()) if p else 0
        status = f"{total} assemblies" if p else "FAILED"
        print(f"  {'✓' if p else '✗'} {m}: {status}")

    if len(active) < 2:
        print("\nERROR: Too few successful proposals. Aborting.")
        return

    # Stage 2
    peer_reviews = stage_2_peer_review(all_proposals, phase_sequences, cfg)

    # Stage 3
    try:
        cell_assemblies = stage_3_chairman_synthesis(
            all_proposals, peer_reviews, phase_sequences, cfg
        )
    except Exception as e:
        print(f"\n! Chairman synthesis failed: {e}")
        return

    # Stage 4 (optional)
    assembly_neuron_map          = {ca["id"]: [] for ca in cell_assemblies}
    neuron_assembly_memberships  = {}

    if not skip_assignment:
        try:
            assembly_neuron_map         = stage_4_neuron_assignment(
                phase_sequences, cell_assemblies, neuron_weights, cfg
            )
            neuron_assembly_memberships = build_neuron_assembly_memberships(
                cell_assemblies, assembly_neuron_map
            )
        except Exception as e:
            print(f"\n! Neuron assignment failed: {e}")
            print("  Saving output without assignments.")

    # Save
    print("\nSaving output...")
    save_output(
        all_proposals, peer_reviews, cell_assemblies,
        assembly_neuron_map, neuron_assembly_memberships,
        cfg, output_path,
    )
    print_summary(cell_assemblies, assembly_neuron_map, cfg)

    print("\n" + "=" * 60)
    print("CA Council session complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CA Council: LLM Council for Cell Assembly Generation"
    )
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which superorganism to generate cell assemblies for (default: us)"
    )
    parser.add_argument(
        "--skip-assignment", action="store_true",
        help="Skip Stage 4 neuron assignment (produce canonical assembly list only)"
    )
    args = parser.parse_args()
    run(scope=args.scope, skip_assignment=args.skip_assignment)
