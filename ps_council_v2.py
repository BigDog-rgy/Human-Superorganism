"""
PS Council V2 — Phase Sequence Generation with CA-PS Assignment

Reads final_ranked_{scope}.json and ca_canon_v2_{scope}.json to generate
canonical phase sequences and CA-PS membership links.

Stages:
    1. Proposals — top 2*sqrt(top_n) anchor neurons (shuffled, scores stripped)
       and their cell assemblies are shown to a 4-model LLM council. Each model
       independently proposes 10-12 phase sequences.
    2. Peer Review — each model reviews the other 3 proposals anonymously and
       ranks them.
    3. Chairman Synthesis — Claude synthesizes the official canonical PS list
       from all proposals and reviews.
    4. CA-PS Assignment — the full CA list is batched (CA_BATCH_SIZE CAs/batch)
       and sent to all 4 models for PS assignment proposals. Majority vote
       (>= MAJORITY_THRESHOLD of 4 models) determines candidates per batch.
       A single final chairman pass reviews all majority candidates and
       unassigned CAs to produce the definitive ca_ps_map.

Output: ps_canon_v2_{scope}.json

Usage:
    python ps_council_v2.py --scope us
    python ps_council_v2.py --scope global
    python ps_council_v2.py --scope us --ca-batch-size 50
    python ps_council_v2.py --scope global --from-stage 4   # rerun CA-PS only
"""

import os
import json
import math
import random
import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from google import genai

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# SCOPE CONFIG
# ---------------------------------------------------------------------------

SCOPE_CONFIG = {
    "us": {
        "label":             "US",
        "ps_id_prefix":      "DPS",
        "ps_bias_field":     "sector_bias",
        "ps_bias_options":   (
            "Federal Executive, Technology, AI/ML, Finance, Defense, "
            "Media, Energy, Judiciary, or Cross-sector"
        ),
        "scope_description": (
            "US domestic power dynamics — politics, technology, finance, "
            "media, defense, energy, judiciary"
        ),
        "n_ps_target":       "8-10",
        "input_file":        "final_ranked_us.json",
        "ca_canon_file":     "ca_canon_v2_us.json",
        "output_file":       "ps_canon_v2_us.json",
        "top_n":             150,
    },
    "global": {
        "label":             "Global",
        "ps_id_prefix":      "PS",
        "ps_bias_field":     "hemisphere_bias",
        "ps_bias_options":   (
            "West, East, Bridge, Split (contested between West and East), or Global"
        ),
        "scope_description": (
            "Global geopolitics — US-China competition, military, finance, "
            "technology, information, energy, regional realignments"
        ),
        "n_ps_target":       "10-12",
        "input_file":        "final_ranked_global.json",
        "ca_canon_file":     "ca_canon_v2_global.json",
        "output_file":       "ps_canon_v2_global.json",
        "top_n":             300,
    },
}

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CA_BATCH_SIZE_DEFAULT = 25
MAJORITY_THRESHOLD    = 3      # min votes out of 4 to include in majority candidates

CLAUDE_COUNCIL_MODEL   = "claude-sonnet-4-6"   # used when Claude acts as a council member
CLAUDE_CHAIRMAN_MODEL  = "claude-opus-4-6"     # used when Claude acts as chairman
GPT_MODEL              = "gpt-5.2"
GROK_MODEL             = "grok-4-1-fast-reasoning"
GEMINI_MODEL           = "gemini-3-pro-preview"

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


def load_ranked_neurons(path: Path, top_n: int) -> list:
    """Load final_ranked_{scope}.json and return top_n entries."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("superorganism_list", [])
    return entries[:top_n]


def load_ca_canon(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_anchor_neurons(neurons: list, top_n: int) -> list:
    """
    Take the top 2*sqrt(top_n) neurons, shuffle them, and strip ELO/rank
    metadata so council members see names and titles only.
    """
    anchor_n = round(2 * math.sqrt(top_n))
    anchor_n = min(anchor_n, len(neurons))
    selected = [
        {"name": n["name"], "title": n.get("title", "")}
        for n in neurons[:anchor_n]
    ]
    random.shuffle(selected)
    return selected


def build_anchor_context(anchor_neurons: list, ca_canon: dict) -> tuple[list, str]:
    """
    Build the neuron + CA context block for Stage 1 and Stage 3 prompts.
    Returns (ca_ids_shown, context_text).
    Each anchor neuron is shown with its cell assemblies (name + description).
    """
    cell_assemblies = ca_canon.get("cell_assemblies", [])
    neuron_ca_map   = ca_canon.get("neuron_ca_map", {})
    ca_by_id        = {ca["id"]: ca for ca in cell_assemblies}

    ca_ids_shown: set = set()
    lines: list       = []

    for neuron in anchor_neurons:
        name  = neuron["name"]
        title = neuron.get("title", "")
        header = name + (f" — {title}" if title else "")
        lines.append(header)

        ca_ids = neuron_ca_map.get(name, [])
        if ca_ids:
            for ca_id in ca_ids:
                ca = ca_by_id.get(ca_id)
                if ca:
                    lines.append(f"  • [{ca_id}] {ca['name']}: {ca.get('description', '')}")
                    ca_ids_shown.add(ca_id)
        else:
            lines.append("  • (no cell assemblies assigned)")
        lines.append("")

    return list(ca_ids_shown), "\n".join(lines)


# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------

def checkpoint_path(scope: str, stage: int) -> Path:
    return SCRIPT_DIR / f"ps_council_v2_{scope}_s{stage}_checkpoint.json"


def save_checkpoint(scope: str, stage: int, **data):
    path = checkpoint_path(scope, stage)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [checkpoint saved: {path.name}]")


def load_checkpoint(scope: str, stage: int) -> dict | None:
    path = checkpoint_path(scope, stage)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STAGE 1: PROPOSALS
# ---------------------------------------------------------------------------

def build_proposal_prompt(
    anchor_neurons: list,
    anchor_context: str,
    ca_ids_shown: list,
    cfg: dict,
    total_neurons: int,
    total_cas: int,
) -> str:
    label        = cfg["label"]
    prefix       = cfg["ps_id_prefix"]
    n_target     = cfg["n_ps_target"]
    bias_field   = cfg["ps_bias_field"]
    bias_options = cfg["ps_bias_options"]
    scope_desc   = cfg["scope_description"]
    anchor_n     = len(anchor_neurons)
    ca_n         = len(ca_ids_shown)

    return f"""You are helping design canonical phase sequences for the {label} superorganism model.

## CONTEXT

In this Hebbian superorganism framework:
- **Neurons** = individual prime movers (key actors)
- **Cell assemblies** = organizations, coalitions, or institutional clusters that neurons belong to
- **Phase sequences** = major ongoing structural dynamics that cell assemblies and neurons participate in

The full {label} actor pool contains {total_neurons} neurons and {total_cas} cell assemblies.
You are seeing {anchor_n} representative anchor neurons and {ca_n} of their associated assemblies
as a structural sample. Your proposed phase sequences must be broad enough to cover the full
range of dynamics across the complete pool — not just these anchor actors.

## ANCHOR NEURONS AND THEIR CELL ASSEMBLIES

{anchor_context}

## YOUR TASK

Propose {n_target} canonical phase sequences for the {label} superorganism.
Focus on: {scope_desc}

Each phase sequence should:
- Represent a major, ongoing structural dynamic (not a one-time event)
- Be meaningfully distinct from the others (minimal overlap)
- Be operationalizable for weekly news tracking
- Be broad enough that multiple cell assemblies from the full pool participate in it

For each phase sequence, provide:
- `id`: sequential ID ({prefix}-01, {prefix}-02, ...)
- `name`: a concise label (2-5 words)
- `definition`: one clear sentence explaining what this phase sequence represents and how it can be tracked
- `{bias_field}`: one of {bias_options}

Return only a JSON array, no preamble:
[
  {{
    "id": "{prefix}-01",
    "name": "...",
    "definition": "...",
    "{bias_field}": "..."
  }},
  ...
]"""


def _call_proposal(model_name: str, prompt: str,
                   anthropic_client, openai_client, xai_client, gemini_client) -> list:
    if model_name == "claude":
        msg = anthropic_client.messages.create(
            model=CLAUDE_COUNCIL_MODEL, max_tokens=2048, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_list(msg.content[0].text)
    elif model_name == "chatgpt":
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL, temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an expert analyst of power dynamics and institutional behavior."},
                {"role": "user",   "content": prompt},
            ],
        )
        return extract_json_list(resp.choices[0].message.content)
    elif model_name == "grok":
        resp = xai_client.chat.completions.create(
            model=GROK_MODEL, temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an expert analyst of power dynamics and institutional behavior."},
                {"role": "user",   "content": prompt},
            ],
        )
        return extract_json_list(resp.choices[0].message.content)
    elif model_name == "gemini":
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return extract_json_list(resp.text)
    raise ValueError(f"Unknown model: {model_name}")


def stage_1_proposals(
    anchor_neurons: list,
    anchor_context: str,
    ca_ids_shown: list,
    cfg: dict,
    total_neurons: int,
    total_cas: int,
) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 1: INDEPENDENT PROPOSALS")
    print("=" * 60)
    print(f"\n  Anchor neurons: {len(anchor_neurons)}  |  CAs in context: {len(ca_ids_shown)}")
    print(f"  Full pool: {total_neurons} neurons, {total_cas} CAs\n")

    prompt = build_proposal_prompt(
        anchor_neurons, anchor_context, ca_ids_shown, cfg, total_neurons, total_cas
    )

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
            print(f"  ✓ {model_name}: {len(proposals)} phase sequences proposed")
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")
            results[model_name] = []

    return results


# ---------------------------------------------------------------------------
# STAGE 2: PEER REVIEW
# ---------------------------------------------------------------------------

def build_peer_review_prompt(other_proposals: dict, cfg: dict) -> str:
    label        = cfg["label"]
    review_block = ""
    for lbl, proposals in other_proposals.items():
        review_block += f"\n### {lbl}'s Proposals:\n"
        for ps in proposals:
            review_block += (
                f"  {ps.get('id','?')}: {ps.get('name','?')} — {ps.get('definition','')}\n"
            )

    return f"""You previously proposed a set of {label} phase sequences for a superorganism model.

Now review these three independent proposals from other models (labeled Model A, B, C):
{review_block}
Rank these three responses from best to worst based on:
1. Coverage — do they capture the most important {label} power dynamics across the full actor pool?
2. Distinctiveness — are the sequences meaningfully separate (no major overlaps)?
3. Operationalizability — can each sequence be tracked with weekly news?
4. Breadth — are the sequences general enough to cover dynamics beyond just the anchor neurons shown?

Return only a JSON object:
{{
  "rankings": [
    {{"model": "Model A", "rank": 1, "reasoning": "brief explanation"}},
    {{"model": "Model B", "rank": 2, "reasoning": "brief explanation"}},
    {{"model": "Model C", "rank": 3, "reasoning": "brief explanation"}}
  ]
}}"""


def _call_peer_review(reviewer: str, prompt: str,
                      anthropic_client, openai_client, xai_client, gemini_client) -> dict:
    if reviewer == "claude":
        msg = anthropic_client.messages.create(
            model=CLAUDE_COUNCIL_MODEL, max_tokens=1024, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(msg.content[0].text)
    elif reviewer == "chatgpt":
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif reviewer == "grok":
        resp = xai_client.chat.completions.create(
            model=GROK_MODEL, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_obj(resp.choices[0].message.content)
    elif reviewer == "gemini":
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return extract_json_obj(resp.text)
    raise ValueError(f"Unknown reviewer: {reviewer}")


def stage_2_peer_review(all_proposals: dict, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 2: PEER REVIEW")
    print("=" * 60)
    print("\n  Each model reviews the other three proposals...\n")

    model_names  = [m for m in ("claude", "chatgpt", "grok", "gemini") if all_proposals.get(m)]
    peer_reviews = {}

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    for reviewer in model_names:
        other   = {}
        lbl_map = {}
        labels  = ["Model A", "Model B", "Model C"]
        idx     = 0
        for m in model_names:
            if m != reviewer and all_proposals.get(m):
                other[labels[idx]]   = all_proposals[m]
                lbl_map[labels[idx]] = m
                idx += 1

        prompt = build_peer_review_prompt(other, cfg)
        print(f"  Getting review from {reviewer}...")
        try:
            review_data = _call_peer_review(
                reviewer, prompt,
                anthropic_client, openai_client, xai_client, gemini_client,
            )
            for ranking in review_data.get("rankings", []):
                ranking["actual_model"] = lbl_map.get(ranking["model"], "unknown")
            peer_reviews[reviewer] = review_data
            print(f"  ✓ {reviewer} review complete")
        except Exception as e:
            print(f"  ✗ {reviewer}: {e}")
            peer_reviews[reviewer] = None

    return peer_reviews


# ---------------------------------------------------------------------------
# STAGE 3: CHAIRMAN SYNTHESIS
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    all_proposals: dict,
    peer_reviews: dict,
    anchor_context: str,
    cfg: dict,
) -> str:
    prefix     = cfg["ps_id_prefix"]
    bias_field = cfg["ps_bias_field"]
    n_target   = cfg["n_ps_target"]
    label      = cfg["label"]

    proposals_block = ""
    for model_name, proposals in all_proposals.items():
        if not proposals:
            continue
        proposals_block += f"\n### {model_name.upper()}'s Proposals:\n"
        for ps in proposals:
            proposals_block += (
                f"  {ps.get('id','?')}: {ps.get('name','?')} "
                f"[{ps.get(bias_field,'')}] — {ps.get('definition','')}\n"
            )

    reviews_block = ""
    for reviewer, review in peer_reviews.items():
        if not review:
            continue
        reviews_block += f"\n### {reviewer.upper()}'s Rankings:\n"
        for r in review.get("rankings", []):
            reviews_block += f"  {r['rank']}. {r['actual_model']} — {r.get('reasoning','')}\n"

    return f"""You are the Chairman of a PS Council defining canonical phase sequences for the {label} superorganism model.

You have received independent proposals from four leading AI models and their peer reviews.
All proposals were based on the representative anchor neurons and cell assemblies shown below.

## ANCHOR CONTEXT (what the council members saw)

{anchor_context}

## ALL PROPOSALS
{proposals_block}

## PEER REVIEWS
{reviews_block}

## YOUR TASK

Synthesize these inputs into a definitive canonical list of {n_target} phase sequences for the {label} superorganism.

Rules:
- Choose the count that best captures the major dynamics (within the {n_target} range)
- Each sequence must be meaningfully distinct — merge any overlapping proposals
- Sequences must be broad enough to cover dynamics across the full actor pool, not just the anchor neurons
- Assign sequential IDs: {prefix}-01, {prefix}-02, ...
- Definitions must be one sentence and operationalizable for weekly news tracking
- Use your judgment — curate, don't just average

Return only a JSON array, no preamble:
[
  {{
    "id": "{prefix}-01",
    "name": "...",
    "definition": "...",
    "{bias_field}": "..."
  }},
  ...
]"""


def stage_3_chairman_synthesis(
    all_proposals: dict,
    peer_reviews: dict,
    anchor_context: str,
    cfg: dict,
) -> list:
    print("\n" + "=" * 60)
    print("STAGE 3: CHAIRMAN SYNTHESIS")
    print("=" * 60)
    print("\n  Claude (Chairman) synthesizing final phase sequences...\n")

    client  = get_anthropic_client()
    prompt  = build_synthesis_prompt(all_proposals, peer_reviews, anchor_context, cfg)
    message = client.messages.create(
        model=CLAUDE_CHAIRMAN_MODEL, max_tokens=4096, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    phase_sequences = extract_json_list(message.content[0].text)
    print(f"  ✓ Chairman synthesis complete: {len(phase_sequences)} phase sequences")
    for ps in phase_sequences:
        print(f"    {ps.get('id','?')}: {ps.get('name','?')}")
    return phase_sequences


# ---------------------------------------------------------------------------
# STAGE 4: CA-PS ASSIGNMENT
# ---------------------------------------------------------------------------

def build_ca_batch_prompt(
    ca_batch: list,
    phase_sequences: list,
    cfg: dict,
    batch_num: int,
    total_batches: int,
) -> str:
    label      = cfg["label"]
    bias_field = cfg["ps_bias_field"]

    ps_block = ""
    for ps in phase_sequences:
        ps_block += (
            f"  {ps['id']}: {ps['name']} [{ps.get(bias_field, '')}]\n"
            f"    {ps.get('definition', '')}\n\n"
        )

    ca_block = ""
    for ca in ca_batch:
        members      = ca.get("member_neurons", [])
        members_str  = ", ".join(members[:8])
        if len(members) > 8:
            members_str += f" +{len(members) - 8} more"
        ca_block += (
            f"  {ca['id']}: {ca['name']}\n"
            f"    Description: {ca.get('description', '')}\n"
            f"    Members: {members_str}\n\n"
        )

    return f"""You are assigning cell assemblies to phase sequences in the {label} superorganism model.

## PHASE SEQUENCES (canonical — do not modify)

{ps_block}

## YOUR TASK (batch {batch_num} of {total_batches})

For each cell assembly below, identify which phase sequence(s) it participates in.
A cell assembly participates in a phase sequence if it routinely activates as part of that structural dynamic.

Rules:
- A CA may belong to 0, 1, or multiple phase sequences
- Leave ps_ids empty if the CA does not clearly fit any phase sequence
- Only assign when there is a genuine structural relationship, not just thematic proximity

## CELL ASSEMBLIES

{ca_block}

Return only a JSON array covering every CA in this batch:
[
  {{"ca_id": "CA-001", "ps_ids": ["PS-01", "PS-03"]}},
  {{"ca_id": "CA-002", "ps_ids": []}},
  ...
]"""


def _call_ca_assignment(model_name: str, prompt: str,
                        anthropic_client, openai_client, xai_client, gemini_client) -> list:
    if model_name == "claude":
        msg = anthropic_client.messages.create(
            model=CLAUDE_COUNCIL_MODEL, max_tokens=2048, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_list(msg.content[0].text)
    elif model_name == "chatgpt":
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_list(resp.choices[0].message.content)
    elif model_name == "grok":
        resp = xai_client.chat.completions.create(
            model=GROK_MODEL, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_json_list(resp.choices[0].message.content)
    elif model_name == "gemini":
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return extract_json_list(resp.text)
    raise ValueError(f"Unknown model: {model_name}")


def compute_majority_votes(
    batch_results: list,   # one list of assignments per model
    ca_batch: list,
    threshold: int,
) -> dict:
    """
    For each CA in the batch, tally how many models assigned each PS.
    A (CA, PS) link is included if vote count >= threshold.

    Returns:
      ca_ps_batch: {ca_id: [ps_id, ...]}  — empty list for CAs with no majority link
    """
    votes: dict = defaultdict(lambda: defaultdict(int))
    for model_result in batch_results:
        if not model_result:
            continue
        for item in model_result:
            ca_id = item.get("ca_id", "")
            for ps_id in item.get("ps_ids", []):
                votes[ca_id][ps_id] += 1

    ca_ps_batch: dict = {}
    for ca in ca_batch:
        ca_id  = ca["id"]
        agreed = [
            ps_id for ps_id, count in votes.get(ca_id, {}).items()
            if count >= threshold
        ]
        ca_ps_batch[ca_id] = agreed

    return ca_ps_batch


def stage_4_ca_ps_assignment(
    cell_assemblies: list,
    phase_sequences: list,
    cfg: dict,
    ca_batch_size: int,
    vote_threshold: int,
) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 4: CA-PS ASSIGNMENT")
    print("=" * 60)
    print(f"\n  Vote threshold: ≥{vote_threshold}/4 models to assign a link")

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    batches       = [
        cell_assemblies[i:i + ca_batch_size]
        for i in range(0, len(cell_assemblies), ca_batch_size)
    ]
    total_batches = len(batches)
    print(f"  {len(cell_assemblies)} CAs → {total_batches} batches of ≤{ca_batch_size}\n")

    ca_ps_map: dict = {}

    for batch_idx, ca_batch in enumerate(batches, start=1):
        print(f"  Batch {batch_idx}/{total_batches} ({len(ca_batch)} CAs)...")
        prompt = build_ca_batch_prompt(
            ca_batch, phase_sequences, cfg, batch_idx, total_batches
        )

        batch_results = []
        for model_name in ("claude", "chatgpt", "grok", "gemini"):
            try:
                result = _call_ca_assignment(
                    model_name, prompt,
                    anthropic_client, openai_client, xai_client, gemini_client,
                )
                batch_results.append(result)
                print(f"    ✓ {model_name}")
            except Exception as e:
                print(f"    ✗ {model_name}: {e}")
                batch_results.append([])

        batch_map     = compute_majority_votes(batch_results, ca_batch, vote_threshold)
        ca_ps_map.update(batch_map)
        assigned      = sum(1 for v in batch_map.values() if v)
        unassigned    = sum(1 for v in batch_map.values() if not v)
        print(f"    Assigned: {assigned} | Unassigned: {unassigned}\n")

    assigned_count   = sum(1 for v in ca_ps_map.values() if v)
    unassigned_count = sum(1 for v in ca_ps_map.values() if not v)
    print(f"  Stage 4 complete: {assigned_count} CAs assigned, {unassigned_count} unassigned")

    return ca_ps_map


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_output(
    scope: str,
    cfg: dict,
    all_proposals: dict,
    peer_reviews: dict,
    phase_sequences: list,
    ca_ps_map: dict,
    anchor_neurons: list,
    output_path: Path,
    vote_threshold: int,
):
    assigned_count   = sum(1 for v in ca_ps_map.values() if v)
    unassigned_count = sum(1 for v in ca_ps_map.values() if not v)

    output = {
        "metadata": {
            "date":              date.today().isoformat(),
            "scope":             scope,
            "label":             cfg["label"],
            "source_neurons":    cfg["input_file"],
            "source_ca_canon":   cfg["ca_canon_file"],
            "anchor_n":          len(anchor_neurons),
            "total_n":           cfg["top_n"],
            "n_phase_sequences": len(phase_sequences),
            "ca_ps_vote_threshold": vote_threshold,
            "n_cas_assigned":    assigned_count,
            "n_cas_unassigned":  unassigned_count,
        },
        "stage_1_proposals":   all_proposals,
        "stage_2_peer_reviews": peer_reviews,
        "phase_sequences":     phase_sequences,
        "ca_ps_map":           ca_ps_map,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Output saved: {output_path.name}")
    print(f"  Phase sequences: {len(phase_sequences)}")
    print(f"  CAs assigned: {assigned_count}  |  CAs unassigned: {unassigned_count}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(scope: str, ca_batch_size: int, from_stage: int, vote_threshold: int):
    cfg         = SCOPE_CONFIG[scope]
    input_path  = SCRIPT_DIR / cfg["input_file"]
    ca_path     = SCRIPT_DIR / cfg["ca_canon_file"]
    output_path = SCRIPT_DIR / cfg["output_file"]

    print("=" * 60)
    print(f"PS COUNCIL V2 [{cfg['label'].upper()}]")
    print(f"Date: {date.today().isoformat()}")
    print(f"CA batch size: {ca_batch_size}  |  Vote threshold: ≥{vote_threshold}/4")
    if from_stage > 1:
        print(f"Resuming from Stage {from_stage}")
    print("=" * 60)

    # Validate inputs
    if not input_path.exists():
        print(f"ERROR: {input_path.name} not found.")
        return
    if not ca_path.exists():
        print(f"ERROR: {ca_path.name} not found.")
        print(f"  Run: python ca_council_v2.py --scope {scope}")
        return

    # Load source data (always, so anchor context is available for output metadata)
    neurons         = load_ranked_neurons(input_path, cfg["top_n"])
    ca_canon        = load_ca_canon(ca_path)
    cell_assemblies = ca_canon.get("cell_assemblies", [])
    total_neurons   = len(neurons)
    total_cas       = len(cell_assemblies)

    print(f"\nLoaded {total_neurons} neurons from {input_path.name}")
    print(f"Loaded {total_cas} cell assemblies from {ca_path.name}")

    # Build anchor context (fresh each run — shuffle is intentional)
    anchor_neurons               = select_anchor_neurons(neurons, cfg["top_n"])
    ca_ids_shown, anchor_context = build_anchor_context(anchor_neurons, ca_canon)
    print(f"Anchor neurons: {len(anchor_neurons)}  |  CAs in context: {len(ca_ids_shown)}")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if from_stage <= 1:
        all_proposals = stage_1_proposals(
            anchor_neurons, anchor_context, ca_ids_shown, cfg, total_neurons, total_cas
        )
        save_checkpoint(scope, 1,
                        all_proposals=all_proposals,
                        anchor_neurons=anchor_neurons,
                        anchor_context=anchor_context,
                        ca_ids_shown=ca_ids_shown)
    else:
        ckpt = load_checkpoint(scope, 1)
        if not ckpt:
            print("ERROR: Stage 1 checkpoint not found. Run without --from-stage.")
            return
        all_proposals  = ckpt["all_proposals"]
        anchor_neurons = ckpt["anchor_neurons"]
        anchor_context = ckpt["anchor_context"]
        ca_ids_shown   = ckpt["ca_ids_shown"]
        total_proposed = sum(len(v) for v in all_proposals.values())
        print(f"\n  Stage 1 restored: {total_proposed} proposals across "
              f"{sum(1 for v in all_proposals.values() if v)} models")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if from_stage <= 2:
        peer_reviews = stage_2_peer_review(all_proposals, cfg)
        save_checkpoint(scope, 2, peer_reviews=peer_reviews)
    else:
        ckpt = load_checkpoint(scope, 2)
        if not ckpt:
            print("ERROR: Stage 2 checkpoint not found.")
            return
        peer_reviews = ckpt["peer_reviews"]
        print(f"\n  Stage 2 restored: reviews from {list(peer_reviews.keys())}")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if from_stage <= 3:
        phase_sequences = stage_3_chairman_synthesis(
            all_proposals, peer_reviews, anchor_context, cfg
        )
        save_checkpoint(scope, 3, phase_sequences=phase_sequences)
    else:
        ckpt = load_checkpoint(scope, 3)
        if not ckpt:
            print("ERROR: Stage 3 checkpoint not found.")
            return
        phase_sequences = ckpt["phase_sequences"]
        print(f"\n  Stage 3 restored: {len(phase_sequences)} phase sequences")
        for ps in phase_sequences:
            print(f"    {ps.get('id','?')}: {ps.get('name','?')}")

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    ca_ps_map = stage_4_ca_ps_assignment(
        cell_assemblies, phase_sequences, cfg, ca_batch_size, vote_threshold
    )

    save_output(
        scope, cfg, all_proposals, peer_reviews,
        phase_sequences, ca_ps_map, anchor_neurons, output_path, vote_threshold
    )


def main():
    parser = argparse.ArgumentParser(description="PS Council V2 — Phase Sequence Generation")
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which superorganism to process (default: us)"
    )
    parser.add_argument(
        "--ca-batch-size", type=int, default=CA_BATCH_SIZE_DEFAULT,
        help=f"CAs per batch in Stage 4 (default: {CA_BATCH_SIZE_DEFAULT})"
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, choices=[1, 2, 3, 4],
        help=(
            "Resume from a stage using its saved checkpoint (default: 1 = full run). "
            "2 = skip Stage 1; 3 = skip Stages 1-2; 4 = skip Stages 1-3 (rerun CA-PS only)"
        )
    )
    parser.add_argument(
        "--vote-threshold", type=int, default=MAJORITY_THRESHOLD,
        help=(
            f"Minimum votes (out of 4) for a CA-PS link to be included (default: {MAJORITY_THRESHOLD}). "
            "Lower = more links, higher = stricter."
        )
    )
    args = parser.parse_args()
    run(args.scope, args.ca_batch_size, args.from_stage, args.vote_threshold)


if __name__ == "__main__":
    main()
