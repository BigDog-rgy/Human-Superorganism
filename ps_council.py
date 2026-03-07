"""
PS Council — LLM Council for Phase Sequence Generation
Three-stage council process to generate canonical phase sequences
and initial neuron-PS weights for the US or global superorganism.

Usage:
    python ps_council.py --scope us
    python ps_council.py --scope global
    python ps_council.py --scope global --no-weights
    python ps_council.py --scope us --prime-movers-file custom.json

Stages:
    1. Independent proposals — all 4 LLMs propose phase sequences
    2. Peer review           — each reviews the other 3 anonymously
    3. Chairman synthesis    — Claude synthesizes final canonical list
    4. Neuron-PS weights     — Claude assigns initial weights (anchor-augmented batching)

Output: ps_canon_{scope}.json
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

# ---------------------------------------------------------------------------
# SCOPE CONFIG
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

SCOPE_CONFIG = {
    "us": {
        "label":             "US",
        "ps_id_prefix":      "DPS",
        "ps_bias_field":     "sector_bias",
        "scope_description": (
            "US domestic power dynamics — politics, technology, finance, "
            "media, defense, energy, judiciary"
        ),
        "ps_bias_options":   (
            "Federal Executive, Technology, AI/ML, Finance, Defense, "
            "Media, Energy, Judiciary, or Cross-sector"
        ),
        "input_file":        "final_ranked_us.json",
        "output_file":       "ps_canon_us.json",
        "n_ps_target":       "8-10",
    },
    "global": {
        "label":             "Global",
        "ps_id_prefix":      "PS",
        "ps_bias_field":     "hemisphere_bias",
        "scope_description": (
            "Global geopolitics — US-China competition, military, finance, "
            "technology, information, energy, regional realignments"
        ),
        "ps_bias_options":   (
            "West, East, Bridge, Split (contested between West and East), or Global"
        ),
        "input_file":        "final_ranked_global.json",
        "output_file":       "ps_canon_global.json",
        "n_ps_target":       "10-12",
    },
}

CLAUDE_MODEL = "claude-opus-4-6"
GPT_MODEL    = "gpt-5.2"
GROK_MODEL   = "grok-4-1-fast-reasoning"
GEMINI_MODEL = "gemini-3-pro-preview"

WEIGHTS_BATCH_SIZE  = 28   # new neurons per batch in anchor-augmented batching
WEIGHTS_ANCHOR_SIZE = 4    # anchors from top tier shown in each subsequent batch


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

def load_prime_movers(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both direct array (final_ranked_*.json) and legacy LLM council format
    return data if isinstance(data, list) else data["stage_3_final_list"]


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


# ---------------------------------------------------------------------------
# STAGE 1: INDEPENDENT PROPOSALS
# ---------------------------------------------------------------------------

def build_proposal_prompt(prime_movers: list, cfg: dict) -> str:
    people_block = "\n".join(
        f"  {p['rank']}. {p['name']} | {p.get('title', p.get('domain', ''))}"
        for p in prime_movers
    )
    prefix       = cfg["ps_id_prefix"]
    n_target     = cfg["n_ps_target"]
    bias_field   = cfg["ps_bias_field"]
    bias_options = cfg["ps_bias_options"]
    scope_desc   = cfg["scope_description"]
    label        = cfg["label"]

    return f"""You are helping design the canonical phase sequences for a {label} superorganism model.

## CONTEXT

In this Hebbian superorganism framework:
- **Neurons** = individual prime movers (the people listed below)
- **Phase sequences** = major ongoing structural dynamics that these neurons participate in
- Phase sequences represent the key forces and competitions that define {label} power right now

## PRIME MOVERS TO CONSIDER

These are the most influential {label} actors as of {date.today().isoformat()}:

{people_block}

## YOUR TASK

Propose {n_target} canonical phase sequences for the {label} superorganism.

Focus on: {scope_desc}

Each phase sequence should:
- Represent a major, ongoing structural dynamic (not a one-time event)
- Be distinct from the others (no meaningful overlap)
- Involve at least 2-3 of the prime movers above
- Be operationalizable for weekly news tracking

For each phase sequence, provide:
- `id`: sequential ID ({prefix}-01, {prefix}-02, ...)
- `name`: a concise label (2-5 words)
- `definition`: one clear sentence explaining what this phase sequence represents
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
            model=CLAUDE_MODEL, max_tokens=2048, temperature=0.7,
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


def stage_1_proposals(prime_movers: list, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 1: INDEPENDENT PROPOSALS")
    print("=" * 60)
    print(f"\nQuerying 4 LLMs for {cfg['label']} phase sequence proposals...\n")

    prompt = build_proposal_prompt(prime_movers, cfg)

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
    label = cfg["label"]
    review_block = ""
    for lbl, proposals in other_proposals.items():
        review_block += f"\n### {lbl}'s Proposals:\n"
        for ps in proposals[:7]:
            review_block += f"  {ps.get('id','?')}: {ps.get('name','?')} — {ps.get('definition','')}\n"
        if len(proposals) > 7:
            review_block += f"  ... and {len(proposals) - 7} more\n"

    return f"""You previously proposed a set of {label} phase sequences for a superorganism model.

Now review these three independent proposals from other models (labeled Model A, B, C):
{review_block}
Rank these three responses from best to worst based on:
1. Coverage — do they capture the most important {label} power dynamics?
2. Distinctiveness — are the sequences meaningfully separate (no major overlaps)?
3. Operationalizability — can each sequence be tracked with weekly news?
4. Relevance to the actual prime movers in this scope

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
            model=CLAUDE_MODEL, max_tokens=1024,
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
    print("\nEach model reviews the other three proposals...\n")

    model_names = [m for m in ("claude", "chatgpt", "grok", "gemini") if all_proposals.get(m)]
    peer_reviews = {}

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    for reviewer in model_names:
        other    = {}
        lbl_map  = {}
        labels   = ["Model A", "Model B", "Model C"]
        idx      = 0
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

def build_synthesis_prompt(all_proposals: dict, peer_reviews: dict, cfg: dict) -> str:
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

## ALL PROPOSALS
{proposals_block}

## PEER REVIEWS
{reviews_block}

## YOUR TASK

Synthesize these inputs into a definitive canonical list of {n_target} phase sequences for the {label} superorganism.

Rules:
- Choose the count that best captures the major dynamics (within {n_target} range)
- Each sequence must be meaningfully distinct — merge any overlapping proposals
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


def stage_3_chairman_synthesis(all_proposals: dict, peer_reviews: dict, cfg: dict) -> list:
    print("\n" + "=" * 60)
    print("STAGE 3: CHAIRMAN SYNTHESIS")
    print("=" * 60)
    print("\nClaude (Chairman) synthesizing final phase sequences...\n")

    client  = get_anthropic_client()
    prompt  = build_synthesis_prompt(all_proposals, peer_reviews, cfg)
    message = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    phase_sequences = extract_json_list(message.content[0].text)
    print(f"✓ Chairman synthesis complete: {len(phase_sequences)} phase sequences")
    return phase_sequences


# ---------------------------------------------------------------------------
# STAGE 4: INITIAL NEURON-PS WEIGHTS (anchor-augmented batching)
# ---------------------------------------------------------------------------

def build_weights_prompt(prime_movers: list, phase_sequences: list, cfg: dict,
                         anchors=None) -> str:
    ps_block = "\n".join(
        f"  {ps['id']}: {ps['name']} — {ps['definition']}"
        for ps in phase_sequences
    )
    people_block = "\n".join(
        f"  {p['rank']}. {p['name']} | {p.get('title', p.get('domain', ''))}"
        for p in prime_movers
    )
    anchor_block = ""
    if anchors:
        anchor_block = (
            "\n## CALIBRATION ANCHORS\n"
            "(These weights are already computed — use them as your reference scale.)\n\n"
        )
        for name, weights in anchors.items():
            w_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(weights.items()))
            anchor_block += f"  {name}: {{{w_str}}}\n"

    label = cfg["label"]
    return f"""You are assigning initial neuron–phase sequence weights for the {label} superorganism model.

## PHASE SEQUENCES

{ps_block}

## NEURONS TO WEIGHT

{people_block}
{anchor_block}
## WEIGHT SCALE

Assign a weight 0.0–1.0 for each (neuron, phase sequence) pair:
  0.0  = not materially involved (omit these entirely)
  0.2  = peripheral (occasional minor role)
  0.5  = moderate (regular, meaningful participation)
  0.8  = major driver (central to this phase sequence)
  1.0  = primary architect (dominant force in this sequence)

Only include non-zero entries.

Return only a JSON object:
{{
  "Name": {{"{phase_sequences[0]['id']}": 0.85, "{phase_sequences[1]['id'] if len(phase_sequences) > 1 else phase_sequences[0]['id']}": 0.4}},
  ...
}}"""


def stage_4_weights(prime_movers: list, phase_sequences: list, cfg: dict) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 4: INITIAL NEURON-PS WEIGHTS")
    print("=" * 60)

    client = get_anthropic_client()
    n      = len(prime_movers)

    # Single call if the full list fits in one batch
    if n <= WEIGHTS_BATCH_SIZE + WEIGHTS_ANCHOR_SIZE:
        print(f"  Single call for all {n} neurons...")
        prompt  = build_weights_prompt(prime_movers, phase_sequences, cfg)
        message = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4096, temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        all_weights = extract_json_obj(message.content[0].text)
        print(f"  ✓ Weights computed for {len(all_weights)} neurons")
        return all_weights

    # Multi-batch: process anchor tier first, then batches with anchors as references
    anchor_n = max(WEIGHTS_ANCHOR_SIZE, 17)
    anchor_tier = prime_movers[:anchor_n]
    print(f"  Batch 0 (anchor tier): {len(anchor_tier)} neurons...")
    prompt  = build_weights_prompt(anchor_tier, phase_sequences, cfg)
    message = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    anchor_weights = extract_json_obj(message.content[0].text)
    print(f"  ✓ Anchor weights: {len(anchor_weights)} neurons")

    all_weights = dict(anchor_weights)

    # Pick 4 exemplar anchors for subsequent batches (top ranked)
    exemplar_names = [p["name"] for p in anchor_tier[:WEIGHTS_ANCHOR_SIZE]]
    anchor_exemplars = {n: anchor_weights[n] for n in exemplar_names if n in anchor_weights}

    # Shuffle remaining neurons to avoid rank-order bias within non-anchor batches
    remaining = prime_movers[anchor_n:]
    random.shuffle(remaining)

    for batch_idx, start in enumerate(range(0, len(remaining), WEIGHTS_BATCH_SIZE), start=1):
        batch = remaining[start:start + WEIGHTS_BATCH_SIZE]
        print(f"  Batch {batch_idx}: {len(batch)} neurons (+ {len(anchor_exemplars)} anchors)...")
        prompt  = build_weights_prompt(batch, phase_sequences, cfg, anchors=anchor_exemplars)
        message = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4096, temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        batch_weights = extract_json_obj(message.content[0].text)
        # Only add weights for neurons not already in anchor_weights
        for name, w in batch_weights.items():
            if name not in anchor_weights:
                all_weights[name] = w

    print(f"  ✓ Total: {len(all_weights)} neurons weighted")
    return all_weights


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_output(all_proposals: dict, peer_reviews: dict, phase_sequences: list,
                neuron_weights, cfg: dict, output_path: Path):
    output = {
        "metadata": {
            "date":               date.today().isoformat(),
            "scope":              cfg["label"],
            "models_queried":     ["claude", "chatgpt", "grok", "gemini"],
            "n_phase_sequences":  len(phase_sequences),
            "n_neurons_weighted": len(neuron_weights) if neuron_weights is not None else 0,
        },
        "stage_1_proposals":  all_proposals,
        "stage_2_peer_reviews": {k: v for k, v in peer_reviews.items() if v is not None},
        "phase_sequences":    phase_sequences,
    }
    if neuron_weights is not None:
        output["initial_neuron_weights"] = neuron_weights

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved → {output_path.name}")


def print_summary(phase_sequences: list, neuron_weights, cfg: dict):
    bias_field = cfg["ps_bias_field"]
    print("\n" + "=" * 60)
    print(f"{cfg['label'].upper()} PHASE SEQUENCES ({len(phase_sequences)} total)")
    print("=" * 60)
    for ps in phase_sequences:
        bias = ps.get(bias_field, "")
        defn = ps.get("definition", "")
        print(f"  {ps['id']}: {ps['name']}  [{bias}]")
        print(f"       {defn[:90]}{'...' if len(defn) > 90 else ''}")

    if neuron_weights:
        print(f"\nNeuron-PS weights: {len(neuron_weights)} neurons")
        shown = 0
        for name, weights in neuron_weights.items():
            top = sorted(weights.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
            print(f"  {name}: {top_str}")
            shown += 1
            if shown >= 5:
                break
        if len(neuron_weights) > 5:
            print(f"  ... and {len(neuron_weights) - 5} more")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(scope: str, use_weights: bool, prime_movers_file=None):
    cfg = SCOPE_CONFIG[scope]

    input_path  = SCRIPT_DIR / (prime_movers_file or cfg["input_file"])
    output_path = SCRIPT_DIR / cfg["output_file"]

    print("=" * 60)
    print(f"PS COUNCIL [{cfg['label'].upper()}]")
    print(f"Date: {date.today().isoformat()}")
    print("=" * 60)

    if not input_path.exists():
        print(f"ERROR: {input_path.name} not found.")
        print("Run llm_council.py (or us_llm_council.py) first.")
        return

    prime_movers = load_prime_movers(input_path)
    print(f"\nLoaded {len(prime_movers)} prime movers from {input_path.name}")

    # Stage 1
    all_proposals = stage_1_proposals(prime_movers, cfg)

    active = {m: p for m, p in all_proposals.items() if p}
    print(f"\n--- Stage 1 Summary ---")
    for m, p in all_proposals.items():
        status = f"{len(p)} sequences" if p else "FAILED"
        print(f"  {'✓' if p else '✗'} {m}: {status}")

    if len(active) < 2:
        print("\nERROR: Too few successful proposals to run peer review. Aborting.")
        return

    # Stage 2
    peer_reviews = stage_2_peer_review(all_proposals, cfg)

    # Stage 3
    try:
        phase_sequences = stage_3_chairman_synthesis(all_proposals, peer_reviews, cfg)
    except Exception as e:
        print(f"\n! Chairman synthesis failed: {e}")
        return

    # Stage 4 (optional)
    neuron_weights = None
    if use_weights:
        try:
            neuron_weights = stage_4_weights(prime_movers, phase_sequences, cfg)
        except Exception as e:
            print(f"\n! Weight computation failed: {e}")
            print("  Saving output without weights.")

    # Save
    print("\nSaving output...")
    save_output(all_proposals, peer_reviews, phase_sequences, neuron_weights, cfg, output_path)
    print_summary(phase_sequences, neuron_weights, cfg)

    print("\n" + "=" * 60)
    print("PS Council session complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS Council: LLM Council for Phase Sequence Generation"
    )
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which superorganism to generate phase sequences for (default: us)"
    )
    parser.add_argument(
        "--no-weights", action="store_true",
        help="Skip Stage 4 neuron-PS weight generation"
    )
    parser.add_argument(
        "--prime-movers-file", type=str, default=None,
        help="Override the default prime movers input file (relative to script dir)"
    )
    args = parser.parse_args()
    run(scope=args.scope, use_weights=not args.no_weights,
        prime_movers_file=args.prime_movers_file)
