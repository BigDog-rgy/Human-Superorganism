"""
CA Council V2 — Neuron-Centric Cell Assembly Generation

Reads ps_canon_{scope}.json (produced by ps_council.py) and generates
canonical cell assemblies via a neuron-centric approach.

A cell assembly = a real organization, coalition, or institutional cluster
that a key actor (neuron) belongs to or leads.

Stages:
    1. Neuron Batch Proposals — all neurons are shuffled into random batches
       of 10. A 4-model LLM council proposes 1-3 CAs per neuron with examples.
    2. Deduplication — raw proposals are flattened, then fed through overlapping
       windows of 50 to Claude, which identifies semantic duplicates. Union-find
       merges clusters under a canonical name; neuron sets are merged too.
       Repeated for DEDUP_PASSES passes.
    3. ELO Tournament — Swiss-style pairings; Claude judges batches of
       ELO_PAIRS_PER_CALL pairs per call. After ELO_ROUNDS rounds, a neuron-
       retention constraint ensures each neuron keeps ≥1 CA, then the list is
       cut to TARGET_CAS.

Output: ca_canon_v2_{scope}.json

Usage:
    python ca_council_v2.py --scope us
    python ca_council_v2.py --scope global
    python ca_council_v2.py --scope us --target-cas 150 --elo-rounds 5
    python ca_council_v2.py --scope us --proposals-only   # stage 1 only
"""

import os
import json
import random
import argparse
import re
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
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
        "label":               "US",
        "ps_canon_file":       "ps_canon_us.json",
        "neuron_source_file":  "us_superorganism_model.json",  # top-150 neurons
        "output_file":         "ca_canon_v2_us.json",
        "ca_id_prefix":        "CA",
        "scope_description": (
            "US domestic power dynamics — politics, technology, finance, "
            "media, defense, energy, judiciary"
        ),
        "examples": """\
Examples of good cell assemblies per neuron (US scope):

  Executive / White House:
    Elon Musk (DOGE head, SpaceX CEO, X owner)
      → "Department of Government Efficiency (DOGE)"  [the specific WH advisory entity he leads]
      → "SpaceX"  [the aerospace company — not a division of it]
      → "X Corp"  [the social platform — its own CA, not "X Media Division"]
      → "xAI"     [his AI lab — a separate CA from SpaceX or X]
    Donald Trump → "Trump White House", "Trump Organization"

  Judiciary:
    Clarence Thomas → "Supreme Court of the United States", "Federalist Society"
    Leonard Leo     → "Federalist Society", "Judicial Crisis Network"
    NOTE: SCOTUS justices share "Supreme Court of the United States" as one CA —
    do NOT create "SCOTUS Conservative Bloc" or "SCOTUS Liberal Wing" as separate CAs;
    the court is the assembly.

  Finance:
    Jerome H. Powell  → "Federal Reserve"  [not "Federal Reserve Board of Governors / FOMC"]
    Scott Bessent     → "US Treasury"
    Ken Griffin       → "Citadel LLC"

  AI / Technology:
    Sam Altman (CEO, OpenAI)
      → "OpenAI"  [his primary org — Stargate is a sub-project of OpenAI, not a separate CA]
      → "Worldcoin / Tools for Humanity"  [his separate biometrics company]
      → "Merge Labs"  [his VR startup — a distinct org]
    Jensen Huang → "NVIDIA"  [not "NVIDIA Data Center Division"]
    Satya Nadella → "Microsoft"

  Congress:
    John Thune (Senate Majority Leader)
      → "US Senate"  [the institution he serves in]
      → "Senate Republican Conference"  [the caucus that functions as a coordinated unit]
    NOTE: For most senators and representatives, "US Senate" or "US House of Representatives"
    is the right CA. Only list a specific committee (e.g., "Senate Armed Services Committee",
    "House Ways and Means Committee") if the neuron chairs it or it is central to their power.
    Do NOT list both "US Senate" and "Senate Finance Committee" for a rank-and-file member.

  Media:
    Sean Hannity → "Fox News"  [the organization — not "Fox News Prime Time Bloc"]
    Jeff Bezos   → "The Washington Post", "Amazon"  [two distinct CAs]

  Defense:
    Pete Hegseth → "Department of Defense"  [not "DoD Civilian Leadership"]

GUIDING PRINCIPLE: Name the real top-level organization. Prefer "SpaceX" over
"SpaceX Starshield", "Fox News" over "Fox News Prime Time", "US Senate" over
"Senate Armed Services Subcommittee on X". Sub-units are sub-assemblies, not CAs.
One neuron can belong to multiple CAs; multiple neurons can share a CA.""",
    },
    "global": {
        "label":            "Global",
        "ps_canon_file":    "ps_canon_global.json",
        "neuron_source_file": "final_ranked_global.json",  # ranked list; capped by neuron_top_n
        "neuron_top_n":     300,                           # cap at top-300 neurons
        "target_cas":       400,                           # global gets more assemblies than US
        "output_file":      "ca_canon_v2_global.json",
        "ca_id_prefix":     "CA",
        "scope_description": (
            "Global geopolitics — US-China competition, military, finance, "
            "technology, information, energy, regional realignments"
        ),
        "examples": """\
Examples of good cell assemblies per neuron (global scope):
  - Xi Jinping       → "CCP Politburo Standing Committee",
                        "PLA Central Military Commission",
                        "Belt and Road Steering Council"
  - Larry Fink       → "BlackRock Executive Committee",
                        "World Economic Forum Board"
  - Christine Lagarde→ "ECB Governing Council", "G7 Finance Ministers"
  - Vladimir Putin   → "FSB/Siloviki Network", "Kremlin Inner Circle"
  - Elon Musk        → "Starlink Global Infrastructure Group", "SpaceX"

A CA must be a nameable organization or coalition — never a vague category
like "Western Leaders" or "Global Financiers".""",
    },
}

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CLAUDE_MODEL  = "claude-sonnet-4-6"
GPT_MODEL     = "gpt-5.2"
GROK_MODEL    = "grok-4-1-fast-reasoning"
GEMINI_MODEL  = "gemini-3-pro-preview"

NEURON_BATCH_SIZE  = 10    # neurons per batch in Stage 1
CAS_PER_NEURON_MIN = 1
CAS_PER_NEURON_MAX = 3

DEDUP_WINDOW      = 50    # CAs per alphabetical dedup window (Stage 2a)
DEDUP_STEP        = 25    # step between windows (overlap = WINDOW − STEP = 25)
EMBED_MODEL       = "text-embedding-3-small"
EMBED_SIM_THRESH  = 0.88  # cosine similarity threshold for flagging duplicates (Stage 2b)
EMBED_BATCH_SIZE  = 100   # CAs per embedding API call

ELO_PAIRS_PER_CALL = 10    # pairs judged per Claude call in Stage 3
ELO_INITIAL        = 1500.0
ELO_K              = 32.0
ELO_ROUNDS_DEFAULT = 10

TARGET_CAS_DEFAULT = 200


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


def load_neurons_from_superorganism(path: Path, top_n: int | None = None) -> list[dict]:
    """
    Load ranked neurons from a superorganism model file or a flat ranked list.
    Handles both formats:
      - superorganism model: {"superorganism_list": [{name, title, ...}, ...]}
      - flat ranked list:    [{name, title, ...}, ...]
    Returns list of {"name": ..., "title": ...} in rank order, capped at top_n if given.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("superorganism_list", [])
    result = [{"name": n["name"], "title": n.get("title", "")} for n in entries]
    if top_n:
        result = result[:top_n]
    return result


def extract_json_obj(text: str) -> dict:
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def extract_json_list(text: str) -> list:
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation/spaces — used for exact-match dedup."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# UNION-FIND  (for dedup clustering)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px  # py cluster merges into px


# ---------------------------------------------------------------------------
# STAGE 1: NEURON BATCH PROPOSALS
# ---------------------------------------------------------------------------

def build_proposal_prompt(neuron_batch: list[dict], cfg: dict) -> str:
    """neuron_batch: list of {"name": ..., "title": ...}"""
    label      = cfg["label"]
    scope_desc = cfg["scope_description"]
    examples   = cfg["examples"]

    neuron_lines = "\n".join(
        f"  {i+1}. {n['name']}" + (f" — {n['title']}" if n.get("title") else "")
        for i, n in enumerate(neuron_batch)
    )

    return f"""You are proposing cell assemblies for the {label} superorganism model.

## CONTEXT

Scope: {scope_desc}

A **cell assembly** is a real, nameable organization, coalition, or institutional
cluster that a key actor belongs to or leads. It fires together as a unit within
the power landscape.

{examples}

## YOUR TASK

For each of the following {len(neuron_batch)} key actors (neurons), propose
{CAS_PER_NEURON_MIN}–{CAS_PER_NEURON_MAX} cell assemblies they belong to.

Rules:
- Each CA must be a real, specific, trackable organization (not a vague category)
- Use the organization's actual name or a precise, widely-recognized label
- Include a one-sentence description of the CA's role in {label} power dynamics
- A neuron may belong to multiple CAs; CAs may be shared by multiple neurons

## NEURONS

{neuron_lines}

Return only a JSON object mapping each neuron's exact name to their CA list:
{{
  "Neuron Name": [
    {{"name": "CA Name", "description": "one sentence: what this org is and its structural role"}},
    ...
  ],
  ...
}}"""


def _call_proposal_claude(prompt: str, client) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json_obj(msg.content[0].text)


def _call_proposal_openai(prompt: str, client, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model, temperature=0.7,
        messages=[
            {"role": "system", "content": "You are an expert analyst of organizational power dynamics."},
            {"role": "user",   "content": prompt},
        ],
    )
    return extract_json_obj(resp.choices[0].message.content)


def _call_proposal_gemini(prompt: str, client) -> dict:
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return extract_json_obj(resp.text)


def build_chairman_prompt(
    neuron_batch: list[dict], batch_raw: dict[str, dict], cfg: dict
) -> str:
    """
    neuron_batch: list of {"name": ..., "title": ...}
    batch_raw: {model_name: {neuron_name: [{name, description}, ...]}}
    """
    label    = cfg["label"]
    examples = cfg["examples"]

    neuron_lines = "\n".join(
        f"  {i+1}. {n['name']}" + (f" — {n['title']}" if n.get("title") else "")
        for i, n in enumerate(neuron_batch)
    )

    proposal_lines = []
    for n in neuron_batch:
        neuron_name = n["name"]
        proposal_lines.append(f"\n**{neuron_name}**")
        for model_name, proposals in batch_raw.items():
            cas = proposals.get(neuron_name, [])
            if cas:
                ca_str = ", ".join(f'"{ca["name"]}"' for ca in cas)
            else:
                ca_str = "(no proposals)"
            proposal_lines.append(f"  {model_name}: {ca_str}")
    proposals_block = "\n".join(proposal_lines)

    return f"""You are the chairman of a cell assembly council for the {label} superorganism model.

Four models have proposed cell assemblies for 10 neurons. Your task: review all
proposals and produce the final, authoritative list for this batch.

{examples}

## NEURONS

{neuron_lines}

## COUNCIL PROPOSALS

{proposals_block}

## YOUR TASK

For each neuron, select or synthesize the best {CAS_PER_NEURON_MIN}–{CAS_PER_NEURON_MAX} cell assemblies:
- Prefer proposals that name the real top-level organization (not sub-units)
- Merge obvious duplicates (e.g. "Federal Reserve" / "The Fed" → keep one)
- Drop vague, overlapping, or sub-unit entries
- Use the most widely recognized, concise official name
- Include a one-sentence description of each CA's structural role

Return only a JSON object mapping each neuron's exact name to their final CA list:
{{
  "Neuron Name": [
    {{"name": "CA Name", "description": "one sentence: what this org is and its structural role"}},
    ...
  ],
  ...
}}"""


def _call_chairman(prompt: str, client) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json_obj(msg.content[0].text)


def stage_1_neuron_proposals(neurons: list[dict], cfg: dict) -> tuple[dict, dict]:
    """
    neurons: list of {"name": ..., "title": ...}
    Returns: (all_proposals, chairman_proposals)
      all_proposals:      {model_name: {neuron_name: [{name, description}, ...]}}  — raw, for logging
      chairman_proposals: {neuron_name: [{name, description}, ...]}                — consolidated by chairman
    """
    print("\n" + "=" * 60)
    print("STAGE 1: NEURON BATCH PROPOSALS + CHAIRMAN CONSOLIDATION")
    print("=" * 60)

    shuffled = neurons[:]
    random.shuffle(shuffled)
    batches = [shuffled[i:i + NEURON_BATCH_SIZE]
               for i in range(0, len(shuffled), NEURON_BATCH_SIZE)]
    print(f"\n{len(neurons)} neurons → {len(batches)} batches of {NEURON_BATCH_SIZE}")

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    xai_client       = get_xai_client()
    gemini_client    = get_gemini_client()

    all_proposals: dict[str, dict[str, list]] = {
        "claude":  {},
        "chatgpt": {},
        "grok":    {},
        "gemini":  {},
    }

    model_dispatch = {
        "claude":  lambda p: _call_proposal_claude(p, anthropic_client),
        "chatgpt": lambda p: _call_proposal_openai(p, openai_client, GPT_MODEL),
        "grok":    lambda p: _call_proposal_openai(p, xai_client, GROK_MODEL),
        "gemini":  lambda p: _call_proposal_gemini(p, gemini_client),
    }

    chairman_proposals: dict[str, list] = {}
    chair_success = chair_fail = 0

    for bi, batch in enumerate(batches):
        proposal_prompt = build_proposal_prompt(batch, cfg)
        batch_raw: dict[str, dict] = {}

        # Collect proposals from all 4 models for this batch
        for model_name, call_fn in model_dispatch.items():
            try:
                proposals = call_fn(proposal_prompt)
                batch_raw[model_name] = proposals
                for neuron, cas in proposals.items():
                    if neuron in all_proposals[model_name]:
                        all_proposals[model_name][neuron].extend(cas)
                    else:
                        all_proposals[model_name][neuron] = cas
            except Exception as e:
                batch_raw[model_name] = {}
                print(f"    [{model_name}] batch {bi+1} failed: {e}")

        # Chairman consolidates this batch
        chair_prompt = build_chairman_prompt(batch, batch_raw, cfg)
        try:
            consolidated = _call_chairman(chair_prompt, anthropic_client)
            for neuron, cas in consolidated.items():
                chairman_proposals[neuron] = cas
            chair_success += 1
        except Exception as e:
            chair_fail += 1
            print(f"    [CHAIRMAN] batch {bi+1} failed, using union fallback: {e}")
            # Fallback: union of all model proposals for this batch
            for model_proposals in batch_raw.values():
                for neuron, cas in model_proposals.items():
                    if neuron not in chairman_proposals:
                        chairman_proposals[neuron] = []
                    chairman_proposals[neuron].extend(cas)

        print(f"  batch {bi+1}/{len(batches)} — chairman {'✓' if chair_fail == 0 or chair_success > 0 else '✗'}")

    total_raw = sum(
        sum(len(v) for v in m.values()) for m in all_proposals.values()
    )
    total_chair = sum(len(v) for v in chairman_proposals.values())
    print(f"\n  Council: {total_raw} raw proposals across 4 models")
    print(f"  Chairman: {chair_success} batches consolidated → {total_chair} CAs for {len(chairman_proposals)} neurons")

    return all_proposals, chairman_proposals


def flatten_proposals(chairman_proposals: dict) -> tuple[dict, dict]:
    """
    Build ca_registry and neuron_ca_map from chairman_proposals.
      chairman_proposals: {neuron_name: [{name, description}, ...]}
      ca_registry:  {canonical_name → {"name", "description", "neurons": set, "count"}}
      neuron_ca_map: {neuron → [ca_name, ...]}

    Exact-match dedup (case-insensitive, punctuation-stripped) handles any
    residual cross-batch name variants; Stage 2 handles the rest.
    """
    norm_to_canonical: dict[str, str] = {}
    ca_registry: dict[str, dict] = {}
    neuron_ca_map: dict[str, list] = {}

    for neuron, cas in chairman_proposals.items():
        if neuron not in neuron_ca_map:
            neuron_ca_map[neuron] = []
        for ca in cas:
            name = ca.get("name", "").strip()
            desc = ca.get("description", "").strip()
            if not name:
                continue
            norm = normalize_name(name)
            if norm not in norm_to_canonical:
                norm_to_canonical[norm] = name
                ca_registry[name] = {
                    "name":        name,
                    "description": desc,
                    "neurons":     set(),
                    "count":       0,
                }
            canonical = norm_to_canonical[norm]
            ca_registry[canonical]["neurons"].add(neuron)
            ca_registry[canonical]["count"] += 1
            if canonical not in neuron_ca_map[neuron]:
                neuron_ca_map[neuron].append(canonical)

    print(f"\n  Flattened → {len(ca_registry)} unique CAs (after exact-match dedup)")
    return ca_registry, neuron_ca_map


# ---------------------------------------------------------------------------
# STAGE 2: DEDUPLICATION
#   2a — Alphabetical-window pass  (catches name variants)
#   2b — Embedding-similarity pass (catches semantic equivalents)
# ---------------------------------------------------------------------------

# ── Shared merge helper ──────────────────────────────────────────────────────

def _apply_merges_to_registry(
    ca_registry: dict,
    neuron_ca_map: dict,
    merge_pairs: list[tuple[str, str]],
) -> int:
    """
    merge_pairs: [(canonical_name, to_absorb_name), ...]
    The canonical is kept; to_absorb is deleted and its neurons/count folded in.
    The higher-count entry always wins the canonical slot when both exist in the
    registry, so Claude's suggested canonical is used as a preference only.
    Returns the number of merges successfully applied.
    """
    norm_to_key: dict[str, str] = {normalize_name(n): n for n in ca_registry}

    applied = 0
    for canonical_name, absorb_name in merge_pairs:
        can_key    = norm_to_key.get(normalize_name(canonical_name))
        absorb_key = norm_to_key.get(normalize_name(absorb_name))

        if not can_key or not absorb_key or can_key == absorb_key:
            continue
        if can_key not in ca_registry or absorb_key not in ca_registry:
            continue

        # If the "absorb" entry actually has more proposals, swap so the more
        # popular name wins (Claude's canonical preference is a tie-breaker).
        if ca_registry[absorb_key]["count"] > ca_registry[can_key]["count"]:
            can_key, absorb_key = absorb_key, can_key

        ca_registry[can_key]["neurons"].update(ca_registry[absorb_key]["neurons"])
        ca_registry[can_key]["count"]   += ca_registry[absorb_key]["count"]
        # Keep whichever description is longer/richer
        if len(ca_registry[absorb_key]["description"]) > len(ca_registry[can_key]["description"]):
            ca_registry[can_key]["description"] = ca_registry[absorb_key]["description"]

        del ca_registry[absorb_key]
        norm_to_key[normalize_name(absorb_name)] = can_key  # redirect

        for ca_list in neuron_ca_map.values():
            if absorb_key in ca_list:
                ca_list.remove(absorb_key)
                if can_key not in ca_list:
                    ca_list.append(can_key)

        applied += 1

    return applied


# ── Stage 2a: Alphabetical-window pass ──────────────────────────────────────

def build_alpha_dedup_prompt(window_cas: list[dict], cfg: dict) -> str:
    label = cfg["label"]
    ca_lines = "\n".join(
        f"  {i+1:3d}. {ca['name']}\n        {ca['description']}"
        for i, ca in enumerate(window_cas)
    )
    return f"""You are consolidating cell assemblies for the {label} superorganism model.

Below are {len(window_cas)} cell assemblies listed alphabetically. Your task is
to identify and merge any that refer to the same real organization under
different names, abbreviations, or minor wording variations.

MERGE only when two entries are genuinely the same organization — e.g.:
  "Federal Reserve" / "The Federal Reserve" / "US Federal Reserve"
  "Department of Defense" / "DoD" / "Dept. of Defense"
  "Supreme Court of the United States" / "SCOTUS" / "US Supreme Court"

Do NOT merge organizations that are merely related, co-owned, or in the same
domain — e.g., "SpaceX" and "Tesla" are distinct even though Elon Musk owns
both. "OpenAI" and "Anthropic" are distinct even though both are AI labs.

GUIDING PRINCIPLE: Name the real top-level organization. "SpaceX" is better
than "SpaceX Inc." or "SpaceX Starshield". "Fox News" is better than "Fox
News Corporation" or "Fox News Channel". For each merged group, use the most
widely recognized, concise, official name.

## CELL ASSEMBLIES

{ca_lines}

Return a JSON array covering every CA in the list. For merged groups, include
one entry with "absorbed" listing the exact names (from the list above) that
are folded in. Unmerged CAs have an empty "absorbed" list.

[
  {{
    "name": "canonical name",
    "description": "best one-sentence description",
    "absorbed": ["exact name A from list", "exact name B from list"]
  }},
  ...
]"""


def _call_alpha_dedup(prompt: str, client) -> list:
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096, temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json_list(msg.content[0].text)


def _stage_2a_alpha_pass(
    ca_registry: dict, neuron_ca_map: dict, cfg: dict, client
) -> dict:
    """
    Sort CAs alphabetically, slide overlapping windows of DEDUP_WINDOW,
    ask Claude to consolidate each window. Collect (canonical, absorbed)
    pairs and apply once per pass.
    """
    ca_names  = sorted(ca_registry.keys())   # alphabetical order
    n_before  = len(ca_names)
    print(f"\n  2a — Alphabetical pass: {n_before} CAs, "
          f"window={DEDUP_WINDOW}, step={DEDUP_STEP}")

    merge_pairs: list[tuple[str, str]] = []
    n_windows = 0

    for start in range(0, max(1, n_before - DEDUP_WINDOW + 1), DEDUP_STEP):
        window_names = ca_names[start : start + DEDUP_WINDOW]
        window_cas   = [
            {"name": n, "description": ca_registry[n]["description"]}
            for n in window_names if n in ca_registry
        ]
        if len(window_cas) < 2:
            continue

        prompt = build_alpha_dedup_prompt(window_cas, cfg)
        try:
            consolidated = _call_alpha_dedup(prompt, client)
            for entry in consolidated:
                canonical = entry.get("name", "").strip()
                for absorbed in entry.get("absorbed", []):
                    absorbed = absorbed.strip()
                    if absorbed and absorbed != canonical:
                        merge_pairs.append((canonical, absorbed))
            n_windows += 1
        except Exception as e:
            print(f"    window {start}–{start+DEDUP_WINDOW} failed: {e}")

    applied = _apply_merges_to_registry(ca_registry, neuron_ca_map, merge_pairs)
    n_after  = len(ca_registry)
    print(f"     {n_windows} windows, {len(merge_pairs)} merge suggestions, "
          f"{applied} applied → {n_after} CAs (−{n_before - n_after})")
    return {
        "pass":              "2a_alpha",
        "ca_count_before":   n_before,
        "windows_processed": n_windows,
        "merge_suggestions": len(merge_pairs),
        "merges_applied":    applied,
        "ca_count_after":    n_after,
    }


# ── Stage 2b: Embedding-similarity pass ─────────────────────────────────────

def _embed_cas(ca_names: list[str], ca_registry: dict, openai_client) -> np.ndarray:
    """
    Embed each CA as "name: description". Returns float32 matrix (N × D),
    rows in the same order as ca_names.
    """
    texts = [
        f"{n}: {ca_registry[n]['description']}" for n in ca_names
    ]
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp  = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        # Response data is ordered to match input
        vectors.extend(item.embedding for item in resp.data)

    mat = np.array(vectors, dtype=np.float32)
    # L2-normalise rows so dot product == cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _find_similar_pairs(
    ca_names: list[str], embeddings: np.ndarray, threshold: float
) -> list[tuple[str, str]]:
    """
    Return all (name_i, name_j) pairs with cosine similarity ≥ threshold.
    Uses batched matrix multiply to avoid O(N²) Python loops.
    """
    sim_matrix = embeddings @ embeddings.T   # shape (N, N)
    rows, cols = np.where(
        (sim_matrix >= threshold) &
        (np.triu(np.ones_like(sim_matrix, dtype=bool), k=1))
    )
    return [(ca_names[r], ca_names[c]) for r, c in zip(rows.tolist(), cols.tolist())]


def _cluster_pairs(pairs: list[tuple[str, str]], ca_names: list[str]) -> list[list[str]]:
    """Union-find clustering of flagged pairs. Returns list of clusters (size ≥ 2)."""
    uf = UnionFind(ca_names)
    for a, b in pairs:
        uf.union(a, b)
    groups: dict[str, list[str]] = {}
    for name in ca_names:
        root = uf.find(name)
        groups.setdefault(root, []).append(name)
    return [members for members in groups.values() if len(members) >= 2]


def build_cluster_confirm_prompt(cluster_cas: list[dict], cfg: dict) -> str:
    label = cfg["label"]
    ca_lines = "\n".join(
        f"  {i+1}. {ca['name']}\n     {ca['description']}"
        for i, ca in enumerate(cluster_cas)
    )
    return f"""The following {len(cluster_cas)} cell assemblies (from the {label} superorganism model)
scored high embedding similarity and may refer to the same real organization.

{ca_lines}

Are these genuinely the same organization? Use the same CA guidance as before:
"Federal Reserve" and "The Fed" → YES. "SpaceX" and "Blue Origin" → NO even
though both are commercial space companies.

Return JSON:
{{
  "same_org": true,
  "canonical": "most widely recognized, concise name",
  "absorbed": ["exact names to fold in — everything except the canonical"],
  "reason": "one sentence"
}}
— OR, if they are distinct organizations —
{{
  "same_org": false,
  "reason": "one sentence"
}}"""


def _call_cluster_confirm(prompt: str, client) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=256, temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json_obj(msg.content[0].text)


def _stage_2b_embedding_pass(
    ca_registry: dict, neuron_ca_map: dict, cfg: dict,
    anthropic_client, openai_client,
) -> dict:
    """
    Embed all remaining CAs, find pairs above EMBED_SIM_THRESH, cluster,
    ask Claude to confirm each cluster, apply confirmed merges.
    """
    ca_names = list(ca_registry.keys())
    n_before = len(ca_names)
    print(f"\n  2b — Embedding pass: {n_before} CAs, threshold={EMBED_SIM_THRESH}")

    print(f"     Embedding {n_before} CAs via {EMBED_MODEL}...")
    embeddings = _embed_cas(ca_names, ca_registry, openai_client)

    similar_pairs = _find_similar_pairs(ca_names, embeddings, EMBED_SIM_THRESH)
    print(f"     {len(similar_pairs)} pairs above threshold → clustering...")

    clusters = _cluster_pairs(similar_pairs, ca_names)
    print(f"     {len(clusters)} clusters to confirm with Claude")

    merge_pairs: list[tuple[str, str]] = []
    confirmed = rejected = 0

    for cluster in clusters:
        cluster_cas = [
            {"name": n, "description": ca_registry[n]["description"]}
            for n in cluster if n in ca_registry
        ]
        if len(cluster_cas) < 2:
            continue
        prompt = build_cluster_confirm_prompt(cluster_cas, cfg)
        try:
            result = _call_cluster_confirm(prompt, anthropic_client)
            if result.get("same_org"):
                canonical = result.get("canonical", "").strip()
                for absorbed in result.get("absorbed", []):
                    absorbed = absorbed.strip()
                    if absorbed and absorbed != canonical:
                        merge_pairs.append((canonical, absorbed))
                confirmed += 1
            else:
                rejected += 1
        except Exception as e:
            print(f"     cluster confirm failed: {e}")

    applied = _apply_merges_to_registry(ca_registry, neuron_ca_map, merge_pairs)
    n_after  = len(ca_registry)
    print(f"     {confirmed} clusters confirmed, {rejected} rejected, "
          f"{applied} merges applied → {n_after} CAs (−{n_before - n_after})")
    return {
        "pass":              "2b_embedding",
        "ca_count_before":   n_before,
        "similar_pairs":     len(similar_pairs),
        "clusters_found":    len(clusters),
        "clusters_confirmed": confirmed,
        "clusters_rejected": rejected,
        "merges_applied":    applied,
        "ca_count_after":    n_after,
    }


# ── Stage 2 orchestrator ─────────────────────────────────────────────────────

def stage_2_dedup(
    ca_registry: dict, neuron_ca_map: dict, cfg: dict
) -> tuple[dict, list]:
    """
    Two-pass deduplication:
      2a — alphabetical windows  → catches name variants / abbreviations
      2b — embedding similarity  → catches semantic equivalents with different names
    Returns (updated ca_registry, dedup_log).
    """
    print("\n" + "=" * 60)
    print("STAGE 2: DEDUPLICATION")
    print("=" * 60)

    anthropic_client = get_anthropic_client()
    openai_client    = get_openai_client()
    dedup_log        = []

    log_a = _stage_2a_alpha_pass(ca_registry, neuron_ca_map, cfg, anthropic_client)
    dedup_log.append(log_a)

    log_b = _stage_2b_embedding_pass(
        ca_registry, neuron_ca_map, cfg, anthropic_client, openai_client
    )
    dedup_log.append(log_b)

    print(f"\n  Dedup complete: {log_a['ca_count_before']} → {log_b['ca_count_after']} CAs")
    return ca_registry, dedup_log


# ---------------------------------------------------------------------------
# STAGE 3: ELO TOURNAMENT
# ---------------------------------------------------------------------------

def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def apply_elo(players: dict, winner_name: str, loser_name: str) -> None:
    w = players[winner_name]
    l = players[loser_name]
    e_w = elo_expected(w["elo"], l["elo"])
    w["elo"]     += ELO_K * (1.0 - e_w)
    l["elo"]     += ELO_K * (0.0 - (1.0 - e_w))
    w["wins"]    += 1
    l["losses"]  += 1
    w["matches"] += 1
    l["matches"] += 1


def make_swiss_pairings(players: dict) -> list[tuple[str, str]]:
    """Pair similar-ELO CAs that haven't met yet, Swiss-style."""
    ordered = sorted(players.values(), key=lambda p: -p["elo"])
    paired: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for player in ordered:
        name = player["name"]
        if name in paired:
            continue
        played_vs: set[str] = player.get("played_against", set())
        chosen = None
        for opp in ordered:
            opp_name = opp["name"]
            if opp_name in paired or opp_name == name:
                continue
            if opp_name not in played_vs:
                chosen = opp_name
                break
        if chosen is None:
            for opp in ordered:
                opp_name = opp["name"]
                if opp_name not in paired and opp_name != name:
                    chosen = opp_name
                    break
        if chosen:
            pairs.append((name, chosen))
            paired.add(name)
            paired.add(chosen)
            player.setdefault("played_against", set()).add(chosen)
            players[chosen].setdefault("played_against", set()).add(name)

    return pairs


def build_elo_batch_prompt(pairs: list[tuple[str, str]], players: dict, cfg: dict) -> str:
    label = cfg["label"]
    pair_lines = ""
    for i, (a_name, b_name) in enumerate(pairs, 1):
        a = players[a_name]
        b = players[b_name]
        pair_lines += (
            f"\nPair {i}:\n"
            f"  A: {a_name} — {a['description']}\n"
            f"  B: {b_name} — {b['description']}\n"
        )

    return f"""You are ranking cell assemblies by structural significance in the {label} power landscape.

For each pair below, decide which cell assembly is more significant as an
organizational unit. Consider:
  • Scale and reach within {label} power dynamics
  • Coherence as a coordinated institutional actor
  • Uniqueness — not easily substituted by another assembly
  • Current relevance (2025–2026)
{pair_lines}
Return only a JSON object with your verdict for each pair (A or B):
{{
  "1": "A",
  "2": "B",
  ...
}}"""


def _call_elo_batch(prompt: str, client, n_pairs: int) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=256, temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    result = extract_json_obj(msg.content[0].text)
    # Validate: keys "1".."n_pairs", values "A" or "B"
    verdicts = {}
    for i in range(1, n_pairs + 1):
        v = result.get(str(i), "").strip().upper()
        if v in ("A", "B"):
            verdicts[str(i)] = v
    return verdicts


def stage_3_elo(ca_registry: dict, cfg: dict, n_rounds: int) -> dict:
    """
    Swiss ELO tournament. Returns players dict:
    {ca_name: {name, description, elo, wins, losses, matches, neurons}}
    """
    print("\n" + "=" * 60)
    print("STAGE 3: ELO TOURNAMENT")
    print("=" * 60)
    print(f"\n  {len(ca_registry)} CAs, {n_rounds} rounds, "
          f"{ELO_PAIRS_PER_CALL} pairs/call, K={ELO_K}")

    client = get_anthropic_client()

    # Initialize players
    players: dict[str, dict] = {}
    for name, entry in ca_registry.items():
        players[name] = {
            "name":           name,
            "description":    entry["description"],
            "elo":            ELO_INITIAL,
            "wins":           0,
            "losses":         0,
            "matches":        0,
            "neurons":        entry["neurons"],
            "played_against": set(),
        }

    for round_num in range(1, n_rounds + 1):
        pairs   = make_swiss_pairings(players)
        batches = [pairs[i:i + ELO_PAIRS_PER_CALL]
                   for i in range(0, len(pairs), ELO_PAIRS_PER_CALL)]
        wins_this_round = 0

        for batch in batches:
            prompt = build_elo_batch_prompt(batch, players, cfg)
            try:
                verdicts = _call_elo_batch(prompt, client, len(batch))
                for i, (a_name, b_name) in enumerate(batch, 1):
                    verdict = verdicts.get(str(i))
                    if verdict == "A":
                        apply_elo(players, a_name, b_name)
                        wins_this_round += 1
                    elif verdict == "B":
                        apply_elo(players, b_name, a_name)
                        wins_this_round += 1
            except Exception as e:
                print(f"    round {round_num} batch failed: {e}")

        ranked = sorted(players.values(), key=lambda p: -p["elo"])
        print(f"  Round {round_num:2d}: {len(pairs)} matches, "
              f"leader={ranked[0]['name'][:40]} ({ranked[0]['elo']:.0f})")

    return players


# ---------------------------------------------------------------------------
# FINAL SELECTION
# ---------------------------------------------------------------------------

def select_final_assemblies(
    players: dict,
    neuron_ca_map: dict,
    target: int,
    ca_id_prefix: str,
) -> tuple[list, dict]:
    """
    1. Sort all CAs by ELO descending.
    2. For each neuron, protect its highest-ELO CA.
    3. Fill slots: protected CAs first, then top-ELO remaining, until target.
    4. If protected CAs exceed target, keep them all (target is soft).

    Returns:
        cell_assemblies: [{"id", "name", "description", "elo_score",
                           "rank", "member_neurons", "neuron_count"}]
        neuron_ca_map_final: {neuron: [ca_id, ...]}
    """
    ranked = sorted(players.values(), key=lambda p: -p["elo"])

    # Map: neuron → best-ELO CA name
    neuron_best_ca: dict[str, str] = {}
    for neuron, ca_list in neuron_ca_map.items():
        best_name = None
        best_elo  = -1.0
        for ca_name in ca_list:
            if ca_name in players and players[ca_name]["elo"] > best_elo:
                best_elo  = players[ca_name]["elo"]
                best_name = ca_name
        if best_name:
            neuron_best_ca[neuron] = best_name

    protected: set[str] = set(neuron_best_ca.values())
    print(f"\n  {len(protected)} protected CAs (neuron-retention constraint)")

    # Build final selection
    selected: list[str] = []
    selected_set: set[str] = set()

    # First: protected CAs (sorted by ELO)
    for ca in ranked:
        if ca["name"] in protected:
            selected.append(ca["name"])
            selected_set.add(ca["name"])

    # Then: fill remaining slots from top-ELO non-protected
    for ca in ranked:
        if len(selected) >= target:
            break
        if ca["name"] not in selected_set:
            selected.append(ca["name"])
            selected_set.add(ca["name"])

    # Sort final selection by ELO (protected may be scattered)
    selected.sort(key=lambda n: -players[n]["elo"])

    # Assign IDs and build output
    cell_assemblies = []
    ca_name_to_id: dict[str, str] = {}
    for rank, ca_name in enumerate(selected, 1):
        ca_id = f"{ca_id_prefix}-{rank:03d}"
        p     = players[ca_name]
        cell_assemblies.append({
            "id":             ca_id,
            "name":           ca_name,
            "description":    p["description"],
            "elo_score":      round(p["elo"], 1),
            "rank":           rank,
            "member_neurons": sorted(p["neurons"]),
            "neuron_count":   len(p["neurons"]),
            "wins":           p["wins"],
            "losses":         p["losses"],
            "matches":        p["matches"],
        })
        ca_name_to_id[ca_name] = ca_id

    # Build neuron → [ca_id] map (only selected CAs)
    neuron_ca_map_final: dict[str, list] = {}
    for neuron, ca_list in neuron_ca_map.items():
        ids = [ca_name_to_id[c] for c in ca_list if c in ca_name_to_id]
        if ids:
            neuron_ca_map_final[neuron] = ids

    return cell_assemblies, neuron_ca_map_final


# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------

CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

def _checkpoint_path(scope: str, stage: int) -> Path:
    return CHECKPOINTS_DIR / f"ca_v2_checkpoint_{scope}_s{stage}.json"


def _prep_for_json(obj):
    """Recursively convert sets → sorted lists for JSON serialisation."""
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, dict):
        return {k: _prep_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_prep_for_json(v) for v in obj]
    return obj


def save_checkpoint(scope: str, stage: int, **data):
    path = _checkpoint_path(scope, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_prep_for_json(data), f, indent=2, ensure_ascii=False)
    print(f"  ✓ Checkpoint saved → {path.name}")


def load_checkpoint(scope: str, stage: int) -> dict | None:
    path = _checkpoint_path(scope, stage)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  ✓ Checkpoint loaded ← {path.name}")
    return data


def _registry_from_json(raw: dict) -> dict:
    """Restore ca_registry from JSON (neurons lists → sets)."""
    return {
        name: {**entry, "neurons": set(entry.get("neurons", []))}
        for name, entry in raw.items()
    }


def _players_from_json(raw: dict) -> dict:
    """Restore players dict from JSON (neurons/played_against lists → sets)."""
    return {
        name: {
            **p,
            "neurons":        set(p.get("neurons", [])),
            "played_against": set(p.get("played_against", [])),
        }
        for name, p in raw.items()
    }


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_output(
    scope: str,
    cfg: dict,
    all_proposals: dict,
    dedup_log: list,
    cell_assemblies: list,
    neuron_ca_map_final: dict,
    output_path: Path,
):
    # Convert sets to lists for JSON serialisation
    def _prep(obj):
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, dict):
            return {k: _prep(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_prep(v) for v in obj]
        return obj

    output = {
        "metadata": {
            "date":              date.today().isoformat(),
            "scope":             cfg["label"],
            "version":           "v2",
            "models_queried":    ["claude", "chatgpt", "grok", "gemini"],
            "n_cell_assemblies": len(cell_assemblies),
            "n_neurons_covered": len(neuron_ca_map_final),
        },
        "stage_1_proposals":     _prep(all_proposals),
        "stage_2_dedup_log":     dedup_log,
        "cell_assemblies":       cell_assemblies,
        "neuron_ca_map":         neuron_ca_map_final,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved → {output_path.name}")


def print_summary(cell_assemblies: list, neuron_ca_map_final: dict, cfg: dict):
    print("\n" + "=" * 60)
    print(f"{cfg['label'].upper()} CELL ASSEMBLIES V2 ({len(cell_assemblies)} total)")
    print("=" * 60)
    for ca in cell_assemblies[:30]:
        neurons_preview = ", ".join(ca["member_neurons"][:3])
        if len(ca["member_neurons"]) > 3:
            neurons_preview += f" +{len(ca['member_neurons'])-3} more"
        print(f"  {ca['id']}: {ca['name']}  (ELO {ca['elo_score']:.0f})")
        print(f"       {ca['description'][:75]}")
        if neurons_preview:
            print(f"       Neurons: {neurons_preview}")
    if len(cell_assemblies) > 30:
        print(f"  ... and {len(cell_assemblies) - 30} more")
    print(f"\n  Neurons covered: {len(neuron_ca_map_final)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(scope: str, target_cas: int, elo_rounds: int,
        proposals_only: bool, from_stage: int):
    cfg         = SCOPE_CONFIG[scope]
    canon_path  = SCRIPT_DIR / cfg["ps_canon_file"]
    output_path = SCRIPT_DIR / cfg["output_file"]
    target_cas  = cfg.get("target_cas", target_cas)  # scope config overrides CLI default

    print("=" * 60)
    print(f"CA COUNCIL V2 [{cfg['label'].upper()}]")
    print(f"Date: {date.today().isoformat()}")
    print(f"Target CAs: {target_cas}  |  ELO rounds: {elo_rounds}")
    if from_stage > 1:
        print(f"Resuming from Stage {from_stage}")
    print("=" * 60)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if from_stage <= 1:
        neuron_source = cfg.get("neuron_source_file")
        if neuron_source:
            neuron_path = SCRIPT_DIR / neuron_source
            if not neuron_path.exists():
                print(f"ERROR: {neuron_path.name} not found.")
                return
            print(f"\nLoading neurons from {neuron_path.name}...")
            neurons = load_neurons_from_superorganism(neuron_path, top_n=cfg.get("neuron_top_n"))
            print(f"  {len(neurons)} ranked neurons loaded")
        else:
            if not canon_path.exists():
                print(f"ERROR: {canon_path.name} not found.")
                print(f"  Run: python ps_council.py --scope {scope}")
                return
            print(f"\nLoading {canon_path.name}...")
            ps_canon = load_ps_canon(canon_path)
            neurons  = [{"name": n, "title": ""} for n in ps_canon.get("initial_neuron_weights", {})]
            print(f"  {len(neurons)} neurons loaded from ps_canon")

        if not neurons:
            print("ERROR: No neurons found.")
            return

        all_proposals, chairman_proposals = stage_1_neuron_proposals(neurons, cfg)
        ca_registry, neuron_ca_map = flatten_proposals(chairman_proposals)

        save_checkpoint(scope, 1,
                        all_proposals=all_proposals,
                        chairman_proposals=chairman_proposals,
                        ca_registry=ca_registry,
                        neuron_ca_map=neuron_ca_map)

        if proposals_only:
            print("\n[--proposals-only] Stopping after Stage 1.")
            save_output(scope, cfg, all_proposals, [], [], {}, output_path)
            return
    else:
        ckpt = load_checkpoint(scope, 1)
        if not ckpt:
            print(f"ERROR: Stage 1 checkpoint not found for scope '{scope}'.")
            print(f"  Run without --from-stage to generate it.")
            return
        all_proposals      = ckpt["all_proposals"]
        chairman_proposals = ckpt.get("chairman_proposals", {})
        ca_registry        = _registry_from_json(ckpt["ca_registry"])
        neuron_ca_map      = ckpt["neuron_ca_map"]
        print(f"  Stage 1 restored: {len(ca_registry)} CAs, "
              f"{len(neuron_ca_map)} neurons")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if from_stage <= 2:
        ca_registry, dedup_log = stage_2_dedup(ca_registry, neuron_ca_map, cfg)
        print(f"\n  After dedup: {len(ca_registry)} unique CAs")

        save_checkpoint(scope, 2,
                        ca_registry=ca_registry,
                        neuron_ca_map=neuron_ca_map,
                        dedup_log=dedup_log)
    else:
        ckpt = load_checkpoint(scope, 2)
        if not ckpt:
            print(f"ERROR: Stage 2 checkpoint not found for scope '{scope}'.")
            print(f"  Run with --from-stage 2 or lower to generate it.")
            return
        ca_registry   = _registry_from_json(ckpt["ca_registry"])
        neuron_ca_map = ckpt["neuron_ca_map"]
        dedup_log     = ckpt["dedup_log"]
        print(f"  Stage 2 restored: {len(ca_registry)} CAs after dedup")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if from_stage <= 3:
        players = stage_3_elo(ca_registry, cfg, n_rounds=elo_rounds)

        save_checkpoint(scope, 3,
                        players=players,
                        neuron_ca_map=neuron_ca_map,
                        dedup_log=dedup_log)
    else:
        ckpt = load_checkpoint(scope, 3)
        if not ckpt:
            print(f"ERROR: Stage 3 checkpoint not found for scope '{scope}'.")
            print(f"  Run with --from-stage 3 or lower to generate it.")
            return
        players       = _players_from_json(ckpt["players"])
        neuron_ca_map = ckpt["neuron_ca_map"]
        dedup_log     = ckpt["dedup_log"]
        print(f"  Stage 3 restored: {len(players)} CAs with ELO scores")

    # ── Final Selection ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SELECTION")
    print("=" * 60)
    cell_assemblies, neuron_ca_map_final = select_final_assemblies(
        players, neuron_ca_map, target_cas, cfg["ca_id_prefix"]
    )
    print(f"  Selected {len(cell_assemblies)} cell assemblies "
          f"covering {len(neuron_ca_map_final)} neurons")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_output(
        scope, cfg, all_proposals, dedup_log,
        cell_assemblies, neuron_ca_map_final, output_path,
    )
    print_summary(cell_assemblies, neuron_ca_map_final, cfg)

    print("\n" + "=" * 60)
    print("CA Council V2 session complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CA Council V2: Neuron-Centric Cell Assembly Generation"
    )
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which superorganism to process (default: us)"
    )
    parser.add_argument(
        "--target-cas", type=int, default=TARGET_CAS_DEFAULT,
        help=f"Target number of final cell assemblies (default: {TARGET_CAS_DEFAULT})"
    )
    parser.add_argument(
        "--elo-rounds", type=int, default=ELO_ROUNDS_DEFAULT,
        help=f"Number of ELO tournament rounds (default: {ELO_ROUNDS_DEFAULT})"
    )
    parser.add_argument(
        "--proposals-only", action="store_true",
        help="Stop after Stage 1 (neuron proposals only)"
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, choices=[1, 2, 3, 4],
        help=(
            "Resume from a stage using its saved checkpoint (default: 1 = full run). "
            "2 = skip Stage 1, load s1 checkpoint; "
            "3 = skip Stages 1-2, load s2 checkpoint; "
            "4 = skip Stages 1-3, load s3 checkpoint (re-run final selection only)"
        )
    )
    args = parser.parse_args()
    run(
        scope=args.scope,
        target_cas=args.target_cas,
        elo_rounds=args.elo_rounds,
        proposals_only=args.proposals_only,
        from_stage=args.from_stage,
    )
