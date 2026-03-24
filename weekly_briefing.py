"""
Weekly Briefing Generator
Fetches news on US or global prime movers via Perplexity, then synthesizes
via Claude into a structured weekly report.

Usage:
    python weekly_briefing.py                          # US run
    python weekly_briefing.py --scope global           # Global run

Firing model:
    Conscious layer  — PS news scored 1–10; top-ranked PSes fire assemblies via network
                       membership (probabilistic, dampened by member count). Neurons are
                       then selected PS by PS: rank-1 PS picks NEURONS_FOR_TOP_PS neurons,
                       ranks 2–4 each pick NEURONS_FOR_OTHER_PS. Eligible pool = all neurons
                       with PS membership; weight = BASE_WEIGHT + CA-membership bonus +
                       N-N score bonus (vs already-selected PS co-members only).
    Spontaneous layer — Skip-counter probability (10% base + 10%/skipped week), sorted by
                        most-overdue first, capped at sqrt(n). SPONTANEOUS_CA_COUNT CAs are
                        also drawn from spontaneous neurons' memberships for display/synthesis
                        (excluded from coactivation learning).
    Together they determine which neurons and assemblies get news fetched this week.
"""

import os
import json
import math
import random
import argparse
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# SCOPE CONFIG
# ---------------------------------------------------------------------------

SONAR_MODEL  = "sonar-pro"
CLAUDE_MODEL = "claude-opus-4-6"

SCRIPT_DIR    = Path(__file__).parent
BRIEFINGS_DIR = SCRIPT_DIR / "briefings"
STATE_DIR     = SCRIPT_DIR / "state"
FETCH_STATE_DIR = STATE_DIR / "fetch"
COACTIVATION_STATE_DIR = STATE_DIR / "coactivation"

SCOPE_CONFIG = {
    "us": {
        "label":                    "US",
        "model_file":               "us_superorganism_model.json",
        "fetch_state_file":         "fetch_state.json",
        "coactivation_state_file":  "us_coactivation_state.json",
        "briefing_prefix":          "weekly_briefing",
        "md_title":                 "Weekly Prime Mover Briefing",
        "synthesis_scope":          "US superorganism",
        "news_system_prompt":       "You are a concise news analyst. Report factual recent events only.",
        "news_hard_categories": (
            "policy decisions, business moves, legal actions, "
            "geopolitical events, significant public statements"
        ),
        "ps_system_prompt":         "You are a concise US policy and power analyst.",
        # n=150, cap=13 → 0.90/13≈6.9% → floor to 6% for stochasticity; 100% at 15 weeks
        "spontaneous_weekly_increase": 0.06,
        "spontaneous_counter_cap":     15,
    },
    "global": {
        "label":                    "Global",
        "model_file":               "superorganism_model.json",
        "fetch_state_file":         "global_fetch_state.json",
        "coactivation_state_file":  "global_coactivation_state.json",
        "briefing_prefix":          "global_weekly_briefing",
        "md_title":                 "Global Prime Mover Briefing",
        "synthesis_scope":          "global superorganism",
        "news_system_prompt":       "You are a concise geopolitical analyst. Report factual recent events only.",
        "news_hard_categories": (
            "policy decisions, geopolitical moves, military actions, "
            "economic initiatives, diplomatic events, significant public statements"
        ),
        "ps_system_prompt":         "You are a concise geopolitical and power analyst.",
        # n=300, cap=18 → 0.90/18=5.0%; 100% at 18 weeks
        "spontaneous_weekly_increase": 0.05,
        "spontaneous_counter_cap":     18,
    },
}

ASSEMBLIES_FOR_TOP_PS    = 5    # assemblies fired for highest-scoring PS
ASSEMBLIES_FOR_OTHER_PS  = 3    # assemblies fired for each of up to 3 other PSes
MAX_ASSEMBLY_PS_COUNT    = 4    # max PSes that can fire assemblies
CONSCIOUS_NEURON_CAP     = 10   # max neurons selected via conscious layer
NEURONS_FOR_TOP_PS       = 4    # neurons selected for rank-1 PS
NEURONS_FOR_OTHER_PS     = 2    # neurons selected for each rank-2/3/4 PS
SPONTANEOUS_CA_COUNT     = 2    # CAs drawn from spontaneous neurons each week
ASSEMBLY_MEMBER_WEIGHT   = 0.5  # member-count influence on probabilistic assembly selection
BASE_WEIGHT              = 1.0  # base selection weight for CAs and neurons (ensures chance for all)
NEURON_MEMBERSHIP_WEIGHT = 0.5  # per-fired-CA membership bonus for neuron base score
SELECTION_SCORE_ALPHA    = 0.5  # exponent for score-to-weight dampening (< 1 compresses spread)


# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------

def get_perplexity_client() -> OpenAI:
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key:
        raise ValueError("PERPLEXITY_API_KEY not set in .env")
    return OpenAI(api_key=key, base_url="https://api.perplexity.ai")


def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_superorganism(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



# ---------------------------------------------------------------------------
# FETCH STATE (sparse fetch persistence)
# ---------------------------------------------------------------------------

def load_fetch_state(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_fetch_state(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_fetch_state(state: dict, fetched_people: list, skipped_people: list, counter_cap: int) -> dict:
    new_state = dict(state)
    for person in fetched_people:
        new_state[person["name"]] = 0
    for person in skipped_people:
        name = person["name"]
        new_state[name] = min(new_state.get(name, 0) + 1, counter_cap)
    return new_state


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def weighted_sample_without_replacement(candidates: list, weights: list, k: int) -> list:
    """Draw k items from candidates without replacement, weighted by weights."""
    selected = []
    pool = list(zip(candidates, weights))
    k = min(k, len(pool))
    for _ in range(k):
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        r = random.uniform(0, total)
        cumsum = 0.0
        for i, (c, w) in enumerate(pool):
            cumsum += w
            if r <= cumsum:
                selected.append(c)
                pool.pop(i)
                break
    return selected


def ps_ids_for_person(person: dict) -> list:
    return [ps["id"] for ps in person.get("superorganism", {}).get("phase_sequences", [])]


def pair_key(a: str, b: str) -> str:
    """Canonical sorted key so A|||B == B|||A."""
    return "|||".join(sorted([a, b]))


def load_coactivation_state(path: Path) -> dict:
    """Load coactivation state file; return empty state if file doesn't exist."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STAGE 1A: PS news via Perplexity
# ---------------------------------------------------------------------------

def fetch_news_for_ps(
    client: OpenAI, ps: dict, week_start: str, week_end: str, cfg: dict
) -> dict:
    """
    One Perplexity call per phase sequence.
    Returns summary + integer activation score 1-10.
    Assembly selection is handled downstream by the network, not by Perplexity.
    """
    prompt = (
        f"Summarize the most significant developments in the following domain from the past 7 days "
        f"({week_start} to {week_end}).\n\n"
        f"Domain: {ps['name']}\n"
        f"Definition: {ps['definition']}\n\n"
        f"Focus on: structural changes, major decisions, policy moves, business actions, or events "
        f"that indicate momentum in this area. 3-5 sentences.\n\n"
        f"After your summary, on a new line write a single integer from 1 to 10:\n"
        f"1-3 = quiet week, minimal activity, no major changes\n"
        f"4-6 = moderate activity, some notable developments\n"
        f"7-10 = high activity, major developments, clear momentum"
    )

    print(f"  [{ps['id']}: {ps['name']}]...")
    response = client.chat.completions.create(
        model=SONAR_MODEL,
        messages=[
            {"role": "system", "content": cfg["ps_system_prompt"]},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Parse activation score from last non-empty line
    activation_score = 5  # default to middle
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            activation_score = max(1, min(10, int(stripped)))
            break

    return {
        "id":               ps["id"],
        "name":             ps["name"],
        "summary":          raw,
        "activation_score": activation_score,
    }


# ---------------------------------------------------------------------------
# STAGE 1.5: Per-assembly news via Perplexity
# ---------------------------------------------------------------------------

def fetch_news_for_assembly(
    client: OpenAI, assembly: dict, ps_names: list, week_start: str, week_end: str, cfg: dict
) -> dict:
    """One Perplexity call per fired cell assembly — targeted 7-day organizational news digest."""
    name       = assembly["name"]
    role       = assembly.get("role", assembly.get("description", ""))
    ps_context = f" Active domains: {', '.join(ps_names)}." if ps_names else ""
    role_line  = f"Organization type: {role}\n\n" if role else ""

    prompt = (
        f"{role_line}"
        f"Find the most significant news about {name} "
        f"from the past 7 days (approximately {week_start} to {week_end}).{ps_context}\n\n"
        f"Focus on: decisions made, actions taken, positions announced, alliances formed, "
        f"or organizational moves that affect power dynamics. 3-4 sentences.\n\n"
        f"Hard news only: {cfg['news_hard_categories']}. Skip routine appearances."
    )

    print(f"  [{assembly['id']}: {name}]...")
    response = client.chat.completions.create(
        model=SONAR_MODEL,
        messages=[
            {"role": "system", "content": cfg["news_system_prompt"]},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    return {
        "id":             assembly["id"],
        "name":           name,
        "ps_memberships": assembly.get("ps_memberships", []),
        "raw_news":       response.choices[0].message.content,
    }


# ---------------------------------------------------------------------------
# STAGE 1B: Coactivation-guided assembly sequence building
# ---------------------------------------------------------------------------

def build_ca_sequence_for_ps(
    candidates: list,
    ca_coactivation: dict,
    cap: int,
) -> list:
    """
    Build an ordered sequence of up to `cap` CAs to fire for a given PS.
    Seed: uniform random from candidates.
    Subsequent CAs: BASE_WEIGHT + |coactivation score| bonus.
      - Step 2: 100% bonus from seed.
      - Step 3+, reinforcing/neutral prev: 2/3 * |cov(t-1)| + 1/3 * |cov(t-2)|.
      - Step 3+, adversarial prev (score < 0): max(0, score(candidate, t)) — reinforcing candidates
        get a bonus, adversarial candidates compete at BASE_WEIGHT only (memory reset).
    Returns list of selected CA dicts in firing order.
    """
    if not candidates or cap <= 0:
        return []

    cap  = min(cap, len(candidates))
    pool = list(candidates)

    seed = random.choice(pool)
    sequence = [seed]
    pool.remove(seed)

    while len(sequence) < cap and pool:
        prev1_id = sequence[-1]["id"]
        prev2_id = sequence[-2]["id"] if len(sequence) >= 2 else None

        if prev2_id is not None:
            score_p1_p2 = ca_coactivation.get(pair_key(prev1_id, prev2_id), {}).get("score", 0.0)
            adversarial_break = score_p1_p2 < 0
        else:
            adversarial_break = False

        weights = []
        for ca in pool:
            ca_id  = ca["id"]
            score_prev1 = ca_coactivation.get(pair_key(ca_id, prev1_id), {}).get("score", 0.0)
            if adversarial_break:
                # After adversarial firing: only reinforce t, suppress adversarial candidates
                bonus = max(0.0, score_prev1)
            elif prev2_id is None:
                bonus = abs(score_prev1) ** SELECTION_SCORE_ALPHA
            else:
                abs_prev1 = abs(score_prev1) ** SELECTION_SCORE_ALPHA
                abs_prev2 = abs(ca_coactivation.get(pair_key(ca_id, prev2_id), {}).get("score", 0.0)) ** SELECTION_SCORE_ALPHA
                bonus     = (2 / 3) * abs_prev1 + (1 / 3) * abs_prev2
            weights.append(BASE_WEIGHT + bonus)

        selected = weighted_sample_without_replacement(pool, weights, 1)
        if not selected:
            break
        sequence.append(selected[0])
        pool.remove(selected[0])

    return sequence


def build_fired_assemblies(
    ps_news_items: list,
    ps_assemblies_map: dict,
    ca_coactivation: dict,
) -> tuple[list, dict, list]:
    """
    Build fired CA sequences for each active PS using coactivation-guided selection.
    Returns:
      fired_ordered   — list of (ca_id, ps_id) in firing sequence order across all PSes
      ps_to_fired     — dict of ps_id -> [ca_id, ...] in sequence order
      ps_ranked_items — ps_news_items in rank order, filtered to only those that fired CAs
    """
    ranked     = sorted(ps_news_items, key=lambda x: x["activation_score"], reverse=True)
    fired_ordered: list = []
    seen_ca_ids: set    = set()
    ps_to_fired: dict   = {}

    for rank, ps_item in enumerate(ranked[:MAX_ASSEMBLY_PS_COUNT]):
        ps_id      = ps_item["id"]
        cap        = ASSEMBLIES_FOR_TOP_PS if rank == 0 else ASSEMBLIES_FOR_OTHER_PS
        candidates = [ca for ca in ps_assemblies_map.get(ps_id, []) if ca["id"] not in seen_ca_ids]
        if not candidates:
            continue

        sequence = build_ca_sequence_for_ps(candidates, ca_coactivation, cap)
        for ca in sequence:
            seen_ca_ids.add(ca["id"])
            fired_ordered.append((ca["id"], ps_id))
            ps_to_fired.setdefault(ps_id, []).append(ca["id"])

    ps_ranked_items = [ps_item for ps_item in ranked[:MAX_ASSEMBLY_PS_COUNT]
                       if ps_item["id"] in ps_to_fired]
    return fired_ordered, ps_to_fired, ps_ranked_items


def select_neurons_conscious(
    ps_ranked_items: list,
    ps_to_fired: dict,
    ps_to_neurons: dict,
    assembly_to_neurons: dict,
    neuron_coactivation: dict,
) -> list:
    """
    Select conscious neurons PS by PS in rank order.

    For each ranked PS:
      - Eligible pool: all neurons with PS membership, excluding already-selected.
      - Per-neuron weight:
          BASE_WEIGHT
          + (# of this PS's fired CAs the neuron belongs to) * NEURON_MEMBERSHIP_WEIGHT
          + Σ|N-N score vs (globally selected ∩ this PS's members)|
      - Quota: NEURONS_FOR_TOP_PS for rank-1, NEURONS_FOR_OTHER_PS for rank-2/3/4.

    Deduplication is global: a neuron selected by an earlier PS cannot appear in a
    later PS's pool.  Returns up to CONSCIOUS_NEURON_CAP person dicts.
    """
    if not ps_ranked_items:
        return []

    selected_names: list  = []
    name_to_person: dict  = {}

    for rank, ps_item in enumerate(ps_ranked_items):
        ps_id  = ps_item["id"]
        quota  = NEURONS_FOR_TOP_PS if rank == 0 else NEURONS_FOR_OTHER_PS

        # All persons with membership to this PS
        ps_members = ps_to_neurons.get(ps_id, [])
        ps_member_names = set()
        for person in ps_members:
            name_to_person[person["name"]] = person
            ps_member_names.add(person["name"])

        # How many of this PS's fired CAs does each neuron belong to?
        membership_count: dict = {}
        for ca_id in ps_to_fired.get(ps_id, []):
            for person in assembly_to_neurons.get(ca_id, []):
                n = person["name"]
                membership_count[n] = membership_count.get(n, 0) + 1
                name_to_person[n]   = person

        # Already-selected PS co-members (for N-N bonus)
        selected_in_ps = [n for n in selected_names if n in ps_member_names]

        # Eligible = PS members not yet globally selected
        eligible = [n for n in ps_member_names if n not in selected_names]

        for _ in range(quota):
            if not eligible:
                break
            weights = []
            for name in eligible:
                base     = BASE_WEIGHT + membership_count.get(name, 0) * NEURON_MEMBERSHIP_WEIGHT
                nn_bonus = sum(
                    abs(neuron_coactivation.get(pair_key(name, sel), {}).get("score", 0.0)) ** SELECTION_SCORE_ALPHA
                    for sel in selected_in_ps
                )
                weights.append(base + nn_bonus)

            chosen = weighted_sample_without_replacement(eligible, weights, 1)
            if not chosen:
                break
            selected_names.append(chosen[0])
            selected_in_ps.append(chosen[0])
            eligible.remove(chosen[0])

    return [name_to_person[n] for n in selected_names if n in name_to_person]


# ---------------------------------------------------------------------------
# STAGE 1C: Spontaneous neuron selection (skip-counter driven)
# ---------------------------------------------------------------------------

def select_neurons_spontaneous(
    individuals: list,
    fetch_state: dict,
    cap: int,
    exclude_names: set,
    weekly_increase: float,
) -> tuple[list, list]:
    """
    Probabilistic selection capped at `cap` neurons.
    Candidates sorted by skip count (most overdue first) for priority.
    Returns (fetched, skipped) — skipped includes all non-selected candidates.
    """
    candidates = [p for p in individuals if p["name"] not in exclude_names]
    candidates.sort(key=lambda p: fetch_state.get(p["name"], 0), reverse=True)

    fetched = []
    skipped = []

    for person in candidates:
        if len(fetched) >= cap:
            skipped.append(person)
            continue
        skipped_weeks = fetch_state.get(person["name"], 0)
        prob = min(0.10 + skipped_weeks * weekly_increase, 1.0)
        if random.random() < prob:
            fetched.append(person)
        else:
            skipped.append(person)

    return fetched, skipped


# ---------------------------------------------------------------------------
# STAGE 1C.5: Spontaneous CA selection
# ---------------------------------------------------------------------------

def select_spontaneous_cas(
    spontaneous_neurons: list,
    person_to_cas: dict,
    consciously_fired_ca_ids: set,
    n: int = SPONTANEOUS_CA_COUNT,
) -> list:
    """
    Draw up to n CA ids from the CAs that spontaneous neurons belong to,
    excluding any CA already consciously fired this week.
    If a drawn CA was already consciously activated, it is simply skipped
    (the week loses one unconscious CA activation rather than falling back).
    Returns a list of up to n CA ids (may be shorter if not enough candidates).
    """
    candidates = set()
    for person in spontaneous_neurons:
        for ca_id in person_to_cas.get(person["name"], []):
            if ca_id not in consciously_fired_ca_ids:
                candidates.add(ca_id)
    candidates = list(candidates)
    random.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# STAGE 1D: Per-person news via Perplexity
# ---------------------------------------------------------------------------

def fetch_news_for_person(
    client: OpenAI, person: dict, week_start: str, week_end: str, cfg: dict
) -> dict:
    """One Perplexity call per neuron — targeted 7-day news digest."""
    name   = person["name"]
    domain = person.get("title", person.get("domain", ""))

    ps_names = [
        ps["name"]
        for ps in person.get("superorganism", {}).get("phase_sequences", [])
    ]
    ps_context = f" Their key areas of activity: {', '.join(ps_names)}." if ps_names else ""

    prompt = (
        f"Find the most significant news about {name} ({domain}) "
        f"from the past 7 days (approximately {week_start} to {week_end}).{ps_context}\n\n"
        "Provide 3-5 items. For each: one-sentence summary of what happened, "
        "and one sentence on why it matters for their influence or position.\n\n"
        f"Hard news only: {cfg['news_hard_categories']}. Skip routine appearances."
    )

    print(f"  [{name}]...")
    response = client.chat.completions.create(
        model=SONAR_MODEL,
        messages=[
            {"role": "system", "content": cfg["news_system_prompt"]},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    return {
        "name":     name,
        "domain":   domain,
        "raw_news": response.choices[0].message.content,
    }



# ---------------------------------------------------------------------------
# STAGE 2: Claude synthesis
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    ps_news_items: list,
    assembly_news_items: list,
    news_items: list,
    phase_sequences: list,
    ps_to_fired_assemblies: dict,
    week_start: str,
    week_end: str,
    cfg: dict,
) -> str:

    ps_vocab_block = "\n".join(
        f"  {ps['id']}: {ps['name']} — {ps['definition']}"
        for ps in phase_sequences
    )

    ps_news_block = "\n\n".join(
        (
            f"### {item['id']}: {item['name']} [score: {item['activation_score']}/10]"
            + (f" — assemblies: {', '.join(ps_to_fired_assemblies.get(item['id'], []))}"
               if ps_to_fired_assemblies.get(item['id']) else "")
            + f"\n{item['summary']}"
        )
        for item in ps_news_items
    )

    valid_asm = [i for i in assembly_news_items if i["raw_news"] != "No data retrieved."]
    assembly_block = ""
    if valid_asm:
        asm_section = "\n\n".join(
            f"### {item['id']}: {item['name']}"
            + (f" [{', '.join(item['ps_memberships'])}]" if item.get("ps_memberships") else "")
            + f"\n{item['raw_news']}"
            for item in valid_asm
        )
        assembly_block = f"\n\n## CELL ASSEMBLY NEWS (organizational activity this week)\n\n{asm_section}"

    news_block = "\n\n".join(
        f"### {item['name']} ({item.get('domain', '')})\n{item['raw_news']}"
        for item in news_items
    )

    all_names   = [item["name"] for item in news_items]
    all_ps_ids  = [ps["id"] for ps in phase_sequences]
    all_asm_ids = [item["id"] for item in valid_asm]

    if all_asm_ids:
        asm_schema = """  "assembly_updates": [
    {
      "id": "CA-01",
      "name": "Assembly name",
      "signal": "active|quiet",
      "summary": "1-2 sentences on what this organization did this week and why it matters"
    }
  ],"""
        asm_rules = (
            f"- assembly_updates must include ALL of these: {json.dumps(all_asm_ids)}\n"
            "- assembly signal: \"active\" = took meaningful action this week; "
            "\"quiet\" = minimal or no notable activity"
        )
    else:
        asm_schema = ""
        asm_rules  = "- assembly_updates: [] (no assemblies fired this week)"

    return f"""You are synthesizing a weekly intelligence briefing on {cfg['synthesis_scope']} prime movers for the week of {week_start} to {week_end}.

## CANONICAL PHASE SEQUENCES

{ps_vocab_block}

## PHASE SEQUENCE NEWS (this week's structural activity)

{ps_news_block}{assembly_block}

## RAW NEWS BY INDIVIDUAL

{news_block}

## YOUR TASK

Synthesize the above into a structured weekly briefing JSON object. Schema:

{{
  "week_ending": "{week_end}",
  "executive_summary": "2-3 sentence overview of the most important developments this week across the {cfg['synthesis_scope']}",
  "person_updates": [
    {{
      "name": "Full Name",
      "signal": "active|quiet",
      "summary": "2-3 sentences on their most significant actions or events this week"
    }}
  ],
  {asm_schema}
  "top_stories": [
    {{
      "headline": "Brief headline",
      "persons": ["Name1", "Name2"],
      "ps_id": "{all_ps_ids[0] if all_ps_ids else 'PS-01'}",
      "valence": "adversarial|cooperative|neutral",
      "significance": "One sentence on why this matters for power dynamics"
    }}
  ]
}}

Rules:
- person_updates must include ALL of these people (even if quiet): {json.dumps(all_names)}
- signal definitions: "active" = default; assign unless there is genuinely no notable activity at all this week; "quiet" = reserve for members with no meaningful news whatsoever
- top_stories: 3-5 items only, the genuinely most significant
- top_stories valence: "adversarial" = direct conflict, legal opposition, or zero-sum competition; "cooperative" = explicit partnership, deal, or aligned action; "neutral" = informational update or parallel action without direct interaction between the named persons
{asm_rules}
- Return only the JSON object, no preamble or explanation"""


def build_phase_sequence_updates(
    ps_news_items: list,
    ps_to_fired_assemblies: dict,
) -> list:
    """
    Auto-build phase_sequence_updates from Perplexity PS news.
    Momentum is derived from activation score: 7-10 = accelerating, 4-6 = stable, 1-3 = decelerating.
    Summary is the Perplexity-written PS summary (first paragraph, stripped of the score line).
    assemblies_fired comes from the firing model.
    """
    updates = []
    for item in ps_news_items:
        # Strip the trailing activation score line Perplexity appends
        lines   = [l for l in item["summary"].splitlines() if not l.strip().isdigit()]
        summary = "\n".join(lines).strip()
        updates.append({
            "id":               item["id"],
            "name":             item["name"],
            "summary":          summary,
            "assemblies_fired": ps_to_fired_assemblies.get(item["id"], []),
        })
    return updates


def synthesize_briefing(
    client: anthropic.Anthropic,
    ps_news_items: list,
    assembly_news_items: list,
    news_items: list,
    phase_sequences: list,
    ps_to_fired_assemblies: dict,
    week_start: str,
    week_end: str,
    cfg: dict,
) -> dict:
    prompt = build_synthesis_prompt(
        ps_news_items, assembly_news_items, news_items,
        phase_sequences, ps_to_fired_assemblies, week_start, week_end, cfg,
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = message.content[0].text
    json_start = response_text.find("{")
    json_end   = response_text.rfind("}") + 1
    briefing   = json.loads(response_text[json_start:json_end])

    briefing["phase_sequence_updates"] = build_phase_sequence_updates(
        ps_news_items, ps_to_fired_assemblies
    )
    return briefing


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_json(briefing: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON  → {path.name}")


def save_markdown(briefing: dict, ps_news_items: list, path: Path, cfg: dict):
    signal_icon = {"active": "▲", "quiet": "—"}

    md  = f"# {cfg['md_title']} — {briefing['week_ending']}\n\n"
    md += f"## Executive Summary\n\n{briefing['executive_summary']}\n\n---\n\n"

    md += "## Phase Sequence Pulse\n\n"
    for ps in briefing.get("phase_sequence_updates", []):
        ps_news = next((n for n in ps_news_items if n["id"] == ps["id"]), None)
        score_str = f" [{ps_news['activation_score']}/10]" if ps_news else ""
        fired = ps.get("assemblies_fired") or []
        assemblies_str = f" — *{', '.join(fired)}*" if fired else ""
        md += f"**{ps['id']}: {ps['name']}**{score_str}{assemblies_str}  \n"
        md += f"{ps['summary']}\n\n"

    if briefing.get("assembly_updates"):
        md += "---\n\n## Cell Assembly Activity\n\n"
        for au in briefing.get("assembly_updates", []):
            sig = "▲" if au.get("signal") == "active" else "—"
            md += f"### {sig} {au['name']}\n\n"
            md += f"{au['summary']}\n\n"
            md += "---\n\n"

    md += "---\n\n## Top Stories\n\n"
    for story in briefing.get("top_stories", []):
        persons     = ", ".join(story.get("persons", []))
        ps_tag      = f" `{story['ps_id']}`" if story.get("ps_id") else ""
        valence     = story.get("valence", "")
        valence_tag = f" `{valence}`" if valence and valence != "neutral" else ""
        md += f"**{story['headline']}**{ps_tag}{valence_tag}  \n"
        md += f"*{persons}* — {story['significance']}\n\n"

    md += "---\n\n## Individual Updates\n\n"
    for p in briefing.get("person_updates", []):
        icon = signal_icon.get(p["signal"], "—")
        md += f"### {icon} {p['name']}\n\n"
        md += f"{p['summary']}\n\n"
        md += "---\n\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ MD    → {path.name}")


def print_console_summary(briefing: dict, fetched_count: int, skipped_count: int, cfg: dict):
    signal_icon = {"active": "▲", "quiet": "—"}
    print("\n" + "=" * 60)
    print(f"{cfg['label'].upper()} BRIEFING SUMMARY")
    print("=" * 60)
    print(f"\n{briefing['executive_summary']}\n")
    print(f"Coverage: {fetched_count} fetched, {skipped_count} skipped this week\n")
    print("Signal by person:")
    for p in briefing.get("person_updates", []):
        icon = signal_icon.get(p["signal"], "—")
        print(f"  {icon}  {p['name']}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(scope: str = "us"):
    cfg        = SCOPE_CONFIG[scope]
    week_end   = date.today().isoformat()
    week_start = (date.today() - timedelta(days=7)).isoformat()

    model_path       = SCRIPT_DIR / cfg["model_file"]
    fetch_state_path = FETCH_STATE_DIR / cfg["fetch_state_file"]

    BRIEFINGS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"WEEKLY BRIEFING GENERATOR [{cfg['label'].upper()}]")
    print(f"Week: {week_start} → {week_end}")
    print("=" * 60)

    # Stage 0: Load model + state
    if not model_path.exists():
        print(f"ERROR: {model_path.name} not found. Run superorganism_assembler.py --scope {scope} first.")
        return

    print(f"\nLoading {model_path.name}...")
    model           = load_superorganism(model_path)
    individuals     = model["superorganism_list"]
    phase_sequences = model["canonical_vocabulary"]["phase_sequences"]
    n               = len(individuals)
    spontaneous_cap = math.ceil(math.sqrt(n))
    print(f"  {n} individuals, {len(phase_sequences)} phase sequences")
    print(f"  Spontaneous cap: sqrt({n}) = {spontaneous_cap}")

    # Build assembly lookup maps from the loaded model
    cell_assemblies = model.get("canonical_vocabulary", {}).get("cell_assemblies", [])

    # ps_id → list of assembly dicts associated with that PS
    ps_assemblies_map: dict = {}
    for ca in cell_assemblies:
        for ps_id in ca.get("ps_memberships", []):
            ps_assemblies_map.setdefault(ps_id, []).append(ca)

    # ca_id → list of person dicts that are members of that assembly
    assembly_to_neurons: dict = {}
    for person in individuals:
        for ca in person.get("superorganism", {}).get("cell_assemblies", []):
            assembly_to_neurons.setdefault(ca["id"], []).append(person)

    # ps_id → list of person dicts with membership to that PS
    ps_to_neurons: dict = {}
    for person in individuals:
        for ps_id in ps_ids_for_person(person):
            ps_to_neurons.setdefault(ps_id, []).append(person)

    # person name → list of CA ids the person belongs to
    person_to_cas: dict = {}
    for person in individuals:
        name = person["name"]
        for ca in person.get("superorganism", {}).get("cell_assemblies", []):
            person_to_cas.setdefault(name, []).append(ca["id"])

    print(f"  {len(cell_assemblies)} cell assemblies ({len(assembly_to_neurons)} with members)")

    fetch_state     = load_fetch_state(fetch_state_path)

    coactivation_path  = COACTIVATION_STATE_DIR / cfg["coactivation_state_file"]
    coactivation_state = load_coactivation_state(coactivation_path)
    ca_coactivation    = coactivation_state.get("ca_coactivation", {})
    neuron_coactivation = coactivation_state.get("neuron_coactivation", {})
    if coactivation_state:
        print(f"  Coactivation state loaded ({len(ca_coactivation)} CA pairs, {len(neuron_coactivation)} N-N pairs)")
    else:
        print(f"  No coactivation state found — selection will be random (run coactivation_updater.py --bootstrap --scope {scope})")

    # Stage 1A: PS news
    print(f"\nStage 1A: Fetching PS news via Perplexity ({len(phase_sequences)} calls)...")
    perplexity    = get_perplexity_client()
    ps_news_items = []
    for ps in phase_sequences:
        try:
            result = fetch_news_for_ps(perplexity, ps, week_start, week_end, cfg)
            ps_news_items.append(result)
        except Exception as e:
            print(f"  ! {ps['id']}: {e}")
            ps_news_items.append({
                "id": ps["id"], "name": ps["name"],
                "summary": "No data retrieved.", "activation_score": 0,
            })

    scores = [p["activation_score"] for p in ps_news_items]
    print(f"  ✓ PS activation scores: min={min(scores)}, max={max(scores)}, mean={sum(scores)/len(scores):.1f}")

    # Stage 1B: Coactivation-guided assembly sequence building
    fired_assemblies_ordered, ps_to_fired_assemblies, ps_ranked_items = build_fired_assemblies(
        ps_news_items, ps_assemblies_map, ca_coactivation
    )

    print(f"\nStage 1B: Assembly sequences — {len(fired_assemblies_ordered)} assemblies fired:")
    for ps_id, ca_ids in ps_to_fired_assemblies.items():
        print(f"  [{ps_id}] → {', '.join(ca_ids)}")

    # Stage 1B→C: PS-gated conscious neuron selection
    conscious_neurons = select_neurons_conscious(
        ps_ranked_items, ps_to_fired_assemblies, ps_to_neurons,
        assembly_to_neurons, neuron_coactivation
    )
    print(f"\nStage 1B→C: Conscious selection — {len(conscious_neurons)} neurons:")
    for p in conscious_neurons:
        print(f"  • {p['name']}")

    # Stage 1.5: Assembly news for network-selected assemblies
    ca_lookup   = {ca["id"]: ca for ca in cell_assemblies}
    ps_name_map = {ps["id"]: ps["name"] for ps in phase_sequences}
    fired_ca_ids = [ca_id for ca_id, _ in fired_assemblies_ordered if ca_id in ca_lookup]

    assembly_news_items: list = []
    if fired_ca_ids:
        print(f"\nStage 1.5: Fetching assembly news via Perplexity ({len(fired_ca_ids)} calls)...")
        for ca_id in fired_ca_ids:
            ca       = ca_lookup[ca_id]
            ps_names = [ps_name_map[pid] for pid in ca.get("ps_memberships", []) if pid in ps_name_map]
            try:
                result = fetch_news_for_assembly(perplexity, ca, ps_names, week_start, week_end, cfg)
                assembly_news_items.append(result)
            except Exception as e:
                print(f"  ! {ca_id}: {e}")
                assembly_news_items.append({
                    "id": ca_id, "name": ca["name"],
                    "ps_memberships": ca.get("ps_memberships", []),
                    "raw_news": "No data retrieved.",
                })
        print(f"  ✓ {len(assembly_news_items)} assembly summaries collected")
    else:
        print("\nStage 1.5: No assemblies fired this week — skipping assembly news")

    # Stage 1C: Spontaneous neuron selection
    exclude_names = {p["name"] for p in conscious_neurons}
    spontaneous_neurons, skipped_neurons = select_neurons_spontaneous(
        individuals, fetch_state, spontaneous_cap, exclude_names,
        weekly_increase=cfg["spontaneous_weekly_increase"],
    )
    print(f"\nStage 1C: Spontaneous selection — {len(spontaneous_neurons)}/{spontaneous_cap} cap:")
    for p in spontaneous_neurons:
        skips = fetch_state.get(p["name"], 0)
        print(f"  • {p['name']} (skipped {skips}w)")
    if skipped_neurons:
        print(f"  Skipping {len(skipped_neurons)} neurons:")
        for p in skipped_neurons:
            skips    = fetch_state.get(p["name"], 0)
            next_prob = min(10 + (skips + 1) * 100 * cfg["spontaneous_weekly_increase"], 100)
            print(f"    · {p['name']} (skipped {skips}w → {next_prob:.0f}% next week)")

    all_fetched = conscious_neurons + spontaneous_neurons
    all_skipped = [p for p in individuals if p["name"] not in {f["name"] for f in all_fetched}]

    # Stage 1C.5: Spontaneous CA selection + news fetch
    consciously_fired_ca_ids = set(ca_id for ca_id, _ in fired_assemblies_ordered)
    spontaneous_ca_ids = select_spontaneous_cas(
        spontaneous_neurons, person_to_cas, consciously_fired_ca_ids
    )
    print(f"\nStage 1C.5: Spontaneous CAs — {len(spontaneous_ca_ids)} drawn:")
    for ca_id in spontaneous_ca_ids:
        print(f"  • {ca_id}")

    if spontaneous_ca_ids:
        print(f"  Fetching news for {len(spontaneous_ca_ids)} spontaneous CAs...")
        for ca_id in spontaneous_ca_ids:
            if ca_id not in ca_lookup:
                print(f"  ! {ca_id}: not found in model — skipping")
                continue
            ca       = ca_lookup[ca_id]
            ps_names = [ps_name_map[pid] for pid in ca.get("ps_memberships", []) if pid in ps_name_map]
            try:
                result = fetch_news_for_assembly(perplexity, ca, ps_names, week_start, week_end, cfg)
                result["spontaneous"] = True
                assembly_news_items.append(result)
            except Exception as e:
                print(f"  ! {ca_id}: {e}")
                assembly_news_items.append({
                    "id": ca_id, "name": ca["name"],
                    "ps_memberships": ca.get("ps_memberships", []),
                    "raw_news": "No data retrieved.",
                    "spontaneous": True,
                })
        print(f"  ✓ spontaneous assembly news fetched")

    # Stage 1D: Individual neuron news
    print(f"\nStage 1D: Fetching individual news via Perplexity ({len(all_fetched)} calls)...")
    news_items = []
    for person in all_fetched:
        try:
            result = fetch_news_for_person(perplexity, person, week_start, week_end, cfg)
            news_items.append(result)
        except Exception as e:
            print(f"  ! {person['name']}: {e}")
            news_items.append({
                "name":     person["name"],
                "domain":   person.get("title", person.get("domain", "")),
                "raw_news": "No data retrieved.",
            })
    print(f"  ✓ {len(news_items)} news summaries collected")

    # Update and persist fetch state (all fetched neurons reset; all skipped increment)
    new_fetch_state = update_fetch_state(fetch_state, all_fetched, all_skipped, cfg["spontaneous_counter_cap"])
    save_fetch_state(new_fetch_state, fetch_state_path)
    print(f"\n  ✓ Fetch state saved → {fetch_state_path.name}")

    # Save raw data so synthesis can be retried without re-running Perplexity
    raw_path = BRIEFINGS_DIR / f"{cfg['briefing_prefix']}_raw_{week_end}.json"
    raw_data = {
        "scope":                    scope,
        "week_start":               week_start,
        "week_end":                 week_end,
        "ps_news_items":            ps_news_items,
        "assembly_news_items":      assembly_news_items,
        "news_items":               news_items,
        "ps_assemblies_fired":      ps_to_fired_assemblies,
        "assemblies_spontaneous":   spontaneous_ca_ids,
        "neurons_conscious":        [p["name"] for p in conscious_neurons],
        "neurons_spontaneous":      [p["name"] for p in spontaneous_neurons],
        "neurons_skipped":          [p["name"] for p in all_skipped],
        "spontaneous_cap":          spontaneous_cap,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Raw data saved  → {raw_path.name}")

    # Filter failed fetches before synthesis
    synthesis_items = [i for i in news_items if i["raw_news"] != "No data retrieved."]
    if not synthesis_items:
        print("  ! No news items to synthesize. Aborting.")
        return

    # Stage 2: Claude synthesis
    print(f"\nStage 2: Synthesizing with Claude ({len(synthesis_items)} neurons + {len(ps_news_items)} PS summaries)...")
    try:
        claude   = get_anthropic_client()
        briefing = synthesize_briefing(
            claude, ps_news_items, assembly_news_items, synthesis_items,
            phase_sequences, ps_to_fired_assemblies, week_start, week_end, cfg,
        )
    except Exception as e:
        print(f"  ! Synthesis failed: {e}")
        return

    briefing["_metadata"] = {
        "generated":                date.today().isoformat(),
        "scope":                    scope,
        "week_start":               week_start,
        "week_end":                 week_end,
        "source_model":             model_path.name,
        "news_model":               SONAR_MODEL,
        "synthesis_model":          CLAUDE_MODEL,
        "neurons_conscious":        [p["name"] for p in conscious_neurons],
        "neurons_spontaneous":      [p["name"] for p in spontaneous_neurons],
        "neurons_skipped":          [p["name"] for p in all_skipped],
        "spontaneous_cap":          spontaneous_cap,
        "ps_activation_scores":     {p["id"]: p["activation_score"] for p in ps_news_items},
        "ps_assemblies_fired":      ps_to_fired_assemblies,
        "assemblies_spontaneous":   spontaneous_ca_ids,
        "assemblies_fetched":       [a["id"] for a in assembly_news_items],
    }

    # Save outputs
    print("\nSaving outputs...")
    stem = f"{cfg['briefing_prefix']}_{week_end}"
    save_json(briefing, BRIEFINGS_DIR / f"{stem}.json")
    save_markdown(briefing, ps_news_items, BRIEFINGS_DIR / f"{stem}.md", cfg)

    print_console_summary(briefing, len(all_fetched), len(all_skipped), cfg)
    print("=" * 60)
    print("Done!")
    print("=" * 60)


def run_synthesis_only(scope: str = "us"):
    """
    Re-run Stage 2 only, loading the most recent raw data file saved by a previous run.
    Use after a synthesis failure without re-fetching Perplexity news.
    """
    cfg = SCOPE_CONFIG[scope]
    BRIEFINGS_DIR.mkdir(exist_ok=True)

    # Find most recent raw data file for this scope
    raw_files = sorted(BRIEFINGS_DIR.glob(f"{cfg['briefing_prefix']}_raw_*.json"))
    if not raw_files:
        print(f"ERROR: No raw data file found in {BRIEFINGS_DIR}/")
        print(f"  Run without --synthesize-only first to collect news.")
        return
    raw_path = raw_files[-1]
    print(f"Loading raw data from {raw_path.name}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ps_news_items          = raw["ps_news_items"]
    assembly_news_items    = raw.get("assembly_news_items", [])
    news_items             = raw["news_items"]
    ps_to_fired_assemblies = raw.get("ps_assemblies_fired", {})
    week_start             = raw["week_start"]
    week_end               = raw["week_end"]

    # Load model just for phase_sequences vocabulary
    model_path = SCRIPT_DIR / cfg["model_file"]
    if not model_path.exists():
        print(f"ERROR: {model_path.name} not found.")
        return
    model           = load_superorganism(model_path)
    phase_sequences = model["canonical_vocabulary"]["phase_sequences"]

    synthesis_items = [i for i in news_items if i["raw_news"] != "No data retrieved."]
    print(f"Synthesizing {len(synthesis_items)} neurons + {len(ps_news_items)} PS summaries...")

    try:
        claude   = get_anthropic_client()
        briefing = synthesize_briefing(
            claude, ps_news_items, assembly_news_items, synthesis_items,
            phase_sequences, ps_to_fired_assemblies, week_start, week_end, cfg,
        )
    except Exception as e:
        print(f"  ! Synthesis failed: {e}")
        return

    briefing["_metadata"] = {
        "generated":               date.today().isoformat(),
        "scope":                   scope,
        "week_start":              week_start,
        "week_end":                week_end,
        "source_model":            model_path.name,
        "news_model":              SONAR_MODEL,
        "synthesis_model":         CLAUDE_MODEL,
        "neurons_conscious":       raw.get("neurons_conscious", []),
        "neurons_spontaneous":     raw.get("neurons_spontaneous", []),
        "neurons_skipped":         raw.get("neurons_skipped", []),
        "spontaneous_cap":         raw.get("spontaneous_cap", 0),
        "ps_activation_scores":    {p["id"]: p["activation_score"] for p in ps_news_items},
        "ps_assemblies_fired":     ps_to_fired_assemblies,
        "assemblies_fetched":      [a["id"] for a in assembly_news_items],
        "synthesize_only":         True,
    }

    print("\nSaving outputs...")
    stem = f"{cfg['briefing_prefix']}_{week_end}"
    save_json(briefing, BRIEFINGS_DIR / f"{stem}.json")
    save_markdown(briefing, ps_news_items, BRIEFINGS_DIR / f"{stem}.md", cfg)
    print_console_summary(briefing, len(synthesis_items), len(raw.get("neurons_skipped", [])), cfg)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly prime mover briefing")
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which superorganism model to run (default: us)"
    )
    parser.add_argument(
        "--synthesize-only", action="store_true",
        help="Skip Perplexity calls; re-run Claude synthesis from the most recent saved raw data"
    )
    args = parser.parse_args()
    if args.synthesize_only:
        run_synthesis_only(scope=args.scope)
    else:
        run(scope=args.scope)
