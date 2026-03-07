"""
Weekly Briefing Generator
Fetches news on US or global prime movers via Perplexity, then synthesizes
via Claude into a structured weekly report.

Usage:
    python weekly_briefing.py                          # US run
    python weekly_briefing.py --scope global           # Global run

Firing model:
    Conscious layer  — PS news fetched first; active PSes recruit neurons via fired
                       cell assemblies → weighted sampling from assembly members.
                       Fallback: direct PS membership sampling if no assemblies fired.
    Spontaneous layer — Skip-counter probability (10% base + 10%/skipped week), sorted by
                        most-overdue first, capped at sqrt(n). No rank-based guarantees.
    Together they determine which neurons get individual news fetched this week.
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

SCOPE_CONFIG = {
    "us": {
        "label":               "US",
        "model_file":          "us_superorganism_model.json",
        "state_file":          "superorganism_state.json",
        "fetch_state_file":    "fetch_state.json",
        "briefing_prefix":     "weekly_briefing",
        "md_title":            "Weekly Prime Mover Briefing",
        "synthesis_scope":     "US superorganism",
        "news_system_prompt":  "You are a concise news analyst. Report factual recent events only.",
        "news_hard_categories": (
            "policy decisions, business moves, legal actions, "
            "geopolitical events, significant public statements"
        ),
        "ps_system_prompt":    "You are a concise US policy and power analyst.",
    },
    "global": {
        "label":               "Global",
        "model_file":          "superorganism_model.json",
        "state_file":          "global_superorganism_state.json",
        "fetch_state_file":    "global_fetch_state.json",
        "briefing_prefix":     "global_weekly_briefing",
        "md_title":            "Global Prime Mover Briefing",
        "synthesis_scope":     "global superorganism",
        "news_system_prompt":  "You are a concise geopolitical analyst. Report factual recent events only.",
        "news_hard_categories": (
            "policy decisions, geopolitical moves, military actions, "
            "economic initiatives, diplomatic events, significant public statements"
        ),
        "ps_system_prompt":    "You are a concise geopolitical and power analyst.",
    },
}

# Assembly-aware conscious selection: assemblies and neurons recruited per PS activation level
ASSEMBLIES_PER_ACTIVATION = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
NEURONS_PER_ACTIVATION    = {"HIGH": 3, "MEDIUM": 1, "LOW": 0}


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


def load_hebbian_state(path: Path) -> dict:
    """Load Hebbian state. Returns empty dicts if file missing (graceful degradation)."""
    if not path.exists():
        print(f"  ! Hebbian state not found at {path.name} — conscious selection will use default weights.")
        return {"neuron_dps_weights": {}, "dps_dominance": {}}
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_fetch_state(state: dict, fetched_people: list, skipped_people: list) -> dict:
    new_state = dict(state)
    for person in fetched_people:
        new_state[person["name"]] = 0
    for person in skipped_people:
        name = person["name"]
        new_state[name] = new_state.get(name, 0) + 1
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


# ---------------------------------------------------------------------------
# STAGE 1A: PS news via Perplexity
# ---------------------------------------------------------------------------

def fetch_news_for_ps(
    client: OpenAI, ps: dict, ps_assemblies: list, week_start: str, week_end: str, cfg: dict
) -> dict:
    """
    One Perplexity call per phase sequence.
    Asks Perplexity which known cell assemblies were implicated in this week's events.
    Returns summary + activation signal + assemblies_fired.
    """
    assembly_block = ""
    if ps_assemblies:
        lines = "\n".join(f"  - {ca['id']}: {ca['name']}" for ca in ps_assemblies)
        assembly_block = (
            f"\n\nKnown active coalitions/organizations in this domain:\n{lines}\n\n"
            f"After your summary and activation signal, on a new line list the IDs of any "
            f"coalitions clearly implicated in this week's events (comma-separated).\n"
            f"Format: ASSEMBLIES: CA-01, CA-05\n"
            f"If none were clearly implicated, write: ASSEMBLIES: none"
        )

    prompt = (
        f"Summarize the most significant developments in the following domain from the past 7 days "
        f"({week_start} to {week_end}).\n\n"
        f"Domain: {ps['name']}\n"
        f"Definition: {ps['definition']}\n\n"
        f"Focus on: structural changes, major decisions, policy moves, business actions, or events "
        f"that indicate momentum in this area. 3-5 sentences.\n\n"
        f"After your summary, on a new line, write exactly one of: HIGH, MEDIUM, or LOW\n"
        f"HIGH = major developments, clear momentum, multiple significant events this week\n"
        f"MEDIUM = some activity, moderate developments, normal background level\n"
        f"LOW = quiet week, minimal developments, no major changes"
        f"{assembly_block}"
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

    # Parse activation signal and assemblies_fired from response lines
    activation_signal = "MEDIUM"
    assemblies_fired = []

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper in ("HIGH", "MEDIUM", "LOW"):
            activation_signal = upper
        elif upper.startswith("ASSEMBLIES:"):
            rest = stripped[11:].strip()  # len("ASSEMBLIES:") == 11
            assemblies_fired = (
                [] if rest.upper() == "NONE"
                else [a.strip() for a in rest.split(",") if a.strip()]
            )

    # Cap assemblies to maximum allowed for this activation level
    max_a = ASSEMBLIES_PER_ACTIVATION.get(activation_signal, 0)
    assemblies_fired = assemblies_fired[:max_a]

    return {
        "id":                ps["id"],
        "name":              ps["name"],
        "summary":           raw,
        "activation_signal": activation_signal,
        "assemblies_fired":  assemblies_fired,
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
# STAGE 1B: Conscious neuron selection (assembly-aware, PS-driven)
# ---------------------------------------------------------------------------

def select_neurons_from_assemblies(
    ps_news_items: list,
    individuals: list,
    assembly_to_neurons: dict,
    neuron_dps_weights: dict,
) -> list:
    """
    For each active PS (HIGH/MEDIUM), route through fired assemblies to recruit neurons.
    Fallback: if no assemblies fired or no members found, sample directly from PS members.
    Returns deduplicated list of person dicts.
    """
    selected_names = set()
    selected = []

    for ps_item in ps_news_items:
        signal = ps_item["activation_signal"]
        k = NEURONS_PER_ACTIVATION.get(signal, 0)
        if k == 0:
            continue

        ps_id = ps_item["id"]
        assemblies_fired = ps_item.get("assemblies_fired", [])

        # Collect candidates: from fired assemblies (preferred)
        candidates = []
        if assemblies_fired:
            seen_in_batch = set()
            for ca_id in assemblies_fired:
                for p in assembly_to_neurons.get(ca_id, []):
                    if p["name"] not in seen_in_batch:
                        seen_in_batch.add(p["name"])
                        candidates.append(p)

        # Fallback: direct PS membership if no assembly candidates found
        if not candidates:
            candidates = [p for p in individuals if ps_id in ps_ids_for_person(p)]

        if not candidates:
            continue

        weights = [
            max(neuron_dps_weights.get(p["name"], {}).get(ps_id, 0.05), 0.01)
            for p in candidates
        ]

        drawn = weighted_sample_without_replacement(candidates, weights, k)
        for person in drawn:
            if person["name"] not in selected_names:
                selected_names.add(person["name"])
                selected.append(person)

    return selected


# ---------------------------------------------------------------------------
# STAGE 1C: Spontaneous neuron selection (skip-counter driven)
# ---------------------------------------------------------------------------

def select_neurons_spontaneous(
    individuals: list,
    fetch_state: dict,
    cap: int,
    exclude_names: set,
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
        prob = min(0.10 + skipped_weeks * 0.10, 1.0)
        if random.random() < prob:
            fetched.append(person)
        else:
            skipped.append(person)

    return fetched, skipped


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
            f"### {item['id']}: {item['name']} [{item['activation_signal']}]"
            + (f" — assemblies: {', '.join(item['assemblies_fired'])}" if item.get("assemblies_fired") else "")
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
        asm_schema = f"""  "assembly_updates": [
    {{
      "id": "CA-01",
      "name": "Assembly name",
      "signal": "active|quiet",
      "summary": "1-2 sentences on what this organization did this week and why it matters",
      "ps_ids": ["{all_ps_ids[0] if all_ps_ids else 'PS-01'}"]
    }}
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
      "signal": "notable|quiet|concerning",
      "summary": "2-3 sentences on their most significant actions or events this week",
      "ps_impacts": ["{all_ps_ids[0] if all_ps_ids else 'PS-01'}"]
    }}
  ],
  {asm_schema}
  "phase_sequence_updates": [
    {{
      "id": "{all_ps_ids[0] if all_ps_ids else 'PS-01'}",
      "name": "Phase sequence name",
      "momentum": "accelerating|stable|decelerating",
      "summary": "1-2 sentences on key developments for this phase sequence this week",
      "assemblies_fired": ["CA-01"]
    }}
  ],
  "top_stories": [
    {{
      "headline": "Brief headline",
      "persons": ["Name1", "Name2"],
      "ps_id": "{all_ps_ids[0] if all_ps_ids else 'PS-01'}",
      "valence": "adversarial|cooperative|neutral",
      "significance": "One sentence on why this matters for power dynamics"
    }}
  ],
  "edge_signals": [
    {{
      "person_a": "Name1",
      "person_b": "Name2",
      "ps_id": "{all_ps_ids[0] if all_ps_ids else 'PS-01'}",
      "valence": "adversarial|cooperative|neutral",
      "evidence": "One sentence citing the specific event or dynamic between these two people"
    }}
  ]
}}

Rules:
- person_updates must include ALL of these people (even if quiet): {json.dumps(all_names)}
- phase_sequence_updates must include ALL of these: {json.dumps(all_ps_ids)}
- signal definitions: "notable" = significant positive action or influence gain; "concerning" = meaningful setback, loss of influence, or serious threat to position; "quiet" = no major developments
- ps_impacts: only list PS IDs concretely activated by this person's actions this week
- assemblies_fired: list only assembly IDs (from the PS news above) that were clearly active; omit the field or use [] if none fired
- top_stories: 3-5 items only, the genuinely most significant
- top_stories valence: "adversarial" = direct conflict, legal opposition, or zero-sum competition; "cooperative" = explicit partnership, deal, or aligned action; "neutral" = informational update or parallel action without direct interaction between the named persons
- edge_signals: derived from top_stories entries that name exactly 2 persons; include only "adversarial" and "cooperative" entries (omit neutral); one entry per qualifying story; if a story names only 1 person, omit it from edge_signals
{asm_rules}
- Return only the JSON object, no preamble or explanation"""


def synthesize_briefing(
    client: anthropic.Anthropic,
    ps_news_items: list,
    assembly_news_items: list,
    news_items: list,
    phase_sequences: list,
    week_start: str,
    week_end: str,
    cfg: dict,
) -> dict:
    prompt = build_synthesis_prompt(
        ps_news_items, assembly_news_items, news_items,
        phase_sequences, week_start, week_end, cfg,
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
    return json.loads(response_text[json_start:json_end])


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_json(briefing: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON  → {path.name}")


def save_markdown(briefing: dict, ps_news_items: list, path: Path, cfg: dict):
    signal_icon   = {"notable": "▲", "quiet": "—", "concerning": "▼"}
    momentum_icon = {"accelerating": "↑", "stable": "→", "decelerating": "↓"}
    activation_icon = {"HIGH": "▲", "MEDIUM": "→", "LOW": "▼"}

    md  = f"# {cfg['md_title']} — {briefing['week_ending']}\n\n"
    md += f"## Executive Summary\n\n{briefing['executive_summary']}\n\n---\n\n"

    md += "## Phase Sequence Pulse\n\n"
    for ps in briefing.get("phase_sequence_updates", []):
        icon = momentum_icon.get(ps["momentum"], "→")
        ps_news = next((n for n in ps_news_items if n["id"] == ps["id"]), None)
        activation = (
            f" [{activation_icon.get(ps_news['activation_signal'], '')} {ps_news['activation_signal']}]"
            if ps_news else ""
        )
        # Show assemblies fired (from briefing first, fall back to ps_news)
        fired = ps.get("assemblies_fired") or (ps_news.get("assemblies_fired") if ps_news else [])
        assemblies_str = f" — *{', '.join(fired)}*" if fired else ""
        md += f"**{icon} {ps['id']}: {ps['name']}** `{ps['momentum']}`{activation}{assemblies_str}  \n"
        md += f"{ps['summary']}\n\n"

    if briefing.get("assembly_updates"):
        md += "---\n\n## Cell Assembly Activity\n\n"
        for au in briefing.get("assembly_updates", []):
            sig      = "▲" if au.get("signal") == "active" else "—"
            ps_tags  = " ".join(f"`{pid}`" for pid in au.get("ps_ids", []))
            md += f"### {sig} {au['name']}\n\n"
            md += f"{au['summary']}\n\n"
            if ps_tags:
                md += f"**Phase sequences:** {ps_tags}\n\n"
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
        icon    = signal_icon.get(p["signal"], "—")
        ps_tags = " ".join(f"`{ps}`" for ps in p.get("ps_impacts", []))
        md += f"### {icon} {p['name']}\n\n"
        md += f"{p['summary']}\n\n"
        if ps_tags:
            md += f"**Phase sequences:** {ps_tags}\n\n"
        md += "---\n\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ MD    → {path.name}")


def print_console_summary(briefing: dict, fetched_count: int, skipped_count: int, cfg: dict):
    signal_icon = {"notable": "▲", "quiet": "—", "concerning": "▼"}
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
    state_path       = SCRIPT_DIR / cfg["state_file"]
    fetch_state_path = SCRIPT_DIR / cfg["fetch_state_file"]

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

    print(f"  {len(cell_assemblies)} cell assemblies ({len(assembly_to_neurons)} with members)")

    hebbian         = load_hebbian_state(state_path)
    neuron_weights  = hebbian.get("neuron_dps_weights", {})
    fetch_state     = load_fetch_state(fetch_state_path)

    # Stage 1A: PS news
    print(f"\nStage 1A: Fetching PS news via Perplexity ({len(phase_sequences)} calls)...")
    perplexity    = get_perplexity_client()
    ps_news_items = []
    for ps in phase_sequences:
        try:
            ps_assemblies = ps_assemblies_map.get(ps["id"], [])
            result = fetch_news_for_ps(perplexity, ps, ps_assemblies, week_start, week_end, cfg)
            ps_news_items.append(result)
        except Exception as e:
            print(f"  ! {ps['id']}: {e}")
            ps_news_items.append({
                "id": ps["id"], "name": ps["name"],
                "summary": "No data retrieved.", "activation_signal": "LOW",
                "assemblies_fired": [],
            })

    active_counts = {s: sum(1 for p in ps_news_items if p["activation_signal"] == s)
                     for s in ("HIGH", "MEDIUM", "LOW")}
    print(f"  ✓ PS activation: {active_counts['HIGH']} HIGH, {active_counts['MEDIUM']} MEDIUM, {active_counts['LOW']} LOW")

    # Stage 1B: Conscious neuron selection (assembly-aware)
    conscious_neurons = select_neurons_from_assemblies(
        ps_news_items, individuals, assembly_to_neurons, neuron_weights
    )
    print(f"\nStage 1B: Conscious selection — {len(conscious_neurons)} neurons via assemblies:")
    for ps_item in ps_news_items:
        if ps_item.get("assemblies_fired"):
            print(f"  [{ps_item['id']}] assemblies fired: {', '.join(ps_item['assemblies_fired'])}")
    for p in conscious_neurons:
        print(f"  • {p['name']}")

    # Stage 1.5: Assembly news for fired assemblies
    ca_lookup   = {ca["id"]: ca for ca in cell_assemblies}
    ps_name_map = {ps["id"]: ps["name"] for ps in phase_sequences}
    fired_ca_ids: list = []
    seen_ca_ids:  set  = set()
    for ps_item in ps_news_items:
        for ca_id in ps_item.get("assemblies_fired", []):
            if ca_id not in seen_ca_ids and ca_id in ca_lookup:
                seen_ca_ids.add(ca_id)
                fired_ca_ids.append(ca_id)

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
        individuals, fetch_state, spontaneous_cap, exclude_names
    )
    print(f"\nStage 1C: Spontaneous selection — {len(spontaneous_neurons)}/{spontaneous_cap} cap:")
    for p in spontaneous_neurons:
        skips = fetch_state.get(p["name"], 0)
        print(f"  • {p['name']} (skipped {skips}w)")
    if skipped_neurons:
        print(f"  Skipping {len(skipped_neurons)} neurons:")
        for p in skipped_neurons:
            skips    = fetch_state.get(p["name"], 0)
            next_prob = min(10 + (skips + 1) * 10, 100)
            print(f"    · {p['name']} (skipped {skips}w → {next_prob}% next week)")

    all_fetched = conscious_neurons + spontaneous_neurons
    all_skipped = [p for p in individuals if p["name"] not in {f["name"] for f in all_fetched}]

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
    new_fetch_state = update_fetch_state(fetch_state, all_fetched, all_skipped)
    save_fetch_state(new_fetch_state, fetch_state_path)
    print(f"\n  ✓ Fetch state saved → {fetch_state_path.name}")

    # Save raw data so synthesis can be retried without re-running Perplexity
    raw_path = BRIEFINGS_DIR / f"{cfg['briefing_prefix']}_raw_{week_end}.json"
    raw_data = {
        "scope":               scope,
        "week_start":          week_start,
        "week_end":            week_end,
        "ps_news_items":       ps_news_items,
        "assembly_news_items": assembly_news_items,
        "news_items":          news_items,
        "neurons_conscious":   [p["name"] for p in conscious_neurons],
        "neurons_spontaneous": [p["name"] for p in spontaneous_neurons],
        "neurons_skipped":     [p["name"] for p in all_skipped],
        "spontaneous_cap":     spontaneous_cap,
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
            phase_sequences, week_start, week_end, cfg,
        )
    except Exception as e:
        print(f"  ! Synthesis failed: {e}")
        return

    briefing["_metadata"] = {
        "generated":              date.today().isoformat(),
        "scope":                  scope,
        "week_start":             week_start,
        "week_end":               week_end,
        "source_model":           model_path.name,
        "news_model":             SONAR_MODEL,
        "synthesis_model":        CLAUDE_MODEL,
        "neurons_conscious":      [p["name"] for p in conscious_neurons],
        "neurons_spontaneous":    [p["name"] for p in spontaneous_neurons],
        "neurons_skipped":        [p["name"] for p in all_skipped],
        "spontaneous_cap":        spontaneous_cap,
        "ps_activation_signals":  {p["id"]: p["activation_signal"] for p in ps_news_items},
        "ps_assemblies_fired":    {p["id"]: p.get("assemblies_fired", []) for p in ps_news_items if p.get("assemblies_fired")},
        "assemblies_fetched":     [a["id"] for a in assembly_news_items],
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

    ps_news_items       = raw["ps_news_items"]
    assembly_news_items = raw.get("assembly_news_items", [])
    news_items          = raw["news_items"]
    week_start     = raw["week_start"]
    week_end       = raw["week_end"]

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
            phase_sequences, week_start, week_end, cfg,
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
        "ps_activation_signals":   {p["id"]: p["activation_signal"] for p in ps_news_items},
        "ps_assemblies_fired":     {p["id"]: p.get("assemblies_fired", []) for p in ps_news_items if p.get("assemblies_fired")},
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
