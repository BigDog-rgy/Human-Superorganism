"""
Weekly Briefing Generator (US)
Fetches recent news on each US prime mover via Perplexity,
then synthesizes via Claude into a structured weekly report.

Usage:
    python weekly_briefing.py                # full run (news + social signals)
    python weekly_briefing.py --no-social    # skip Grok social signals call
"""

import os
import json
import argparse
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SONAR_MODEL = "sonar-pro"
CLAUDE_MODEL = "claude-opus-4-6"
GROK_MODEL = "grok-4-1-fast-reasoning"

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "us_superorganism_model.json"
BRIEFINGS_DIR = SCRIPT_DIR / "briefings"


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


def get_xai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_superorganism(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STAGE 1A: Per-person news via Perplexity
# ---------------------------------------------------------------------------

def fetch_news_for_person(client: OpenAI, person: dict, week_start: str, week_end: str) -> dict:
    """One Perplexity call per person — targeted 7-day news digest."""
    name = person["name"]
    domain = person["domain"]

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
        "Hard news only: policy decisions, business moves, legal actions, "
        "geopolitical events, significant public statements. Skip routine appearances."
    )

    print(f"  [{name}]...")
    response = client.chat.completions.create(
        model=SONAR_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise news analyst. Report factual recent events only."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return {
        "name": name,
        "domain": domain,
        "raw_news": response.choices[0].message.content
    }


# ---------------------------------------------------------------------------
# STAGE 1B: Social signals via Grok (optional)
# ---------------------------------------------------------------------------

def fetch_social_signals(client: OpenAI, top_names: list, week_end: str) -> str:
    """One Grok call for X/Twitter social signals on the top 6 prime movers."""
    names_str = ", ".join(top_names)
    prompt = (
        f"What were the significant social media and public discourse moments this past week "
        f"(ending {week_end}) around these US power figures: {names_str}?\n\n"
        "Focus on: major viral moments, elite/political reactions, "
        "significant sentiment shifts, noteworthy X/Twitter activity that signals "
        "changing influence or political dynamics.\n\n"
        "2-3 sentences per person. Hard signals only — skip minor news."
    )

    print("  [Social signals via Grok]...")
    response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a political analyst monitoring US power and media dynamics."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# STAGE 2: Claude synthesis
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    news_items: list,
    social_signals: str,
    phase_sequences: list,
    week_start: str,
    week_end: str
) -> str:

    ps_block = "\n".join(
        f"  {ps['id']}: {ps['name']} — {ps['definition']}"
        for ps in phase_sequences
    )

    news_block = "\n\n".join(
        f"### {item['name']} ({item['domain']})\n{item['raw_news']}"
        for item in news_items
    )

    social_block = (
        f"\n\n## Social / Narrative Signals (via Grok / X)\n\n{social_signals}"
        if social_signals else ""
    )

    all_names = [item["name"] for item in news_items]
    all_dps_ids = [ps["id"] for ps in phase_sequences]

    return f"""You are synthesizing a weekly intelligence briefing on US prime movers for the week of {week_start} to {week_end}.

## CANONICAL PHASE SEQUENCES (US Domestic)

{ps_block}

## RAW NEWS BY INDIVIDUAL

{news_block}{social_block}

## YOUR TASK

Synthesize the above into a structured weekly briefing JSON object. Schema:

{{
  "week_ending": "{week_end}",
  "executive_summary": "2-3 sentence overview of the most important developments this week across the US superorganism",
  "person_updates": [
    {{
      "name": "Full Name",
      "signal": "notable|quiet|concerning",
      "summary": "2-3 sentences on their most significant actions or events this week",
      "ps_impacts": ["DPS-01", "DPS-03"]
    }}
  ],
  "phase_sequence_updates": [
    {{
      "id": "DPS-01",
      "name": "Phase sequence name",
      "momentum": "accelerating|stable|decelerating",
      "summary": "1-2 sentences on key developments for this phase sequence this week"
    }}
  ],
  "top_stories": [
    {{
      "headline": "Brief headline",
      "persons": ["Name1", "Name2"],
      "ps_id": "DPS-XX",
      "valence": "adversarial|cooperative|neutral",
      "significance": "One sentence on why this matters for US power dynamics"
    }}
  ],
  "edge_signals": [
    {{
      "person_a": "Name1",
      "person_b": "Name2",
      "ps_id": "DPS-XX",
      "valence": "adversarial|cooperative|neutral",
      "evidence": "One sentence citing the specific event or dynamic between these two people"
    }}
  ]
}}

Rules:
- person_updates must include ALL of these people (even if quiet): {json.dumps(all_names)}
- phase_sequence_updates must include ALL of these: {json.dumps(all_dps_ids)}
- signal definitions: "notable" = significant positive action or influence gain; "concerning" = meaningful setback, loss of influence, or serious threat to position; "quiet" = no major developments
- ps_impacts: only list DPS IDs concretely activated by this person's actions this week
- top_stories: 3-5 items only, the genuinely most significant
- top_stories valence: "adversarial" = direct conflict, legal opposition, or zero-sum competition; "cooperative" = explicit partnership, deal, or aligned action; "neutral" = informational update or parallel action without direct interaction between the named persons
- edge_signals: derived from top_stories entries that name exactly 2 persons; include only "adversarial" and "cooperative" entries (omit neutral); one entry per qualifying story; if a story names only 1 person, omit it from edge_signals
- Return only the JSON object, no preamble or explanation"""


def synthesize_briefing(
    client: anthropic.Anthropic,
    news_items: list,
    social_signals: str,
    phase_sequences: list,
    week_start: str,
    week_end: str
) -> dict:
    print("\nSynthesizing with Claude...")
    prompt = build_synthesis_prompt(
        news_items, social_signals, phase_sequences, week_start, week_end
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = message.content[0].text
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    return json.loads(response_text[json_start:json_end])


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_json(briefing: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON  → {path.name}")


def save_markdown(briefing: dict, path: Path):
    signal_icon = {"notable": "▲", "quiet": "—", "concerning": "▼"}
    momentum_icon = {"accelerating": "↑", "stable": "→", "decelerating": "↓"}

    md = f"# Weekly Prime Mover Briefing — {briefing['week_ending']}\n\n"
    md += f"## Executive Summary\n\n{briefing['executive_summary']}\n\n---\n\n"

    md += "## Top Stories\n\n"
    for story in briefing.get("top_stories", []):
        persons = ", ".join(story.get("persons", []))
        ps_tag = f" `{story['ps_id']}`" if story.get("ps_id") else ""
        valence = story.get("valence", "")
        valence_tag = f" `{valence}`" if valence and valence != "neutral" else ""
        md += f"**{story['headline']}**{ps_tag}{valence_tag}  \n"
        md += f"*{persons}* — {story['significance']}\n\n"

    md += "---\n\n## Individual Updates\n\n"
    for p in briefing.get("person_updates", []):
        icon = signal_icon.get(p["signal"], "—")
        ps_tags = " ".join(f"`{ps}`" for ps in p.get("ps_impacts", []))
        md += f"### {icon} {p['name']}\n\n"
        md += f"{p['summary']}\n\n"
        if ps_tags:
            md += f"**Phase sequences:** {ps_tags}\n\n"
        md += "---\n\n"

    md += "## Phase Sequence Pulse\n\n"
    for ps in briefing.get("phase_sequence_updates", []):
        icon = momentum_icon.get(ps["momentum"], "→")
        md += f"**{icon} {ps['id']}: {ps['name']}** `{ps['momentum']}`  \n"
        md += f"{ps['summary']}\n\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ MD    → {path.name}")


def print_console_summary(briefing: dict):
    signal_icon = {"notable": "▲", "quiet": "—", "concerning": "▼"}
    print("\n" + "=" * 60)
    print("BRIEFING SUMMARY")
    print("=" * 60)
    print(f"\n{briefing['executive_summary']}\n")
    print("Signal by person:")
    for p in briefing.get("person_updates", []):
        icon = signal_icon.get(p["signal"], "—")
        print(f"  {icon}  {p['name']}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(use_social: bool = True):
    week_end = date.today().isoformat()
    week_start = (date.today() - timedelta(days=7)).isoformat()

    BRIEFINGS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("WEEKLY BRIEFING GENERATOR (US)")
    print(f"Week: {week_start} → {week_end}")
    print("=" * 60)

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run us_superorganism_mapper.py first.")
        return

    print(f"\nLoading {MODEL_PATH.name}...")
    model = load_superorganism(MODEL_PATH)
    individuals = model["superorganism_list"]
    phase_sequences = model["canonical_vocabulary"]["phase_sequences"]
    print(f"  {len(individuals)} individuals, {len(phase_sequences)} phase sequences")

    # Stage 1A: Per-person news
    print(f"\nStage 1: Fetching news via Perplexity ({len(individuals)} calls)...")
    perplexity = get_perplexity_client()
    news_items = []
    for person in individuals:
        try:
            result = fetch_news_for_person(perplexity, person, week_start, week_end)
            news_items.append(result)
        except Exception as e:
            print(f"  ! {person['name']}: {e}")
            news_items.append({
                "name": person["name"],
                "domain": person["domain"],
                "raw_news": "No data retrieved."
            })
    print(f"  ✓ {len(news_items)} news summaries collected")

    # Stage 1B: Social signals (optional)
    social_signals = ""
    if use_social:
        print("\nStage 1B: Fetching social signals via Grok...")
        try:
            grok = get_xai_client()
            top_names = [p["name"] for p in individuals[:6]]
            social_signals = fetch_social_signals(grok, top_names, week_end)
            print("  ✓ Social signals collected")
        except Exception as e:
            print(f"  ! Grok social signals failed: {e}")

    # Stage 2: Claude synthesis
    print("\nStage 2: Synthesizing with Claude...")
    try:
        claude = get_anthropic_client()
        briefing = synthesize_briefing(
            claude, news_items, social_signals, phase_sequences, week_start, week_end
        )
    except Exception as e:
        print(f"  ! Synthesis failed: {e}")
        return

    briefing["_metadata"] = {
        "generated": date.today().isoformat(),
        "week_start": week_start,
        "week_end": week_end,
        "source_model": MODEL_PATH.name,
        "news_model": SONAR_MODEL,
        "synthesis_model": CLAUDE_MODEL,
        "social_signals_included": use_social and bool(social_signals)
    }

    # Save outputs
    print("\nSaving outputs...")
    stem = f"weekly_briefing_{week_end}"
    save_json(briefing, BRIEFINGS_DIR / f"{stem}.json")
    save_markdown(briefing, BRIEFINGS_DIR / f"{stem}.md")

    print_console_summary(briefing)
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly US prime mover briefing")
    parser.add_argument(
        "--no-social",
        action="store_true",
        help="Skip Grok social signals call (saves ~1 API call)"
    )
    args = parser.parse_args()
    run(use_social=not args.no_social)
