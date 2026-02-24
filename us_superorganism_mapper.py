"""
US Superorganism Mapper
Reads the US LLM Council stage_3_final_list and augments each individual
with Hebbian superorganism annotations using a US-specific canonical vocabulary.

Framework:
  - Neurons       = individual US prime movers
  - Cell assemblies = organizations that form around neurons
  - Phase sequences = goal-directed behaviors from linked assemblies (DPS-xx prefix)
  - Sectors        = 7 US institutional domains replacing global hemispheres
"""

import os
import json
from datetime import date
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# CANONICAL VOCABULARY — US-specific, defined once
# ---------------------------------------------------------------------------

US_POWER_PHASE_SEQUENCES = [
    {
        "id": "DPS-01",
        "name": "MAGA Realignment",
        "definition": "The structural reorientation of the Republican Party and federal government under Trump's 2025-2026 executive agenda — deregulation, tariff-based industrial policy, executive power consolidation, and dismantling of legacy administrative state norms."
    },
    {
        "id": "DPS-02",
        "name": "AI & Compute Dominance",
        "definition": "The race among US private actors and government to lead in frontier AI model development, chip design, and data center infrastructure — determining American technological supremacy for the next decade."
    },
    {
        "id": "DPS-03",
        "name": "Financial System Control",
        "definition": "Competition and coordination among the Federal Reserve, Treasury, Wall Street asset managers, and cryptocurrency actors over interest rates, capital allocation, and the architecture of US and global financial markets."
    },
    {
        "id": "DPS-04",
        "name": "Platform & Narrative Hegemony",
        "definition": "Dominance over the US information environment via social media platforms, algorithmic curation, podcast networks, and cable news — who shapes what 330M Americans see, believe, and vote for."
    },
    {
        "id": "DPS-05",
        "name": "Defense & Industrial Mobilization",
        "definition": "The shift toward great-power-competition defense spending, reshoring of critical manufacturing, and integration of private defense contractors into national security priorities — the US war economy restructuring."
    },
    {
        "id": "DPS-06",
        "name": "Energy Dominance",
        "definition": "The Trump administration's strategy of maximizing US fossil fuel production, rolling back climate mandates, and leveraging American LNG and oil exports as geopolitical tools."
    },
    {
        "id": "DPS-07",
        "name": "Judicial & Constitutional Architecture",
        "definition": "The long-run reshaping of US law through Supreme Court appointments, federal judiciary transformation, and litigation-based policy battles — determining the constitutional boundaries of executive power, regulatory authority, and civil rights."
    },
    {
        "id": "DPS-08",
        "name": "Capital & Wealth Concentration",
        "definition": "The acceleration of wealth consolidation among the top 0.1% through tax policy, financial deregulation, and technology-driven productivity gains — reshaping US class structure, political influence, and consumer market dynamics."
    },
]

US_SECTOR_MAP = {
    "Federal Executive": {
        "character": "Direct executive branch authority — White House, cabinet, executive agencies, and executive order power"
    },
    "Technology": {
        "character": "Private sector tech company leadership — product, platform, and compute control over the digital economy"
    },
    "AI / ML": {
        "character": "Frontier AI research and deployment — models, infrastructure, safety governance, and the AGI race"
    },
    "Finance / Capital Markets": {
        "character": "Capital allocation, monetary policy, asset management, and market-making — control over the financial nervous system"
    },
    "Defense / Security": {
        "character": "Military procurement, intelligence adjacency, and national security apparatus — hardware and software of US power"
    },
    "Media / Narrative": {
        "character": "Platform ownership, editorial control, and narrative reach over US public opinion and political discourse"
    },
    "Energy": {
        "character": "Fossil fuel production, utility control, and energy regulatory influence — the metabolic base of the US economy"
    },
}

# Valid primary sectors
VALID_SECTORS = [
    "Federal Executive", "Technology", "AI / ML",
    "Finance / Capital Markets", "Defense / Security",
    "Media / Narrative", "Energy"
]

US_CANONICAL_CELL_ASSEMBLIES = [
    # Political / Executive
    "Trump Administration",
    "MAGA Movement",
    "Republican Party",
    "Democratic Party",
    "DOGE (Department of Government Efficiency)",
    "White House Office",
    "US Senate",
    "US House of Representatives",
    "US Department of Treasury",
    "US Department of Defense",
    "US Department of Justice",
    # Technology (overlaps with global — exact spelling preserved)
    "OpenAI",
    "Microsoft",
    "Azure Cloud",
    "GitHub",
    "Meta",
    "Facebook",
    "Instagram",
    "Apple",
    "App Store Ecosystem",
    "NVIDIA",
    "CUDA Ecosystem",
    "SpaceX",
    "Tesla",
    "xAI",
    "X (Twitter)",
    "Google / Alphabet",
    "Google DeepMind",
    # Finance (overlaps with global — exact spelling preserved)
    "Federal Reserve System",
    "FOMC (Federal Open Market Committee)",
    "BlackRock",
    "Aladdin Platform",
    # Finance (US-specific)
    "Goldman Sachs",
    "JP Morgan",
    "US Treasury / Bond Market",
    # Media (US-specific)
    "Fox News",
    "Podcast Industrial Complex",
    # Defense (US-specific)
    "Palantir",
    "Anduril",
]


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_us_mapping_prompt(individuals: list) -> str:
    ps_block = "\n".join(
        f"  {ps['id']}: {ps['name']} — {ps['definition']}"
        for ps in US_POWER_PHASE_SEQUENCES
    )

    sector_block = "\n".join(
        f"  {sector}: {data['character']}"
        for sector, data in US_SECTOR_MAP.items()
    )

    assemblies_block = "\n".join(f"  - {a}" for a in US_CANONICAL_CELL_ASSEMBLIES)

    individuals_block = "\n".join(
        f"  {p['rank']}. {p['name']} | Domain: {p['domain']} | {p['justification']}"
        for p in individuals
    )

    schema_example = json.dumps({
        "name": "Elon Musk",
        "hemisphere": "West",
        "primary_sector": "Technology",
        "secondary_sectors": ["AI / ML", "Federal Executive", "Media / Narrative"],
        "neuron_role": "Hypomanic burst neuron — fires across multiple US assemblies simultaneously with extraordinary amplitude, reorganizing executive, tech, and narrative networks in real time",
        "neuron_type": "Burst-firing excitatory neuron — intermittent but extremely high-amplitude influence across Federal Executive, Technology, and Media sectors simultaneously",
        "cell_assemblies": [
            {"name": "SpaceX", "role": "Primary industrial executor of US aerospace dominance"},
            {"name": "xAI", "role": "Competing actor in DPS-02 AI & Compute Dominance"},
            {"name": "DOGE (Department of Government Efficiency)", "role": "Primary vehicle for DPS-01 MAGA Realignment — metabolic restructuring of federal bureaucracy"},
            {"name": "X (Twitter)", "role": "Primary platform for DPS-04 Platform & Narrative Hegemony"}
        ],
        "phase_sequences": [
            {"id": "DPS-01", "name": "MAGA Realignment", "role": "Co-architect via DOGE — dismantling administrative state from inside executive branch"},
            {"id": "DPS-02", "name": "AI & Compute Dominance", "role": "Competing actor via xAI; controls critical AI infrastructure (Starlink compute, data)"},
            {"id": "DPS-04", "name": "Platform & Narrative Hegemony", "role": "Platform owner with direct reach to 100M+ US users via X"}
        ]
    }, indent=2)

    return f"""You are annotating 12 domestically influential Americans using a Hebbian superorganism framework.

## FRAMEWORK

In this model:
- **Neurons** = individual US prime movers (the 12 people below)
- **Cell assemblies** = organizations that form around neurons through repeated co-activation
- **Phase sequences** = goal-directed behaviors that emerge when linked cell assemblies fire in succession
- **Sectors** = 7 US institutional domains (replacing the global hemisphere model)

## CANONICAL PHASE SEQUENCES — US DOMESTIC (use ONLY these — do not invent new ones)

{ps_block}

## US SECTOR MAP (assign each individual to one primary sector)

{sector_block}

Valid values for primary_sector: {', '.join(VALID_SECTORS)}

IMPORTANT: Because all 12 individuals are Americans, set hemisphere = "West" for all of them.

## CANONICAL CELL ASSEMBLIES (use ONLY names from this list — exact spelling)

{assemblies_block}

## INDIVIDUALS TO ANNOTATE

{individuals_block}

## YOUR TASK

For each of the 12 individuals, produce a superorganism annotation object. Rules:
1. `hemisphere`: ALWAYS "West" (all are Americans)
2. `primary_sector`: must be one of the 7 valid sectors listed above
3. `secondary_sectors`: list of other sectors this person meaningfully influences (can be empty)
4. `neuron_role`: one sentence describing how this neuron functions within the US superorganism
5. `neuron_type`: one sentence describing their firing pattern and influence style
6. `cell_assemblies`: list of objects with "name" (exact from canonical list) and "role" (brief, reference DPS IDs where applicable)
7. `phase_sequences`: list of objects with "id", "name", and "role" — only include DPS sequences this person materially participates in
8. Do NOT invent phase sequences (use only DPS-01 through DPS-08) or cell assemblies outside the canonical list

## OUTPUT FORMAT

Return a JSON array of exactly 12 objects. Each object has this structure:
{{
  "name": "Full Name",
  "hemisphere": "West",
  "primary_sector": "...",
  "secondary_sectors": [...],
  "neuron_role": "...",
  "neuron_type": "...",
  "cell_assemblies": [
    {{"name": "...", "role": "..."}}
  ],
  "phase_sequences": [
    {{"id": "DPS-XX", "name": "...", "role": "..."}}
  ]
}}

## EXAMPLE (Elon Musk)

{schema_example}

Now annotate all 12 individuals. Return only the JSON array, no preamble.
"""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def load_individuals(results_path: str) -> list:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stage_3_final_list"]


def call_claude(prompt: str) -> list:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Calling Claude Opus 4.6 for US superorganism mapping...")

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text
    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1
    raw_annotations = json.loads(response_text[json_start:json_end])

    # Validate: filter out any phase sequences not in the canonical DPS list
    valid_dps_ids = {ps["id"] for ps in US_POWER_PHASE_SEQUENCES}
    for annotation in raw_annotations:
        original_ps = annotation.get("phase_sequences", [])
        filtered_ps = [ps for ps in original_ps if ps.get("id") in valid_dps_ids]
        if len(filtered_ps) < len(original_ps):
            dropped = [ps["id"] for ps in original_ps if ps.get("id") not in valid_dps_ids]
            print(f"  ! Warning: dropped non-canonical phase sequences for {annotation.get('name', '?')}: {dropped}")
        annotation["phase_sequences"] = filtered_ps

    return raw_annotations


def merge_annotations(individuals: list, annotations: list) -> list:
    annotation_map = {a["name"]: a for a in annotations}
    merged = []
    for person in individuals:
        annotation = annotation_map.get(person["name"])
        if annotation is None:
            # Fuzzy fallback: try substring match
            for ann_name, ann in annotation_map.items():
                if person["name"].lower() in ann_name.lower() or ann_name.lower() in person["name"].lower():
                    annotation = ann
                    break
        entry = dict(person)
        if annotation:
            superorganism = {k: v for k, v in annotation.items() if k != "name"}
            entry["superorganism"] = superorganism
            print(f"  + {person['name']}: {superorganism['primary_sector']} | {superorganism['hemisphere']}")
        else:
            print(f"  ! Warning: no annotation found for {person['name']}")
        merged.append(entry)
    return merged


def save_output(merged: list, output_path: str):
    output = {
        "metadata": {
            "date": str(date.today()),
            "source": "us_llm_council_results.json",
            "focus": "US domestic prime movers",
            "annotated_by": "claude-opus-4-6",
            "methodology": (
                "Hebbian superorganism annotation with US-specific DPS phase sequences: "
                "canonical phase sequences (DPS-01 to DPS-08) and cell assemblies "
                "defined a priori; individuals mapped by Claude Opus 4.6 at temperature=0.3"
            )
        },
        "canonical_vocabulary": {
            "phase_sequences": US_POWER_PHASE_SEQUENCES,
            "sectors": US_SECTOR_MAP,
            "cell_assemblies": US_CANONICAL_CELL_ASSEMBLIES
        },
        "superorganism_list": merged
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "us_llm_council_results.json")
    output_path = os.path.join(script_dir, "us_superorganism_model.json")

    print("=" * 60)
    print("US SUPERORGANISM MAPPER")
    print("=" * 60)

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run us_llm_council.py first.")
        return

    print("\nLoading individuals from stage_3_final_list...")
    individuals = load_individuals(input_path)
    print(f"Loaded {len(individuals)} individuals")

    prompt = build_us_mapping_prompt(individuals)

    try:
        annotations = call_claude(prompt)
        print(f"Received {len(annotations)} annotations from Claude\n")
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return

    print("Merging annotations...")
    merged = merge_annotations(individuals, annotations)

    save_output(merged, output_path)

    print("\n" + "=" * 60)
    print("US superorganism mapping complete!")
    print("=" * 60)


if __name__ == "__main__":
    run()
