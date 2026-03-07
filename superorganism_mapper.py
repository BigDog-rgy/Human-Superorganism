"""
Superorganism Mapper
Reads the LLM Council stage_3_final_list and augments each individual
with Hebbian superorganism annotations using a pre-defined canonical vocabulary.

Framework:
  - Neurons       = individual prime movers
  - Cell assemblies = organizations that form around neurons
  - Phase sequences = goal-directed behaviors from linked assemblies
  - Brain regions  = 7 macro-civilizational structures mapped to neural anatomy
"""

import os
import json
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# CANONICAL VOCABULARY — defined once, referenced by all individuals
# ---------------------------------------------------------------------------

POWER_PHASE_SEQUENCES = [
    {
        "id": "PS-01",
        "name": "Global AI Supremacy",
        "hemisphere_bias": "West",
        "definition": "Race among state and corporate actors to develop, deploy, and leverage dominant AI capabilities — the primary technological competition of the era."
    },
    {
        "id": "PS-02",
        "name": "US-China Strategic Bifurcation",
        "hemisphere_bias": "Split",
        "definition": "Technological, economic, and military decoupling of the world's two largest powers into parallel global systems — trade blocs, chip stacks, payment rails, and alliances."
    },
    {
        "id": "PS-03",
        "name": "Multiplanetary Expansion",
        "hemisphere_bias": "West",
        "definition": "Establishment of permanent human presence beyond Earth, driven by private space enterprise and competing national space programs."
    },
    {
        "id": "PS-04",
        "name": "Dollar Hegemony Defense",
        "hemisphere_bias": "West",
        "definition": "Coordinated maintenance of the USD as the world's reserve currency through monetary policy, sanctions architecture, and geopolitical leverage against alternatives (BRICS currencies, digital yuan)."
    },
    {
        "id": "PS-05",
        "name": "Eurasian Resistance Arc",
        "hemisphere_bias": "East",
        "definition": "Russia-China-Iran strategic coordination to counterbalance the Western-led international order — shared interest in multipolar world, sanctions evasion, and alternative institutions."
    },
    {
        "id": "PS-06",
        "name": "Middle East Power Realignment",
        "hemisphere_bias": "East",
        "definition": "Post-Gaza restructuring of regional alliances, normalization dynamics, and Iranian containment — reshaping the security and diplomatic architecture of the Middle East."
    },
    {
        "id": "PS-07",
        "name": "Semiconductor Sovereignty Race",
        "hemisphere_bias": "Split",
        "definition": "National competition for control of advanced chip design, fabrication, and supply chain — the decisive chokepoint of the AI and defense technology era."
    },
    {
        "id": "PS-08",
        "name": "Global Energy Transition",
        "hemisphere_bias": "West",
        "definition": "Shift from fossil fuel dependence to renewable energy while managing near-term energy security — a decades-long metabolic transformation of the global economy."
    },
    {
        "id": "PS-09",
        "name": "Information & Narrative Control",
        "hemisphere_bias": "West",
        "definition": "Platform-mediated competition over global information flows, algorithmic curation, and ideological framing — who controls what billions of people see and believe."
    },
    {
        "id": "PS-10",
        "name": "Western Alliance Reformation",
        "hemisphere_bias": "West",
        "definition": "Restructuring of NATO, transatlantic economic ties, and liberal international institutions under new geopolitical pressures — redefining who bears costs and who calls shots."
    },
]

HEMISPHERE_MAP = {
    "West": {
        "regions": ["US", "Europe"],
        "character": "Individualist, rule-of-law, market-driven, liberal institutions"
    },
    "East": {
        "regions": ["China", "Russia"],
        "character": "Collective, state-centric, long-horizon, hierarchical — Russia firmly East-aligned post-2022"
    },
    "Bridge": {
        "regions": ["India", "Middle East"],
        "character": "Non-aligned, leverage-based swing actors who maintain optionality between East and West systems"
    },
    "Ancestral": {
        "regions": ["Africa"],
        "character": "Oldest civilizational substrate — foundational patterns; not yet prime-mover-generating in the current model snapshot"
    },
}

# Valid primary regions (geographic, not hemispherical)
VALID_REGIONS = ["US", "Europe", "China", "Russia", "India", "Middle East", "Africa"]

CANONICAL_CELL_ASSEMBLIES = [
    # Xi Jinping
    "CCP (Chinese Communist Party)",
    "PLA (People's Liberation Army)",
    "Belt and Road Initiative (BRI)",
    "CIPS (Cross-Border Interbank Payment System)",
    # Trump
    "Trump Administration",
    "MAGA Movement",
    "Republican Party",
    # Musk
    "SpaceX",
    "Tesla",
    "xAI",
    "X (Twitter)",
    "DOGE (Department of Government Efficiency)",
    # Putin
    "Russian Federation Executive",
    "FSB",
    "Gazprom / Rosneft",
    "CSTO (Collective Security Treaty Organization)",
    # Modi
    "BJP (Bharatiya Janata Party)",
    "Digital India Initiative",
    "Make in India",
    "RSS (Rashtriya Swayamsevak Sangh)",
    # Jensen Huang
    "NVIDIA",
    "CUDA Ecosystem",
    # Powell
    "Federal Reserve System",
    "FOMC (Federal Open Market Committee)",
    # MBS
    "Saudi Aramco",
    "OPEC+",
    "Public Investment Fund (PIF)",
    "Vision 2030",
    # Altman
    "OpenAI",
    "Worldcoin",
    # Nadella
    "Microsoft",
    "Azure Cloud",
    "GitHub",
    # Von der Leyen
    "European Commission",
    "EU Regulatory Apparatus",
    # Zuckerberg
    "Meta",
    "Facebook",
    "Instagram",
    "WhatsApp",
    # Netanyahu
    "IDF (Israel Defense Forces)",
    "Mossad",
    "Likud",
    # Khamenei
    "IRGC (Islamic Revolutionary Guard Corps)",
    "Axis of Resistance (Hezbollah / Hamas / Houthis)",
    # Cook
    "Apple",
    "App Store Ecosystem",
    # Fink
    "BlackRock",
    "Aladdin Platform",
    # Lagarde
    "European Central Bank (ECB)",
    "Eurozone Banking System",
    # Shared / Multi-person
    "NATO",
    "G7 / G20",
    "UN Security Council",
    "WTO (World Trade Organization)",
    "IMF (International Monetary Fund)",
    "TSMC (Taiwan Semiconductor Manufacturing Company)",
]


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def _load_ps_from_canon(path: Path):
    """Load phase sequences from ps_canon file. Returns None if file missing or has no sequences."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ps = data.get("phase_sequences")
    if ps:
        print(f"  Loaded PS from {path.name} ({len(ps)} sequences)")
    return ps or None


def build_mapping_prompt(individuals: list, phase_sequences: list = None) -> str:
    if phase_sequences is None:
        phase_sequences = POWER_PHASE_SEQUENCES
    ps_block = "\n".join(
        f"  {ps['id']}: {ps['name']} [{ps.get('hemisphere_bias', '')}] — {ps['definition']}"
        for ps in phase_sequences
    )

    hemisphere_block = "\n".join(
        f"  {hemi} ({', '.join(data['regions'])}): {data['character']}"
        for hemi, data in HEMISPHERE_MAP.items()
    )

    assemblies_block = "\n".join(f"  - {a}" for a in CANONICAL_CELL_ASSEMBLIES)

    individuals_block = "\n".join(
        f"  {p['rank']}. {p['name']} | {p.get('title', p.get('domain', ''))}"
        for p in individuals
    )

    schema_example = json.dumps({
        "name": "Elon Musk",
        "hemisphere": "West",
        "primary_region": "US",
        "secondary_regions": ["Europe", "China"],
        "neuron_role": "Motor initiation — fires novel, high-amplitude signals that reorganize large assemblies across the organism",
        "neuron_type": "Burst-firing excitatory neuron — intermittent but high-amplitude influence across multiple assemblies simultaneously",
        "cell_assemblies": [
            {"name": "SpaceX", "role": "Primary driver of PS-03"},
            {"name": "xAI", "role": "Competitor in PS-01"},
            {"name": "DOGE (Department of Government Efficiency)", "role": "Metabolic regulator of US executive function"},
            {"name": "X (Twitter)", "role": "Primary platform for PS-09"}
        ],
        "phase_sequences": [
            {"id": "PS-03", "name": "Multiplanetary Expansion", "role": "Primary architect and sole industrial executor"},
            {"id": "PS-01", "name": "Global AI Supremacy", "role": "Competing actor via xAI; also controls critical AI infrastructure (Starlink, compute)"},
            {"id": "PS-09", "name": "Information & Narrative Control", "role": "Platform owner with direct global reach through X"}
        ]
    }, indent=2)

    return f"""You are annotating 17 globally influential individuals using a Hebbian superorganism framework.

## FRAMEWORK

In this model:
- **Neurons** = individual prime movers (the 17 people below)
- **Cell assemblies** = organizations that form around neurons through repeated co-activation
- **Phase sequences** = goal-directed behaviors that emerge when linked cell assemblies fire in succession
- **Hemispheres** = 4 macro-civilizational blocs (West, East, Bridge, Ancestral)

## CANONICAL PHASE SEQUENCES (use ONLY these — do not invent new ones)

{ps_block}

## HEMISPHERE MAP (assign each individual to one of these 4 hemispheres)

{hemisphere_block}

Valid values for primary_region: US, Europe, China, Russia, India, Middle East, Africa

## CANONICAL CELL ASSEMBLIES (use ONLY names from this list — exact spelling)

{assemblies_block}

## INDIVIDUALS TO ANNOTATE

{individuals_block}

## YOUR TASK

For each of the 17 individuals, produce a superorganism annotation object. Rules:
1. `hemisphere`: must be "West", "East", "Bridge", or "Ancestral"
2. `primary_region`: must be one of the 7 valid regions listed above
3. `secondary_regions`: list of other regions this person meaningfully influences (can be empty)
4. `neuron_role`: one sentence describing how this neuron functions within the global superorganism
5. `neuron_type`: one sentence describing their firing pattern and influence style
6. `cell_assemblies`: list of objects with "name" (exact from canonical list) and "role" (brief, reference PS IDs where applicable)
7. `phase_sequences`: list of objects with "id", "name", and "role" — only include phase sequences this person materially participates in
8. Do NOT invent phase sequences or cell assemblies outside the canonical lists

## OUTPUT FORMAT

Return a JSON array of exactly 17 objects. Each object has this structure:
{{
  "name": "Full Name",
  "hemisphere": "...",
  "primary_region": "...",
  "secondary_regions": [...],
  "neuron_role": "...",
  "neuron_type": "...",
  "cell_assemblies": [
    {{"name": "...", "role": "..."}}
  ],
  "phase_sequences": [
    {{"id": "PS-XX", "name": "...", "role": "..."}}
  ]
}}

## EXAMPLE (Elon Musk)

{schema_example}

Now annotate all 17 individuals. Return only the JSON array, no preamble.
"""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def load_individuals(results_path: str) -> list:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["stage_3_final_list"]


def call_claude(prompt: str) -> list:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("Calling Claude Opus 4.6 for superorganism mapping...")

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text
    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1
    return json.loads(response_text[json_start:json_end])


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
            print(f"  + {person['name']}: {superorganism['primary_region']} | {superorganism['hemisphere']}")
        else:
            print(f"  ! Warning: no annotation found for {person['name']}")
        merged.append(entry)
    return merged


def save_output(merged: list, output_path: str, phase_sequences: list = None):
    output = {
        "metadata": {
            "date": str(date.today()),
            "source": "final_ranked_global.json",
            "annotated_by": "claude-opus-4-6",
            "methodology": (
                "Hebbian superorganism annotation: canonical phase sequences and cell assemblies "
                "defined a priori; individuals mapped by Claude Opus 4.6 at temperature=0.3"
            )
        },
        "canonical_vocabulary": {
            "phase_sequences": phase_sequences if phase_sequences is not None else POWER_PHASE_SEQUENCES,
            "hemispheres": HEMISPHERE_MAP,
            "cell_assemblies": CANONICAL_CELL_ASSEMBLIES
        },
        "superorganism_list": merged
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "final_ranked_global.json")
    output_path = os.path.join(script_dir, "superorganism_model.json")

    print("=" * 60)
    print("SUPERORGANISM MAPPER")
    print("=" * 60)

    # Load phase sequences from ps_canon if available, else use hardcoded fallback
    ps_canon_path = Path(script_dir) / "ps_canon_global.json"
    phase_sequences = _load_ps_from_canon(ps_canon_path)
    if phase_sequences is None:
        print("  Using hardcoded POWER_PHASE_SEQUENCES (ps_canon_global.json not found)")
        phase_sequences = POWER_PHASE_SEQUENCES

    print("\nLoading individuals from final_ranked_global.json...")
    individuals = load_individuals(input_path)
    print(f"Loaded {len(individuals)} individuals")

    prompt = build_mapping_prompt(individuals, phase_sequences)

    try:
        annotations = call_claude(prompt)
        print(f"Received {len(annotations)} annotations from Claude\n")
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return

    print("Merging annotations...")
    merged = merge_annotations(individuals, annotations)

    save_output(merged, output_path, phase_sequences)

    print("\n" + "=" * 60)
    print("Superorganism mapping complete!")
    print("=" * 60)


if __name__ == "__main__":
    run()
