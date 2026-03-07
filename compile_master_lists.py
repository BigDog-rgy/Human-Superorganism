"""
Compile Master Prime Mover Lists
Aggregates all candidate pools into a US list and a Global list.
Each entry: name + title + source (used by tournament_filter.py for domain-aware pairing)

Sources and their tournament_filter.py treatment:
  council        — LLM council output; gets a bye in round 1 (already curated)
  forbes_400     — US billionaires; round 1 paired within source only
  congress       — US legislators; round 1 paired within source only
  forbes_global  — Global billionaires; round 1 paired within source only
  world_leaders  — Heads of state; round 1 paired within source only

US list sources:
  - forbes_400_names.json           (Forbes 400 US billionaires)
  - congress_candidates.json        (US senators and representatives)
  - other_candidates_us.json        (stage 3 final lists, all categories)

Global list sources:
  - forbes_billionaires_names.json  (Forbes global billionaires)
  - world_leaders_candidates.json   (heads of state, 193 countries)
  - other_candidates_global.json    (stage 3 final lists, all categories)

Outputs: master_list_us.json / master_list_global.json
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent


def load_json(filename: str) -> dict | list:
    with open(BASE_DIR / filename, encoding='utf-8') as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# US list                                                             #
# ------------------------------------------------------------------ #

def compile_us_list() -> list[dict]:
    entries = []

    # Forbes 400 — names only, no title in source
    for name in load_json('forbes_400_names.json'):
        entries.append({'name': name, 'title': 'Billionaire (Forbes 400)', 'source': 'forbes_400'})

    # US Congress — chamber, state, party available
    congress = load_json('congress_candidates.json')
    for c in congress['candidates']:
        chamber = c.get('chamber', '')
        state   = c.get('state', '')
        party   = c.get('party', '')
        title   = f"{chamber} ({state}, {party})"
        entries.append({'name': c['name'], 'title': title, 'source': 'congress'})

    # Other candidates US — use role from stage_3_final_list; gets bye in round 1
    other_us = load_json('other_candidates_us.json')
    us_categories = [
        'academics_intellectuals',
        'religious_leaders',
        'media_cultural',
        'ngo_foundation',
        'political_operators',
        'officials_judiciary',
    ]
    for cat in us_categories:
        if cat not in other_us:
            continue
        for person in other_us[cat].get('stage_3_final_list', []):
            entries.append({
                'name':   person.get('name', ''),
                'title':  person.get('role', ''),
                'source': 'council',
            })

    return entries


# ------------------------------------------------------------------ #
# Global list                                                         #
# ------------------------------------------------------------------ #

def compile_global_list() -> list[dict]:
    entries = []

    # Forbes global billionaires — names only, no title in source
    for name in load_json('forbes_billionaires_names.json'):
        entries.append({'name': name, 'title': 'Billionaire (Forbes Global)', 'source': 'forbes_global'})

    # World leaders — country and subcategory available
    world_leaders = load_json('world_leaders_candidates.json')
    for c in world_leaders['candidates']:
        country = c.get('country', '')
        subcat  = c.get('subcategory', 'head_of_state').replace('_', ' ').title()
        title   = f"{subcat}, {country}"
        entries.append({'name': c['name'], 'title': title, 'source': 'world_leaders'})

    # Other candidates global — use role from stage_3_final_list; gets bye in round 1
    other_global = load_json('other_candidates_global.json')
    global_categories = [
        'academics_intellectuals',
        'religious_leaders',
        'media_cultural',
        'ngo_foundation',
        'global_operators',
        'global_officials',
    ]
    for cat in global_categories:
        if cat not in other_global:
            continue
        for person in other_global[cat].get('stage_3_final_list', []):
            entries.append({
                'name':   person.get('name', ''),
                'title':  person.get('role', ''),
                'source': 'council',
            })

    return entries


# ------------------------------------------------------------------ #
# Output                                                              #
# ------------------------------------------------------------------ #

def save_list(entries: list[dict], filename_base: str) -> None:
    json_path = BASE_DIR / f'{filename_base}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {len(entries)} entries → {json_path.name}")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    print("Compiling US master list...")
    us_list = compile_us_list()
    save_list(us_list, 'master_list_us')

    print("\nCompiling Global master list...")
    global_list = compile_global_list()
    save_list(global_list, 'master_list_global')

    print("\nDone.")
