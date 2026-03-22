"""
Superorganism Assembler

Reads final_ranked_{scope}.json + ps_canon_v2_{scope}.json + ca_canon_v2_{scope}.json
and assembles the superorganism model consumed by combined_viz.py,
hebbian_updater.py, and weekly_briefing.py.

No API calls — pure data assembly.
  - Neuron → CA memberships come from ca_canon_v2 (neuron_ca_map)
  - CA → PS memberships come from ps_canon_v2 (ca_ps_map)
  - Neuron → PS memberships are derived by chaining neuron → CAs → PSes

Usage:
    python superorganism_assembler.py --scope us
    python superorganism_assembler.py --scope global
"""

import json
import argparse
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SCOPE_CONFIG = {
    "us": {
        "label": "US",
        "input_file": "final_ranked_us.json",
        "ps_canon_file": "ps_canon_v2_us.json",
        "ca_canon_file": "ca_canon_v2_us.json",
        "output_file": "us_superorganism_model.json",
        "default_hemisphere": "West",
        "top_n": 150,   # Only top-ranked neurons enter the model; full list kept in input file
        # Sectors carried through to canonical_vocabulary for combined_viz.py
        "sectors": {
            "Federal Executive": {
                "character": (
                    "Direct executive branch authority — White House, cabinet, "
                    "executive agencies, and executive order power"
                )
            },
            "Technology": {
                "character": (
                    "Private sector tech company leadership — product, platform, "
                    "and compute control over the digital economy"
                )
            },
            "AI / ML": {
                "character": (
                    "Frontier AI research and deployment — models, infrastructure, "
                    "safety governance, and the AGI race"
                )
            },
            "Finance / Capital Markets": {
                "character": (
                    "Capital allocation, monetary policy, asset management, and "
                    "market-making — control over the financial nervous system"
                )
            },
            "Defense / Security": {
                "character": (
                    "Military procurement, intelligence adjacency, and national "
                    "security apparatus — hardware and software of US power"
                )
            },
            "Media / Narrative": {
                "character": (
                    "Platform ownership, editorial control, and narrative reach "
                    "over US public opinion and political discourse"
                )
            },
            "Energy": {
                "character": (
                    "Fossil fuel production, utility control, and energy regulatory "
                    "influence — the metabolic base of the US economy"
                )
            },
        },
    },
    "global": {
        "label": "Global",
        "input_file": "final_ranked_global.json",
        "ps_canon_file": "ps_canon_v2_global.json",
        "ca_canon_file": "ca_canon_v2_global.json",
        "output_file": "superorganism_model.json",
        "default_hemisphere": "West",   # fallback; hemisphere coloring not yet in ranked list
        "top_n": 300,    # Match ps_council_v2 anchor set size
        "sectors": {},
    },
}


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_prime_movers(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both direct array (new format) and legacy stage_3_final_list wrapper
    return data if isinstance(data, list) else data["stage_3_final_list"]


def load_ps_canon(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ca_canon(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ASSEMBLY
# ---------------------------------------------------------------------------

def derive_cell_assemblies(name: str, memberships: dict, ca_lookup: dict) -> list:
    """
    Build the superorganism.cell_assemblies list for a neuron.
    combined_viz.py tooltip accesses a['name'] and a['role'] on each entry.
    """
    result = []
    for ca_id in memberships.get(name, []):
        ca = ca_lookup.get(ca_id, {})
        result.append({
            "id": ca_id,
            "name": ca.get("name", ca_id),
            "role": ca.get("description", ""),   # tooltip key — must be present
            "ps_memberships": ca.get("ps_memberships", []),
        })
    return result


def derive_phase_sequences(
    neuron_ca_ids: list,
    ca_ps_map: dict,
    ps_lookup: dict,
) -> list:
    """
    Derive a neuron's PS memberships by chaining neuron → CAs → PSes.
    neuron_ca_ids: [ca_id, ...] from neuron_ca_map
    ca_ps_map:     {ca_id: [ps_id, ...]} from ps_canon_v2
    ps_lookup:     {ps_id: {name, definition, ...}}

    Returns deduplicated list of {"id", "name", "role"} in PS id order.
    `role` must be present — tooltip renderer (combined_viz.py) accesses it.
    """
    seen: set = set()
    result = []
    for ca_id in neuron_ca_ids:
        for ps_id in ca_ps_map.get(ca_id, []):
            if ps_id in seen:
                continue
            seen.add(ps_id)
            ps_meta = ps_lookup.get(ps_id, {})
            result.append({
                "id":   ps_id,
                "name": ps_meta.get("name", ps_id),
                "role": ps_meta.get("definition", ""),
            })
    result.sort(key=lambda x: x["id"])
    return result


def assemble(scope: str, top_n_override: int = None):
    cfg = SCOPE_CONFIG[scope]
    input_path  = SCRIPT_DIR / cfg["input_file"]
    canon_path  = SCRIPT_DIR / cfg["ps_canon_file"]
    ca_path     = SCRIPT_DIR / cfg["ca_canon_file"]
    output_path = SCRIPT_DIR / cfg["output_file"]

    print("=" * 60)
    print(f"SUPERORGANISM ASSEMBLER  [{cfg['label']}]")
    print("=" * 60)

    if not input_path.exists():
        print(f"ERROR: {input_path.name} not found.")
        return
    if not canon_path.exists():
        print(f"ERROR: {canon_path.name} not found.")
        print(f"  Run: python ps_council_v2.py --scope {scope}")
        return

    print(f"\nLoading prime movers from {input_path.name}...")
    people = load_prime_movers(input_path)
    total = len(people)
    top_n = cfg.get("top_n") or top_n_override
    if top_n and len(people) > top_n:
        people = people[:top_n]
        print(f"  {total} individuals loaded — capped to top {top_n}")
    else:
        print(f"  {total} individuals loaded")

    print(f"Loading phase sequences from {canon_path.name}...")
    ps_canon        = load_ps_canon(canon_path)
    phase_sequences = ps_canon.get("phase_sequences", [])
    ca_ps_map       = ps_canon.get("ca_ps_map", {})
    print(f"  {len(phase_sequences)} phase sequences")
    print(f"  CA-PS links for {len(ca_ps_map)} assemblies")

    if not phase_sequences:
        print("ERROR: ps_canon has no phase_sequences.")
        return

    # id → full PS metadata (name, definition, ...)
    ps_lookup = {ps["id"]: ps for ps in phase_sequences}

    print(f"Loading cell assemblies from {ca_path.name}...")
    ca_canon = load_ca_canon(ca_path)
    if ca_canon:
        cell_assemblies = ca_canon.get("cell_assemblies", [])
        ca_memberships  = ca_canon.get("neuron_ca_map", ca_canon.get("neuron_assembly_memberships", {}))
        ca_lookup       = {ca["id"]: ca for ca in cell_assemblies}
        # Inject ps_memberships onto each CA from ca_ps_map
        for ca_id, ca in ca_lookup.items():
            ca["ps_memberships"] = ca_ps_map.get(ca_id, [])
        assigned_cas = sum(1 for ca_id in ca_lookup if ca_ps_map.get(ca_id))
        print(f"  {len(cell_assemblies)} cell assemblies, {len(ca_memberships)} neurons assigned")
        print(f"  {assigned_cas}/{len(cell_assemblies)} CAs have PS memberships")
    else:
        print(f"  Not found — cell_assemblies will be empty (run ca_council_v2.py --scope {scope})")
        cell_assemblies = []
        ca_memberships  = {}
        ca_lookup       = {}

    print(f"\nAssembling...")
    unmatched = []
    superorganism_list = []

    for person in people:
        name       = person["name"]
        ca_ids     = ca_memberships.get(name, [])

        if not ca_ids:
            unmatched.append(name)

        ps_list = derive_phase_sequences(ca_ids, ca_ps_map, ps_lookup)
        ca_list = derive_cell_assemblies(name, ca_memberships, ca_lookup)

        superorganism = {
            "hemisphere": cfg["default_hemisphere"],
            "primary_sector": "",
            "secondary_sectors": [],
            "neuron_role": "",
            "neuron_type": "",
            "cell_assemblies": ca_list,
            "phase_sequences": ps_list,
        }

        entry = dict(person)
        entry["superorganism"] = superorganism
        superorganism_list.append(entry)
        print(f"  + {name}: {len(ps_list)} PS, {len(ca_list)} assemblies")

    if unmatched:
        print(f"\n  ! No CA memberships found for {len(unmatched)} individuals:")
        for n in unmatched:
            print(f"      {n}")
        print("    Re-run ca_council_v2.py to generate CA memberships for all neurons.")

    # Build output — structure must match what combined_viz.py + hebbian_updater.py expect
    output = {
        "metadata": {
            "date": str(date.today()),
            "scope": scope,
            "source": cfg["input_file"],
            "ps_canon": cfg["ps_canon_file"],
            "ca_canon": cfg["ca_canon_file"] if ca_canon else None,
            "focus": f"{cfg['label']} prime movers",
            "assembled_by": "superorganism_assembler.py",
            "top_n": top_n,
            "methodology": (
                "V2 assembly: cell_assemblies from ca_canon_v2 neuron_ca_map; "
                "phase_sequences derived by chaining neuron → CA → PS via ca_ps_map. "
                "No LLM calls."
            ),
        },
        "canonical_vocabulary": {
            "phase_sequences": phase_sequences,
            "sectors": cfg.get("sectors", {}),
            "cell_assemblies": cell_assemblies,
        },
        "superorganism_list": superorganism_list,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {output_path.name}")
    print(f"\n{'=' * 60}")
    print(f"{cfg['label']} superorganism assembly complete!")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble superorganism model from ps_canon_v2 and ca_canon_v2"
    )
    parser.add_argument(
        "--scope", choices=["us", "global"], default="us",
        help="Which model to assemble: 'us' (default) or 'global'"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Override the per-scope top_n cap (e.g. --top-n 200)"
    )
    args = parser.parse_args()
    assemble(args.scope, top_n_override=args.top_n)


if __name__ == "__main__":
    main()
