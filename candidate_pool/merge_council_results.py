"""
Merge Tech Executives Council Results into Final Rankings

Reads tech_executives_results.json, deduplicates against the existing
final_ranked_*.json lists, then locates each new candidate's insertion
rank using binary search (bisection) through the existing ranked list.

Calibration method — binary search:
  - Each new candidate plays duels against ranked entries using bisection.
  - Duel 1 is at the midpoint of the full list. A win narrows the search
    to the upper (better) half; a loss narrows to the lower (worse) half.
  - This repeats until the range collapses to a single insertion point.
  - ~log2(N) duels needed: ≈9 for US (502 entries), ≈11 for Global (1289).
  - Maximum 20 duels per candidate (hard cap).
  - Anchor ELO scores are NEVER updated — they are established benchmarks.
  - Calibration ELO is tracked for information only and NOT used for
    insertion (avoids the scale mismatch between the closed main-tournament
    ELO system and the open calibration system).

After calibration, new candidates are inserted at their bisected rank,
all ranks are renumbered, and filtered_list_*.json is updated for record-keeping.

Outputs:
  final_ranked_us.json     (overwritten in-place)
  final_ranked_global.json (overwritten in-place)
  filtered_list_us.json    (new candidates appended)
  filtered_list_global.json (new candidates appended)

Backups already created as *.bak.json before running this script.

Usage:
  python merge_council_results.py          # both us and global
  python merge_council_results.py us       # US only
  python merge_council_results.py global   # global only
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

BASE_DIR    = Path(__file__).parent
MODEL       = 'claude-sonnet-4-6'
K_FACTOR    = 32       # match main tournament K — used only for ELO tracking, not insertion
INITIAL_ELO = 1000.0
DELAY       = 0.25
MAX_DUELS   = 20       # hard cap; log2(1289) ≈ 11, so 20 is more than sufficient

SCOPE_CONFIG = {
    'us': {
        'council_key':   'us_tech_executives',
        'final_ranked':  'final_ranked_us.json',
        'filtered_list': 'filtered_list_us.json',
        'context':       'US domestic influence — political, economic, cultural, and social power',
        'cutoff':        150,
    },
    'global': {
        'council_key':   'global_tech_executives',
        'final_ranked':  'final_ranked_global.json',
        'filtered_list': 'filtered_list_global.json',
        'context':       'global influence — geopolitical, economic, cultural, and institutional power',
        'cutoff':        300,
    },
}


# ------------------------------------------------------------------ #
# ELO math                                                            #
# ------------------------------------------------------------------ #

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def calibrate_elo(candidate: dict, anchor_elo: float, candidate_wins: bool) -> None:
    """Update candidate ELO only — anchor is frozen."""
    e = expected_score(candidate['elo'], anchor_elo)
    score = 1.0 if candidate_wins else 0.0
    candidate['elo'] += K_FACTOR * (score - e)


# ------------------------------------------------------------------ #
# Duel                                                                #
# ------------------------------------------------------------------ #

def build_duel_prompt(a: dict, b: dict, context: str) -> str:
    return (
        f"You are ranking figures by {context}.\n\n"
        f"Who currently has greater real-world influence and power?\n\n"
        f"A. {a['name']} — {a['title']}\n"
        f"B. {b['name']} — {b['title']}\n\n"
        f"Consider: current decision-making power, control of resources or "
        f"narratives, and ability to move systems — not historical legacy alone.\n\n"
        f"Reply with only the letter A or B."
    )


def run_duel(client: anthropic.Anthropic,
             candidate: dict,
             anchor: dict,
             context: str) -> bool:
    """Returns True if candidate (A) wins, False if anchor (B) wins."""
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=16,
            messages=[{
                'role': 'user',
                'content': build_duel_prompt(candidate, anchor, context)
            }],
        )
        answer = msg.content[0].text.strip().upper()
        return not answer.startswith('B')
    except Exception as exc:
        print(f'[API error: {exc}]', end=' ')
        return True   # default: candidate wins (conservative — avoids placing them too low)


# ------------------------------------------------------------------ #
# Binary-search calibration                                           #
# ------------------------------------------------------------------ #

def bisect_rank(client: anthropic.Anthropic,
                candidate: dict,
                ranked_list: list[dict],
                context: str,
                max_duels: int = MAX_DUELS) -> tuple[int, list[str]]:
    """
    Binary search through ranked_list to find where candidate belongs.

    lo/hi are exclusive-upper-bound indices into ranked_list (0-indexed).
    A win against ranked_list[mid] means candidate is better → hi = mid.
    A loss means candidate is worse → lo = mid + 1.
    Terminates when lo == hi or duel budget exhausted.

    Returns (estimated_rank, anchors_played) where estimated_rank is 1-indexed.
    Candidate ELO is updated for tracking only.
    """
    lo = 0
    hi = len(ranked_list)
    anchors_played: list[str] = []
    duel_count = 0

    while lo < hi and duel_count < max_duels:
        mid = (lo + hi) // 2
        entry = ranked_list[mid]
        anchor = {
            'name':  entry['name'],
            'title': entry['title'],
            'elo':   entry.get('elo', INITIAL_ELO),
        }
        anchor_rank = entry.get('rank', mid + 1)

        won = run_duel(client, candidate, anchor, context)
        calibrate_elo(candidate, anchor['elo'], won)
        result = 'W' if won else 'L'
        anchors_played.append(anchor['name'])

        print(f'      [{result}] vs rank {anchor_rank:>4} {anchor["name"][:32]:<32} '
              f'range [{lo + 1}–{hi}] → ', end='')

        if won:
            hi = mid
        else:
            lo = mid + 1

        duel_count += 1
        print(f'[{lo + 1}–{hi}]  candidate ELO → {candidate["elo"]:.0f}')
        time.sleep(DELAY)

    estimated_rank = lo + 1  # 1-indexed insertion position
    return estimated_rank, anchors_played


# ------------------------------------------------------------------ #
# Main merge logic                                                    #
# ------------------------------------------------------------------ #

def run_merge(scope: str) -> None:
    cfg    = SCOPE_CONFIG[scope]
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Load council results
    council_path = BASE_DIR / 'tech_executives_results.json'
    if not council_path.exists():
        print(f'ERROR: tech_executives_results.json not found. '
              f'Run tech_executives_council.py first.')
        sys.exit(1)

    council_data = json.loads(council_path.read_text(encoding='utf-8'))
    council_key  = cfg['council_key']
    stage3       = council_data.get(council_key, {}).get('stage_3_final_list', [])

    if not stage3:
        print(f'ERROR: No stage_3_final_list found for "{council_key}".')
        sys.exit(1)

    # Load existing final rankings
    final_path   = BASE_DIR / cfg['final_ranked']
    ranked_list  = json.loads(final_path.read_text(encoding='utf-8-sig'))

    existing_names_lower = {e['name'].lower() for e in ranked_list}

    # Deduplicate — only process people not already in the list
    new_candidates = []
    skipped        = []
    for person in stage3:
        name = person.get('name', '').strip()
        if not name:
            continue
        if name.lower() in existing_names_lower:
            skipped.append(name)
        else:
            new_candidates.append({
                'name':    name,
                'title':   person.get('role', ''),
                'elo':     INITIAL_ELO,
                'wins':    0,
                'losses':  0,
                'matches': 0,
                'source':  'council',
                'calibration_anchors': [],
            })

    print(f'\n{"=" * 60}')
    print(f'MERGE: {scope.upper()}')
    print(f'  Existing entries:   {len(ranked_list)}')
    print(f'  Council candidates: {len(stage3)}')
    print(f'  Already present:   {len(skipped)}')
    if skipped:
        for s in skipped:
            print(f'    - {s}')
    print(f'  New to calibrate:  {len(new_candidates)}')
    print(f'  Method: binary search (max {MAX_DUELS} duels per candidate)')
    print(f'{"=" * 60}')

    if not new_candidates:
        print('No new candidates to add. Exiting.')
        return

    # Binary-search calibration for each new candidate
    for i, candidate in enumerate(new_candidates, 1):
        print(f'\n  Calibrating [{i}/{len(new_candidates)}]: {candidate["name"]}')
        estimated_rank, anchors_played = bisect_rank(
            client, candidate, ranked_list, cfg['context'])
        candidate['calibration_anchors'] = anchors_played
        candidate['calibration_elo']     = round(candidate['elo'], 1)
        candidate['elo']                 = None   # not comparable to main-tournament ELO
        candidate['estimated_rank']      = estimated_rank
        print(f'  → Duels played:                    {len(anchors_played)}')
        print(f'  → Calibration ELO (tracking only): {candidate["calibration_elo"]}')
        print(f'  → Estimated insertion rank:        {candidate["estimated_rank"]}')

    # Insert new candidates at their bisected rank positions.
    # Process from lowest estimated_rank to highest (reversed primary key) so earlier
    # insertions don't shift the target positions of later ones.
    new_candidates.sort(key=lambda c: c['estimated_rank'], reverse=True)
    combined = list(ranked_list)
    for candidate in new_candidates:
        insert_at = min(candidate['estimated_rank'] - 1, len(combined))
        combined.insert(insert_at, candidate)

    # Re-number ranks
    for idx, entry in enumerate(combined):
        entry['rank'] = idx + 1

    # Save final_ranked
    final_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f'\n✓ Saved {len(combined)} entries to {final_path.name}')

    # Append to filtered_list for pipeline record-keeping
    filtered_path = BASE_DIR / cfg['filtered_list']
    filtered      = json.loads(filtered_path.read_text(encoding='utf-8-sig'))
    for c in new_candidates:
        filtered.append({
            'name':   c['name'],
            'title':  c['title'],
            'source': 'council',
        })
    filtered_path.write_text(
        json.dumps(filtered, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f'✓ Appended {len(new_candidates)} entries to {filtered_path.name}')

    # Report where new candidates landed
    cutoff = cfg['cutoff']
    print(f'\n  New candidate placements (cutoff = top {cutoff}):')
    for c in sorted(new_candidates, key=lambda x: x.get('calibration_elo', 0), reverse=True):
        rank   = next(e['rank'] for e in combined if e['name'] == c['name'])
        marker = ' IN' if rank <= cutoff else 'OUT'
        print(f'    [{marker}] rank {rank:>4}  {c["name"]:<40}  '
              f'calibration ELO {c.get("calibration_elo", "N/A")}')

    # Boundary warning
    boundary_entries = [
        c for c in new_candidates
        if abs(next(e['rank'] for e in combined if e['name'] == c['name']) - cutoff) <= 25
    ]
    if boundary_entries:
        print(f'\n  ⚠ {len(boundary_entries)} new candidate(s) landed within ±25 of '
              f'the cutoff — consider re-running boundary_elo.py for refined ranking.')


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    args      = sys.argv[1:]
    scope_arg = args[0].lower() if args else 'both'
    scopes    = ['us', 'global'] if scope_arg == 'both' else [scope_arg]

    for s in scopes:
        if s not in SCOPE_CONFIG:
            print(f'Unknown scope "{s}". Use: us, global, or both.')
            sys.exit(1)
        run_merge(s)
