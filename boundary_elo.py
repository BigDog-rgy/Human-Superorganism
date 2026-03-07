"""
Boundary ELO Refinement
Runs additional focused Swiss ELO rounds on the 50 players straddling
each list's inclusion cutoff — where main-tournament ranking noise is
most consequential.

  US:     cutoff = 150  →  window = ranks 126–175  (25 in, 25 out)
  Global: cutoff = 300  →  window = ranks 276–325  (25 in, 25 out)

The 50 players start with fresh ELO (1000) and compete only against
each other for 10 rounds with K=64, allowing fast spread in a small
field. After refinement, they are re-sorted by refined ELO and
re-inserted into their original rank slots. Every player outside the
window keeps their rank unchanged.

Outputs:
  final_ranked_us.json
  final_ranked_global.json
  (both include an `original_rank` and `main_elo` field on boundary entries
   so you can see how much the refinement moved each person)

Usage:
  python boundary_elo.py          # both us and global
  python boundary_elo.py us       # US only
  python boundary_elo.py global   # global only
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
K_FACTOR    = 64       # higher K → faster spread in a short focused tournament
INITIAL_ELO = 1000.0
DELAY       = 0.25

SCOPE_CONFIG = {
    'us': {
        'input':      'elo_ranked_us.json',
        'output':     'final_ranked_us.json',
        'context':    'US domestic influence — political, economic, cultural, and social power',
        'cutoff':     150,   # inclusion threshold: top N make the final list
        'window':     50,    # 25 players on each side of the cutoff
        'num_rounds': 10,
    },
    'global': {
        'input':      'elo_ranked_global.json',
        'output':     'final_ranked_global.json',
        'context':    'global influence — geopolitical, economic, cultural, and institutional power',
        'cutoff':     300,
        'window':     50,
        'num_rounds': 10,
    },
}


# ------------------------------------------------------------------ #
# ELO math (same as swiss_elo.py)                                     #
# ------------------------------------------------------------------ #

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def apply_elo(winner: dict, loser: dict) -> None:
    e_w = expected_score(winner['elo'], loser['elo'])
    winner['elo']     += K_FACTOR * (1.0 - e_w)
    loser['elo']      += K_FACTOR * (0.0 - (1.0 - e_w))
    winner['wins']    += 1
    loser['losses']   += 1
    winner['matches'] += 1
    loser['matches']  += 1


# ------------------------------------------------------------------ #
# Swiss pairing (same as swiss_elo.py)                                #
# ------------------------------------------------------------------ #

def make_pairings(players: list[dict]) -> list[tuple[str, str]]:
    ordered = sorted(players, key=lambda p: p['elo'], reverse=True)
    paired  = set()
    pairs: list[tuple[str, str]] = []

    for i, player in enumerate(ordered):
        if player['name'] in paired:
            continue
        played_vs: set[str] = player.get('played_against', set())
        chosen = None
        for j in range(i + 1, len(ordered)):
            opp = ordered[j]
            if opp['name'] in paired:
                continue
            if opp['name'] not in played_vs:
                chosen = opp
                break
        if chosen is None:
            for j in range(i + 1, len(ordered)):
                opp = ordered[j]
                if opp['name'] not in paired:
                    chosen = opp
                    break
        if chosen is not None:
            pairs.append((player['name'], chosen['name']))
            paired.add(player['name'])
            paired.add(chosen['name'])

    return pairs


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
             a: dict, b: dict,
             context: str) -> tuple[dict, dict]:
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=16,
            messages=[{'role': 'user', 'content': build_duel_prompt(a, b, context)}],
        )
        answer = msg.content[0].text.strip().upper()
        if answer.startswith('B'):
            return b, a
        return a, b
    except Exception as exc:
        print(f'[API error: {exc}]', end=' ')
        return a, b


# ------------------------------------------------------------------ #
# Round                                                               #
# ------------------------------------------------------------------ #

def run_round(client: anthropic.Anthropic,
              players: list[dict],
              context: str,
              ckpt_path: Path) -> None:
    """Runs one Swiss round in-place (mutates player dicts)."""
    player_map = {p['name']: p for p in players}

    if ckpt_path.exists():
        ckpt       = json.loads(ckpt_path.read_text(encoding='utf-8'))
        name_pairs = ckpt['name_pairs']
        start_idx  = ckpt['next_pair']
        for name, state in ckpt['player_state'].items():
            if name in player_map:
                p = player_map[name]
                p['elo']            = state['elo']
                p['wins']           = state['wins']
                p['losses']         = state['losses']
                p['matches']        = state['matches']
                p['played_against'] = set(state['played_against'])
        print(f'  Resuming from pair {start_idx + 1}/{len(name_pairs)}')
    else:
        name_pairs = make_pairings(players)
        start_idx  = 0

    bye_count = len(players) - len(name_pairs) * 2
    print(f'  Pairs: {len(name_pairs)}  Byes: {bye_count}')

    for i, (name_a, name_b) in enumerate(name_pairs[start_idx:], start=start_idx):
        a = player_map[name_a]
        b = player_map[name_b]

        winner, loser = run_duel(client, a, b, context)
        apply_elo(winner, loser)

        a.setdefault('played_against', set()).add(b['name'])
        b.setdefault('played_against', set()).add(a['name'])

        w_name = winner['name'][:34]
        l_name = loser['name'][:34]
        print(f'  [{i + 1:>3}/{len(name_pairs)}]  {w_name:<35} def. {l_name:<35} '
              f'ELO {winner["elo"]:.0f} / {loser["elo"]:.0f}')

        time.sleep(DELAY)

        ckpt_path.write_text(
            json.dumps({
                'next_pair':    i + 1,
                'name_pairs':   name_pairs,
                'player_state': {
                    p['name']: {
                        'elo':            p['elo'],
                        'wins':           p['wins'],
                        'losses':         p['losses'],
                        'matches':        p['matches'],
                        'played_against': list(p.get('played_against', set())),
                    }
                    for p in players
                },
            }, ensure_ascii=False),
            encoding='utf-8',
        )

    ckpt_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Refinement                                                          #
# ------------------------------------------------------------------ #

def run_refinement(scope: str) -> None:
    cfg    = SCOPE_CONFIG[scope]
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    full_list = json.loads((BASE_DIR / cfg['input']).read_text(encoding='utf-8'))

    cutoff    = cfg['cutoff']
    half      = cfg['window'] // 2

    # 0-indexed window: 25 ranks inside cutoff, 25 ranks outside
    win_start = cutoff - half      # e.g. 125 for US  (rank 126 in 1-indexed)
    win_end   = cutoff + half      # e.g. 175 for US  (rank 175 inclusive, 0-indexed)
    win_start = max(0, win_start)
    win_end   = min(len(full_list), win_end)

    boundary_raw = full_list[win_start:win_end]
    num_rounds   = cfg['num_rounds']
    total_duels  = (len(boundary_raw) // 2) * num_rounds

    print(f'\n{"=" * 60}')
    print(f'BOUNDARY REFINEMENT: {scope.upper()}')
    print(f'  Full list:     {len(full_list)} entries')
    print(f'  Cutoff:        top {cutoff}')
    print(f'  Window:        ranks {win_start + 1}–{win_end} ({len(boundary_raw)} players)')
    print(f'                 ({half} inside cutoff, {len(boundary_raw) - half} outside)')
    print(f'  Rounds:        {num_rounds}  (K={K_FACTOR})')
    print(f'  Total duels:   ~{total_duels}')
    print(f'{"=" * 60}')

    # Initialise players with fresh ELO; preserve main-tournament data for reference
    players: list[dict] = [
        {
            'name':           e['name'],
            'title':          e['title'],
            'original_rank':  e.get('rank', win_start + i + 1),
            'main_elo':       e['elo'] if e.get('elo') is not None else INITIAL_ELO,
            'elo':            INITIAL_ELO,
            'wins':           0,
            'losses':         0,
            'matches':        0,
            'played_against': set(),
        }
        for i, e in enumerate(boundary_raw)
    ]

    # Run rounds
    for round_num in range(1, num_rounds + 1):
        print(f'\n--- Round {round_num} / {num_rounds} ---')
        ckpt = BASE_DIR / f'_boundary_ckpt_{scope}_r{round_num}.json'
        run_round(client, players, cfg['context'], ckpt)

        top10 = sorted(players, key=lambda p: p['elo'], reverse=True)[:10]
        print(f'\n  Top 10 after round {round_num}:')
        for rank, p in enumerate(top10, 1):
            print(f'    {rank:>2}. {p["name"]:<42} '
                  f'ELO {p["elo"]:>7.1f}  {p["wins"]}W-{p["losses"]}L')

    # Sort window by refined ELO, re-insert into full list
    players.sort(key=lambda p: p['elo'], reverse=True)

    output: list[dict] = []

    # Entries before the window — unchanged
    output.extend(full_list[:win_start])

    # Refined window — reassigned to same rank slots
    for i, p in enumerate(players):
        output.append({
            'rank':          win_start + i + 1,
            'name':          p['name'],
            'title':         p['title'],
            'elo':           round(p['elo'], 1),
            'wins':          p['wins'],
            'losses':        p['losses'],
            'matches':       p['matches'],
            'main_elo':      round(p['main_elo'], 1) if p['main_elo'] is not None else None,
            'original_rank': p['original_rank'],
            'played_against': sorted(p.get('played_against', set())),
        })

    # Entries after the window — unchanged, ranks renumbered
    for i, entry in enumerate(full_list[win_end:]):
        entry = dict(entry)
        entry['rank'] = win_end + i + 1
        output.append(entry)

    out_path = BASE_DIR / cfg['output']
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\n✓ Saved {len(output)} entries to {out_path.name}')

    # Print the boundary with IN/OUT markers and rank-change deltas
    print(f'\n  Final boundary (ranks {win_start + 1}–{win_end}):')
    for entry in output[win_start:win_end]:
        marker = ' IN' if entry['rank'] <= cutoff else 'OUT'
        orig   = entry.get('original_rank', '?')
        delta  = (orig - entry['rank']) if isinstance(orig, int) else 0
        arrow  = f'▲{delta}' if delta > 0 else (f'▼{abs(delta)}' if delta < 0 else '  =')
        print(f'    [{marker}] {entry["rank"]:>4}. {entry["name"]:<42} '
              f'ELO {entry["elo"]:>7.1f}   {arrow} (was #{orig})')


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    args      = sys.argv[1:]
    scope_arg = args[0].lower() if args else 'both'

    scopes = ['us', 'global'] if scope_arg == 'both' else [scope_arg]

    for s in scopes:
        if s not in SCOPE_CONFIG:
            print(f'Unknown scope "{s}". Use: us, global, or both.')
            sys.exit(1)
        run_refinement(s)
