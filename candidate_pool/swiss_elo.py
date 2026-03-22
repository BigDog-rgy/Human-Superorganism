"""
Swiss ELO Tournament — 5 rounds of head-to-head duels
Runs on filtered_list_us.json and filtered_list_global.json

Each round:
  1. Sort players by current ELO descending (Swiss pairing)
  2. Pair adjacent players, skipping rematches where possible
  3. For each pair, Claude picks the more influential figure
  4. Update ELO scores (K=32, standard formula)
  Odd player out each round gets a bye (ELO unchanged)

After 5 rounds, output is sorted by ELO descending with full stats.

Usage:
  python swiss_elo.py              # both us and global
  python swiss_elo.py us           # US list only
  python swiss_elo.py global       # Global list only
  python swiss_elo.py us 3         # US, 3 rounds (testing)
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
K_FACTOR    = 32
INITIAL_ELO = 1000.0
NUM_ROUNDS  = 5   # default fallback if not set per-scope
DELAY       = 0.25   # seconds between API calls

# Rounds are set per-scope.
# Minimum meaningful rounds ≈ log2(N):
#   US  ~472 players  → log2(472) ≈ 9  → using 7 (good coverage, practical runtime)
#   Global ~1247 players → log2(1247) ≈ 10 → using 10
SCOPE_CONFIG = {
    'us': {
        'input':      'filtered_list_us.json',
        'output':     'elo_ranked_us.json',
        'context':    'US domestic influence — political, economic, cultural, and social power',
        'num_rounds': 7,
    },
    'global': {
        'input':      'filtered_list_global.json',
        'output':     'elo_ranked_global.json',
        'context':    'global influence — geopolitical, economic, cultural, and institutional power',
        'num_rounds': 10,
    },
}


# ------------------------------------------------------------------ #
# ELO math                                                            #
# ------------------------------------------------------------------ #

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def apply_elo(winner: dict, loser: dict) -> None:
    e_w = expected_score(winner['elo'], loser['elo'])
    winner['elo']    += K_FACTOR * (1.0 - e_w)
    loser['elo']     += K_FACTOR * (0.0 - (1.0 - e_w))
    winner['wins']   += 1
    loser['losses']  += 1
    winner['matches'] += 1
    loser['matches']  += 1


# ------------------------------------------------------------------ #
# Swiss pairing                                                       #
# ------------------------------------------------------------------ #

def make_pairings(players: list[dict]) -> list[tuple[str, str]]:
    """
    Sort by ELO descending, pair adjacent players.
    If the natural adjacent pair is a rematch, look ahead for the nearest
    available opponent who hasn't been faced. Falls back to allowing the
    rematch if no alternative exists.
    Returns list of (name_a, name_b) tuples.
    """
    ordered  = sorted(players, key=lambda p: p['elo'], reverse=True)
    paired   = set()
    pairs: list[tuple[str, str]] = []

    for i, player in enumerate(ordered):
        if player['name'] in paired:
            continue

        played_vs: set[str] = player.get('played_against', set())

        # Prefer the nearest ELO-adjacent opponent not yet used and not a rematch
        chosen = None
        for j in range(i + 1, len(ordered)):
            opp = ordered[j]
            if opp['name'] in paired:
                continue
            if opp['name'] not in played_vs:
                chosen = opp
                break

        # Fall back: allow rematch with nearest available
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
        # else: player gets a bye this round

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
    """Returns (winner, loser). Defaults to A on any error."""
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
        return a, b   # default win to A


# ------------------------------------------------------------------ #
# Round                                                               #
# ------------------------------------------------------------------ #

def run_round(client: anthropic.Anthropic,
              players: list[dict],
              context: str,
              round_num: int,
              scope: str) -> None:
    """Runs one Swiss round, mutating player dicts in-place."""

    ckpt_path  = BASE_DIR / f'_elo_ckpt_{scope}_r{round_num}.json'
    player_map = {p['name']: p for p in players}

    # ---- Generate or restore pairings --------------------------------
    if ckpt_path.exists():
        ckpt       = json.loads(ckpt_path.read_text(encoding='utf-8'))
        name_pairs = ckpt['name_pairs']
        start_idx  = ckpt['next_pair']
        # Restore per-player state from checkpoint
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
    print(f'  Pairs: {len(name_pairs)}   Byes: {bye_count}')

    # ---- Run duels ---------------------------------------------------
    for i, (name_a, name_b) in enumerate(name_pairs[start_idx:], start=start_idx):
        a = player_map[name_a]
        b = player_map[name_b]

        winner, loser = run_duel(client, a, b, context)
        apply_elo(winner, loser)

        a.setdefault('played_against', set()).add(b['name'])
        b.setdefault('played_against', set()).add(a['name'])

        label = f'[{i + 1:>{len(str(len(name_pairs)))}}/{len(name_pairs)}]'
        w_name = winner['name'][:34]
        l_name = loser['name'][:34]
        print(f'  {label}  {w_name:<35} def. {l_name:<35} '
              f'ELO {winner["elo"]:.0f} / {loser["elo"]:.0f}')

        time.sleep(DELAY)

        # Checkpoint after every duel
        ckpt_path.write_text(
            json.dumps({
                'next_pair':   i + 1,
                'name_pairs':  name_pairs,
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
# Tournament                                                          #
# ------------------------------------------------------------------ #

def run_tournament(scope: str, num_rounds: int | None = None) -> None:
    cfg        = SCOPE_CONFIG[scope]
    num_rounds = num_rounds if num_rounds is not None else cfg.get('num_rounds', NUM_ROUNDS)
    client     = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    raw = json.loads((BASE_DIR / cfg['input']).read_text(encoding='utf-8'))

    # Initialise player state (preserve any existing ELO if re-running)
    players: list[dict] = [
        {
            'name':           e['name'],
            'title':          e['title'],
            'elo':            float(e.get('elo', INITIAL_ELO)),
            'wins':           e.get('wins', 0),
            'losses':         e.get('losses', 0),
            'matches':        e.get('matches', 0),
            'played_against': set(e.get('played_against', [])),
        }
        for e in raw
    ]

    print(f'\n{"=" * 60}')
    print(f'SWISS ELO TOURNAMENT: {scope.upper()}')
    print(f'  Players:       {len(players)}')
    print(f'  Rounds:        {num_rounds}  (K={K_FACTOR})')
    print(f'  Duels/round:   ~{len(players) // 2}')
    print(f'  Total duels:   ~{(len(players) // 2) * num_rounds}')
    print(f'  Model:         {MODEL}')
    print(f'{"=" * 60}')

    for round_num in range(1, num_rounds + 1):
        print(f'\n--- Round {round_num} / {num_rounds} ---')
        run_round(client, players, cfg['context'], round_num, scope)
        print(f'--- Round {round_num} complete ---')

        # Top 10 snapshot after each round
        top10 = sorted(players, key=lambda p: p['elo'], reverse=True)[:10]
        print(f'\n  Top 10 after round {round_num}:')
        for rank, p in enumerate(top10, 1):
            print(f'    {rank:>2}. {p["name"]:<42} '
                  f'ELO {p["elo"]:>7.1f}  {p["wins"]}W-{p["losses"]}L')

    # ---- Save final results ------------------------------------------
    players.sort(key=lambda p: p['elo'], reverse=True)
    output: list[dict] = []
    for rank, p in enumerate(players, 1):
        output.append({
            'rank':           rank,
            'name':           p['name'],
            'title':          p['title'],
            'elo':            round(p['elo'], 1),
            'wins':           p['wins'],
            'losses':         p['losses'],
            'matches':        p['matches'],
            'played_against': sorted(p.get('played_against', set())),
        })

    out_path = BASE_DIR / cfg['output']
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\n✓ Saved {len(output)} ranked entries to {out_path.name}')

    # Print final top 20
    print(f'\n  Final top 20:')
    for p in output[:20]:
        print(f'  {p["rank"]:>4}. {p["name"]:<42} '
              f'ELO {p["elo"]:>7.1f}  {p["wins"]}W-{p["losses"]}L')


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    args       = sys.argv[1:]
    scope_arg  = args[0].lower() if args else 'both'
    rounds_arg = int(args[1]) if len(args) >= 2 else None  # None → use per-scope default

    scopes = ['us', 'global'] if scope_arg == 'both' else [scope_arg]

    for s in scopes:
        if s not in SCOPE_CONFIG:
            print(f'Unknown scope "{s}". Use: us, global, or both.')
            sys.exit(1)
        run_tournament(s, num_rounds=rounds_arg)
