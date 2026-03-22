"""
Tournament Filter — 6-way comparisons, keep top 4, 2 rounds
Whittles master_list_us.json and master_list_global.json by ~4/9.

Round 1 — domain-isolated, council gets a bye:
  - Entries are grouped by their `source` field.
  - Each source pool is shuffled and batched independently (Forbes vs Forbes,
    Congress vs Congress, world leaders vs world leaders, etc.).
  - Council entries (source='council') skip round 1 entirely and enter at round 2.
  This prevents a religious leader or NGO head from being eliminated simply
  because they were randomly grouped with five billionaires.

Round 2 — cross-domain:
  - All round-1 survivors + council entries are shuffled together.
  - Standard 6→4 batching across all sources.

Usage:
  python tournament_filter.py          # both us and global
  python tournament_filter.py us       # US list only
  python tournament_filter.py global   # Global list only
  python tournament_filter.py us 1     # US, round 1 only (testing)
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

BASE_DIR = Path(__file__).parent
MODEL    = 'claude-sonnet-4-6'

SCOPE_CONFIG = {
    'us': {
        'input':   'master_list_us.json',
        'output':  'filtered_list_us.json',
        'context': 'US domestic influence — political, economic, cultural, and social power',
    },
    'global': {
        'input':   'master_list_global.json',
        'output':  'filtered_list_global.json',
        'context': 'global influence — geopolitical, economic, cultural, and institutional power',
    },
}

BATCH_SIZE    = 6
KEEP_COUNT    = 4
NUM_ROUNDS    = 2
REQUEST_DELAY = 0.3
COUNCIL_SOURCE = 'council'


# ------------------------------------------------------------------ #
# Prompt + query                                                      #
# ------------------------------------------------------------------ #

def build_prompt(batch: list[dict], context: str) -> str:
    keep = min(KEEP_COUNT, len(batch) - 1)
    numbered = '\n'.join(
        f"{i + 1}. {e['name']} — {e['title']}"
        for i, e in enumerate(batch)
    )
    return (
        f"You are building a ranked list of the most consequential living people "
        f"by {context}.\n\n"
        f"From these {len(batch)} figures, select the {keep} who currently have "
        f"the greatest demonstrable real-world influence and power:\n\n"
        f"{numbered}\n\n"
        f"Criteria: current reach and impact — not historical legacy or "
        f"institutional prestige alone. Consider actual decision-making power, "
        f"control of resources or narratives, and ability to move systems.\n\n"
        f'Use the exact names as listed. Return only a JSON array with no extra text:\n'
        f'["Name1", "Name2", ..., "Name{keep}"]'
    )


def query_survivors(client: anthropic.Anthropic,
                    batch: list[dict],
                    context: str) -> list[dict]:
    keep = min(KEEP_COUNT, len(batch) - 1)
    try:
        msg  = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{'role': 'user', 'content': build_prompt(batch, context)}],
        )
        text  = msg.content[0].text.strip()
        start, end = text.find('['), text.rfind(']') + 1
        chosen: list[str] = json.loads(text[start:end])
    except Exception as exc:
        print(f'[error: {exc}] keeping all {len(batch)}')
        return batch

    name_map       = {e['name']: e for e in batch}
    name_map_lower = {e['name'].lower(): e for e in batch}

    survivors: list[dict] = []
    seen: set[str] = set()
    for name in chosen[:keep]:
        entry = name_map.get(name) or name_map_lower.get(name.lower())
        if entry and entry['name'] not in seen:
            survivors.append(entry)
            seen.add(entry['name'])

    # Fill to `keep` if LLM returned too few valid names
    if len(survivors) < keep:
        for e in batch:
            if e['name'] not in seen:
                survivors.append(e)
                seen.add(e['name'])
            if len(survivors) == keep:
                break

    return survivors


# ------------------------------------------------------------------ #
# Batch runner (shared by both round types)                           #
# ------------------------------------------------------------------ #

def run_batches(client: anthropic.Anthropic,
                entries: list[dict],
                context: str,
                ckpt_path: Path,
                batch_label_prefix: str = '') -> list[dict]:
    """
    Splits `entries` into BATCH_SIZE groups, queries each, returns survivors.
    Checkpoints after every batch.
    """
    batches    = [entries[i:i + BATCH_SIZE] for i in range(0, len(entries), BATCH_SIZE)]
    survivors  = []
    start_idx  = 0

    if ckpt_path.exists():
        ckpt      = json.loads(ckpt_path.read_text(encoding='utf-8'))
        survivors = ckpt['survivors']
        start_idx = ckpt['next_batch']
        print(f'    Resuming from batch {start_idx + 1}/{len(batches)} '
              f'({len(survivors)} survivors so far)')

    for i, batch in enumerate(batches[start_idx:], start=start_idx):
        label = f'    {batch_label_prefix}[{i + 1:>{len(str(len(batches)))}}/{len(batches)}]'

        if len(batch) < KEEP_COUNT + 1:
            survivors.extend(batch)
            print(f'{label} bye ({len(batch)} kept)')
        else:
            top = query_survivors(client, batch, context)
            survivors.extend(top)
            print(f'{label} kept {len(top)}/{len(batch)}')
            time.sleep(REQUEST_DELAY)

        ckpt_path.write_text(
            json.dumps({'survivors': survivors, 'next_batch': i + 1},
                       ensure_ascii=False),
            encoding='utf-8',
        )

    ckpt_path.unlink(missing_ok=True)
    return survivors


# ------------------------------------------------------------------ #
# Round 1 — domain-isolated, council bypass                          #
# ------------------------------------------------------------------ #

def run_round_1(client: anthropic.Anthropic,
                entries: list[dict],
                context: str,
                scope: str) -> list[dict]:
    """
    Separates entries by source, runs within-source batching.
    Council entries bypass this round entirely.
    """
    council   = [e for e in entries if e.get('source') == COUNCIL_SOURCE]
    non_council = [e for e in entries if e.get('source') != COUNCIL_SOURCE]

    # Group non-council by source
    by_source: dict[str, list[dict]] = {}
    for e in non_council:
        by_source.setdefault(e.get('source', 'unknown'), []).append(e)

    print(f'  Round 1 (domain-isolated):')
    print(f'    Council entries bypassing round 1: {len(council)}')

    all_survivors: list[dict] = []
    for src, src_entries in by_source.items():
        random.shuffle(src_entries)
        print(f'    Source "{src}": {len(src_entries)} entries')
        ckpt = BASE_DIR / f'_ckpt_{scope}_r1_{src}.json'
        survivors = run_batches(client, src_entries, context, ckpt,
                                batch_label_prefix=f'{src} ')
        print(f'    → {len(survivors)} survivors from {src}')
        all_survivors.extend(survivors)

    return all_survivors + council


# ------------------------------------------------------------------ #
# Round 2+ — cross-domain mixed                                      #
# ------------------------------------------------------------------ #

def run_round_n(client: anthropic.Anthropic,
                entries: list[dict],
                context: str,
                round_num: int,
                scope: str) -> list[dict]:
    """Standard cross-domain round: shuffle all, batch, keep top 4."""
    print(f'  Round {round_num} (cross-domain): {len(entries)} entries')
    random.shuffle(entries)
    ckpt = BASE_DIR / f'_ckpt_{scope}_r{round_num}.json'
    survivors = run_batches(client, entries, context, ckpt)
    return survivors


# ------------------------------------------------------------------ #
# Tournament orchestration                                            #
# ------------------------------------------------------------------ #

def run_tournament(scope: str, num_rounds: int = NUM_ROUNDS) -> None:
    cfg    = SCOPE_CONFIG[scope]
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    entries = json.loads(
        (BASE_DIR / cfg['input']).read_text(encoding='utf-8')
    )

    council_count     = sum(1 for e in entries if e.get('source') == COUNCIL_SOURCE)
    non_council_count = len(entries) - council_count
    expected_r1 = int(non_council_count * KEEP_COUNT / BATCH_SIZE) + council_count
    expected_final = int(expected_r1 * KEEP_COUNT / BATCH_SIZE)

    print(f'\n{"=" * 60}')
    print(f'TOURNAMENT: {scope.upper()}')
    print(f'  Total entries:   {len(entries)}')
    print(f'    Council (bye): {council_count}')
    print(f'    Other pools:   {non_council_count}')
    print(f'  After round 1:   ~{expected_r1}')
    print(f'  After round 2:   ~{expected_final}')
    print(f'  Model:           {MODEL}')
    print(f'{"=" * 60}')

    current = entries

    for round_num in range(1, num_rounds + 1):
        print(f'\n--- Round {round_num} / {num_rounds} ---')
        if round_num == 1:
            current = run_round_1(client, current, cfg['context'], scope)
        else:
            current = run_round_n(client, current, cfg['context'], round_num, scope)
        print(f'--- Round {round_num} complete: {len(current)} survivors ---')

    out_path = BASE_DIR / cfg['output']
    out_path.write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'\n✓ Saved {len(current)} entries to {out_path.name}')


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    args       = sys.argv[1:]
    scope_arg  = args[0].lower() if args else 'both'
    rounds_arg = int(args[1]) if len(args) >= 2 else NUM_ROUNDS

    scopes = ['us', 'global'] if scope_arg == 'both' else [scope_arg]

    for s in scopes:
        if s not in SCOPE_CONFIG:
            print(f'Unknown scope "{s}". Use: us, global, or both.')
            sys.exit(1)
        run_tournament(s, num_rounds=rounds_arg)
