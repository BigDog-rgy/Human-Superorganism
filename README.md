# Human Superorganism — Prime Mover Tracker

Project 2 of 2026 — February 16 to March 29, 2026.

A prototype intelligence pipeline for tracking which living actors most shape
large political, economic, technological and institutional systems — modelled on
Hebbian neural organization: individual actors as *neurons*, recurring groupings
as *cell assemblies*, and higher-order patterns as *phase sequences*.

It runs in two scopes, **US** and **Global**, and does six linked jobs: build a
broad candidate universe, rank it into a scoped roster, map actors into cell
assemblies and phase sequences, fetch weekly evidence of current activity,
maintain a memory of reinforcing and adversarial co-activation, and expose the
result through analyst-facing artifacts and visualization.

Start with **[`project-concept.md`](project-concept.md)** for the full design.

## Layout

| Path | What |
|---|---|
| `candidate_pool/` | scraping and LLM-council ranking to build the candidate universe |
| `ca_council_v2.py` | generates cell assemblies from neurons |
| `ps_council_v2.py` | generates phase sequences, and CA↔PS links |
| `superorganism_assembler.py` | assembles the model JSON for each scope |
| `coactivation_updater.py` | maintains persistent co-activation state |
| `weekly_briefing.py` | weekly evidence gathering |
| `combined_viz.py` → `combined_viz.html` | interactive network visualization |
| `explainer.html` | explainer page |
| `*_canon_v2_*.json`, `superorganism_model.json` | persisted canons and models |
| `briefings/`, `checkpoints/`, `state/` | generated artifacts and run state |

`combined_viz.html` is committed and self-contained — it renders straight from a
clone using `lib/bindings/utils.js`, with no Python environment needed.

## Automation

`.github/workflows/weekly_update.yml` runs the weekly update every Wednesday at
14:00 UTC, and can be triggered manually from the Actions tab.

## Setup

```bash
pip install -r requirements.txt
```

Needs a `.env` with the API keys the councils and briefing scripts read; it is
not committed.

## Documentation

The working journal for this project — 24 daily logs, 6 weekly summaries and a
retrospective overview — lives in a companion repository,
`BigDog-rgy/2026-Project-Documentation`, which is **private**. It is the single
source of truth for documentation across all five 2026 projects.
