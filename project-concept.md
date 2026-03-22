# Prime Mover Tracker - Project Concept

*Updated: March 15, 2026*

## What This Project Is
Prime Mover Tracker is a prototype intelligence system for tracking which living actors most shape large political, economic, technological, and institutional systems.

It is not just a ranking project. The system is trying to do four linked jobs:

- maintain a working roster of relevant actors
- rank their current influence within a scope
- map them into recurring structural dynamics and organizational clusters
- update that picture weekly from fresh evidence

The project currently operates in two scopes:

- US
- Global

## Current Reality
The repo is now a working multi-stage pipeline. It produces ranked lists, superorganism models, weekly briefings, persistent Hebbian state, and an interactive visualization.

The important distinction is this:

- the ranking, briefing, state-update, and visualization loop is implemented
- the next-generation learning model is designed but not yet integrated

That distinction matters because some of the most ambitious ideas in the repo are still proposals, while the live system still runs on neuron-to-phase-sequence Hebbian weights.

## Implemented System

### 1) Candidate Ingestion
Candidate coverage starts from structured public lists and then expands through LLM-assisted sourcing.

Current inputs include:

- `scrape_all_sources.py`
- `scrape_forbes.py`
- `scrape_congress.py`
- `scrape_world_leaders.py`
- council scripts for additional candidate pools such as tech, LLM, and other underrepresented actors
- `compile_master_lists.py`

This produces scope-specific master pools such as:

- `master_list_us.json`
- `master_list_global.json`

The design goal is broad coverage first, then selection pressure.

### 2) Ranking Pipeline
The ranking layer is much more concrete than the original concept implied. The current stack is:

- `tournament_filter.py`
  Uses grouped LLM comparison rounds to reduce the candidate pool
- `swiss_elo.py`
  Produces a broader influence ordering through repeated pairwise comparisons
- `boundary_elo.py`
  Re-ranks candidates near the inclusion cutoff
- `merge_council_results.py`
  Calibrates newly surfaced candidates into an existing ranked list

Key artifacts include:

- `filtered_list_us.json`
- `filtered_list_global.json`
- `elo_ranked_us.json`
- `elo_ranked_global.json`
- `final_ranked_us.json`
- `final_ranked_global.json`

### 3) Canonical Vocabulary
The project treats influence as structured activity, not just a list of names. That structural layer is represented by:

- phase sequences
- cell assemblies

Current scripts:

- `ps_council.py`
  Generates canonical phase sequences plus initial neuron-to-PS weights
- `ca_council.py`
  Generates the original CA canon
- `ca_council_v2.py`
  New neuron-centric CA generation pipeline with batching, deduplication, and ELO narrowing

Current implementation state is asymmetric:

- US assembly generation has already shifted toward `ca_canon_v2_us.json`
- global assembly generation still points at the older CA canon path in the main assembly flow

This is one of the clearest signs that the project is in an active transition rather than a frozen architecture.

### 4) Superorganism Assembly
`superorganism_assembler.py` is the main data assembly step.

It combines:

- final ranked people
- PS canon
- CA canon

Outputs:

- `us_superorganism_model.json`
- `superorganism_model.json`

These models are the hub format used by the weekly briefing layer, the Hebbian updater, and the visualization.

For the US path, the assembler currently reads `ca_canon_v2_us.json`.
For the global path, it still reads `ca_canon_global.json`.

### 5) Weekly Briefing Layer
`weekly_briefing.py` is the active weekly evidence-ingestion engine for both scopes.

Supported entry points:

- `python weekly_briefing.py`
- `python weekly_briefing.py --scope global`

The weekly process is staged:

1. Fetch recent news per phase sequence.
2. Score phase-sequence activation.
3. Fire relevant cell assemblies from the network.
4. Select people for targeted coverage using structural memberships plus Hebbian weighting.
5. Add spontaneous coverage so dormant actors still get sampled over time.
6. Save raw fetch outputs for retry and audit.
7. Synthesize structured weekly briefing output.

Artifacts are written into `briefings/` as JSON and Markdown, with raw JSON snapshots for re-synthesis without re-fetching.

### 6) Persistent Learning State
`hebbian_updater.py` maintains persistent weekly state for both scopes.

Supported entry points:

- `python hebbian_updater.py --bootstrap`
- `python hebbian_updater.py`
- `python hebbian_updater.py --bootstrap --scope global`
- `python hebbian_updater.py --scope global`

What is implemented today:

- neuron-to-phase-sequence weights (`neuron_dps_weights`)
- weekly decay for quiet actors
- strengthening for notable activity
- weakening for concerning activity
- extra adjustment from cooperative or adversarial edge signals

This gives the system actual memory across weeks, but it is still the earlier, flatter learning design.

### 7) Visualization
`combined_viz.py` generates the main analyst-facing visualization.

It currently supports:

- US and global scope switching
- neuron and cell-assembly views
- Hebbian overlays where state exists

The current viz still computes neuron-level edge strength from shared PS weights rather than from learned neuron-neuron edges. That matches the current state model and also marks a clear boundary between the live implementation and the next proposed learning architecture.

## What Is Implemented vs Proposed

### Implemented Now

- candidate sourcing and list compilation
- tournament and ELO-based ranking
- PS canon generation
- CA canon generation, including a newer US-focused v2 path
- superorganism assembly
- weekly news ingestion and synthesis for US and global
- persistent Hebbian updates using neuron-to-PS weights
- interactive HTML visualization

### Proposed / In Progress

- replacement of flat neuron-to-PS weights with:
  - neuron-neuron weights
  - neuron-cell-assembly weights
  - cell-assembly-to-phase-sequence weights
- homeostatic normalization and other drift guardrails
- deeper use of learned CA structure in briefing selection and viz edges
- stronger historical logging and queryability across many weeks

The working design for that next step is captured in `learning_proposal.md`, but it is not yet the behavior of the running pipeline.

## Operating Loop
The practical operating cadence is now:

1. Refresh candidate pools when the universe changes.
2. Re-run ranking when the roster needs updating.
3. Regenerate PS and CA canon when the structural vocabulary is revised.
4. Assemble the superorganism model.
5. Run `weekly_briefing.py` for US and/or global.
6. Run `hebbian_updater.py` to apply the latest weekly evidence.
7. Run `combined_viz.py` to inspect the resulting network.

In practice there are two tempos:

- a slower structural tempo for roster, ranking, and canon work
- a weekly tempo for evidence ingestion and learning-state updates

## What The Project Is Becoming
The clearest trajectory is an evidence-linked longitudinal intelligence system with four layers:

- roster layer: who belongs in the field
- ranking layer: who currently matters most
- structure layer: what assemblies and phase sequences organize the field
- momentum layer: what changed this week and how it shifts the network

If the current pipeline is stabilized, the project could support:

- weekly analyst briefings
- person-level momentum history
- assembly-level and phase-sequence-level change tracking
- coalition and conflict shift detection
- more explicit alerting around sharp network changes

## Main Gaps
The repo is already substantial, but it is still prototype-shaped in several ways:

- orchestration is mostly manual
- naming and entry-point docs have some drift from the live scripts
- historical storage is artifact-based rather than query-first
- provenance is stronger in raw outputs than in the end presentation
- evaluation of ranking quality and briefing quality is still informal
- US and global paths are not perfectly symmetrical yet
- the new hierarchical learning design is not implemented

## Near-Term Direction
The most defensible next sequence is:

1. Align the pipeline documentation with the actual script entry points and files.
2. Finish the CA v2 transition so US and global assembly handling are architecturally consistent.
3. Replace `neuron_dps_weights` with the proposed hierarchical weight system.
4. Add better run logging, historical snapshots, and provenance visibility.
5. Improve evaluation for ranking drift, briefing quality, and false-signal accumulation.

## Guiding Principle
Treat influence as a changing system of actors, organizations, and strategic dynamics under continuous evidence update, not as a static list of famous people.
