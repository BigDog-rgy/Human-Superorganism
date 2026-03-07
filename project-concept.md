# Prime Mover Tracker - Project Concept

*Updated: March 7, 2026*

## What This Project Is
Prime Mover Tracker is a power-mapping and momentum-tracking system for living actors who can move large political, economic, technological, and institutional systems.

Its practical goal is not just to publish a ranked list. The system is trying to do three things at once:

- identify who currently matters most
- model the structural dynamics those people are participating in
- update that model weekly as new evidence arrives

The project currently supports two scopes:

- US: domestic power and institutional influence
- Global: geopolitical and transnational power

## Current System Shape
The implemented system is a staged pipeline, not a single model:

1. Candidate ingestion
   Raw pools are scraped from structured sources such as Forbes lists, Congress, and world leaders, then supplemented with LLM-generated candidate pools for groups that are underrepresented in public rankings.

2. Ranking pipeline
   Candidate pools are compiled into US and global master lists, filtered through tournament-style LLM comparisons, ranked with Swiss ELO, and then refined near the inclusion cutoff.

3. Canon generation
   Separate LLM councils generate the canonical phase sequences and cell assemblies that define the superorganism vocabulary for each scope.

4. Superorganism assembly
   Final ranked people, phase-sequence canon, and assembly canon are merged into a structured model consumed by the briefing, learning, and visualization layers.

5. Weekly update loop
   A weekly briefing process now fetches fresh news by phase sequence first, then by fired cell assembly and selected person, synthesizes the week into structured updates, and then applies a Hebbian-style update to persistent state.

6. Visualization
   A combined HTML visualization renders both scopes and now supports both neuron and cell-assembly views, with state overlays where available.

## Implemented Pipeline

### 1) Candidate Sources
The project already has a concrete ingestion layer:

- `scrape_all_sources.py` for Forbes, Congress, and world leaders
- specialized council scripts for non-obvious candidate pools
- `compile_master_lists.py` to merge source pools into `master_list_us.json` and `master_list_global.json`

The intent here is coverage first, then filtering. Public prestige lists are treated as inputs, not as the final answer.

### 2) Ranking and Selection
The ranking stack is more mature than the old concept doc implied. It currently works like this:

- `tournament_filter.py`
  Shrinks the candidate pool with grouped LLM elimination rounds
- `swiss_elo.py`
  Produces broad rank ordering through repeated pairwise comparisons
- `boundary_elo.py`
  Re-runs focused comparison rounds around the inclusion cutoff
- `merge_council_results.py`
  Inserts newly surfaced candidates into the existing ranking via binary-search calibration

Current core ranking artifacts include:

- `filtered_list_us.json`
- `filtered_list_global.json`
- `elo_ranked_us.json`
- `elo_ranked_global.json`
- `final_ranked_us.json`
- `final_ranked_global.json`

### 3) Canonical Vocabulary
The superorganism vocabulary is explicitly generated and stored:

- `ps_council.py`
  Creates the canonical phase sequences and initial neuron-to-phase-sequence weights
- `ca_council.py`
  Creates canonical cell assemblies and neuron-to-assembly membership

Outputs:

- `ps_canon_us.json`
- `ps_canon_global.json`
- `ca_canon_us.json`
- `ca_canon_global.json`

This layer matters because the project is trying to track influence structurally, not only reputationally. People are mapped into recurring dynamics and institutional clusters rather than treated as isolated names.

### 4) Superorganism Models
`superorganism_assembler.py` is now the main assembly step.

It builds:

- `us_superorganism_model.json`
- `superorganism_model.json`

These models combine:

- ranked individuals
- phase-sequence assignments
- cell-assembly memberships
- minimal metadata needed by the briefing and visualization layers

The assembler is intentionally lightweight and data-driven. It replaces the earlier, heavier mapper-first conception.

### 5) Weekly Briefing Layer
`weekly_briefing.py` is the active briefing engine for both scopes.

It supports:

- `python weekly_briefing.py`
- `python weekly_briefing.py --scope global`
- optional social-signal collection
- raw data checkpointing before synthesis
- synthesis-only reruns

The briefing process is now explicitly multi-stage:

- fetch phase-sequence news first
- detect which cell assemblies were active inside those phase sequences
- fetch targeted news for fired assemblies
- use assembly-aware and PS-aware logic to choose which people merit individual fetching
- add a spontaneous coverage layer so overdue people still get sampled
- save raw fetch output so Claude synthesis can be retried without re-running all news calls
- synthesize the result into structured weekly output

Artifacts are written to `briefings/` as JSON and Markdown, with matching raw JSON snapshots for retry and audit. The weekly briefing is now the evidence bridge between the static ranked model, the assembly layer, and the persistent learning state.

### 6) Persistent Learning State
`hebbian_updater.py` maintains the memory layer.

For each scope it bootstraps from the assembled model and then updates:

- neuron-to-phase-sequence weights
- phase-sequence dominance
- pairwise effects implied by cooperative or adversarial edge signals

This creates stateful behavior across weeks:

- quiet actors gradually decay
- notable activity strengthens relevant links
- concerning activity disrupts them
- recurring momentum accumulates instead of being recomputed from zero

Key state artifacts:

- `superorganism_state.json`
- `global_superorganism_state.json`

### 7) Visualization
`combined_viz.py` now generates a single interactive HTML view with two dimensions of switching:

- scope: Global or US
- view type: neurons or cell assemblies

The visualization is already designed to show:

- relative influence by node size
- hemisphere or scope framing by color
- edge valence and strength
- Hebbian state overlays when available
- assembly-level structure as a first-class network, not just a tooltip detail

The current behavior is asymmetric in a useful but important way:

- US neuron and US assembly views use the latest US briefing plus Hebbian state
- global neuron view uses global Hebbian state
- global assembly view is currently structural rather than briefing-annotated

This is still a generated analyst view, not yet a full product UI.

## Operating Loop
The implemented weekly loop is roughly:

1. Maintain or refresh ranked lists when the candidate universe changes.
2. Regenerate PS and CA canon when the structural vocabulary needs revision.
3. Reassemble the superorganism model.
4. Run `weekly_briefing.py` for US and/or global scope.
   This now produces both final briefing outputs and raw fetch snapshots.
5. Run `hebbian_updater.py` to apply the latest briefing to persistent state.
6. Run `combined_viz.py` to inspect the updated network.

In practice, the project has two cadences:

- slower structural cadence for ranking and canon revision
- faster weekly cadence for evidence ingestion and state update

## What The Project Is Becoming
The clearest trajectory is an evidence-linked longitudinal intelligence system with four layers:

- roster layer: who belongs in the field
- ranking layer: who currently matters most
- structure layer: what dynamics and coalitions organize the field
- momentum layer: what changed this week and how that alters the network

If that matures, the system could support:

- weekly analyst briefings
- person-level influence histories
- phase-sequence momentum tracking
- coalition and conflict change detection
- event-triggered alerts when important actors or dynamics shift sharply

## What Is Still Missing
The current project is substantial, but still prototype-shaped in a few important ways:

- orchestration is mostly manual
- there is no durable historical database for easy querying across many weeks
- provenance is better in the raw artifacts than in the end-user presentation
- evaluation of ranking quality and briefing quality is still informal
- visualization is useful, but still a generated artifact rather than a polished interface
- schema discipline exists in practice, but not yet as a fully formalized contract layer

## Near-Term Direction
The most defensible next build sequence is:

1. Stabilize the operational path end to end.
2. Add run logging and append-only historical snapshots.
3. Improve provenance and confidence visibility in briefing outputs.
4. Add evaluation loops for ranking drift, briefing quality, and false-signal rate.
5. Build a lightweight query surface for person history, PS momentum, and edge changes.

## Guiding Principle
Treat influence as a changing system of actors, coalitions, and strategic dynamics under continual evidence update, not as a static list of famous people.
