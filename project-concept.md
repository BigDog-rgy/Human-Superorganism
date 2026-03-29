# Prime Mover Tracker - Project Concept

*Updated: March 26, 2026*

## What This Project Is
Prime Mover Tracker is a prototype intelligence pipeline for tracking which living actors most shape large political, economic, technological, and institutional systems.

It is not just a ranking exercise. The working system is trying to do six linked jobs:

- build a broad candidate universe
- reduce and rank that universe into a scoped roster of prime movers
- map those actors into recurring cell assemblies and phase sequences
- fetch weekly evidence about what is active right now
- maintain a lightweight memory of reinforcing and adversarial co-activation
- expose the resulting structure through analyst-facing artifacts and visualization

The repository currently operates in two scopes:

- US
- Global

## Current Reality
This repo is no longer just a concept sketch. It already contains a multi-stage pipeline with persisted artifacts for both scopes.

The active system today produces:

- candidate pools and compiled master lists
- filtered and ranked prime mover lists
- phase-sequence and cell-assembly canons
- assembled superorganism model JSON files
- weekly briefings in JSON and Markdown
- persistent coactivation state
- a combined interactive HTML visualization

The most important architectural clarification is this:

- the live weekly memory layer is `coactivation_updater.py`
- the older `hebbian_updater.py` path is referenced in some documents and docstrings, but that file is not present in the current codebase

So the running system has weekly learning, but it is centered on pairwise neuron-neuron and CA-CA coactivation scores, not on the older direct neuron-to-phase-sequence weight file described in some historical notes.

## Implemented Pipeline

### 1) Candidate Ingestion and Pool Building
Candidate coverage starts with structured public lists and then expands through LLM-assisted councils for harder-to-source categories.

Key files in `candidate_pool/`:

- `scrape_forbes.py`
- `scrape_congress.py`
- `scrape_world_leaders.py`
- `scrape_all_sources.py`
- `tech_executives_council.py`
- `us_llm_council.py`
- `other_candidates_us_council.py`
- `other_candidates_global_council.py`
- `compile_master_lists.py`

This layer produces scope-specific pool artifacts such as:

- `master_list_us.json`
- `master_list_global.json`

The current design is coverage-first. Public lists provide baseline recall; council-generated pools widen the field before ranking pressure is applied.

### 2) Ranking Pipeline
The ranking stack is concrete and fairly mature.

Current ranking flow:

1. `candidate_pool/tournament_filter.py`
   Reduces the master pool through two rounds of grouped LLM comparison, with source-aware handling so niche but meaningful domains are not immediately crushed by billionaire-heavy pools.
2. `candidate_pool/swiss_elo.py`
   Runs scope-specific Swiss-style head-to-head ELO rounds on the filtered list.
3. `candidate_pool/boundary_elo.py`
   Re-ranks the cutoff window where inclusion noise matters most:
   - US cutoff at top 150
   - Global cutoff at top 300

Primary outputs:

- `filtered_list_us.json`
- `filtered_list_global.json`
- `elo_ranked_us.json`
- `elo_ranked_global.json`
- `final_ranked_us.json`
- `final_ranked_global.json`

This means the ranked roster is not hand-maintained. It is produced by an explicit reduction-and-refinement pipeline.

### 3) Canonical Vocabulary
The project treats influence as organized behavior, not just a list of names. That structural layer is represented by:

- cell assemblies
- phase sequences

Current scripts:

- `ca_council_v2.py`
  Generates canonical cell assemblies through neuron-centric proposals, deduplication, and ELO narrowing
- `ps_council_v2.py`
  Generates canonical phase sequences and assigns CAs to PS domains through a multi-model council plus chairman synthesis

Primary outputs:

- `ca_canon_v2_us.json`
- `ca_canon_v2_global.json`
- `ps_canon_v2_us.json`
- `ps_canon_v2_global.json`

This is an important shift from earlier versions of the repo. The current assembly flow is built around the v2 canons for both scopes.

### 4) Superorganism Assembly
`superorganism_assembler.py` is the hub assembly step.

It combines:

- ranked prime movers from `final_ranked_{scope}.json`
- phase-sequence canon from `ps_canon_v2_{scope}.json`
- cell-assembly canon from `ca_canon_v2_{scope}.json`

It then writes:

- `us_superorganism_model.json`
- `superorganism_model.json`

What the assembler is doing in practice:

- limits the model to the top ranked roster for that scope
- attaches each person to cell assemblies
- derives phase-sequence memberships by chaining neuron -> CA -> PS
- creates the model format consumed downstream by briefing generation and visualization

This model JSON is the central exchange format of the repo.

### 5) Weekly Briefing Layer
`weekly_briefing.py` is the active weekly evidence-ingestion script for both scopes.

Entry points:

- `python weekly_briefing.py --scope us`
- `python weekly_briefing.py --scope global`

What the script actually does:

1. Loads the relevant superorganism model.
2. Loads sparse fetch state from `state/fetch/`.
3. Loads coactivation state from `state/coactivation/` if available.
4. Fetches recent PS-level news via Perplexity.
5. Scores PS activation.
6. Selects cell assemblies under the active PSes using coactivation-informed sampling.
7. Selects conscious neurons using PS membership, CA membership, and neuron coactivation signals.
8. Adds spontaneous neuron coverage so dormant actors are still periodically sampled.
9. Fetches person-level news for the selected actors.
10. Synthesizes final weekly briefing artifacts.

Outputs are written into `briefings/` as:

- synthesized JSON
- synthesized Markdown
- raw JSON snapshots for audit and regeneration

The weekly process is therefore not just "news summarization." It is a structured attention-and-selection loop over the current superorganism model.

### 6) Persistent Learning State
The live persistent state layer is `coactivation_updater.py`.

Entry points:

- `python coactivation_updater.py --bootstrap --scope us`
- `python coactivation_updater.py --bootstrap --scope global`
- `python coactivation_updater.py --scope us`
- `python coactivation_updater.py --scope global`
- `python coactivation_updater.py --status --scope us`
- `python coactivation_updater.py --status --scope global`

What it stores:

- neuron-neuron coactivation scores
- cell-assembly to cell-assembly coactivation scores
- decay and update metadata

What it does each week:

- reads the latest synthesized briefing for the relevant scope
- decays existing scores toward zero
- evaluates pairwise relationships among conscious active neurons and active assemblies
- records whether pairs were reinforcing, adversarial, or neutral
- updates rolling sparse state in:
  - `state/coactivation/us_coactivation_state.json`
  - `state/coactivation/global_coactivation_state.json`

This is the current memory system that materially affects later briefing selection and visualization edges.

## Visualization
`combined_viz.py` generates the main analyst-facing HTML network view.

It currently supports:

- US and global datasets
- neuron view and cell-assembly view
- coactivation-informed edge rendering when state exists
- briefing-aware overlays and legends
- local serving in a browser

The visualization now relies primarily on coactivation state for learned edges. Some internal comments and fallback text still mention "Hebbian" edges because this code evolved out of the earlier weight-based design, but the active learned layer in the current repo is the coactivation state.

## What Is Implemented vs Proposed

### Implemented Now

- candidate scraping and council-based pool expansion
- master list compilation
- tournament filtering plus Swiss ELO ranking plus cutoff refinement
- v2 cell-assembly canon generation
- v2 phase-sequence canon generation with CA-PS assignment
- superorganism model assembly
- weekly US and global briefing generation
- persistent coactivation learning state
- interactive combined visualization

### Proposed / Not Yet Integrated

The main future-facing design is captured in `learning_proposal.md`.

That proposal describes replacing older flat neuron-to-PS logic with a more structurally faithful hierarchy:

- neuron-neuron weights
- neuron-cell-assembly weights
- cell-assembly to phase-sequence weights
- homeostatic normalization and drift guardrails

Important caveat: this proposal is still a design document. It is not the behavior of the running pipeline today.

## Practical Operating Loop
The working cadence for the repo is now:

1. Refresh candidate pools when the universe changes materially.
2. Rebuild master lists.
3. Re-run ranking when the roster needs to be refreshed.
4. Regenerate CA and PS canons when the structural vocabulary changes.
5. Reassemble the superorganism model.
6. Bootstrap coactivation state if needed.
7. Run `weekly_briefing.py` for US and/or global.
8. Run `coactivation_updater.py` to learn from the latest briefing.
9. Run `combined_viz.py` to inspect the resulting network.

In practice there are two tempos:

- a slower structural tempo for pool building, ranking, and canon work
- a weekly tempo for briefing generation, coactivation updates, and review

## Main Gaps
The repo is already substantive, but it is still prototype-shaped in several important ways:

- orchestration is still mostly manual
- documentation has drifted from the live scripts in several places
- historical "Hebbian updater" language still exists even though the active weekly memory path is coactivation-based
- evaluation of ranking quality and briefing quality is still informal
- provenance is stronger in raw artifacts than in top-level documentation
- the pipeline is artifact-driven rather than query-first
- browser-facing visualization and script comments still reflect some legacy terminology

## Immediate Documentation Truths
Anyone working in this repo should keep these points straight:

- `project-concept.md`, `startup.txt`, and some docstrings previously described a `hebbian_updater.py` loop
- the active weekly updater in the current tree is `coactivation_updater.py`
- `weekly_briefing.py` already consumes coactivation state during selection
- `superorganism_assembler.py` now uses v2 canons for both US and global

Those points explain a lot of the apparent architectural contradiction when reading the repo for the first time.

## Near-Term Direction
The most defensible next sequence appears to be:

1. Finish documentation cleanup so entry points, artifact names, and architectural claims all match the live code.
2. Decide whether coactivation is the intended long-term learning layer or just a bridge toward the richer hierarchical state in `learning_proposal.md`.
3. If the hierarchical design is still the target, implement it in the live pipeline rather than leaving it as a parallel paper design.
4. Add better run logging, historical summaries, and evaluation around ranking drift and briefing quality.
5. Move from artifact inspection toward more queryable historical state.

## Guiding Principle
Treat influence as an evolving system of actors, assemblies, and strategic dynamics under continuous evidence update, not as a static list of famous people.
