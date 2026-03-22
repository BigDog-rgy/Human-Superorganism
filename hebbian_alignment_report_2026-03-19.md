# Hebbian Alignment Report

Date: 2026-03-19
Project: Prime Mover Tracker

## Assessment

At a high level, the project is directionally Hebbian, but only part of the live implementation matches the model in the notes. The strongest alignment is in the ontology and the weekly firing loop: the code explicitly builds `neuron -> cell assembly -> phase sequence` structure in [superorganism_assembler.py](./superorganism_assembler.py), then uses weekly PS activity to fire assemblies and select conscious neurons in [weekly_briefing.py](./weekly_briefing.py). That lines up well with the notes on assemblies forming first, then phase sequences and superordinate organization, and with attention as selective reinforcement rather than pure reaction.

The coactivation layer also fits reasonably well. [coactivation_updater.py](./coactivation_updater.py) applies weekly decay plus reinforcing or adversarial updates, which is a decent abstraction of "fire together, wire together" plus fading traces. The spontaneous neuron sampling in [weekly_briefing.py](./weekly_briefing.py) also loosely matches the note that neurons have endogenous activity even without direct sensory excitation.

## Where The Implementation Matches Well

- The project has an explicit hierarchical representation of neurons, cell assemblies, and phase sequences rather than treating influence as a flat ranked list.
- The assembly step derives phase-sequence membership through assemblies rather than assigning everything independently, which is much closer to the Hebbian picture of intermediate organization.
- The weekly briefing loop uses selective activation and sampling rather than attempting total coverage. That is a good fit for Hebb's emphasis on attention and partial activation.
- The coactivation updater adds persistence, decay, and repeated reinforcement across weeks, which gives the system something like a reverberatory trace and cumulative association history.
- The separation between conscious and spontaneous selection is conceptually strong. It maps well onto the notes' distinction between stimulus-driven activity and partially autonomous background activity.

## Where It Is Lacking Or Divergent

The main divergence is that the persistent learning state is still centered on direct neuron-to-PS weights in [hebbian_updater.py](./hebbian_updater.py). That skips the assembly layer exactly where the notes, and the design in [learning_proposal.md](./learning_proposal.md), say the real Hebbian causal chain should live. So the project has Hebbian language and some Hebbian behavior, but the core long-term plasticity is still flatter than the intended model.

There is also a more immediate implementation problem: the live briefing schema and the updater schema do not match. `weekly_briefing.py` produces `person_updates` with `signal: active|quiet` and `top_stories` with `valence`, while `hebbian_updater.py` expects `ps_impacts`, `notable|concerning`, and `edge_signals`. So in practice the updater is probably mostly decaying weights, not learning from weekly evidence.

That is compounded by a likely bug: `hebbian_updater.py` appears to load the latest file with prefix `weekly_briefing_` without excluding `_raw_`, unlike `coactivation_updater.py`. The current state file indicates that a raw briefing file was applied. That means the updater may be reading the wrong artifact entirely.

Another divergence is that the current selection logic treats positive and negative coactivation almost the same at selection time. `coactivation_updater.py` stores signed reinforcing versus adversarial scores, but `weekly_briefing.py` uses `abs(score)` when building CA and neuron selection bonuses. That means adversarial history still increases future co-selection probability. That is not a good fit for a model where opposition should often inhibit or disrupt a phase sequence rather than strengthen association in the same way.

More broadly, the system still encodes assemblies as mostly structural rather than plastic. In the notes, assemblies are not just labels or containers; they are the primary learned units that can integrate, fractionate, generalize, and change over developmental time. The project has the structure for that, but not yet the persistent learning dynamics.

## Suggestions For Improvement

1. Fix the live learning loop before adding more theory.
   Make `hebbian_updater.py` read synthesized briefing files only, and make the briefing emit the fields the updater actually uses, or update the updater to consume the current schema.

2. Replace the direct neuron-to-PS learning state with the hierarchical model already outlined in [learning_proposal.md](./learning_proposal.md).
   The clearest upgrade is persistent `N-N`, `N-CA`, and `CA-PS` weights.

3. Give assemblies their own persistent plasticity.
   Assemblies should not remain static containers while only neurons learn. This is the biggest mismatch with the notes' theory of cell assemblies.

4. Preserve sign in selection and propagation.
   Reinforcing links should facilitate co-firing; adversarial links should inhibit, compete, or break a firing chain rather than contribute through absolute value.

5. Add homeostatic normalization or another drift control.
   Without this, repeated weekly updates risk rich-get-richer saturation for already central actors and gradual disappearance of quieter but structurally important ones.

6. Add explicit support for fractionation and generalization.
   If an assembly repeatedly fires under varied contexts, the system should eventually differentiate stronger and weaker submemberships rather than treating membership as permanently binary.

## Suggestions For Testing

1. Add fixture-based updater tests using small hand-written briefing JSONs.
   Cover at least one cooperative week, one adversarial week, one quiet week, and one mixed week.

2. Add a regression test that raw briefing files are never selected by the updater.

3. Add an end-to-end replay test over archived briefings.
   The goal is to verify that repeated co-activation produces durable strengthening and that inactive links decay gradually.

4. Add sign-sensitivity tests.
   An adversarial pair should not increase future co-selection the same way a reinforcing pair does.

5. Add assembly-plasticity tests once the hierarchical model is implemented.
   Repeated neuron activity in one assembly should strengthen that neuron's role there without equally strengthening all of its other assemblies.

6. Add stability tests over many simulated weeks.
   Check for saturation, collapse, and loss of coverage among dormant but important actors.

## Bottom Line

The project already matches the Hebbian framework in structure and in some of its weekly activation logic, especially around selective attention, assemblies, and repeated coactivation. But the true long-term plasticity still does not align well enough with the notes because the main persistent learning state bypasses the assembly layer and the current updater appears misaligned with the briefing artifacts it is supposed to learn from.

The biggest conceptual gap is direct neuron-to-PS learning. The biggest practical gap is that the updater likely is not consuming the right schema, so some of the intended Hebbian learning may not actually be happening.
