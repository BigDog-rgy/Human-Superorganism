# Superorganism Weight Expansion: Learning Proposal

**Status**: Draft
**Date**: 2026-03-12
**Scope**: Replace the flat neuron-PS weight layer with three hierarchically structured weight types — neuron-neuron (N-N), neuron-cell assembly (N-CA), and cell assembly-phase sequence (CA-PS) — and define initialization, update mechanics, and drift guardrails for each.

---

## Why Replace Neuron-PS Weights

The current N-PS weights (`neuron_dps_weights`) are the least structurally faithful part of the Hebbian model. They assign neurons directly to phase sequences, skipping the intermediate cell assembly layer entirely. This means:

- A neuron's weight in a PS doesn't reflect whether its *assembly* is active, only whether the neuron appeared in related news
- Assemblies don't accumulate any persistent learning signal; they remain purely structural after initialization
- The only "edge" information between neurons is the viz-time computation `Σ w_a[ps] × w_b[ps]`, which encodes structural co-membership, not behavioral co-activation

The correct causal chain in Hebb's model is: **neuron fires → cell assembly activates → phase sequence propagates**. The three proposed weight types follow this chain directly and together replace everything N-PS was doing.

```
Neuron ←——— N-N ———→ Neuron
   |                     |
 N-CA                  N-CA
   |                     |
Cell Assembly ←— CA-PS —→ Phase Sequence
```

---

## Unified Update Principle

All three weight types use the same two-sided Hebbian rule:

- **Co-activation strengthens**: if both endpoints are active in the same week, the weight increases
- **Inactivity decays**: if unused, weights decay multiplicatively toward zero (never below a floor)

Multiplicative decay (`w ← w × (1 - decay_rate)`) is used throughout because it naturally slows as weights approach zero, preventing complete collapse without needing a hard floor check.

---

## 1. Neuron-Neuron (N-N) Weights

### What they represent
Direct behavioral co-occurrence between two neurons — how often they appear together as active agents in the same news cycle, independent of their structural co-membership in CAs or PSes.

### Initialization
Start all N-N weights at **zero**. Since N-PS weights are being retired, there is no bootstrap shortcut. Weights build from scratch from observed co-activation. This is the cleanest starting point: no inherited structural bias, all edges earned.

Pairs with no co-activation history simply don't exist in the sparse storage.

### Update (weekly)

**Step 1 — Decay**: Multiplicative decay on all stored pairs.
```
w_ab ← w_ab × (1 - decay_rate)
```

**Step 2 — Co-activation boost**: For each edge signal where both parties appear as `notable` in `person_updates` that week and share the same `ps_id`, boost their N-N weight:
```
w_ab += learning_rate    (cooperative valence)
w_ab -= adversarial_drop (adversarial valence)
```
Adversarial N-N weights are allowed to go negative (encoding opposition), with a floor at `WEIGHT_MIN = -1.0`, same as the existing N-PS convention.

**Step 3 — Prune**: Drop entries with `|w| < PRUNE_THRESHOLD`.

### Function in the model
- **Visualization**: replaces the on-the-fly `Σ w_a[ps] × w_b[ps]` edge computation in `combined_viz.py` with stored, historically-earned weights
- **Lateral propagation** (future): when a neuron fires strongly, boost N-N neighbors' selection scores slightly — modeling associative recall

---

## 2. Neuron-Cell Assembly (N-CA) Weights

### What they represent
How central a neuron is to a given cell assembly — not binary membership, but degree of participation. Central members have higher weights; peripheral or dormant members decay toward zero over time.

### Initialization
All members of each CA are initialized to the same flat weight:
```
w(neuron, ca) = 1.0   for all current members
```
No LLM calls, no proposal count lookups needed. Differentiation happens purely through subsequent Hebbian updates. This is the simplest possible starting point that is consistent with "we don't yet have behavioral evidence to distinguish members."

### Update (weekly)

**Step 1 — Decay**: Multiplicative decay on all N-CA weights.

**Step 2 — Activation credit**: For each `person_update` where a neuron has signal `notable` or `concerning`, find all CAs the neuron belongs to whose `ps_memberships` overlap with the neuron's `ps_impacts` that week. Those CA connections receive an update:
```
notable:    w(neuron, ca) += learning_rate
concerning: w(neuron, ca) -= disruption_rate
```
Only CAs in the matching PS domain receive credit — a neuron firing in a trade-war story doesn't strengthen its connection to an assembly in the domestic politics domain.

**Step 3 — Normalize** *(drift guardrail — see below)*

**Step 4 — Prune**: Drop entries below `PRUNE_THRESHOLD`. A decayed entry means "this member has gone dormant" — they remain in the structural membership list but no longer influence selection.

### Function in the model
- **Conscious neuron selection**: replaces the `hebbian_w = neuron_dps_weights[name][ps_id]` lookup. When an assembly fires, each member's contribution is weighted by their N-CA weight:
  ```python
  score += normalized_ps_score * neuron_ca_weights.get(name, {}).get(ca_id, 0.5)
  ```
- **Assembly health**: aggregate N-CA weights across an assembly's members signal whether it is cohering or dissolving

---

## 3. Cell Assembly-Phase Sequence (CA-PS) Weights

### What they represent
How strongly an assembly participates in a given phase sequence. Replaces the binary `ps_memberships` list with a continuous weight that evolves based on observed co-activation.

### Initialization
All existing PS memberships initialized to a flat weight:
```
w(ca, ps) = 0.5   for all (ca, ps) in current ps_memberships
```
Derivable entirely from the existing model JSON, no API calls. Assemblies with multiple PS memberships start equally associated with each.

### Update (weekly)

**Step 1 — Decay**: Multiplicative decay on all CA-PS weights.

**Step 2 — Assembly activation credit**: For each assembly that was active (`assembly_updates` signal) in a week where its associated PS had a high activation score (≥7/10):
```
w(ca, ps) += learning_rate × (ps_activation_score / 10)
```

**Step 3 — Normalize** *(drift guardrail — see below)*

**Step 4 — Prune**: Drop entries below `PRUNE_THRESHOLD`. A fully decayed CA-PS connection means the assembly has become irrelevant to that phase sequence.

### Function in the model
- **Assembly selection probability**: replaces the member-count-based weighting in `select_assemblies_from_network()`. When a PS fires, assemblies are sampled proportional to their CA-PS weight for that PS
- **PS momentum signal**: aggregate CA-PS weights for currently active assemblies feed into the synthesis context — a PS whose strongly-weighted assemblies are all active is clearly accelerating

---

## Guardrails Against Drift

This is the critical design concern. Without guardrails, Hebbian rules cause two failure modes over many weeks:

1. **Saturation**: active neurons/assemblies accumulate max weights everywhere (rich-get-richer)
2. **Collapse**: inactive neurons/assemblies decay to zero and permanently vanish from selection

Both are addressed by a single mechanism borrowed from computational neuroscience: **homeostatic normalization** (also called synaptic scaling).

### Homeostatic normalization

After applying all boosts and before pruning, normalize each entity's outgoing weight vector so the **sum stays bounded at a target value**:

**For N-CA** (per neuron): after each weekly update, if a neuron's total N-CA weight across all their assemblies exceeds `N_CA_SUM_CAP`, scale all their N-CA weights down proportionally:
```python
total = sum(neuron_ca_weights[name].values())
if total > N_CA_SUM_CAP:
    scale = N_CA_SUM_CAP / total
    for ca_id in neuron_ca_weights[name]:
        neuron_ca_weights[name][ca_id] *= scale
```
This creates **competition among a neuron's assemblies**: if one connection strengthens, others weaken slightly. A neuron can't become maximally central to all their assemblies at once.

**For CA-PS** (per assembly): same logic — if an assembly's total CA-PS weight across its PSes exceeds `CA_PS_SUM_CAP`, scale down. An assembly can't become maximally relevant to all its phase sequences simultaneously.

**For N-N** (per neuron): N-N weights use hard bounds (`WEIGHT_MIN` / `WEIGHT_MAX`) rather than sum normalization, since negative values (adversarial edges) would make sum normalization ill-defined. Hard bounds are sufficient here.

### Why this works
- Normalization is multiplicative and applied after Hebbian updates — it scales rather than zeroes weights, so no information is lost
- The system naturally finds an equilibrium: weights that keep getting boosted stay high, weights that decay lose their share
- The `N_CA_SUM_CAP` value can be set to `num_ca_memberships × 0.8` per neuron — so a neuron with 4 assemblies has a cap of 3.2, meaning on average each assembly has weight 0.8 (slightly below the 1.0 init), with the distribution decided by activity

### Additional guardrails
- **Multiplicative decay** (already used in `apply_decay`): `w ← w × (1 - rate)` rather than `w -= rate`. Weights approach zero asymptotically, never go below the prune threshold from a single step
- **Initialization floor**: N-CA and CA-PS weights initialize at 0.5–1.0, not 0.0, so all entities start with representation in selection before any news arrives
- **Prune threshold as floor**: `PRUNE_THRESHOLD = 0.04` means members don't fully vanish until they've been inactive for many consecutive weeks — the model has structural memory

---

## Storage

All three weight types added to `superorganism_state.json`, replacing `neuron_dps_weights`:

```json
{
  "version": 2,
  "config": {
    "decay_rate":          0.10,
    "learning_rate":       0.15,
    "disruption_rate":     0.08,
    "cooperative_boost":   0.05,
    "adversarial_drop":    0.06,
    "prune_threshold":     0.04,
    "n_ca_sum_cap_factor": 0.80,
    "ca_ps_sum_cap_factor": 0.80
  },
  "neuron_neuron_weights": { "Name A": { "Name B": 0.18 } },
  "neuron_ca_weights":     { "Name":   { "CA-001": 0.92 } },
  "ca_ps_weights":         { "CA-001": { "PS-01":  0.61 } }
}
```

`n_ca_sum_cap_factor` and `ca_ps_sum_cap_factor` are multiplied by the entity's membership count to compute its individual cap, making the cap scale naturally with how many connections an entity has.

---

## Script Changes Required

| Script | Change |
|---|---|
| `hebbian_updater.py` | Remove N-PS logic; add N-N, N-CA, CA-PS init and update; add normalization step |
| `weekly_briefing.py` | Load N-CA and CA-PS weights; use in `select_assemblies_from_network()` and `select_neurons_conscious()` |
| `combined_viz.py` | Use stored N-N weights for edge rendering; drop on-the-fly formula |
| `superorganism_assembler.py` | Write N-CA and CA-PS initial weights to state at assembly time |

---

## Open Questions

1. **N-N directionality**: Should N-N weights be symmetric, or asymmetric (Trump→Musk ≠ Musk→Trump)? Asymmetric weights allow encoding influence direction but double the storage and complicate normalization. Recommend starting symmetric.

2. **Credit path for edge signals**: Currently, `edge_signals` carry a `ps_id`. In the new system, should edge signals update N-N directly (losing PS context), or also update the CA-PS weight for the relevant CA? The PS context may be valuable to keep.

3. **Stranded neurons**: A neuron whose N-CA weights all decay below prune threshold would have no representation in selection. Should there be a minimum floor per neuron (e.g., always keep their highest N-CA weight above 0.05), or is full dormancy an acceptable outcome?

4. **Backward compatibility**: Existing `superorganism_state.json` files (version 1) contain only `neuron_dps_weights`. Migration path: on first run with the new code, re-bootstrap from model JSON to produce the version 2 state, discarding the N-PS history.
