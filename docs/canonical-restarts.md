# Fresh basins under one canonical refinement budget

Canonical refinement can explore new native generations after local editing
stalls. This is an experimental search policy; it does not certify that an
arbitrary target is attainable or that distance `1e-4` will be reached.
Canonical scoring, quality callbacks, text/token constraints, deduplication, and
the two fresh singleton checks for epsilon remain unchanged.

## Configuration

`RefinementConfig` adds:

| Field | Default | Meaning |
|---|---:|---|
| `max_restarts` | 3 | Additional fresh native basins; zero disables them. |
| `restart_candidates` | 2 | Maximum fresh generation attempts in each additional basin. |
| `initial_candidates_before_refinement` | 2 | Requested initial candidates to attempt before local refinement. |
| `basin_generation_fraction` | 0.6 | Share of a basin's time slice reserved for its generation phase. |
| `repair_span_mode` | `"words"` | Existing one-to-four-word repair, or experimental `"sentence"`. |

`max_candidates` retains its meaning as the requested initial candidate pool.
For example, six requested initial candidates are scheduled in groups of two,
with local refinement between groups. They are not silently reduced to two.
Later initial groups do not consume `max_restarts`; after that pool is attempted,
up to three additional basins may generate two candidates each. Epsilon,
cancellation, backend errors, or the shared time/query budget can stop before
all requested work is attempted. The trace records what actually happened.

Supplying `initial_text` bypasses the entire initial generation pool. With
`initial_text` and `max_restarts=0`, execution remains a local-repair diagnostic
and never starts a fresh generation. No runtime, or a runtime without
`generate`, also disables fresh-generation restarts.

## Global winner and active basin

Each basin owns its beam, incumbent, and stall counter. These reset when the
search changes basins. A new basin can improve from a worse starting point even
while the previous basin's paragraph remains the global winner. Local repair
uses the current basin's text, not the global winner from a different basin.

The global best and published history remain strictly improving, canonical,
quality-approved results. Starting or improving a worse basin does not publish
it as a new winner. All basins share the same verifier, cache, hard-query cap,
confirmation reserve, quality policy, cancellation event, and global clock.
Fresh coarse and repair requests use distinct seeds; future initial seeds are
reserved so repairs cannot accidentally reuse them.

If a fresh seed produces an identical previously scored paragraph, that exact
cached text can seed the new basin without another canonical query. Current
quality approval is still required; a later rejection cannot be bypassed with
the old verdict. Cache reactivation alone never publishes a global improvement.
It permits subsequent repairs with a fresh seed instead of discarding the basin
because its starting text was a duplicate.

## Fair slices and cooperative stopping

The remaining global rounds and time are divided among the current basin,
pending initial groups, and available restart slots. The first basin has twice
the time weight to accommodate first-use costs. Its generation phase receives
60% of that slice by default, leaving a local-refinement share. This is budget
allocation, not measured evidence that the ratio is optimal. With the default
20 rounds, six initial candidates, and three restarts, round shares leave an
opportunity to reach all three restarts rather than spending every round in the
initial basin.

Native generation checks its operation deadline between passes. Local expiry
does not set the user's cancellation event or reset the global clock. The
controller also temporarily composes the basin stop condition with supported
`should_stop` callbacks on the proposer and its provider or
`allowed_replacements` object. Original callbacks are restored after the call,
including after errors. Providers must not permanently negative-cache a
temporary deadline result. The bundled contextual provider handles this distinction.

A running native forward, encoder call, or custom noncooperative synchronous
callback can finish after its slice. No thread is forcibly interrupted. Actual
elapsed times, cooperative/noncooperative proposal-call counts, unfinished work,
and stop reasons are retained so these overruns are visible. Global quality
policies keep their own time limits; restarting never weakens them.

## Sentence repairs

`repair_span_mode="sentence"` uses the separate sentence-boundary helper,
prioritizes complete sentences containing gradient hotspots, and rotates among
the eligible spans. The runtime still protects tokens outside the selected
span. For these repairs, the actual native request uses a source-grounded
revision prompt instructing preservation of claims, quantities, uncertainty,
negation, entities, causal relationships, and language. The full actual prompt
is in the native trace. This is a proposal prior, not a quality guarantee;
canonical acceptance and the configured quality callback still decide.

## Trace and validation

`basins` records each initial group or extra restart, its trigger, requested and
attempted seeds, time/round slices, local improvements, basin winner, global-best
distance before/after, query counts, and stop reason. `restarts` is the subset of
extra restart records, with `restart_count` and actual attempted candidate count.
Native records and draft events carry `basin_id`; fresh extra generation uses
phase `restart_coarse`. Remaining initial requests are `scheduled_coarse`.

`stalled`, `basin_time_slice`, `basin_round_slice`, `round_limit`, and
`no_new_verified_text` explain local transitions. Global completion distinguishes
confirmed epsilon, cancellation, time/query/round exhaustion,
`restarts_exhausted`, and no valid text. None of the failure statuses proves
unreachability. Compare policies under equal measured global budgets and the
same quality requirements; extra restarts are not an accuracy claim.

The focused tests use fake runtimes and a deterministic canonical scorer. They
cover fresh improvement, a worse basin's independent progress, duplicate seed
reactivation, quality rejection, no runtime, exhausted restarts, initial-pool
scheduling, distinct seeds, cancellation, local versus global deadlines,
cooperative proposal stopping and restoration, shared confirmation queries, and
the sentence repair prompt. No GPU or model-generation probe was used to add
this controller behavior.
