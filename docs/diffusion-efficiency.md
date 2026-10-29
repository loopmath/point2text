# Diffusion efficiency audit of existing traces

This is a read-only audit of saved runs, made while the live demo remains available. No generation was started for this audit. The system uses native diffusion and revises logits inside denoising, but the outer search still generates independent candidates and chooses by canonical embedding distance. These traces establish working components; they do not establish efficient use of diffusion or a reliable quality advantage.

Sources: `docs/experiments/diffusion-probes.json` in the integration worktree; the latest complete persisted diffusion browser run at audit time, ID `6d8e829f0ff44acc9d13bfef6c7072c9`, created **2026-10-29 17:49:56.239720 UTC**; and the integration versions of `diffusion/runtime.py`, `guidance.py`, and `search.py`. The browser record is a local runtime artifact, not a bundled benchmark. All times below are seconds, rounded to three decimals.

## What was actually executed

The paired probes request 32 denoising steps with a 128-token canvas. Native early stopping means the number of executed passes is smaller. A hook call is not necessarily a guided step: the first schedules selected steps against the requested 32-step horizon, and many scheduled steps are never reached.

| Probe / mode | Native passes | Actual guided steps | Encoder loss updates | Prefill | Runtime elapsed | Search elapsed | Sum guidance step times | Sum encoder update times |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paired-probe / unguided | 9 | 0 | 0 | 0.911 | 5.311 | 5.339 | 0 | 0 |
| paired-probe / guided | 10 | 2 | 4 | 0.340 | 3.119 | 3.159 | 0.868 | 0.322 |
| strong-01 / unguided | 9 | 0 | 0 | 1.230 | 5.865 | 5.890 | 0 | 0 |
| strong-01 / guided | 10 | 4 | 8 | 0.296 | 3.011 | 3.032 | 1.412 | 0.339 |
| early-strong-02 / unguided | 9 | 0 | 0 | 0.348 | 4.136 | 4.158 | 0 | 0 |
| early-strong-02 / guided | 13 | 13 | 26 | 0.256 | 3.425 | 3.459 | 2.917 | 0.802 |

Actual guided step indices were `[6,9]`, `[6,7,8,9]`, and `[0..12]`, respectively. Hard-score time was **not separately preserved in these older probe records**. Search-minus-runtime is only an envelope for scoring and other outer overhead: 0.028/0.040, 0.025/0.021, and 0.022/0.034 seconds for unguided/guided. It must not be relabeled as measured encoder scoring time.

The guided calls followed their unguided partners. Prefill and unattributed runtime costs differ substantially, so the shorter guided elapsed times are **not a measured guidance speedup**. There was no randomized or controlled cold/warm comparison. The first pair produced identical text; the stronger later-only run worsened map distance, and the early strong run improved it for this one target. None establishes a quality-per-second benefit.

## Latest complete browser run

This run used six sequential independent candidates, 128-token canvases, up to 32 native passes, guidance on every reached step, and two encoder updates per guided step. Context length was 389 tokens. The settings included top-k 32, strength 10, step size 20, and maximum logit change 4.

| Seed | Native / guided / encoder updates | Prefill | Runtime | Generation wrapper | Hard scoring | Sum guidance | Sum encoder updates |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | 12 / 12 / 24 | 0.494 | 3.960 | 3.965 | 0.011 | 3.235 | 0.594 |
| 43 | 12 / 12 / 24 | 0.427 | 4.075 | 4.080 | 0.012 | 3.419 | 0.649 |
| 44 | 13 / 13 / 26 | 0.368 | 4.198 | 4.203 | 0.013 | 3.582 | 0.747 |
| 45 | 17 / 17 / 34 | 0.357 | 5.350 | 5.355 | 0.014 | 4.668 | 1.161 |
| 46 | 13 / 13 / 26 | 0.333 | 4.077 | 4.083 | 0.016 | 3.491 | 0.895 |
| 47 | 13 / 13 / 26 | 0.329 | 4.098 | 4.103 | 0.015 | 3.512 | 0.974 |
| Sum | **80 / 80 / 160** | **2.307** | **25.759** | **25.789** | **0.081** | **21.909** | **5.020** |

Search trace elapsed was **25.873 seconds**; the surrounding browser run record was **25.876 seconds**. These are different nested scopes. The 0.081-second total for hard-text verification is about 0.3% of the search duration and is not the first optimization target. Prompt prefill represents about 8.9% of the search duration, but that observed share is not a promised saving from reuse.

## Timing attribution matters

**Do not add the columns.** Encoder update time is inside guidance time; guidance time and prefill are inside runtime; runtime is inside generation; generation and hard scoring are inside search. The summed native step traces are also inclusive: 1.702/2.768 seconds for paired-probe, 2.075/2.707 for strong-01, 1.389/3.157 for early-strong-02, and 23.400 for the six browser trajectories. They are not isolated decoder forward times.

MLX evaluates lazily. The runtime starts the pass timer before constructing decoder logits; the guidance hook's first `mx.eval(selected, ids, retained)` can materialize that native decoder work. Consequently, **21.909 seconds is the observed hook envelope, not 21.909 seconds of additional encoder guidance**. The 5.020-second encoder-update envelope includes reward execution and Python update bookkeeping; it is more specific but still not a pure backward-only measurement. These logs cannot separate deferred decoding, full-vocabulary selection, device synchronization, bridge lookup, encoder forward, and backward precisely.

The unguided probe runtime also has large unexplained gaps between prefill plus step sums and overall elapsed time. Runtime elapsed begins before model readiness, lock acquisition, and prompt preparation, so subtracting these scopes does not yield a valid decoder-only cost.

## Concrete optimization order

1. **Fix attribution and eliminate redundant whole-grid checks.** Add synchronized diagnostic timing boundaries around decoder materialization, top-k/retained-mass selection, compact transfer, bridge cache misses, encoder forward/backward, scatter, prefill, and cleanup. Keep those profiling synchronizations optional because they can themselves change execution. The current runtime scans the entire `[1,128,262144]` output for finiteness and maximum hook delta on every hook call, including identity calls. That is 33,554,432 elements per scan. Skip work for an identical returned logits object; for actual compact edits, validate the finite compact updates and obtain delta evidence from the selected entries, with a full-grid check available in diagnostic mode. Preserve tests proving other logits and noneditable rows are unchanged.
2. **Spend guidance where it can affect committed text.** The initial requested-horizon schedule guided only two reached steps; the current all-step schedule spent 160 loss updates over six candidates. Compare a small number of early or entropy-triggered updates with every-step guidance, while keeping clipping and strength fixed. Measure actual reached guided steps, changed token decisions, hard error, and elapsed time. Increasing strength or update count alone is not an efficiency improvement. A one-update variant saves a reward call per guided step but has no KL restoring gradient at its initial proposal; evaluate that tradeoff explicitly.
3. **Reuse prompt work across independent candidates.** Search currently invokes a fresh runtime generation for every seed. Each invocation constructs a new cache and repeats prefill despite the identical prompt; weights and sparse bridge vectors already persist. Cache a prompt prefill keyed by model revision and exact context tokens, then clone or otherwise isolate any mutable state required by each trajectory. Verify identical-seed output and cancellation isolation before keeping it. The current `finally` also synchronizes and clears MLX cache after each request; measure allocation and memory behavior before changing this, because peak memory is already substantial.
4. **Test decoder compilation as one isolated change.** Native decoder construction currently uses `compile_graph=False`. Try the supported compiled decoder path after the eager path is reproducible. Measure first compilation separately from repeated execution; retain the external hook and compare token decisions, numerical behavior, cancellation, peak memory, and hard-text scores. Compilation is a hypothesis, not an established speedup on this hardware.
5. **Reduce repeated bridge and candidate work after measuring it.** The bridge already caches requested token surfaces/vectors across steps and candidates. Batch cache misses or cache reusable selected-vector stacks if the new timings show that lookup remains material. Test a predefined target-tolerance or budget stopping rule for the outer candidate loop. In the recorded six-candidate run, the eventual winner appeared as candidate four, but stopping there is hindsight and cannot be counted as a deployable speedup without a rule chosen in advance. Reusing denoising trajectories or targeted remasking is a later diffusion-specific experiment requiring sampler correctness tests.

## Fair next test

Run after the live session, with no competing generation process. Freeze the model/map, prompt, output policy, target panel, seed list, and safety checks. First isolate implementation changes such as identity-hook checks, prefill reuse, and compilation with the same guidance policy. Separate first-use costs from repeated calls and randomize/interleave variants to limit order effects. Do not compare a cold first run with a warm second run as a speedup.

Then compare unguided best-of-candidates, sparse early/entropy guidance, and current all-step guidance under **equal measured wall-clock budgets**. Record best hard-text error as a function of elapsed time, candidate validity, blinded coherence/task fit, repetition/diversity, memory, actual native passes, and actual encoder updates. Keep the nearest-example prompting policy fixed so that retrieval is not a confounder. A useful optimization improves the measured cost/quality tradeoff; lowering a two-dimensional surrogate or changing a paragraph is insufficient evidence. Residual priors and alternate encoders remain outside this efficiency decision.
