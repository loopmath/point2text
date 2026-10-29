# Path to canonical distance 1e-4

**Change the optimizer, not just the update count.** Keep diffusion for coherent coarse proposals and span repairs; add a closed loop that preserves the best actual text, followed by canonical-token discrete search for fine corrections. The components below are proposed work, not implemented capabilities.

## Objective and current gap

Success means `||PCA(canonical_encode(text)) - target|| <= 1e-4` on the existing map, equivalent to squared error at most `1e-8`. Keep the map, preprocessing, target and units fixed. Rounded display values and soft loss are not acceptance criteria; show at least six decimal places near zero and retain full-precision records. Evaluate numerical accuracy separately from text quality and originality.

Today, each native pass predicts fresh logits. We apply a fixed number of normalized, trust-capped updates to an approximate soft-token representation, then let the sampler advance. Native confidence can stop the trajectory independently of target error. Canonical distance is measured only after the complete candidate; it does not drive a repair of that candidate. Thus the current system has no convergence procedure for the requested tolerance.

## Implementation order and parallel ownership

| Component | Build | Acceptance test |
| --- | --- | --- |
| Canonical controller | Score valid clean drafts every 2-4 native passes; cache duplicates and retain the best actual text independently of sampler state. Stop on verified epsilon or an explicit budget. | No worse draft overwrites the incumbent; success uses ordinary re-tokenization and repeated canonical verification. |
| Diffusion repair runtime | Warm-start from the incumbent, corrupt a small selected span, and preserve the surrounding tokens. | Protected tokens remain fixed through sampling, re-noising, self-conditioning and final argmax; caches and cancellation remain isolated. |
| Fine discrete optimizer | Compute gradients at the incumbent's actual encoder tokens; rank word/span substitutions, insertions and deletions; batch-score the real resulting texts. | Accept measured canonical improvements with validity/quality checks; no cross-tokenizer surrogate is used to certify a result. |
| Evaluation | Freeze target panel, constraints, budgets and epsilon ladder `1e-2 -> 1e-3 -> 1e-4`. | Report success rate and best distance versus time/query count, including failures and quality. |

The runtime and discrete optimizer can develop against the same `Proposal(text, edit_span, provenance)` and canonical verifier. The controller is the integration owner. Numerical fixtures and evaluation can proceed independently.

The repair mask needs explicit runtime support. This DiffusionGemma sampler uses uniform random token corruption, not a generic MASK operation; a seeded canvas does not freeze its tokens. Simply setting guidance gradients to zero cannot protect text from subsequent native sampling or final argmax changes.

## Fine optimization

Use the encoder's exact normal tokenization to obtain input-vector gradients. Rank discrete replacements by the gradient dotted with the proposed embedding change, then decode and re-tokenize every candidate normally. Restrict proposals with word boundaries, frozen-model plausibility and declared output constraints. Gradients propose edits; exact scores decide acceptance. This adapts the gradient-ranked flip/beam idea in [HotFlip](https://aclanthology.org/P18-2006.pdf).

Keep a small beam and test combinations of disjoint edits: two individually coarse coordinate changes may cancel more precisely. Measure each edit's actual two-dimensional response, propose combinations using a local linear model, and verify the combined text because encoder interactions are nonlinear. Retain the best incumbent while allowing bounded exploratory repairs when local search stalls.

Shrink the edit/trust region as error falls. Replace constant normalized steps with a calibrated adaptive step; a damped two-output Jacobian solve is one proposal, but only if predicted changes correlate with actual canonical changes. A better optimizer of the wrong surrogate still misses the target. Use fluency and validity as constraints instead of assuming a fixed weighted KL objective has its minimum at zero map error.

Trajectory refinement has precedent in [PG-DLM](https://arxiv.org/html/2507.08390v5), but its masked-model implementation is not a drop-in solver for this runtime or a guarantee of epsilon accuracy.

## What is feasible to claim

A CPU audit observed maximum canonical coordinate drift `1.087e-7`, roughly 920 times below epsilon. Numerical precision therefore appears adequate in this environment. Forty sampled edits moved points by `0.00137-0.05398`; this motivates finer combinatorial search, but is not a universal edit-resolution floor. Case/spacing variants can have identical encoder tokens and must not count as novel generation. [Measurements](experiments/epsilon-geometry.json) and the reproduction script below retain the evidence.

First use ten hidden witness targets satisfying the same output policy, followed by dense/interpolated and sparse targets whose reachability is unknown. The current short map corpus is neither held out nor a witness under the 60-100-word policy. A witness proves feasibility only for its own constraints; excluding its text as a valid output makes novelty a stronger, separate requirement. Do not give witness text to prompting, retrieval or initialization.

Arbitrary clicks need not have a coherent allowed paragraph within epsilon. Use geometric impossibility diagnostics where justified; otherwise report `stalled` or `budget_exhausted`, not `unreachable`. Successful termination requires a valid candidate to pass two canonical singleton checks at `d <= 1e-4`. Record achieved distance and quality status even when the threshold is missed.

```bash
uv run --extra encoder python scripts/probe_canonical_precision.py --output .cache/epsilon-recheck.json
```
