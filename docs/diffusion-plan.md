# Native diffusion integration and acceptance plan

**Current state:** the combined native runtime, encoder-gradient bridge, hard-text verification, browser path, and per-pass update control work in [draft PR #1](https://github.com/loopmath/point2text/pull/1). The integration gate below passed real local runs; the branch remains separate for review. The next open gate is measured quality and efficiency under equal time budgets.

The immediate deliverable is a local Mac run in which native DiffusionGemma denoising invokes frozen-encoder guidance, produces real decoded research prose, and reports its measured position on the existing fixed map. The authorized window began **2026-10-29 17:28:31 UTC** and ends **18:28:31 UTC**. This is the implementation deadline; it is not a promise of a particular accuracy, speed, or quality improvement.

## Independent lanes and ownership

All three implementation lanes use GPT-6.1-sol with xhigh reasoning. Each lane commits its owned files in its worktree; the integration owner cherry-picks completed commits. Avoid simultaneous edits to contracts, dependencies, or integration entry points.

| Owner | Worktree / branch | Responsibility and current evidence |
|---|---|---|
| Runtime lane | `work/diffusion-runtime` / `diffusion/runtime` | Native MLX checkpoint loading, sampler hook before token commitment, decoding, cancellation, runtime trace. Implemented; lane reports a real unguided native generation. |
| Encoder/bridge lane | `work/diffusion-encoder` / `diffusion/encoder-bridge` | Frozen MiniLM input-vector gradients, PCA objective, sparse tokenizer bridge, encoder/bridge tests. Implemented in `5d329784837ba0a04170cc4772d4498b2d3b61df`; nine small tests and a real cached-MiniLM parity check passed. An actual DiffusionGemma-tokenizer smoke produced finite gradients and reduced its soft objective. |
| Guidance/search lane | `work/diffusion-guidance` / `diffusion/guidance` | Sparse top-k selection/scatter, update schedule, proposal reference, guided/unguided selection, final candidate scoring. Implemented; lane reports unit tests and native MLX sparse-scatter checks. |
| Integration owner | `work/diffusion` / `diffusion/integration` | Shared contracts, CLI/web wiring, dependencies, combined real runs, evidence logs, packaging, user-facing status, and promotion into `main`. Combined guided acceptance remains the immediate gate. |

The main checkout is `point2text` on `main`. Keep experimental integration isolated until its checks pass. Checkpoint weights and generated runtime caches remain outside tracked source; the public package contains source, tests, documentation, and the original conceptual demo corpus.

## Contracts already available

`DiffusionRuntime.generate(request, hook=..., cancel_event=...)` owns native tensors and denoising. `DiffusionStep` supplies logits, current token IDs, editable positions, step information, and the actual generator tokenizer. The hook returns revised logits before the sampler commits token choices. Generator autograd is unnecessary.

The runtime reports a **128-position decoder canvas**, with native tensors batched as logits `[1,128,V]` and token IDs/editable masks `[1,128]`. Guidance selects and transfers only top-k entries to CPU. The reward consumes logits/IDs `[n,k]`, current IDs/editable masks `[n]`, target, temperature, and KL weight; it returns scalar loss, gradients `[n,k]`, and diagnostics. Noneditable positions receive zero gradient.

`TorchEmbeddingReward(encoder, bundle)` uses the current pinned CPU MiniLM transformer, frozen parameters, fixed CLS/SEP, mask-aware mean pooling, L2 normalization, and the saved PCA arrays. The encoder budget is 256 positions, leaving **254 surrogate canvas positions** after fixed CLS/SEP; the current 128-position canvas fits. This does not guarantee that the final decoded text fits: normal re-tokenization can expand it, and over-budget text must be reported as invalid rather than scored through a truncated prefix.

`MapBundle.score(text,target)` remains the authoritative final score. Decode the actual output, apply the runtime's documented structural-header removal, and score that exact resulting string. The surrogate's soft location is a diagnostic, never the measured final location.

The bridge decodes only selected unique generator IDs, tokenizes each surface with the encoder without special tokens, and averages the resulting encoder word embeddings. It caches these detached vectors. Empty/whitespace and structural special surfaces are omitted; mask placeholders use the encoder MASK embedding. Isolated decoding, compressed one-to-many tokens, word boundaries, and lexical channel headers make this an **approximation**. Exact parity is established only for ordinary encoder-token inputs, not for the mismatched tokenizer bridge.

The KL reference is the original top-k proposal, held fixed across inner updates. At the first inner update, the proposal and reference match, so KL and its gradient are zero. KL becomes active after logits move away from that reference; a single-inner-update configuration receives no KL restoring force. Record the update count and do not claim regularization merely because a nonzero KL coefficient is configured.

## Immediate combined acceptance gate

The integration owner runs the following sequence while lane owners address only failures in their own files:

1. Import and construct the runtime, canonical encoder, frozen map, reward, and hook in one process. Confirm model revisions, map identity, array shapes, canvas length, and sampler-hook placement before a long run.
2. Select a technical/research target on the existing map. Run one real **unguided** native trajectory and one real **guided** native trajectory with the same prompt, seed, canvas, denoising settings, and target. The guided run must execute the encoder backward path and alter editable native logits. This is a wiring check; a single pair cannot establish an accuracy benefit.
3. Save both decoded paragraphs and independently re-embed each exact final text through `MapBundle.score`. Confirm finite coordinates and distance, and preserve invalid outputs or scoring errors as failures. Do not substitute a retrieved paragraph, a conventional generator, or a soft representation for the native result.
4. Exercise the user entry point against that same integrated path. Show target, actual final point, distance, elapsed time, and a clear failure/cancellation state. Run the focused runtime, bridge, guidance, and integration checks, then verify package installation and imports include the new modules.

Store a compact machine-readable run record with actual timestamps; checkpoint revisions; encoder/map IDs; prompt and seed; target; canvas and denoising settings; guided-step/inner-update count; top-k, temperature, step size and KL weight; native forward count; reward calls; elapsed time and available peak-memory measurement; exact final text; canonical coordinates/distance; and errors. Preserve per-guidance diagnostics: soft location/loss, gradient norm, bridge coverage, unknown/special/empty/one-to-many mass, and top-k retained mass. Retained mass must come from the full native proposal; selected logits alone cannot determine it.

Passing this gate means the combined native diffusion-and-guidance path is operational and its outputs are verifiably scored. Record the observed result, including cases where guidance worsens hard-text distance or quality. If a gate fails, the log should identify the failing boundary and its reproducible inputs so the owning lane can fix it without changing unrelated components.

## Next scientific gate

After the combined path works, compare scoring-only/unguided and guided strategies under **equal measured wall-clock budgets**, with paired targets and seeds. Freeze the encoder, map, prompt policy, output-length policy, target strata, and settings before comparing methods. Use decoded-text distance, coherence/task fit, repetition, diversity, validity, and latency together; inspect the gap between soft progress and hard-text progress. Report per-target behavior and failures rather than only the best trajectory.

Keep the authored 100-paragraph corpus labeled as a demo, not a held-out research benchmark. A later reconstruction evaluation needs hidden witness texts excluded from retrieval and prompting, with its map and priors fit on the designated training split. Do not infer semantic success from PCA distance alone or claim a quantitative improvement before measuring it.

Residual completion, conditional residual priors, alternate encoders, broader hardware coverage, and more elaborate guidance schedules are separate experiments. They remain outside the critical path to the working native demo. The immediate map-space objective and canonical hard-text verification are sufficient to finish the current integration.
