# Experimental diffusion path

This branch implements real local DiffusionGemma generation, sparse map-gradient steering before native sampler decisions, and canonical decoded-text scoring. It stays isolated from the baseline on `main` until reviewed. It is an operational prototype; useful guidance across targets is not yet established.

## Install and run

Apple silicon and sufficient unified memory are required for the pinned MLX runtime. The reference machine is an M4 Max with 128 GiB; the measured MLX peak for the native runtime probe was about 17.48 GB, excluding a claim about total process/OS memory. Smaller Mac tiers are untested.

```bash
uv sync --python 3.12 --extra encoder --extra diffusion --extra dev
uv run --extra encoder --extra diffusion point2text-diffusion --paired --target-record t01-03
```

First use downloads the public community checkpoint. The runtime pins `mlx-vlm==0.7.11` and `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision `a7a81407613811e8ba63af92ac0d852b809e191f`. It produces a real 128-position diffusion canvas by default. The ordinary MiniLM encoder and map remain the same as the baseline.

`--paired` runs the identical request with guidance off/on, prints both actual paragraphs and measured positions, and saves an experiment record. It is a plumbing probe, not an equal-wall-clock benchmark. Use `--unguided` for native diffusion without a gradient hook. `--target X Y` chooses coordinates; `--target-record ID` uses a corpus paragraph's projected point.

## Browser and tailnet

```bash
uv run --extra encoder --extra diffusion point2text-diffusion --serve --tailnet --port 8478
```

The same browser interface/API now runs the diffusion strategy. The header says which backend is active. Canonical verified candidates, remaining distance, cancellation, and run records work as before. The strategy's native/guidance trace is included in the run JSON. Existing baseline and Tailscale Serve routes are independent.

The browser's **Encoder updates / pass** control sets 1, 2, 4, 8, or 16 inner refinements for each guided native pass (default 2). This adds encoder forward/backward work against the same generator proposal; it does not add diffusion-model passes. The API accepts `guidance_updates` in guided run requests and validates 1-16. Completed candidates show their actual update and native-pass counts. Counts can be lower than the scheduled maximum because native stopping is adaptive. More inner updates can saturate the trust bound or worsen final text; the ordinary encoder score remains decisive.

The exercised, stronger intervention profile is:

```bash
uv run --extra encoder --extra diffusion point2text-diffusion --serve --tailnet --port 8478 \
  --start-fraction 0 --guided-steps 32 --inner-updates 2 \
  --strength 10 --step-size 20 --max-delta 4
```

This profile was selected to visibly exercise the intervention in a development probe, not as a validated optimum. The CLI's more conservative defaults use strength 1, step size 4, max delta 2, and eight scheduled guided steps. Native early stopping may end before all scheduled steps. Start at fraction zero to make guidance available before early convergence; the end fraction defaults to 0.9.

## What the algorithm does

1. The frozen generator predicts native canvas logits. Temperature shaping remains native.
2. The hook selects top-k logits/IDs per position (32 by default); only compact arrays cross to CPU.
3. The bridge decodes each selected generator token in isolation, tokenizes its surface with MiniLM, and averages the corresponding frozen input embeddings. It caches only requested unique IDs. This loses contextual segmentation and one-to-many sequence length; it is an explicit approximation.
4. A differentiable MiniLM body includes fixed CLS/SEP, mask-aware mean pooling, normalization, and the saved PCA projection. Map error and conditional top-k KL produce gradients with respect to selected logits only. Model parameters stay frozen.
5. Bounded gradient updates return to native MLX logits before sampling, confidence/entropy decisions, and self-conditioning. Noneditable positions are preserved. The current entropy-bound sampler can revisit its whole canvas, so its editable mask is all true.
6. Final text is decoded and structurally cleaned, then re-tokenized normally and scored through the ordinary encoder. Only this canonical score ranks candidates.

Gradients are L2-normalized by default, clipped, and limited by a per-logit trust delta. The KL reference is held fixed across inner updates. Its first-update gradient is zero at the original proposal; with two or more inner updates it can exert a restoring force. Top-k retained mass, bridge unknown/special/empty/one-to-many mass, soft loss, raw gradient norm, actual native delta, changed-logit count, native passes, peak MLX memory, hard coordinates, and timings are recorded.

Exact hard-token parity is tested for MiniLM's own tokenizer. It does not make the cross-tokenizer bridge exact. Soft losses are diagnostics and never replace final text scoring. The default 128-position surrogate fits within MiniLM's 256-position budget plus CLS/SEP; final normal re-tokenization can still expand and exceed the budget, in which case scoring rejects the text.

## Observed development probes

All three probes below used the same diffusion model, target `t01-03`, seed 42, prompt context, and 128-token/32-step request. They are engineering observations on one target, not held-out or equal-time effectiveness results.

| Intervention | Unguided hard distance | Guided hard distance | Observation |
| --- | --- | --- | --- |
| Sparse late, strength 1 | 0.07483 | 0.07483 | Native logits and pass count changed; final text stayed identical |
| Strong late, max delta 1 | 0.07483 | 0.09050 | Final text changed; distance worsened |
| Strong early, max delta 4 | 0.07483 | 0.02411 | All 13 executed passes guided; final text changed and distance decreased |

The early/strong probe changed thousands of selected native logits per step and produced finite canonical scores. These observations establish an end-to-end gradient-to-sampler-to-text path. They do not establish a general benefit, a speedup, or acceptable quality across domains. Cold and warm model load/cache costs differed and must not be used as a performance comparison.

## Tests and next work

```bash
uv run --extra encoder --extra diffusion --extra dev pytest
POINT2TEXT_REAL_ENCODER_TEST=1 uv run --extra encoder --extra dev pytest tests/test_diffusion_encoder.py
```

Fixture tests need no checkpoint. Native sampler tests use MLX but tiny tensors; optional real encoder parity uses the cached/downloaded pinned MiniLM. The native smoke script is `tests/smoke_diffusion_runtime.py`. It checks real native generation, seeded identity, native sampler parity, and a controlled pre-commit intervention.

See [orthogonal work plan](diffusion-plan.md) for ownership and acceptance gates. Next measure paired target/seed panels under equal end-to-end time budgets, with quality, diversity, invalidity, and hard map error together. Residual priors and broader encoders/hardware remain separate directions.
