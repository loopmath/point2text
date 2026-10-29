# Generation and guidance planning note

Planning only, 29 October 2026. No models downloaded, implementation performed, or repository files changed. This note predates the first demo implementation; see the project README for current status.

## Recommended split

Keep geometry, candidate generation, guidance, and final verification independent. Geometry supplies a frozen map ID, target coordinates, distance transform, tolerance, and optional separately versioned full-embedding regularizer. It must not require a reconstructed full embedding. The generator supplies a language prior; the verifier decides whether actual text reaches the target.

Start two interchangeable strategies behind the same candidate-result interface:

1. **Scoring-only:** generate independent complete candidates, decode, run the ordinary text encoder, and retain the best valid candidate. Under a deadline, return the best verified candidate so far. Add neighbor-conditioned prompts or revision/remasking as separately named variants; give guidance the same prompt information. Best-of-N is the minimum useful baseline, and works with unrelated tokenizers or text-only encoder APIs.
2. **Frozen-model guidance:** use exactly the same base generator/sampler and prompts, but modify detached token logits before decisions are committed. First try a soft-embedding loss with one update at 4–8 selected steps. Then compare straight-through and entropy-weighted estimators. These counts are proposed starting points, not literature-established optima. Generator parameters and denoising history stay outside autograd; encoder parameters stay frozen while input derivatives remain available.

PG-DLM is a later scoring-only refinement comparator, not a name for generic remasking. It requires reference trajectories, conditional resampling, and partial-reward estimation. Defer it until independent sampling establishes the baseline curve.

## Contracts to agree before implementation

**Generator ordinary interface:** `generate(request, rng, budget) → Candidate`. Request declares prompt, output-content policy, minimum/maximum length, sampler configuration and seed. Candidate includes generated IDs, decoded text, model/tokenizer/sampler revisions, completion/stop reason and timing. Expose optional resumable trajectory operations separately; a generator with only a text API still supports scoring-only search.

**Generator sampler hook:** `steer(StepView) → LogitUpdate`. StepView contains detached pre-sampling logits `[batch, canvas_positions, vocabulary]`, token IDs, editable/committed/prompt/padding masks, absolute output positions and block index, timestep, fixed previously generated context, token-ID semantics, sampling temperature, allowed-token constraints, and RNG state. LogitUpdate preserves shape/device/dtype and changes editable entries only. The native sampler owns masking, token selection, commitment, EOS and state/cache advancement. Declare whether entropy/confidence selection is recomputed from guided logits; guidance must act before both confidence selection and token sampling if those decisions are intended to reflect guidance. An identity hook must reproduce ordinary sampling with the same RNG. The hook cannot silently alter committed text or reuse stale conditioning caches.

**Encoder scoring interface:** `encode_text(texts, EncodingSpec) → embeddings`. EncodingSpec freezes tokenizer revision, prefixes/instructions, normalization, pooling, truncation and maximum length. Ordinary text encoding is the authoritative path for corpus, candidate ranking and final placement.

**Encoder soft adapter:** `embed_surrogate(distributions, fixed_context, masks, AdapterSpec) → embeddings, diagnostics`, differentiable to editable logits. AdapterSpec records generator-to-encoder ID mapping, special-token/position/attention rules, tokenization assumptions, top-k/tail policy and unsupported probability mass. It uses the same encoder body, pooling, instructions and output normalization as ordinary encoding. On compatible token sequences, one-hot surrogate inputs must match ordinary hard encoding numerically; on incompatible tokenizers, explicitly report failure of this equivalence and measured surrogate discrepancy. A one-token-to-many-token bridge changes sequence lengths and positions and needs its own design, not an undocumented lookup substitution.

The target loss can be `map_error + α full_embedding_regularizer + λ KL(q || q0)` on editable positions, with q0 the detached base proposal at the same temperature. Gradient clipping/trust regions, temperature and KL are independent settings. This is a proposed local surrogate objective; neither small soft loss nor small KL proves fluent or accurate final text.

## Verification and comparison

Decode complete proposed outputs using the generator's production decoding rules, extract exactly the content shown to the user, and re-tokenize with the real encoder. Score that text through the frozen map. Intermediate partial canvases can be diagnostically completed and scored, but never count mask placeholders or unfinished text as successful outputs. For multiple blocks, evaluate the full accumulated output under the encoder's declared length policy; silently scoring just the final canvas or a truncated prefix invites objective gaming.

Return text, measured embedding/map location, distance, tolerance success, validity/length status and elapsed time. Keep the best hard-verified candidate throughout sampling so a later surrogate improvement cannot erase a better real result. When nothing reaches tolerance, return the best result with its measured remaining error.

Compare best-of-N and guidance under equal end-to-end wall-clock budgets on the same device, model revisions, length limits, prompts, precision and memory cap. Include encoding, gradient updates, decoding/retokenization, retries and verification; synchronize asynchronous device work at timing boundaries. Report cold setup/load cost separately and warm request latency with identical shared-cache assumptions. Baselines may spend guidance's extra compute on more trajectories. Use paired target/seed panels, multiple repetitions, p50/p95 latency, peak memory, hard distance, tolerance-success rate, coherence, repetition and diversity. Plot verified quality/error against time; equal denoising counts or equal candidate counts alone are not fair. Do not select hyperparameters using the final evaluation targets.

## Approximation risks

- Tokenizer vocabulary sizes or family names prove no compatibility. Overlap must respect bytes, whitespace, normalization, contextual segmentation and special tokens. Zero-filling unmatched tokens discards meaning and provides no direct token-embedding gradient for those entries, although softmax coupling can still change their probabilities. Measure probability-weighted coverage as well as vocabulary coverage, including multilingual cases.
- Top-k mixtures reduce bridge/memory cost but exclude potential solutions; renormalization changes the loss and entropy. Log retained mass, sweep k, refresh support, and compare a short full-vocabulary reference where feasible. If using EntRGi's entropy weighting, compute entropy against the actual full proposal or label its approximation; replacing log vocabulary size with log k changes the estimator. FastGuide's candidate-pool k refers to token-position candidates, not this vocabulary truncation.
- Straight-through estimators use hard forward values and soft backward derivatives. They are biased local search directions, not exact derivatives of decoded-text reward. They do not solve tokenizer mismatch. Soft inputs can be off the encoder's training distribution; monitor the soft/hard score gap and actual discrete improvement.

## Staged feasibility gates

1. **Capability audit, no download needed:** inspect exact checkpoint/runtime/tokenizer metadata and sampler source. Confirm a pre-commit hook, encoder input autograd, one shared execution stack and memory budget. A text-only generator is sufficient to start the baseline; lack of a hook blocks only guidance.
2. **Uncontrolled and scoring-only runs:** once execution is separately authorized, benchmark short ordinary outputs and hard encoding. Establish the best-of-N error/quality/time curve on reachable held-out-text targets, nearby interpolations, sparse regions and outside-support targets. This separates generation/geometry difficulty from gradient difficulty.
3. **Adapter correctness:** verify hard-encoding parity on compatible IDs, zero/identity-hook parity, finite input gradients, unchanged weights, special tokens, pooling, positions and sequence limits. Check numerical derivatives only for the smooth surrogate; do not expect finite differences of hard sampling to validate STE. Quantify bridge discrepancy before using mismatched tokenizers.
4. **Local steering signal:** with short canvases, test whether one small update improves decoded sampled completions more often than no guidance across seeds. Record soft-loss change and hard change separately. If soft gains do not transfer, investigate bridge/estimator behavior before scaling compute.
5. **Budgeted ablation:** compare soft, STE and EntRGi-style estimators; guidance schedule; KL/step size; top-k and paired seeds. Continue only for a better hard-error/coherence/latency frontier. Add PG-DLM or FastGuide-style acceleration after this gate; adaptation to a new sampler is a separate hypothesis.

## Checked primary evidence and discrepancies

**EntRGi v2, 12 May 2026:** the brief correctly describes frozen inference-time guidance separately from RGRL post-training. The method weights hard tokens more at high predictive entropy and soft embeddings more at low entropy, using a stop-gradient correction. Appendix C.6 confirms zero-filled unmatched embeddings and reports 45–55% vocabulary mismatch for LLaDA versus Qwen3/Llama-3. That reported success is evidence for particular reward tasks, not exact retokenization or proof for sentence-embedding map losses. Source: https://arxiv.org/html/2602.05000v2 ; authors' implementation: https://github.com/atutej/entrgi-rgrl .

**PG-DLM v5, 6 August 2026:** the brief accurately identifies full-trajectory particle Gibbs refinement without retraining. Algorithm 1 samples the next reference trajectory by reward weights; merely preserving the best completion is not equivalent. Its partial-reward estimates can require additional model completions and belong in the budget. Its reward-weighted sampling guarantees do not guarantee an arbitrary click is reachable. Source: https://arxiv.org/html/2507.08390v5 ; authors' code: https://github.com/MhDang/pg-dlm .

**FastGuide v1, 28 September 2026:** the brief's main 2.8–4.4× speedup is supported for A5000 experiments with 128-token outputs and particular reward benchmarks. It combines guidance reuse, recomputation, and confidence deferral; simple gradient caching alone is not the full method. Appendix D.2 additionally reports 4.9× for an 8B reward model placed on a second GPU, so 4.4× is the main comparison headline, not a universal maximum. No Mac or arbitrary embedding-loss speedup follows. Source: https://arxiv.org/html/2609.36202v1 .

**DiffusionGemma model card:** confirms a cached prompt encoder and bidirectional diffusion canvas decoder, 25.2B total/3.8B active parameters and 256-token canvases. Completed canvases are added to context, so block boundaries and irreversible previous content must be explicit in the hook contract. This establishes architecture, not availability of a ready-made gradient hook. Source: https://ai.google.dev/gemma/docs/diffusiongemma/model_card .

No fatal citation mismatch was found in the three central references. The principal gap remains transfer: none establishes the complete interchangeable-encoder, arbitrary-click, frozen-model, interactive-Mac system. All performance figures here are source reports, not reproduced measurements.
