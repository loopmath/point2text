# Frozen-Encoder Guidance for Diffusion Text Generation on a Semantic Map

@authors point2text contributors - loopmath
@date Experimental technical report - 29 October 2026

This implementation steers a frozen local diffusion language model toward a clicked point on a fixed embedding map. It adds encoder-input gradients inside denoising while retaining the outer generate-and-verify search. Real probes confirm that guidance changes native sampling, decoded text, and its measured location. They do not establish a general quality or efficiency advantage.

## 1  Target and native runtime

The map contains 100 authored technical paragraphs. A pinned all-MiniLM-L6-v2 encoder [1] produces normalized 384-dimensional vectors; ordinary unwhitened PCA supplies a fixed mean mu and two orthonormal directions V. Canonical scoring uses the normal tokenizer, attention-mask mean pooling, float32 CPU execution, and L2 normalization. Inputs exceeding 256 encoder tokens, including special tokens, are rejected.

$$ \pi(e)=V^{\mathsf T}(e-\mu), \qquad d(x,y^*)=\|\pi(E(x))-y^*\|_2. $$

The generator is the four-bit mlx-community/diffusiongemma-26B-A4B-it-4bit checkpoint, pinned to revision a7a81407613811e8ba63af92ac0d852b809e191f. DiffusionGemma [2] runs through MLX-VLM 0.7.11 [3]. Each trajectory uses a 128-token canvas, vocabulary size 262,144, and a 32-pass maximum. A native prompt KV cache supplies conditioning throughout that trajectory.

The entropy-bound sampler can revise the whole canvas at each pass. Its confidence/stability rule can stop before the pass budget. The hook receives detached, temperature-adjusted native logits before sampling. Its returned logits drive sampling, entropy-based acceptance, self-conditioning, and stopping decisions. Consequently, modifying logits can alter both token values and the subsequent trajectory. No backward graph passes through the generator.

## 2  Sparse encoder surrogate

At a guided pass, the hook selects 32 vocabulary entries per position on the native device. Only compact logits, IDs, masks, and retained probability mass cross to CPU. A token bridge decodes generator tokens in isolation, re-tokenizes their surfaces with MiniLM, and averages the resulting encoder input vectors. Unsupported or empty surfaces are omitted; mixtures are renormalized over represented surfaces. Encoder CLS/SEP tokens, attention masks, mean pooling, and output normalization remain explicit.

This bridge is approximate: isolated surfaces discard contextual segmentation, multiple encoder tokens collapse into one position, and top-k selection removes alternatives. Final EOS clipping and ordinary re-tokenization can also change the scored input. The surrogate is not the canonical text encoder evaluated on a differentiable string.

On fixed selected supports, q_i is the conditional softmax distribution and q_i^0 is the original native proposal. For editable positions M, the loss is:

$$ L(z)=\|\pi(E_{\rm soft}(z))-y^*\|_2^2+\frac{\lambda}{|M|}\sum_{i\in M}\mathrm{KL}(q_i\|q_i^0). $$

The update count is configurable from 1 to 16 per pass (default two), retaining the same reference proposal. The first KL gradient is zero because q_i initially equals q_i^0; the penalty becomes active after movement. The compact gradient is globally L2-normalized, direction-clipped, and scaled. A trust bound clamps cumulative per-logit changes relative to the original proposal. Fixed positions remain unchanged. Zero strength returns the original native tensor.

<!-- page -->

## 3  Outer search and hard verification

The outer loop remains sequential best-of-N generation. Four nearest corpus paragraphs provide the same context for each candidate; seeds differ. Within a trajectory, selected passes additionally perform the encoder-guided updates:

```text
for each candidate seed:
  prefill native prompt cache; initialize canvas
  for each executed denoising pass:
    z0 <- frozen generator logits
    z <- m sparse encoder-loss updates if scheduled
         otherwise z0
    advance native sampler using z; stop if converged
  x <- decode final tokens
  skip empty or duplicate x
  e <- canonical_encode_text(x)
  publish (x, pi(e), distance(pi(e), target))
return the published text with smallest hard distance
```

Only normally re-tokenized final text is ranked. Traces retain losses, applied logit changes, executed passes, and hard coordinates. Cancellation preserves verified candidates.

## 4  What the probes establish

On an M4 Max with 128 GiB memory, paired probes used the same corpus target, neighborhood context, seed 42, checkpoint, and native generation budget.

| Profile | Canonical distance | Observed intervention |
| --- | --- | --- |
| Unguided | 0.07483 | Nine native passes |
| Strong late guidance | 0.09050 | Four guided passes; text changed |
| Strong early guidance | 0.02411 | Thirteen guided passes; text changed |

Early guidance ran at all 13 executed passes, with two encoder updates each and a maximum native logit change of approximately four. Late guidance used a tighter bound and began at pass six. These are different intervention profiles on one development target, not a held-out benefit benchmark. The worsening case demonstrates why reducing surrogate loss cannot substitute for canonical verification. No blind factuality or coherence assessment was performed.

## 5  Computational cost and next gate

Candidates run sequentially, each requiring full native generator passes. Top-k sparsity reduces the reward bridge, not the generator vocabulary computation. The strong early profile adds 26 encoder forward/backward pairs, bridging, transfers, and synchronization. Increasing inner updates adds encoder work, not generator passes. Prompt caching operates within each trajectory; prefill is repeated across candidates. Decoder graph compilation is disabled with compile_graph=False.

Cold/warm timing differences do not demonstrate acceleration. The next meaningful gate is an equal end-to-end wall-clock comparison against spending the same budget on more unguided diffusion samples, across multiple targets. Report hard error together with coherence, factuality, diversity, memory, and latency. Entropy-aware hard/soft estimators [4] are a separate comparison, not an implemented guarantee. Low two-dimensional distance alone does not establish semantic equivalence or useful text.

## References

[1] Sentence Transformers. all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[2] Google. DiffusionGemma model card. https://ai.google.dev/gemma/docs/diffusiongemma/model_card

[3] MLX-VLM. Native inference implementation; version 0.7.11 used here. https://github.com/Blaizzy/mlx-vlm

[4] Tejaswi et al. Entropy Aware Reward Guidance for Diffusion Language Model Alignment. v2, 2026. https://arxiv.org/abs/2602.05000
