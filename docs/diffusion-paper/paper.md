# Frozen-Encoder Guidance for Diffusion Text Generation on a Semantic Map

@authors point2text contributors - loopmath
@date Experimental technical report - 29 October 2026

This implementation steers a frozen local diffusion language model toward a clicked point on a fixed embedding map. It adds encoder-input gradients inside denoising while retaining the outer generate-and-verify search. Real probes confirm that guidance changes native sampling, decoded text, and its measured location. They do not establish a general quality or efficiency advantage.

## 1  Target and native runtime

The fixed map uses 100 authored paragraphs and normalized 384-dimensional all-MiniLM-L6-v2 embeddings [1]. Canonical scoring means ordinary tokenization, frozen encoder inference, attention-mask mean pooling, L2 normalization, and saved unwhitened PCA; inputs over 256 encoder tokens are rejected.

$$ \pi(e)=V^{\mathsf T}(e-\mu), \qquad d(x,y^*)=\|\pi(E(x))-y^*\|_2. $$

The pinned four-bit DiffusionGemma checkpoint [2] runs through MLX-VLM 0.7.11 [3]: 128 canvas positions, 262,144 vocabulary entries, and at most 32 passes. Prompt conditioning is cached within each trajectory; the exact checkpoint revision is recorded in run artifacts.

The entropy-bound sampler can revise the whole canvas and stop early on confidence/stability. Its temperature-adjusted logits are detached before guidance. Revised logits drive sampling, acceptance, self-conditioning, and stopping; the generator receives no backward pass.

## 2  Sparse encoder surrogate

At a guided pass, z0 denotes the native clean-token logits. Each position's top-32 support S_i stays fixed during that pass's inner updates; only its selected logits z are optimized. The conditional distribution is q_i = softmax(z_i / tau).

For generator token v, isolated decoding and encoder re-tokenization produce token list T_v. The bridge averages MiniLM's input lookup vectors W_E, not sentence embeddings. With validity mask a_v, represented rows renormalize probabilities:

$$ B_v=\frac{1}{|T_v|}\sum_{k\in T_v}W_E[k], \qquad u_i=\frac{\sum_{v\in S_i}q_{iv}a_vB_v}{\sum_{v\in S_i}q_{iv}a_v}. $$

The sequence [CLS, u_1, ..., u_n, SEP] passes through the frozen MiniLM body, masked mean pooling, L2 normalization, and PCA. Unrepresented rows use fixed-token fallbacks and masks. The editable-position loss is:

$$ L(z)=\|\pi(E_{\rm soft}(z))-y^*\|_2^2+\frac{\lambda}{|M|}\sum_{i\in M}\mathrm{KL}(q_i\|q_i^0). $$

The original proposal q_i^0 is fixed across inner updates. Gradients follow map loss through pooling, MiniLM inputs, mixtures, and softmax into selected logits; neither hard decoding nor the generator is differentiated. The first KL gradient is zero at q_i = q_i^0.

Soft and canonical inputs differ concretely: two encoder subwords become one averaged vector at one position here; normal encoding retains both positions and their contextual attention. Joining decoded surfaces can also change tokenization. Top-k omissions and final EOS clipping add discrepancies. Thus lower soft loss need not lower completed-text distance.

<!-- page -->

## 3  Outer search and hard verification

For each of N independent seeds, reuse the four-neighbor prompt:

```text
canvas <- initialize(seed); self_condition <- none
cache <- prefill(prompt)
for native pass t = 1,...,T:
  z0 <- frozen_generator(canvas, cache, self_condition)
  z <- detached_copy(z0)
  if guidance_scheduled(t):
    S <- top32(z0); q0 <- softmax(z0[S] / tau)
    repeat m times:
      g <- gradient_z[S] L(z[S]; S, q0, target)
      z[S] <- bounded_gradient_update(z[S], g, z0[S])
  canvas, self_condition <- native_sampler(z)
  stop if native confidence/stability criterion holds
x <- decode(final_tokens)
d <- canonical_distance(x, target)
```

Updates normalize and clip the global gradient direction, take a scaled descent step, then cap cumulative changes from z0. Fixed m (default two) does not test convergence. Canvas and self-conditioning carry influence forward; each pass generates fresh z0, not persistent optimized logits.

Native confidence/stability stopping does not test target epsilon. Canonical scoring of completed, normally re-tokenized text selects the best of N; it never feeds a revision or next candidate. No epsilon = 1e-4 stopping rule is implemented.

## 4  What the probes establish

On an M4 Max with 128 GiB memory, paired probes used the same corpus target, neighborhood context, seed 42, checkpoint, and native generation budget.

| Profile | Canonical distance | Observed intervention |
| --- | --- | --- |
| Unguided | 0.07483 | Nine native passes |
| Strong late guidance | 0.09050 | Four guided passes; text changed |
| Strong early guidance | 0.02411 | Thirteen guided passes; text changed |

These profiles differ on one development target. Early guidance used two updates at all 13 passes with maximum logit change four; late guidance began at pass six with a tighter bound. General benefit and text quality were not established.

## 5  Computational cost and next gate

Candidates run sequentially and repeat prompt prefill. Top-k reduces bridge work, not generator forward cost. The early profile adds 26 encoder forward/backward pairs plus synchronization; more inner updates add encoder work. Decoder compilation is disabled.

Cold/warm differences show no speedup. Compare guidance with more unguided samples under equal wall time across targets, measuring canonical error, quality, diversity, and memory. Hard/soft estimators [4] remain a future comparison. Low map distance alone is not semantic equivalence.

## References

[1] Sentence Transformers. all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[2] Google. DiffusionGemma model card. https://ai.google.dev/gemma/docs/diffusiongemma/model_card

[3] MLX-VLM. Native inference implementation; version 0.7.11 used here. https://github.com/Blaizzy/mlx-vlm

[4] Tejaswi et al. Entropy Aware Reward Guidance for Diffusion Language Model Alignment. v2, 2026. https://arxiv.org/abs/2602.05000
