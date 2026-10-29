# point2text: Local Text Generation Toward a Point in Semantic Space

@authors point2text contributors - loopmath
@date Technical report - 29 October 2026

A click on an embedding map specifies a generation objective, not a complete meaning. point2text generates technical paragraphs, re-embeds the actual text, and displays its measured location. The implemented system has a generate-and-verify baseline and a separate operational diffusion-guidance branch. Both keep model weights frozen and judge outputs through the same canonical encoder and fixed projection.

## 1  Canonical embeddings and map objective

The demo contains 100 authored conceptual paragraphs across ten technical topics. They are context examples, not published abstracts or a held-out benchmark. The encoder is all-MiniLM-L6-v2 [1,2], pinned to revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Corpus and generated text share its ordinary tokenizer, attention-mask mean pooling, float32 CPU execution, and L2 normalization. Inputs exceeding 256 tokens, including special tokens, are rejected before scoring.

$$ e_i=E(x_i)\in\mathbb{R}^{384}, \qquad \|e_i\|_2=1. $$

For corpus C, ordinary unwhitened PCA supplies two orthonormal directions V. The fitted mean and directions remain fixed when projecting new text:

$$ \mu=\frac{1}{|C|}\sum_i e_i, \quad \pi(e)=V^{\mathsf T}(e-\mu), \quad V^{\mathsf T}V=I_2. $$

For clicked coordinates y*, selection minimizes Euclidean distance among the usable candidates A actually generated:

$$ d(x,y^*)=\|\pi(E(x))-y^*\|_2, \qquad \widehat{x}=\operatorname*{argmin}_{x\in A}d(x,y^*). $$

The two components retain approximately 11.74% of corpus embedding variance. Nearby map positions can therefore conceal substantial differences in omitted dimensions, factual content, or quality. Arbitrary clicks need not be attainable by fluent text. Changing the encoder requires rebuilding the map.

## 2  Generate and verify

The baseline uses an installed autoregressive model through Ollama [3]. The four nearest corpus paragraphs supply common context for B independent attempts, with B between one and six. Each uses a distinct seed, temperature 0.8, top-p 0.9, and a bounded token budget. The prompt requests an original, accurate 60–100 word paragraph; factuality and originality are not automatically verified.

```text
context <- four nearest corpus paragraphs to target
for b = 0,...,B-1:
  x <- decode(frozen_generator(context, seed+b))
  skip empty or duplicate x
  e <- canonical_encode_text(x)  # ordinary tokenizer
  publish (x, pi(e), distance(pi(e), target))
return published candidate with smallest distance
```

Attempts bound computation, not accepted-result count. Incomplete transport and length-limited generation raise errors. Cancellation retains already verified candidates. Results record text, coordinates, distance, seeds, and map/model identities.

<!-- page -->

## 3  Initial engineering validation

On an M4 Max with 128 GiB memory, CPU encoding and local gemma4:12b, one warm browser search produced:

| Observation | Result |
| --- | --- |
| Completed paragraphs | 3 in 11.70 s |
| Map distances | 0.05324, 0.08512, 0.14634 |
| Independent canonical re-scoring | All three coordinates and distances matched |

The target was a corpus paragraph's position, and neighboring context was available to generation. This validates the interaction; it establishes neither held-out reconstruction performance nor a latency distribution.

## 4  Scope and extension

PCA distance is not full semantic equivalence. Evaluation must measure hard error with coherence, factuality, diversity, latency, and memory across target panels. For held-out reconstruction tests, witness text must be excluded from retrieval and prompting. Compare strategies under equal end-to-end time budgets rather than equal candidate counts.

A separate diffusion path adds encoder-gradient updates to token logits during denoising; final text still uses the same canonical scoring rule. Its sampler hook, tokenizer surrogate, computational cost, and development probes are described in the separate diffusion report. General benefit remains unestablished.

## References

[1] Sentence Transformers. all-MiniLM-L6-v2 model card. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[2] Reimers and Gurevych. Sentence-BERT. EMNLP, 2019. https://arxiv.org/abs/1908.10084

[3] Ollama. Generate API. https://docs.ollama.com/api/generate
