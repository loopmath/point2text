# Text Optimization on a Frozen Semantic Map

## 1  Objective and system boundary

point2text turns a click on a two-dimensional semantic map into a text-generation objective. It retrieves nearby example paragraphs, generates new text with frozen local models, and measures the actual decoded paragraph with the same frozen encoder and projection used to build the map. Later stages use measured feedback to improve that paragraph. A map coordinate constrains two numbers; it does not specify all scientific claims or guarantee useful prose.

The shipped corpus contains 100 authored conceptual paragraphs across ten technical topics. These are context examples, not published abstracts or a benchmark of factuality. The default sentence encoder is `all-MiniLM-L6-v2` [1], pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Ordinary tokenization, float32 CPU inference, attention-mask mean pooling and L2 normalization produce a 384-dimensional vector. Overlength inputs are rejected rather than silently truncated; the default limit is 256 encoder tokens including special tokens.

$$ e(x)=\mathrm{normalize}\left(\frac{\sum_i a_i h_i(x)}{\sum_i a_i}\right)\in\mathbb{R}^{384}. $$

For corpus embeddings, the implementation centers the data, fits PCA by singular-value decomposition and saves two orthonormal component rows V. It fixes each component's sign deterministically and performs no whitening. Corpus embeddings, mean, components, encoder identity and corpus snapshot determine the immutable map fingerprint.

$$ \pi(e)=V(e-\mu),\qquad VV^{\mathsf{T}}=I_2. $$

For target y*, the canonical objective and reported distance are:

$$ d(x,y^*)=\|\pi(e(x))-y^*\|_2,\qquad \widehat{x}=\arg\min_{x\in A}d(x,y^*). $$

A is the set of eligible texts actually explored. The demo's two components retain approximately 11.74% of corpus embedding variance. Most variation is omitted, so proximity can coexist with different claims, poor grammar or misleading terminology. Even a known attainable target does not imply the search will recover its witness; arbitrary clicks need not be attainable by any fluent paragraph under the output constraints.

Measurements in this report use an Apple M4 Max with 128 GiB memory. First-use model loading, cached execution and judge calls have separate timing scopes.

<!-- column -->

[FIGURE:architecture]

@caption **Figure 1.** Implemented component boundaries. Proposal mechanisms may be approximate; published coordinates come from ordinary encoding of real text. Optional quality policy controls which measured texts may become incumbents.

### Baseline and diffusion paths

The original baseline calls an installed autoregressive Ollama model [2]. Four nearest corpus records supply topic context. Independent seeded attempts use a fixed prompt requesting one original 60-100 word paragraph; decoded results are deduplicated, canonically scored and ranked. This is generate-and-verify search: the encoder selects outputs but does not alter generation. A warm three-candidate development run took 11.70 seconds and produced distances 0.05324, 0.08512 and 0.14634; independent re-scoring matched all three. This validates the interaction, not a latency distribution.

The diffusion path adds two levels of control. Inside a native denoising pass, gradients through a frozen encoder alter selected generator logits. Outside that pass, a controller evaluates clean drafts, proposes canonical token edits, combines measured edits, or repairs a small span. A separate verifier decides whether actual text satisfies the mechanical, distance and optional quality policies.

The map and both main models remain frozen throughout. No generator fine-tuning, encoder fitting to targets, PCA refitting or learned inverse map occurs. The runtime, surrogate, controller, quality gate and storage are separable interfaces; combining them is an inference procedure, not end-to-end training. The report distinguishes working components from demonstrated optimization benefit and retains negative results.

<!-- page -->

## 2  What DiffusionGemma actually does

DiffusionGemma is Google's experimental discrete-diffusion language model built from Gemma 4's sparse mixture-of-experts backbone. The 26B-A4B designation denotes roughly 25.2 billion total parameters and 3.8 billion active parameters; routing uses eight of 128 experts plus a shared expert [3]. Its authors describe supervised bidirectional-denoising adaptation followed by reinforcement learning and sampler distillation [4]. point2text downloads frozen inference weights; it performs none of that training.

The model uses **uniform-state corruption**. A generation canvas begins with vocabulary tokens drawn uniformly, rather than a grid of MASK tokens. During inference, a shared backbone first uses causal attention to prefill prompt KV state. Denoising then uses bidirectional attention within the canvas, conditioned on that cached prompt. Uncertain positions can be re-noised and revised after earlier decisions. Self-conditioning carries a probability-weighted token-embedding representation of the previous prediction into the next pass. General DiffusionGemma inference can append completed blocks to the prompt cache and start another canvas [5]. This adapter implements text-only, single-canvas inference.

The audited stack is MLX-VLM 0.7.11 with MLX 0.32.4 and `mlx-community/diffusiongemma-26B-A4B-it-4bit`, revision `a7a81407613811e8ba63af92ac0d852b809e191f` [6,7]. Vocabulary size is 262,144. Early probes used 128 positions; refinement uses 192 to reduce sentence cutoffs. The maximum requested trajectory is 32 denoising passes, with native early stopping.

For token entropies sorted from smallest to largest, the native entropy-bound selector accepts the prefix satisfying the following rule (beta=0.1 in the checkpoint):

$$ \sum_{j=1}^{k}H_{(j)}-\max_{j\leq k}H_{(j)}\leq\beta. $$

Accepted positions receive samples from the proposed categorical distributions; remaining positions receive fresh uniform vocabulary noise. Native stopping requires stable predictions and mean entropy below 0.005; it does not evaluate the clicked target. The adapter preserves native temperature shaping, prompt caching, sampling, acceptance and self-conditioning. A fresh prompt cache is created for each trajectory; it is reused within that trajectory, not shared across independent candidates.

A hook receives detached, temperature-adjusted logits z0 of shape [1, L, 262144], current token IDs and an editable mask. Every downstream native decision consumes the hook's returned logits. There is no backward pass through MLX generator weights. Native generation is serialized because the MLX random-number generator is global; the hook and draft observer must not consume it. Compilation, multimodal inputs and multiple canvases are outside this adapter.

<!-- column -->

## 3  The sparse gradient bridge

Generator tokens and MiniLM tokens have incompatible vocabularies and boundaries. The adapter therefore builds an explicitly approximate bridge. For generator token v, it decodes that token in isolation, re-tokenizes its surface with MiniLM without special tokens, and averages the corresponding **input lookup vectors** W. These are not sentence embeddings.

$$ B_v=\frac{1}{|T_v|}\sum_{k\in T_v}W[k],\qquad q_{iv}=\mathrm{softmax}(z_i/\tau)_v. $$

At each guided native pass, S_i is the original top-32 support for position i. That support remains fixed across the pass's inner updates. Empty, whitespace-only and structural surfaces are excluded from the lexical mixture; represented probability is renormalized with validity indicator a_v:

$$ u_i=\frac{\sum_{v\in S_i}q_{iv}a_vB_v}{\sum_{v\in S_i}q_{iv}a_v}. $$

The sequence [CLS, u_1, ..., u_L, SEP] passes through the frozen MiniLM body, attention-mask mean pooling, normalization and saved PCA. Noneditable or unrepresented positions use fixed-token fallbacks and masks. The optimized loss is squared map error plus a divergence penalty over editable positions M:

$$ \mathcal{L}(z)=\|\pi(E_{\mathrm{soft}}(z))-y^*\|_2^2+\frac{\lambda}{|M|}\sum_{i\in M}\mathrm{KL}(q_i\|q_i^0). $$

The reference q_i^0 is fixed at the native proposal throughout the inner loop. Gradients flow from map loss through PCA, pooling, the frozen encoder body, vector mixtures and softmax into selected logits. Frozen weights can transmit input gradients without being updated. The generator computation and hard decoding remain outside this graph.

Each update normalizes/clips the gradient direction, takes a scaled descent step and bounds cumulative logit displacement from z0. m updates are a compute budget, not a convergence test. The first KL gradient is zero at q=q0; later updates can incur a restoring penalty. With clipped/normalized direction v, step scale eta and logit bound D, the selected-logit update is:

$$ z^{r+1}=z^0+\mathrm{clip}(z^r-\eta v-z^0,-D,D). $$

A new native pass produces fresh z0 rather than continuing the preceding pass's optimized logits.

The surrogate collapses several encoder subwords into one averaged vector and one position. Actual encoding retains their order, positions and contextual attention. Joining decoded surfaces can change tokenization; top-k support omits mass, and final EOS cleanup changes the scored text. Consequently, lower soft loss does not imply lower canonical distance. Ordinary decoded-text scoring remains the acceptance rule.

<!-- page -->

## 4  Canonical feedback and local repair

The controller closes the gap between a useful proposal signal and the actual objective. It observes clean greedy drafts every three native passes and at termination, applies output checks, and canonically measures eligible drafts. It can keep an earlier good draft even if a later canvas is worse. Observing a draft is passive unless the verifier confirms tolerance and requests a safe stop.

```text
best, beam <- verified initial text or coarse drafts
for bounded refinement round:
  atoms <- propose edits from each beam source
  measured <- canonical_score(actual atom texts)
  combos <- combine same-source measured deltas
  score actual combo texts; apply quality policy
  update beam and best only with eligible texts
  if stalled or scheduled:
    repair a small span with native diffusion
    canonical-score clean repair drafts and final
  if d(best) <= epsilon:
    repeat two fresh singleton encodings
    stop only if both distances <= epsilon
  stop on cancellation, budget or persistent stall
return best verified text and explicit stop reason
```

Eligibility requires configured word/token limits, finite coordinates, sentence-ending punctuation and no model headers, special tokens or detected draft repetition. The verifier deduplicates normalized text and canonical token sequences. These mechanical checks do not establish scientific validity.

### Edits at actual canonical token IDs

`CanonicalEditProposer` starts from the paragraph's real MiniLM IDs and exact character offsets. It differentiates both projected coordinates with respect to the canonical input vectors. For a proposed local vector change delta U, the Jacobian predicts a two-dimensional displacement:

$$ J_i=\frac{\partial\pi(E(U))}{\partial U_i},\quad \widehat{\Delta y}=\sum_i J_i\Delta U_i,\quad g_i=2J_i^{\mathsf{T}}(y-y^*). $$

These gradients rank whole-word edits and pairs with potentially cancelling displacements. Equal-length replacements use corresponding input-vector differences; changed lengths use a mean-vector approximation. This ordinary-token path avoids the generator-surface bridge at the source paragraph. Replacement changes, nonlinear attention and retokenization still make the predicted edit effect approximate. Every resulting paragraph is decoded, normally tokenized and canonically scored.

The native vocabulary screen corrupts one generator token while keeping surrounding tokens fixed, runs one denoiser forward and extracts supported whole-word alternatives. The production screen intersects its top 32 with curated grammatical families, protects technical collocations and permits only exact single-generator-token replacements. Unsupported spans yield no alternatives. It probes at most four spans per text and caches results. This is a contextual vocabulary constraint, not calibrated sequence likelihood or semantic certification.

<!-- column -->

### Protected-span diffusion repair

A repair selects roughly one to four words, maps character offsets to overlapping generator tokens, and uniformly corrupts only those positions. Token-boundary expansion is recorded explicitly. Protected IDs are clamped before/after guidance, during sampling and re-noising, in self-conditioning and at final decoding. New structural/EOS tokens cannot hide a protected suffix. Inputs must round-trip exactly through the generator tokenizer and fit one canvas; there is no silent truncation. Repairs use eight requested native passes by default and may explore worse intermediate text; only verified improvements replace the incumbent.

## 5  Combining measured edit effects

The measured planner uses canonical observations instead of first-order input gradients. For source x and single-span edit j, it stores the actual displacement delta_j between the edited text and source. It then ranks compatible sets using their summed displacement:

$$ \delta_j=\pi(E(x_j))-\pi(E(x)),\qquad \widehat{y}_S=\pi(E(x))+\sum_{j\in S}\delta_j. $$

This can retain an individually worsening edit whose movement cancels another edit's error. Eligible atoms must declare one exact source span and replacement that reconstruct the complete measured text. Overlapping edits and ambiguous insertion boundaries conflict. The default searches two to four edits; configuration allows up to six.

Bounded pair/triple pools retain both nearby partial sums and a deterministic reservoir of distant ones. A two-dimensional nearest-neighbor meet-in-the-middle search retrieves promising compatible sums under explicit visit/materialization limits. It is neither exhaustive nor guaranteed to find the best combination. It performs no encoder call itself, but measuring atoms and certifying materialized combinations costs encoder queries.

[FIGURE:displacements]

@caption **Figure 2.** Measured single-edit vectors can cancel. Their sum predicts the combined paragraph's location; attention interactions make the combined true displacement nonadditive. Canonical re-scoring is mandatory, even when the prediction lands exactly on target.

<!-- page -->

## 6  Meaning-preserving proposal banks

The paraphrase path broadens search beyond native-screened single words. `ParaphraseEditProposer` creates a bounded, frozen bank for each source paragraph from curated phrase alternatives and an optional local provider. It paginates without repeats and exposes all atomic proposals for measured-combination search, including atoms that are individually worse. It uses neither target coordinates nor encoder gradients to generate the bank.

Each proposal changes one exact character span. Screens preserve numeric literals, signs of negation and qualification, comparison/direction signals and entity-like tokens. Capitalization is adjusted at sentence starts. These conservative checks reject obvious claim changes but do not prove equivalence. Banks are bounded per source and across sources; reaching a capacity returns no new bank rather than silently regenerating one.

The optional provider calls an **already installed local** Ollama model such as `gemma4:e2b` (or a configured smaller model). It proposes short same-meaning replacements from source text alone. The adapter records/fixes the model identity and keeps proposal generation separate from target-based ranking. It never trains or pulls a model. A small generator can suggest useful phrasing cheaply, but malformed or meaning-changing suggestions are discarded; a larger proposal count is not evidence of better quality or greater target attainability.

A frozen bank makes experiments interpretable: equal initial text and provider output define the same atomic search space. Changing the bank, source paragraph, provider digest or screening policy changes the experiment. Expanding lexical coverage can improve reachability but creates more opportunities for semantic drift, so distance and quality must be evaluated together.

## 7  The optional scientific quality gate

`TechnicalQualityGate` reviews actual text through a separate local model, default `gemma4:12b`. It evaluates five dimensions: **grammar, coherence, technical plausibility, claim preservation, and no deterioration** in clarity, precision or scientific meaning. With a reference, all five must pass. Without one, preservation and deterioration must explicitly be marked not applicable.

The baseline reference is fixed during refinement. Replacing it with each accepted edit could allow small individually plausible changes to accumulate into a different scientific claim. The gate checks qualifiers, causal direction, entities, quantities, units and uncertainties against that fixed reference. A numeric-literal multiset precheck rejects changed numbers before any model work. It can also reject harmless numeric reformatting and cannot establish that preserved numbers retain their intended associations.

<!-- column -->

### Acceptance, cost and independence

The judge receives candidate and reference as quoted JSON data; target coordinates, distance, gradients and optimization reward are absent. Fixed instructions and a strict schema define acceptance. Every applicable verdict must be pass: uncertain, missing, duplicate or unexpected fields, invalid types, truncated responses, model errors, cancellation and deadline failures reject the candidate. Deterministic decoding settings reduce sampling variability but do not turn the judge into a factuality oracle.

The installed model digest is frozen after inventory; each call rechecks it, including cache hits. Cache keys include the full policy, candidate, reference and model identity. Valid positive and negative judgments are cached; transport/parsing errors are not. Requests use loopback-only Ollama addresses, refuse redirects and cloud entries, and never pull models or choose fallbacks. Input/response sizes, request time, cache and trace counts are bounded. Traces preserve dispatched judge requests and server-reported token/duration costs.

**Late quality evaluation** first obtains canonical coordinates for all candidate atoms, then asks the judge only about texts that can enter the bounded beam. Only quality-accepted text can become an incumbent or beam source. Coordinates from unaccepted atoms may still support a combination prediction, whose actual combined text must pass its own quality check. This saves avoidable judge work without treating numerical proximity as quality evidence.

In two real calibration calls, the authored gradient-descent reference passed, while a fluent paragraph reversing gradient direction and two convergence claims failed technical plausibility, preservation and no deterioration. The calls took 18.73 and 12.20 seconds, about 30.93 seconds together. A separate changed sample count failed the numeric precheck with zero judge calls. This is a sanity check, not a measured accuracy or robustness benchmark; the larger panel in section 8 subsequently exposed false passes. The judge marked coherence as passing even while mentioning a contradiction, showing that its own verdicts can be inconsistent.

A quality policy can reject the numerically closest paragraph and leave a larger residual distance or no accepted result. That is an explicit constraint, not an optimization failure to hide. The research-specific gate does not automatically apply to support replies, fiction or another domain: those uses require an appropriate policy/reference and separate evaluation. Human review remains necessary for scientific correctness claims.

The browser's native-screened controller, and harness/controller configurations adding paraphrases, measured combinations or a quality callback, are separate operating modes. Enabling one module in a test does not establish that every entry point uses it.

<!-- page -->

## 8  Ten-target results and failure analysis

Ten public witness targets were held out from corpus and prompting. Generation received coordinates and nearby examples, without witness text. One 192-token trajectory per target used sparse, one-update guidance; all ten produced mechanically eligible coarse text. Fine search started from a selected coarse paragraph and used the first paraphrase bank, local `gemma4:e2b` proposals, measured combinations and the `gemma4:12b` quality gate. It did not run native span repairs during this fine stage. Per-case limits were 240 seconds, 4,096 queries and epsilon=0.0001.

| Held-out target | Initial d | Final d |
| --- | --- | --- |
| Machine learning | 0.094337 | 0.094337 |
| Distributed systems | 0.046594 | 0.046594 |
| Biology | 0.213210 | 0.208957 |
| Climate science | 0.134163 | 0.126548 |
| Quantum computing | 0.053839 | 0.053839 |
| Robotics | 0.109823 | 0.084704 |
| Neuroscience | 0.050515 | 0.050515 |
| Materials science | 0.123868 | 0.123868 |
| Security | 0.177935 | 0.177935 |
| Astronomy | 0.244574 | 0.226500 |

Four of ten distances improved; median distance changed from 0.116846 to 0.109103. All ten stopped stalled and zero reached epsilon. The saved panel used 117 coarse native passes, 255 fine-stage scored texts, 14 local paraphrase requests and 32 judge requests. Per-case controller time ranged from 13.95 to 88.43 seconds; initial anchor review and coarse generation are outside that controller timing. Counters have different scopes and must not be summed as elapsed time.

**All ten final paragraphs passed the local judge, but independent model-assisted, distance-blinded review flagged four clear false passes.** The machine-learning target's paragraph inherited an incorrect decoherence-as-collapse claim; biology retained an unsupported causal conclusion from observational maps; climate retained a solar-only forcing definition and malformed wording; robotics introduced “lets an autonomous agent to create” and sentence-capitalization regressions. Exact text hashes were matched to saved accepted judgments after review. The four numerical gains are therefore not four demonstrated meaning-preserving successes. The other six paragraphs are not certified correct by one review pass.

A nine-call adversarial audit found no reliable judge improvement: Gemma 4 31B accepted faulty physics. Qwen3.6's raw verdicts passed it, but formatting triggered rejection; 12B's semantic verdict remained unknown after parse failure. Offline parser replay is not new inference [9].

<!-- column -->

### How much each stage has established

| Component | Evidence and limit |
| --- | --- |
| Canonical verifier | Actual-text coordinates and repeated tolerance checks; no meaning guarantee |
| Sparse guidance | Earlier paired error 0.07483 -> 0.02411 with early guidance; 0.09050 with late guidance |
| Feedback / screened edits | Hosted case 0.015393 -> 0.004157 in 26.76 s; stalled above epsilon |
| Protected-span repair | Exact locking safeguards verified; marginal distance benefit unisolated |
| Measured combinations | 4/4 eligible local d=0; six-pair A/B returned identical outputs |
| Paraphrase bank | Two accepted gains in six reviewed anchor cases; general benefit unestablished |
| Quality judge | 10/10 final passes included four clear defects in independent review |
| Profiles / storage / UI | Reusable setup/persistence; no distance gain attributed |

Earlier early guidance improved one-target error 67.8%; late guidance worsened it 20.9%. The hosted screened-edit case improved error 73.0% using 11 canonical queries and 32 native vocabulary probes. Cold/warm ordering and changed screens prevent controlled speedup claims.

A separate CPU local diagnostic constructed three-edit witnesses from each source's curated bank, then gave the optimizer only source text and target coordinates. Four fixtures had enough independent spans; all recovered the exact witness with two fresh singleton checks at d=0: 10-37 queries and 0.093-0.266 search seconds, excluding preparation. Six of ten fixtures failed bank coverage before search. One independent distance-blinded model-assisted review found all four grammatical, coherent and claim-preserving; robotics wording was less idiomatic. This is 4/4 eligible cases, 4/10 fixtures, on deliberately easy bank-generated targets. It demonstrates local feasibility, not target-only generation or general epsilon attainment [10].

Later, a 30-draft blinded review accepted 11, rejected 14 and left five uncertain; four targets lacked an approved anchor. On six approved anchors, atomic edits versus added combinations of up to six spans used equal 60-second/4,096-query limits. Both arms returned identical text in all six: no added combination benefit observed; none reached epsilon. Final independent review accepted two, rejected two inherited SLAM/phase-transition overstatements and left two uncertain, although all edits preserved claims. Only the accepted gains 0.050430 -> 0.045929 and 0.164335 -> 0.163197 support scoped quality-preserving improvement. The first is federated-learning text at a quantum target: proximity does not enforce topic. [A/B evidence](https://github.com/loopmath/point2text/blob/main/docs/optimization-progress.md).

<!-- page -->

## 9  Reuse, browser and reproducibility

A use-case profile chooses the encoder, preprocessing, corpus and writing policy **before** map construction. TOML settings cover checkpoint/revision, maximum length, normalization and encoder prefix; corpus path; and output purpose, domain, style, word limits, neighbor count, content rules and extra instructions. The exact default profile preserves the original MiniLM map fingerprint beginning `6b6f4e24`. Changing encoder or corpus requires a new map. Prompt-only changes can reuse geometry but require a new recorded prompt-policy identity.

Corpus JSONL records have exactly four nonempty fields: id, topic, title and text. Unique IDs and at least three records are required. The bundled six-record support example demonstrates configuration, not a validated support map. The rendered generation prompt can be inspected with actual nearest examples. Content rules can request fictional consistency rather than scientific exposition. Encoder prefixes affect scoring inputs; writing prompts affect generation. They serve different purposes and must not be interchanged.

A backend-independent `TextEncoder` needs an identity describing all preprocessing and `encode(texts)` returning a finite [texts, dimensions >= 2] matrix. This supports map construction, ordinary scoring, unguided generation and measured text search. Gradient guidance additionally needs an audited token-input graph reproducing canonical pooling/normalization. The built-in audit accepts Transformer -> attention-mask mean Pooling -> optional Normalize with fixed CLS/SEP layout and `inputs_embeds`. Extra projection/router modules, different pooling, prompt exclusion, prefixes or dimension truncation need a verified adapter. Unsupported graphs reject actionably; there is no hidden MiniLM fallback.

The browser uses zoom/pan on a fixed map, displays clicked targets, canonical candidate coordinates, tolerance and explicit stop status, and animates clean native denoising drafts. Host SQLite stores generation overlays keyed to the map identity; adding generated text does not refit PCA or alter corpus points. A FastAPI service initializes the map asynchronously and runs model work in a worker; streamed events update the browser without encoding on every pointer move. Stored results carry model/map identities, text and run traces so a browser refresh can recover the same measurements. History is evidence of explored text, not extra training data.

Browser defaults are six initial candidates, 16 inner updates, tolerance 0.01 and a 120-second budget, with controls up to 64 candidates/updates. Removing the clock limit leaves other query, round and stall limits active. The research controller default remains epsilon=0.0001. The looser interactive tolerance is a usability choice, not evidence that the harder objective has been solved.

<!-- column -->

### Stopping and exact success semantics

Tolerance uses unsquared Euclidean distance: d <= epsilon. At epsilon=1e-4, the corresponding squared map loss is 1e-8. Rounded labels, a small surrogate loss, native confidence, or one batched score cannot establish success. The verifier requires two fresh singleton canonical encodings of the real text and reports the worse distance. Confirmation is numerical evidence for that text/map only; it does not certify meaning or general attainability.

Default refinement bounds are 1,024 scored texts, 20 rounds, beam width four, 64 proposals per round and four rounds without meaningful improvement. Time/cancellation checks occur between native passes or probes; an in-flight forward or encoder batch finishes its current operation. Results retain the best eligible incumbent and distinguish tolerance reached, stall, budget, cancellation and error.

The repository is MIT-licensed; separately downloaded model weights and dependencies retain their own licenses. On Apple Silicon, install the encoder and native runtime, then launch the refinement server:

```sh
uv sync --python 3.12 --extra encoder \
  --extra diffusion --extra dev
uv run point2text-diffusion \
  --refine --serve
```

Profiles use `--profile PATH`. For comparison runs, save full-precision targets, immutable model revisions, map and prompt IDs, seed lists, all budgets, every final text and rejection/stop reason. Held-out witness paragraphs must be absent from the retrieval corpus, prompt and coordinates-only initialization. The six-page PDF rebuilds with the optional `paper` dependencies. Source, full-precision experiment records, fixtures and tests remain reviewable in the public repository [8].

## References

[1] Sentence Transformers. [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Pinned encoder revision stated in section 1.

[2] Ollama. [Generate API](https://docs.ollama.com/api/generate). Local autoregressive baseline and proposal/judge transport.

[3] Google DeepMind. [DiffusionGemma model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card). Architecture and released sampler configuration.

[4] DiffusionGemma Team. [DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146), 2026. Training and model background.

[5] Google. [Diffusion in Text Generation Explained](https://gemma.ai/models/diffusiongemma/explained/). Uniform-state corruption, cached prefill and self-conditioning.

[6] MLX-VLM. [Audited native sampler source](https://github.com/Blaizzy/mlx-vlm/blob/217548e24d0d1586b4981c3b50d2f1d3a8e64975/mlx_vlm/generate/diffusion.py). Installed version 0.7.11.

[7] MLX Community. [Pinned four-bit checkpoint configuration](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit/blob/a7a81407613811e8ba63af92ac0d852b809e191f/config.json).

[8] point2text contributors. [Source and experiment records](https://github.com/loopmath/point2text). The ten-target receipts are [coarse generation](https://github.com/loopmath/point2text/blob/main/docs/experiments/quality-coarse-pool-v1.json), [fine search](https://github.com/loopmath/point2text/blob/main/docs/experiments/quality-multi-target-v1.json) and [blinded review](https://github.com/loopmath/point2text/blob/main/docs/experiments/quality-blind-review-v1.json). See `evaluation/README.md` for fixture boundaries.

[9] point2text contributors. [Adversarial judge calibration](https://github.com/loopmath/point2text/blob/main/docs/experiments/semantic-quality-adversarial-calibration.json). Nine calls; three-case costs: 12B 28.70 s, 31B 52.85 s, Qwen 24.95 s. Original format rejection and offline parser replay are separate observations.

[10] point2text contributors. [Measured reconstruction](https://github.com/loopmath/point2text/blob/main/docs/experiments/measured-reconstruction.json) and [blinded review](https://github.com/loopmath/point2text/blob/main/docs/experiments/measured-reconstruction-review.json). Constructed local targets; six coverage failures retained.
