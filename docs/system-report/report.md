# Text Optimization on a Frozen Semantic Map

## 1  Objective, current result and operating modes

point2text converts a click into a text-generation objective on a frozen two-dimensional map. It retrieves nearby public examples, generates a paragraph, then measures the actual decoded text with the original encoder and projection. Subsequent search proposes edits, combines measured changes and restarts from fresh generations. Models and map remain fixed; this is inference-time search, with no trained inverse decoder.

**Current result.** In an adaptive six-click development panel, retained independently reviewed text improved for five targets. The best canonical distance is **0.009275**; none reached the research objective **1e-4**. Several numerically closer outputs failed quality review and were discarded. These results do not establish generalization or a quality guarantee. This report describes implementation snapshot **0fdfde3**, including native and optional hybrid steps.

The default corpus contains 100 authored technical paragraphs across ten topics, not published abstracts. The pinned all-MiniLM-L6-v2 encoder [1] uses ordinary tokenization, float32 CPU inference, attention-mask mean pooling and L2 normalization. Inputs beyond 256 encoder tokens, including special tokens, are rejected.

$$ e(x)=\mathrm{normalize}\left(\frac{\sum_i a_i h_i(x)}{\sum_i a_i}\right)\in\mathbb{R}^{384}. $$

PCA is fitted once by centered singular-value decomposition. Two saved orthonormal rows V have deterministic signs; there is no whitening. Encoder/preprocessing identity, corpus and fitted arrays determine the immutable map fingerprint.

$$ \pi(e)=V(e-\mu),\qquad VV^{\mathsf{T}}=I_2. $$

For target y*, search ranks eligible texts A by unsquared Euclidean distance:

$$ d(x,y^*)=\|\pi(e(x))-y^*\|_2,\qquad \widehat{x}=\arg\min_{x\in A}d(x,y^*). $$

The two components retain about 11.74% of corpus embedding variance. Proximity therefore neither fixes the topic nor preserves scientific claims. Arbitrary points may be inaccessible under the text constraints; failure of a finite search does not prove that they are.

Measurements use an Apple M4 Max with 128 GiB memory. Loading, denoising, encoder and proposal/judge work have separate ledgers.

<!-- column -->

[FIGURE:architecture]

@caption **Figure 1.** Current dataflow. A shared controller retains the global incumbent while exploring local basins. Only ordinary encoding supplies reported coordinates. Proposal branches are configurable; a quality callback is optional.

### Which components run together?

| Mode | Proposal and acceptance path |
| --- | --- |
| Scoring-only baseline | Installed Ollama model; ordinary encoding ranks independent paragraphs |
| Default refinement browser | DiffusionGemma drafts/word-span repairs; canonical gradient edits with native vocabulary screening; mechanical checks |
| Optional research paths | Whole-sentence native repair, measured combinations, sentence banks and quality callbacks |

The baseline and native generator receive a writing policy plus nearest corpus examples. The default policy requests an original 60-100 word technical paragraph. Example text is context, not an instruction. Profiles can change purpose, domain, style, constraints and neighbor count.

The **hybrid sentence-bank path** starts from diffusion-generated text but uses a separate local autoregressive Ollama model for paraphrase proposals. It is not a claim that DiffusionGemma generated every edit. The browser does not automatically enable this bank, measured combinations or a scientific judge.


<!-- page -->

## 2  What DiffusionGemma actually does

DiffusionGemma is Google's experimental discrete-diffusion language model built from Gemma 4's sparse mixture-of-experts backbone. The 26B-A4B designation denotes roughly 25.2 billion total parameters and 3.8 billion active parameters; routing uses eight of 128 experts plus a shared expert [3]. Its authors describe supervised bidirectional-denoising adaptation followed by reinforcement learning and sampler distillation [4]. point2text downloads frozen inference weights; it performs none of that training.

The model uses **uniform-state corruption**. A generation canvas begins with vocabulary tokens drawn uniformly, rather than a grid of MASK tokens. During inference, a shared backbone first uses causal attention to prefill prompt KV state. Denoising then uses bidirectional attention within the canvas, conditioned on that cached prompt. Uncertain positions can be re-noised and revised after earlier decisions. Self-conditioning carries a probability-weighted token-embedding representation of the previous prediction into the next pass. General DiffusionGemma inference can append completed blocks to the prompt cache and start another canvas [5]. This adapter implements text-only, single-canvas inference.

The audited stack is MLX-VLM 0.7.11 with MLX 0.32.4 and `mlx-community/diffusiongemma-26B-A4B-it-4bit`, revision `a7a81407613811e8ba63af92ac0d852b809e191f` [6,7]. Vocabulary size is 262,144. Early probes used 128 positions; default coarse generation in refinement uses 192 to reduce cutoffs. Protected repairs use the incumbent's exact encoded length. Coarse generation requests 32 denoising passes by default, with native early stopping.

For token entropies sorted from smallest to largest, the native entropy-bound selector accepts the prefix satisfying the following rule (beta=0.1 in the checkpoint):

$$ \sum_{j=1}^{k}H_{(j)}-\max_{j\leq k}H_{(j)}\leq\beta. $$

Accepted positions receive samples from the proposed categorical distributions; remaining positions receive fresh uniform vocabulary noise. Native stopping requires stable predictions and mean entropy below 0.005; it does not evaluate the clicked target. The adapter preserves native temperature shaping, prompt caching, sampling, acceptance and self-conditioning. A fresh prompt cache is created for each trajectory; it is reused within that trajectory, not shared across independent candidates.

A hook receives detached, temperature-adjusted logits z0 of shape [1, L, 262144], current token IDs and an editable mask. Every downstream native decision consumes the hook's returned logits. There is no backward pass through MLX generator weights. Native generation is serialized because the MLX random-number generator is global; the hook and draft observer must not consume it. Compilation, multimodal inputs and multiple canvases are outside this adapter.

<!-- column -->

## 3  The sparse gradient bridge

Generator tokens and MiniLM tokens have incompatible vocabularies and boundaries. The adapter therefore builds an explicitly approximate bridge. For generator token v, it decodes that token in isolation, re-tokenizes its surface with MiniLM without special tokens, and averages the corresponding **input lookup vectors** W. These are not sentence embeddings.

$$ B_v=\frac{1}{|T_v|}\sum_{k\in T_v}W[k]. $$

At each guided pass, S_i is the original top-32 support at position i, fixed across inner updates. Both q_i and its reference are conditional on this support; the omitted vocabulary tail is excluded from KL:

$$ q_{iv}=\frac{\exp(z_{iv}/\tau)}{\sum_{w\in S_i}\exp(z_{iw}/\tau)},\qquad v\in S_i. $$

Empty, whitespace-only and structural surfaces are excluded from the lexical mixture, with validity indicator a_v:

$$ u_i=\frac{\sum_{v\in S_i}q_{iv}a_vB_v}{\sum_{v\in S_i}q_{iv}a_v}. $$

The sequence [CLS, u_1, ..., u_L, SEP] passes through the frozen MiniLM body, attention-mask mean pooling, normalization and saved PCA. Noneditable or unrepresented positions use fixed-token fallbacks and masks. The optimized loss is squared map error plus a divergence penalty over editable positions M:

$$ \mathcal{L}(z)=\|\pi(E_{\mathrm{soft}}(z))-y^*\|_2^2+\frac{\lambda}{|M|}\sum_{i\in M}\mathrm{KL}(q_i\|q_i^0). $$

The reference q_i^0 is fixed at the native proposal throughout the inner loop. Gradients flow from map loss through PCA, pooling, the frozen encoder body, vector mixtures and softmax into selected logits. Frozen weights can transmit input gradients without being updated. The generator computation and hard decoding remain outside this graph.

Each update normalizes/clips the gradient direction, takes a scaled descent step and bounds cumulative logit displacement from z0. m updates are a compute budget, not a convergence test. The first KL gradient is zero at q=q0; later updates can incur a restoring penalty. With clipped/normalized direction v, step scale eta and logit bound D, the selected-logit update is:

$$ z^{r+1}=z^0+\mathrm{clip}(z^r-\eta v-z^0,-D,D). $$

A new native pass produces fresh z0 rather than continuing the preceding pass's optimized logits.

The surrogate collapses several encoder subwords into one averaged vector and one position. Actual encoding retains their order, positions and contextual attention. Joining decoded surfaces can change tokenization; top-k support omits mass, and final EOS cleanup changes the scored text. Consequently, lower soft loss does not imply lower canonical distance. Ordinary decoded-text scoring remains the acceptance rule.
<!-- page -->

## 4  Canonical feedback under shared budgets

The controller observes clean native drafts at periodic checkpoints and termination. With UI streaming enabled, every native pass can update the unverified draft panel, while ordinary encoding remains periodic, by default every three passes. Earlier eligible text can survive a worse later canvas. Observers do not consume native random numbers; an eligible tolerance hit can request a safe stop.

```text
best <- none; create shared verifier and budgets
for scheduled initial group or fresh restart basin:
  reset local beam, local best and stall count
  generate unused seeds; verify clean checkpoints
  while local slice and global budgets remain:
    propose atoms from the basin's beam
    measure atoms and actual combined paragraphs
    if due or stalled: run protected native repair
    retain eligible local states
    publish only strict global-best improvements
    if distance <= epsilon:
      require two fresh singleton text encodings
      stop only after both confirm tolerance
  preserve global best; move to the next basin
return best and explicit completion/stop reason
```

For a continuation, initial_text supplies an existing paragraph and skips the initial generation pool. It is re-scored under the same policy. With restarts disabled or no generative runtime, only local refinement runs; this is not itself new target-to-text generation.

Eligibility checks word/token bounds, finite coordinates, completed sentence punctuation, structural tokens and detected repetition. Text and canonical token sequences are deduplicated. A separate callback can impose a quality policy; absent that callback, eligibility is mechanical only. Neither path establishes factual correctness.

### Exact numerical success

The verifier reserves two queries for fresh singleton encodings and reports their worse distance. Thus a 1,024-query cap leaves at most 1,022 search queries. A query counts one scored text, not a batch or every encoder forward used by guidance.

$$ d(x,y^*)\leq 10^{-4}\quad\Longleftrightarrow\quad d(x,y^*)^2\leq 10^{-8}. $$

A small relaxed loss, one batch score, native confidence or a rounded coordinate cannot satisfy this rule. Quality review remains separate even after numerical confirmation. Cancellation and budget expiry retain the best eligible incumbent and an explicit reason; they do not imply unreachability.

<!-- column -->

### Fresh generation after local progress stalls

Six requested initial candidates are interleaved in cohorts of two with local refinement. Additional restarts default to three basins, with two fresh trajectories each. Pending initial cohorts do not consume those extra restart slots. Limits can stop execution before all requested work occurs.

A basin has its own best text, beam and stall count. Even a basin starting worse than the global winner can make local progress; its repairs use its own text. The global best, strictly improving history, canonical cache, query reserve, deadline and quality policy remain shared. Restarting never resets the total budget.

Remaining rounds and time are divided over planned basins. If B includes the current basin and w is two for the first basin, otherwise one, its time share is:

$$ T_b=T_{\mathrm{remaining}}\frac{w}{B-1+w},\qquad T_{\mathrm{generation}}=0.6T_b. $$

The first weight accommodates warmup; the 60% generation allocation leaves local-search time. These are scheduling heuristics, not established optimal ratios. Native passes and supported proposer/provider callbacks poll local stop signals. An in-flight forward or noncooperative call can overrun a slice without becoming a user cancellation.

Fresh coarse and repair requests use recorded distinct seeds. An identical cached coarse paragraph can seed another basin without another canonical query, subject to current quality approval. All cached candidates are considered in local ordering; a stale verdict or padded-text lookup cannot bypass a current rejection. Temporary stop results are not permanently negative-cached.

### Protected words or whole sentences

Default repair covers roughly one to four words. Optional sentence mode selects a complete sentence, prioritizing gradient hotspots, and uses a source-grounded prompt to preserve claims, quantities, entities, negation, uncertainty and language. Both modes map character spans to generator tokens and record boundary expansion.

Protected tokens are clamped through guidance, sampling, re-noising, self-conditioning and final decoding; structural/EOS changes cannot hide a protected suffix. Inputs must round-trip exactly and fit one canvas. Repairs request eight passes by default, every second round or on a stall. Actual repaired text still needs canonical and configured quality checks.

<!-- page -->

## 5  Local proposals and measured combinations

### Canonical token gradients

CanonicalEditProposer starts from real encoder IDs and exact character offsets. It differentiates both projected coordinates with respect to input vectors, yielding a local two-coordinate Jacobian:

$$ J_i=\frac{\partial\pi(E(U))}{\partial U_i},\qquad \widehat{\Delta y}=\sum_iJ_i\Delta U_i. $$

This ranks whole-word edits and potentially cancelling pairs without the generator-token bridge. Equal-length replacements use corresponding lookup-vector differences; changed lengths use a mean-vector approximation. Retokenization and nonlinear attention still require ordinary scoring of every proposed paragraph.

The browser intersects curated grammatical families with native contextual top-32 alternatives. The screen corrupts one generator token, holds context fixed and runs one denoiser forward. It permits exact single-token word replacements and four probed spans per source. This constrains vocabulary, not scientific correctness.

### Combining actual single-edit displacements

For each same-source atomic edit j, the planner records its measured change and predicts a compatible combination S by addition:

$$ \delta_j=\pi(E(x_j))-\pi(E(x)),\qquad \widehat{y}_S=\pi(E(x))+\sum_{j\in S}\delta_j. $$

Atoms declare exact source spans and replacements that reconstruct the complete edited paragraph. Overlaps and ambiguous insertion boundaries conflict. Individually worsening atoms can still be useful in a cancelling combination.

Bounded pair/triple pools retain nearby and diverse partial sums. Two-dimensional meet-in-the-middle search ranks sets of two to four edits, optionally six, under visit/materialization caps. The planner makes no encoder call; atoms and combined texts require scoring.

[FIGURE:displacements]

@caption **Figure 2.** Actual single-edit vectors need not add to the actual combined displacement. One retained improving combination in the latest hybrid study had 0.00605 prediction error. Final selection always used the canonical result.

<!-- column -->

## 6  Target-blind phrase and sentence banks

ParaphraseEditProposer supplies curated phrases and optional short local-model edits. Its conservative 1-8-word restriction remains intact. The separate SentenceParaphraseBank broadens syntax by replacing one whole sentence at an exact offset. Repeated identical sentences are distinguished by position.

Banks are frozen and paginated per exact source hash. Supplied JSON or an already installed local Ollama model can provide variants. The provider receives the paragraph and selected sentence, **never target coordinates, encoder state or reward**. Only subsequent canonical ranking is target-conditioned. Hashes and exact-span provenance let the measured planner combine disjoint sentences.

Screens preserve numeric literals and supported unit pairings, negation, modality, scope signals, technical referents and entity-like tokens. A narrow eleven-word English sentence-initial capitalization exception requires unchanged casefolded occurrence counts. It permits certain clause reorderings while retaining the lexical words. These English heuristics both overreject valid prose and miss semantic changes.

The optional provider fixes an installed model digest and bounds calls, time, context/output size and caches. It streams cancellable requests, never pulls a model, and retains bounded raw/error responses, variants and rejection reasons. Static review metadata is provenance only. A composed paragraph still needs its own checks.

## 7  Three distinct quality decisions

**Browser checks** reject detected mechanical defects. The default browser has no scientific quality callback and labels text quality unverified.

**Optional online policy.** TechnicalQualityGate uses a local model to assess grammar, coherence, technical plausibility, claim preservation and no deterioration. With a fixed reference, all must pass. Missing, uncertain, malformed, truncated or errored decisions fail closed. Numeric literals are prechecked; model digest, prompts, decisions and costs are recorded. The reference stays fixed so incremental edits cannot redefine the original claims.

Late quality evaluation scores atoms first and judges likely beam entrants. When configured, only callback-approved text enters that beam or becomes an incumbent; unapproved atom coordinates can still predict a combination that must pass separately.

**Offline experimental review.** Distance-blinded GPT-6.1-sol agents assess exact final text against public sources and approved anchors. Local edits must preserve anchor claims; fresh generations can differ from the old anchor but must satisfy the declared source policy. Rejected or uncertain finals fall back to approved starts in reported retained results. This is a later research selection step, not an online browser guarantee.


<!-- page -->

## 8  Actual target-to-text evidence

Six clicks were fixed from public corpus geometry before generation: positive mixtures of topic centroids, the global centroid and a seeded corpus point. None equals a corpus position. No hidden witness paragraph defined a target or entered a prompt. This tests generation toward coordinates, not known edit-bank reconstruction.

Two native seeds per click were paired with identical-request guidance-off runs. All guided trajectories were treated: **142 native passes, 28 guided passes and 448 inner updates**. Blinded review of twelve texts per arm accepted two with guidance and three without; six in each were rejected, leaving four/three uncertain. This panel isolates no validated quality-preserving guidance benefit.

Generation then used reviewed public neighbors and preservation instructions. A 40-70-word, 2-3-sentence variant supplied approved generated starts for four clicks. Two seeds from neighbor ranks 2-4 supplied the missing two. This combined pool has 36 trajectories and 233 native passes, **zero gradient guidance**, and six selected approved starts. Prompt, length, source and seed changes make it an adaptive sequence.

Across native repair/restart and optional hybrid stages, retained quality-reviewed distances are:

| Fixed click | Approved start d | Retained d |
| --- | --- | --- |
| 01 | 0.033695 | 0.033695 |
| 02 | 0.046790 | 0.023555 |
| 03 | 0.033952 | 0.009275 |
| 04 | 0.076398 | 0.027391 |
| 05 | 0.050411 | 0.046745 |
| 06 | 0.041483 | 0.037211 |

**Five of six improve; zero reach 1e-4.** The best is about 93 times above epsilon. Individual blinded assessments are not factuality certificates. Selected heterogeneous stages are neither a held-out benchmark nor an equal-compute causal comparison [9].

### Native repairs and restarts

The three arms used word repair, sentence repair, or sentence repair with two extra basins of two seeds. Curated phrase proposals and measured combinations were shared. Each allowed 60 seconds, 600 queries and 20 rounds. Guidance used strength 0.1, logit bound 0.5, 16 inner updates and a start-at-zero schedule. Executed compute differed.

Reviewed examples include 0.033952 to 0.009275 with sentence repair plus restarts, and 0.076398 to 0.027391 with sentence repair. Closer finals with broken syntax or changed claims were excluded. Version hashes, failures and reviews are retained; audits found later cache fixes inactive in older runs.

<!-- column -->

### What the hybrid sentence bank contributed

From the best approved native texts, the local AR bank used gemma4:12b, six calls and six variants per call as ceilings, a 90-second/600-query limit, eight rounds, beam width one and combinations up to four edits. No native runtime ran in this fine stage.

Actual work was **22 Ollama calls, 77 raw variants, 21 admitted atoms and 40 canonical queries**, in 119.7 seconds across six targets. There were zero diffusion passes or encoder-gradient updates. All six finals passed the declared offline source-and-anchor review; four were exactly unchanged. The two reviewed gains were 0.045217 to 0.023555 and 0.050411 to 0.046745 [10].

Thirty-seven variants failed capitalization screening and nineteen failed qualification/technical screens. Nine source banks were visited; eight contained atoms. Five targets had rank-two measured displacement sets, while one had one atom. Rank two alone did not provide fine enough or suitably directed motion. An improving combination's additive prediction error was 0.00605, far larger than epsilon.

### An isolated filter change with no extra gain

A fixed-bank replay reused the same 77 raw variants, map, starts and budgets. Only the old versus corrected capitalization screen differed; neither arm called a model. The narrow exception admitted five more atoms, 21 to 26, yet both returned the same final text for every target. Twelve reviewed rows were six duplicated candidate/reference/anchor triples, transparently reusing prior judgments; they were not twelve new quality observations.

Exact textual enumeration gave 40 legacy and 60 corrected feasible compositions in total, matching scored-text/cache counts. Under the frozen controller's restricted query family, this supports finite-pool coverage. No independent per-query text ledger was retained. Exhausting those small sets neither proves global unreachability nor makes a larger bank unnecessary [10].

### Historical controls and validation

The earlier ten-target trial reached no epsilon despite numerical reductions: all ten passed the local judge, but independent review found four clear defects. Larger local judges also missed faulty physics [11]. Separately, four eligible bank-constructed local witnesses were recovered at d=0 with two fresh singleton checks; six fixtures lacked coverage. Those easy known-bank recoveries are not actual arbitrary-click generation. Another six-anchor word-versus-combination comparison returned identical outputs [11].

Current checks passed **395 tests**, with one optional local skip; CI separately tested the actual pinned encoder. Two installed-wheel HTTP searches entered fresh restarts and streamed 23/38 draft updates. Their roughly 30-second, mechanical-only results validate lifecycle behavior, not quality or epsilon benefit [8].

<!-- page -->

## 9  Reuse, interface and reproduction

### Choose geometry once at setup

A TOML use-case profile selects checkpoint/revision, encoder prefix, token limit and normalization; a JSONL corpus; and writing purpose, domain, style, word limits, content rules, extra instructions and neighbor count. Changing encoder or corpus creates a new map. Prompt-only changes can reuse geometry with a new recorded prompt identity. The reference map fingerprint begins 6b6f4e24.

A TextEncoder supplies preprocessing identity and a finite text-by-dimension matrix. This supports ordinary scoring, unguided generation and measured edits with other embedding models. Gradient guidance additionally needs an audited input-vector graph matching canonical tokenization, pooling and normalization. The built-in adapter supports Transformer, masked-mean Pooling and optional Normalize with CLS/SEP and inputs_embeds. Prefixes, alternative pooling or extra modules require verified adapters; unsupported cases reject explicitly.

Writing prompts and encoder prefixes have different roles. The exact rendered generation prompt can be inspected before model loading. Scientific exposition, support text and fiction require different content/quality policies; an English scientific judge is not a universal acceptance policy. The repository's use-case skill documents setup and representative validation.

### Browser and persistence

The fixed map supports wheel/pinch zoom, pan, keyboard movement and saved-generation overlays. Host SQLite scopes saved text and provenance to map identity. Saving does not refit PCA. Native drafts animate changed words and display phase/basin labels; unverified drafts remain distinct from measured paragraphs.

Interactive defaults are **6 initial candidates, 16 updates per scheduled guided pass, tolerance 0.01, 120 seconds, 3 extra restarts and 2 candidates per restart**. Counts can increase; unlimited time removes only the clock. Other limits remain 1,024 queries, 20 rounds, beam four, 64 proposals per round and a four-round stall threshold. Browser searches choose fresh recorded seeds; CLI/API default to seed 42.

To launch the default native refinement path:

```sh
uv sync --python 3.12 --extra encoder --extra diffusion
uv run point2text-diffusion --refine --serve
```

Sentence repairs use the additional flag --repair-span-mode sentence. Sentence banks and measured combinations have separate Python/evaluation interfaces; the CLI flag does not enable the hybrid provider. Research runs explicitly request epsilon=1e-4.

<!-- column -->

## 10  Remaining path to 1e-4

The current banks terminate far below their query ceilings because they run out of admissible text. More inner updates optimize an approximate bridge, not its correctness; more queries cannot improve an exhausted finite pool. Broader coherent sentence/clause proposals and a deliberately diverse frontier of approved ancestors are the next inference experiments. Online calibration should first test whether measured nonlinear residuals improve candidate ranking on held-out probes.

Nested SMC motivates completed-text lookahead but requires correct proposal weights [13]. Zero2Text motivates online alignment, yet reconstructs full embeddings rather than arbitrary two-coordinate targets [14]. Neither method is implemented here or establishes this tolerance. The primary-source research review separates usable inference ideas from methods requiring trained inverse models [12].

Reproduction requires the frozen map and model revisions, exact prompts, seed lists, budgets, rejected texts, review hashes and executed work. Requested denoising steps and configured updates are not actual counts. The bare GuidanceConfig defaults differ from CLI settings; each receipt is authoritative. Map changes require new targets. Project code, original corpus and documentation are MIT-licensed; external weights/dependencies keep their licenses.

## References

[1] Sentence Transformers. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.

[2] Ollama. [Local generation API](https://docs.ollama.com/api/generate).

[3] Google DeepMind. [DiffusionGemma model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card).

[4] DiffusionGemma Team. [Technical Report](https://arxiv.org/abs/2608.00146), 2026.

[5] Google. [Diffusion in Text Generation Explained](https://gemma.ai/models/diffusiongemma/explained/).

[6] MLX-VLM. [Audited sampler source](https://github.com/Blaizzy/mlx-vlm/blob/217548e24d0d1586b4981c3b50d2f1d3a8e64975/mlx_vlm/generate/diffusion.py), version 0.7.11.

[7] MLX Community. [Pinned four-bit checkpoint](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit/blob/a7a81407613811e8ba63af92ac0d852b809e191f/config.json).

[8] point2text. [Implementation and validation](https://github.com/loopmath/point2text/blob/main/docs/validation-restarts.md), snapshot 0fdfde3.

[9] point2text. [Reviewed six-click progress and stage receipts](https://github.com/loopmath/point2text/blob/main/docs/optimization-progress.md).

[10] point2text. [Hybrid diagnostics](https://github.com/loopmath/point2text/blob/main/docs/experiments/fresh-hybrid-sentence-bank-diagnostics-v1.json) and [fixed-bank replay](https://github.com/loopmath/point2text/blob/main/docs/experiments/fresh-fixed-bank-replay-diagnostics-v1.json).

[11] point2text. [Historical trials and quality failures](https://github.com/loopmath/point2text/blob/main/docs/semantic-quality.md); receipts linked in [9].

[12] point2text. [2025-2026 primary-source research review](https://github.com/loopmath/point2text/blob/main/docs/research-2025-2026.md).

[13] Chanchu et al. [Nested sequential Monte Carlo for discrete diffusion control](https://arxiv.org/abs/2608.20123), 2026.

[14] Kim et al. [Zero2Text](https://arxiv.org/abs/2602.01757), 2026.
