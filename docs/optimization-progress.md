# Optimization results and current bottleneck

The research objective remains canonical Euclidean distance `<=1e-4` with usable
text. The browser's requested default is `0.01`. Neither a lower distance nor a
local model's approval establishes scientific correctness.

## Current evidence

The [six-page report](system-report/point2text-system-report.pdf) contains the
first ten-target experiment and the four independently reviewed local witness
recoveries. All four eligible **known-bank local** cases recovered exact
three-edit witnesses and passed two fresh canonical checks at zero distance;
six fixtures lacked enough independent edits. These easy, bank-constructed
targets establish local feasibility, not new generation from arbitrary points.

We then generated six native candidates with sixteen encoder updates per pass
for each of the ten targets. This changes two factors together and uses more
compute, so it is not an isolated guidance ablation. The
[complete compressed pool](experiments/quality-coarse-six.json.gz) retains all
requests, drafts, runtime traces and scores.

A distance-blinded reviewer examined the three closest valid drafts per target
in shuffled order: **11 accepted, 14 rejected, 5 uncertain**. Four targets had
no approved candidate among those inspected; unreviewed drafts remain
unreviewed. These are individual assessments, not an estimated judge accuracy.
The [review](experiments/coarse-six-quality-review.json) preserves reasons and
text hashes.

From approved anchors, we compared curated atomic edits against the same edits
plus measured combinations of up to six disjoint spans. Each arm had the same
60-second/4,096-query limits, beam width four and 256 proposals per round. No
generator or local-judge calls occurred during this component experiment.

| Target ID | Anchor | Atomic | With combinations | Final independent review |
| --- | ---: | ---: | ---: | --- |
| hidden-systems | .047184 | .042669 | .042669 | Uncertain: protocol-specific scope |
| hidden-quantum | .050430 | .045929 | .045929 | Accepted; generated subject is federated learning |
| hidden-robotics | .100518 | .089009 | .089009 | Rejected: inherited SLAM overstatement |
| hidden-neuroscience | .046069 | .039149 | .039149 | Uncertain: mechanism/scope |
| hidden-materials | .081609 | .076503 | .076503 | Rejected: inherited phase-transition generalization |
| hidden-astronomy | .164335 | .163197 | .163197 | Accepted |

Both arms returned identical text in all six cases: **no additional benefit
from combinations was observed here**. None reached epsilon. All edits retained
their source claims, but the second reviewer found inherited scientific defects
in two anchors and withheld judgment on two more. Only the two accepted results
support a scoped quality-preserving improvement claim. The target IDs identify
witness locations; a two-dimensional match does not enforce the witness's topic.

[Full ablation](experiments/reviewed-edit-ablation.json) and
[blinded final review](experiments/reviewed-edit-quality.json) include failures,
counts, texts, hashes and qualifications.

## Next work

The main bottleneck is obtaining sound, target-relevant coarse text and a larger
set of admissible edits. Stronger local judges also missed a physics error in
the [adversarial audit](experiments/semantic-quality-adversarial-calibration.json).
We should improve grounding and domain-specific review before expanding the
search budget, then measure whether additional independent paraphrases improve
resolution. Preserve the fixed map, repeated epsilon checks and explicit
failure records throughout. Do not weaken quality criteria to improve a score.

To rerun the component comparison with the reference map:

```sh
uv run --extra encoder --extra diffusion python scripts/evaluate_reviewed_edits.py \
  --pool docs/experiments/quality-coarse-six.json.gz \
  --reviews docs/experiments/coarse-six-quality-review.json \
  --output .cache/reviewed-edits.json
```

The runner rejects a different map identity. If a platform or embedding setup
produces a different fingerprint, generate a new pool with
`scripts/evaluate_quality_refinement.py prepare` and review its actual texts;
do not reinterpret the reference coordinates as belonging to the new map.

## Fresh off-corpus clicks

A new [six-click panel](experiments/fresh-click-targets-v1.json) uses only public
corpus geometry: far-apart topic centroids mixed with the global centroid and a
seeded corpus point using positive weights. Thus the clicks lie inside the
corpus hull, but none equals a corpus location. No hidden witness text defines
these targets, enters prompts, or serves as a reconstruction reference.

We generated two fresh native seeds per click at 192 tokens and 32 requested
denoising steps. The first arm requested 16 encoder updates per guided pass;
actual early stopping produced **142 native passes, 28 guided passes and 448
encoder updates** across all twelve trajectories. Every trajectory received
guidance; per-trajectory update counts were 32–64, not `16 × 32`. A paired arm
used exactly the same requests/seeds with no guidance hook. The
[programmatic pairing/count audit](experiments/fresh-native-paired-summary-v1.json)
verifies identical targets and requests.

The distance-blinded reviewer examined all twelve selected texts per arm:
the closest mechanically accepted draft if available, otherwise a raw final or
last-draft fallback. Both current arms had **12 mechanically accepted selections
and zero raw fallbacks**; mechanical acceptance can still admit broken grammar
and scientific errors. No selected text was excluded from the review counts:
**2 accepted, 6 rejected, 4 uncertain** with guidance, and
**3 accepted, 6 rejected, 3 uncertain** without it. These individual assessments
do not establish judge accuracy or a population quality effect. None of the
mechanical results reached `1e-4`, and most targets still lacked an approved
generated anchor. Complete [guided](experiments/fresh-native-basins-v1.json.gz)
and [unguided](experiments/fresh-native-unguided-v1.json.gz) pools retain requests,
raw drafts, mechanical failures and native/guidance traces. Both
[guided review](experiments/fresh-native-root-review-v1.json) and
[unguided review](experiments/fresh-native-unguided-root-review-v1.json) retain
text hashes, verdicts and reasons.

We next independently reviewed the six nearest public corpus references; all
six were accepted as cautious references. An exploratory, source-grounded arm
then generated four fresh native paraphrases per click without guidance. It
uses one public reference, explicit preservation of claims and qualifications,
and an experimental **40–100-word** policy because those sources contain only
40–46 words. This changes prompt, seed pool and word policy together and is
not an isolated guidance ablation. The application's 60–100-word default is
unchanged. Literal complete-reference-sentence copies are excluded from
candidate ranking, while their raw drafts remain in the evidence.

The [24-trajectory grounded pool](experiments/fresh-grounded-basins-v1.json.gz)
and its [distance-blinded packet](experiments/fresh-grounded-blind-review-v1.json)
are preserved for independent review (24 mechanical selections, zero raw
fallbacks). Mechanical acceptance alone is not a
claim of scientific validity, claim preservation, or epsilon attainment. Local
refinement must start from approved **generated** text, not from the public
reference itself. The harness supports equal-budget word-span versus whole-
sentence repair and fresh-restart comparisons once the reviewed anchors and
restart core are available.

Reproduce the paired initial pools with the same target file and frozen map:

```sh
uv run --extra encoder --extra diffusion python scripts/evaluate_fresh_basins.py \
  generate --targets docs/experiments/fresh-click-targets-v1.json \
  --cache-dir .cache/point2text --output .cache/fresh-guided.json.gz \
  --attempts 2 --inner-updates 16 --seconds 1200

uv run --extra encoder --extra diffusion python scripts/evaluate_fresh_basins.py \
  generate --targets docs/experiments/fresh-click-targets-v1.json \
  --cache-dir .cache/point2text --output .cache/fresh-unguided.json.gz \
  --attempts 2 --inner-updates 16 --seconds 1200 --unguided
```

The grounded command additionally uses `--grounded --unguided --min-words 40
--attempts 4 --seed 2973431`. `summary` checks paired request equality and reports
actual native/guidance work. `packet` omits target labels, coordinates, seeds and
distances from reviewer items; its separate key preserves linkage.

Independent [grounded-v1 review](experiments/fresh-grounded-independent-review-v1.json)
accepted 9 selected texts, rejected 12, and marked 3 uncertain. Approved generated
anchors covered only **three of the six targets**; the missing targets remain
explicit failures in the [reviewed summary](experiments/fresh-grounded-reviewed-summary-v1.json).

A further [conservative-v2 arm](experiments/fresh-grounded-conservative-v2.json.gz)
generated four new seeds per target with no guidance, a 40–70-word policy, and
an explicit 2–3-sentence/source-modality-preservation prompt. This is another
exploratory prompt/length/seed change, not a controlled single-factor benefit
claim. All 24 selected texts were independently reviewed against their public
references: **11 accepted, 10 rejected, 3 uncertain**. Approved anchors covered
four targets; neuroscience and climate/cosmology clicks still had none. The
[review](experiments/fresh-grounded-conservative-independent-review-v2.json),
[summary](experiments/fresh-grounded-conservative-reviewed-summary-v2.json), raw
drafts and failures remain available. No epsilon success follows from this
coverage improvement. Word-span, sentence-span and fresh-restart refinement are
now being compared from approved generated anchors under equal per-arm budgets,
with separate blinded post-review required for their new text.

## Reviewed word, sentence and restart refinements

The [completed three-arm comparison](experiments/fresh-approved-refinement-v2.json.gz)
started from the four independently approved conservative-v2 generated anchors.
All six original targets remain in its receipt: targets 01 and 02 have explicit
`not_run` entries because neither had an approved generated anchor. Each actual
arm allowed 60 seconds, 600 canonical queries and 20 rounds, with the exact
40–70-word prompt policy. Word and sentence repair disabled restarts; the third
arm allowed two fresh restart basins, each with two new coarse seeds. Weak
guidance used strength .1, maximum logit delta .5, 16 inner updates and an
explicit start-at-zero schedule. Actual executed work is in every run receipt;
equal budget limits do not imply equal realized compute.

Before final review, the policy was fixed as scientific accuracy, grammar,
coherence and public-source fidelity for **all** arms. Anchor-claim preservation
was graded separately and additionally required for local word/sentence repair.
Fresh restart output could differ from the old generated anchor while remaining
faithful to the public source. Both public source and initial anchor were shown
to the reviewer; targets, distances and arms were hidden. The
[independent review](experiments/fresh-refinement-independent-review-v2.json)
accepted 4 finals, rejected 7 and marked 1 uncertain.

| Target | Approved initial | Words | Sentence | Sentence + restarts |
| --- | ---: | --- | --- | --- |
| 01 | None | Not run | Not run | Not run |
| 02 | None | Not run | Not run | Not run |
| 03 | .033952 | .027826 accepted | .017356 uncertain | .009275 accepted |
| 04 | .076398 | .067402 rejected | .027391 accepted | .025401 rejected |
| 05 | .050411 | .048940 rejected | .039121 rejected | .019595 rejected |
| 06 | .041483 | .030183 rejected | .037211 accepted | .026463 rejected |

Distances in rejected or uncertain cells are mechanical measurements, **not
quality-preserving gains**. The best reviewed initial-or-final distance improves
for three of the four eligible targets; target 05 retains its approved initial
text. No arm attained epsilon. In particular, .009275 remains roughly 93 times
the requested `1e-4`. Some superficially better outputs contained punctuation
artifacts or changed technical claims, and are retained rather than repaired
silently. The [programmatic reviewed summary](experiments/fresh-refinement-reviewed-summary-v2.json)
applies the predeclared quality policy and preserves missing targets.

This phase executed the frozen `53fe8ab` implementation. Later cache fixes were
audited against saved traces: every logged curated source was recovered, all
atomic edits and up-to-four disjoint combinations were enumerated as a cache
candidate superset, and no cross-basin coarse match was found. The quality-key
fix did not apply because the experiment used no live quality callback.
[Audit code](../scripts/audit_fresh_cohort_cache.py) and its
[receipt](experiments/fresh-refinement-cache-audit-v2.json) preserve the method
and verify the original source hash. New phases use the corrected implementation.

To improve coverage for the two missing targets, a separate source-diversity
phase generated two fresh native seeds for each of public-neighbor ranks 2–4.
All six additional public references were reviewed before generation. The
[combined raw pool](experiments/fresh-source-diversity-v3.json.gz) preserves all
twelve outcomes, individual source/seed settings and hashes of the unchanged
per-rank pools. The [independent source-diversity review](experiments/fresh-source-diversity-independent-review-v3.json)
accepted 6 texts, rejected 3 and marked 3 uncertain. Both previously missing
clicks now have approved **generated** anchors. The
[combined selection receipt](experiments/fresh-all-six-reviewed-anchor-summary-v3.json)
preserves the separate pools/reviews and all six fixed targets; no reference
paragraph is substituted for generated text.

Those two newly eligible targets then received the same three native-refinement
arms and 60-second/600-query/20-round limits. This separate stage executed
`94a7614`, with controller/runtime file hashes recorded in the
[complete receipt](experiments/fresh-approved-refinement-v3.json.gz). The earlier
stage used `53fe8ab`; source diversity and implementation version both differ,
so comparisons between stages are exploratory. The
[independent final review](experiments/fresh-refinement-independent-review-v3.json)
accepted one text and rejected five. Only the dark-matter word-edit final passed
the full source-quality and anchor-preservation policy, improving .046790 to
.045217. The mechanically closer .003456 neural final and .011638 cosmology
final were rejected and establish no quality-preserving gain. All rejected
texts and their reasons remain available in the
[reviewed summary](experiments/fresh-refinement-reviewed-summary-v3.json).

Across both native stages, the best independently approved initial-or-final
distances for the original six clicks are **.033695, .045217, .009275, .027391,
.050411 and .037211**. All six have approved genuinely generated anchors; none
has an approved epsilon result. The
[start-selection receipt](experiments/fresh-best-reviewed-native-starts-v3.json)
selects by canonical distance only after applying existing independent reviews,
retaining source identity and each heterogeneous native start's provenance.

A further exploratory hybrid arm uses those approved native starts with
local autoregressive `gemma4:12b` whole-sentence proposals, canonical scoring
and up-to-four disjoint measured combinations. It performs **no native diffusion
calls** during refinement. Per target, limits are 90 seconds, 600 canonical
queries, 8 rounds, beam width one and at most 6 provider calls with 6 variants
per sentence; it retains the exact 40–70-word policy. The predeclared final
quality gate requires scientific accuracy, grammar, coherence, public-source
fidelity **and** preservation of the approved starting anchor's claims. The
provider is target blind; canonical selection is target conditioned. Raw
responses, screening failures, all initial/rebased bank atoms and normal
controller measurements are retained. This arm is hybrid research, not a claim
that DiffusionGemma alone supplied the new proposals. Its outputs require a new
blinded review before any quality-preserving progress is claimed.

The [completed hybrid receipt](experiments/fresh-hybrid-sentence-bank-v1.json.gz)
used frozen source `a00c2d6`. It executed 22 local autoregressive calls and 40
canonical queries in 119.7 seconds across the six targets; actual native
trajectories, diffusion passes and gradient updates were all zero. Of 77 raw
variants, 21 entered the frozen sentence banks. Screening rejected 37 variants
for capitalized-entity differences and 19 for qualification/quantity/technical
differences. All six searches stalled before their budget limits. Two finals improved mechanically. The
[independent blinded review](experiments/fresh-hybrid-sentence-bank-independent-review-v1.json)
accepted all six exact final texts and judged every anchor's claims preserved;
four finals were exactly unchanged. Under the predeclared full quality gate,
click 02 improves from .045217 to .023555 and click 05 from .050411 to .046745.
The other four retain their approved starting distances, apart from floating-
point rounding at about `1e-16`. The
[strict reviewed summary](experiments/fresh-hybrid-sentence-bank-reviewed-summary-v1.json)
links exact text hashes to the judgments. These are individual reviewer
assessments, not a guarantee or a calibrated accuracy estimate. No output
reached epsilon; the best reviewed distance remains .009275.

The [complete actual bank export](experiments/fresh-hybrid-sentence-bank-atoms-v1.json.gz)
includes all eight initial/rebased source banks and their 21 atoms, not just
winners. The [diagnostic summary](experiments/fresh-hybrid-sentence-bank-diagnostics-v1.json)
uses existing controller measurements only. Five targets had numerical rank-two
single-edit displacement sets; the remaining target had only one accepted atom.
Rank two does not establish sufficiently fine spatial resolution or feasible
quality-preserving movement. The one improving combination retained in history
had a `.00605` error between its additive prediction and verified coordinate;
only the actual canonical result was used for selection. The archive does not
claim prediction errors for unretained combinations.

Literal copy audits found no exact public-reference paragraph or complete
public-reference sentence in any final. This is a narrow literal test, not a
claim of novelty or factual validity. No byte-bound transport error occurred;
raw responses, rejected variants, installed model digest and provider token
counts are preserved. Normal provider/model/cache errors would remain failures
rather than silently disappearing from the denominator.

Observed guard failures suggest the next bounded intervention: distinguish
sentence-initial capitalization from named entities while keeping quantities,
qualifications and true technical entities protected. Several passive-voice or
fronted-clause paraphrases were rejected solely because the initial capitalized
word changed. Adversarial guard tests and a new blinded quality review should
precede claims that this yields more admissible movement. Increasing a budget
that currently exhausts its tiny bank cannot by itself demonstrate `1e-4`.

To reproduce this hybrid phase after deriving the reviewed starts:

```sh
uv run --extra encoder --extra diffusion python scripts/evaluate_fresh_basins.py \
  refine --cache-dir .cache/point2text \
  --pool docs/experiments/fresh-all-six-generated-anchor-pool-v3.json.gz \
  --reviews docs/experiments/fresh-all-six-generated-anchor-reviews-v3.json \
  --starts docs/experiments/fresh-best-reviewed-native-starts-v3.json \
  --arm hybrid-sentence-bank --proposal-model gemma4:12b \
  --provider-calls 6 --provider-variants 6 \
  --seconds 90 --session-seconds 650 --queries 600 --rounds 8 --restarts 0 \
  --seed 1973901 --output .cache/fresh-hybrid.json.gz \
  --packet .cache/fresh-hybrid-review.json
```

Reproduction also requires the installed local Ollama model with the recorded
digest and the frozen map identity. The evidence receipt records exact
controller, provider and harness file hashes. An implementation, model digest,
or input change defines a new run rather than overwriting these artifacts.
