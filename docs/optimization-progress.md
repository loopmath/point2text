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
