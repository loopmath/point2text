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
