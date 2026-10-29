# Onboarding an agent

The repository includes a reusable [point2text-use-case skill](../skills/point2text-use-case/SKILL.md) for adapting a local installation to a writing task. It ships with the source so the guide and implementation can evolve together. The [profile guide](adapting.md) contains the configuration schema and examples.

You can give an agent this prompt directly; installation in a global skills directory is optional:

> Read `skills/point2text-use-case/SKILL.md` in this checkout and use it to set up point2text for short customer-support replies. Use my provided corpus and chosen embedding checkpoint, keep that embedding choice fixed for the map, show the actual prompts, and validate text quality separately from map distance. Use an installed local generator and hand off a working local URL with retained data.

Replace the task, corpus, model, and access preference with your own. An agent with skill discovery enabled can invoke `$point2text-use-case` after the user chooses to copy or symlink the skill folder into their own skill directory. Reading this guide does not require changing global agent configuration.

## Where to extend it

| Need | Repository seam |
| --- | --- |
| Another corpus or writing style | Named profile and JSONL corpus; see [adapting.md](adapting.md) |
| Another supported SentenceTransformer | Profile `[encoder]`; pin checkpoint revision and input contract |
| A different embedding backend | `TextEncoder` in `src/point2text/contracts.py`; implement stable `identity` and batch `encode(texts)` |
| Different domain acceptance checks | Quality callback on `CanonicalRefinementSearch`; see below |
| A changed guidance graph | A separately validated reward/gradient adapter, not a profile-only change |

`TextEncoder.encode()` returns a finite array with one row per input, a stable dimension, and the same semantics for corpus and generated text. Its identity must change when vector-producing behavior changes. Scoring-only support does not imply differentiable guidance support. Keep the hard scorer as the authority for reported decoded-text coordinates even when a surrogate guides generation.

## Domain quality adapters

`point2text.diffusion.quality.TechnicalQualityGate` screens technical/scientific text using a local judge. Its verdict is a heuristic, not factual verification or general-domain certification. For fiction, support replies, or another domain, supply a suitable callback before enabling refinement, or use the scoring baseline with domain quality explicitly unvalidated.

`CanonicalRefinementSearch(..., quality_callback=gate)` accepts a callable `gate(text)`. A useful adapter returns a dictionary such as:

```python
{"accepted": False, "reasons": ["promises an unsupported refund"]}
```

The verifier accepts boolean or tuple verdicts too; a dictionary makes the adapter's own diagnostic records easier to inspect. The verifier uses the acceptance value and aggregates callback rejection/error counts, so retain detailed reasons in the adapter's trace if needed. Exceptions reject the proposal. Do not turn service failure or unparseable judge output into acceptance. Inspect the current `late_quality` behavior when deciding which proposals are screened; only a screened, accepted final text can support a quality-pass claim.

Define the rubric from the actual task: support replies may require a permitted resolution, no invented account details, and a specified tone; fiction may require consistent viewpoint and respect for supplied character facts. Supply examples of acceptable text and plausible failures. Keep endpoint, model identity, judge prompt, verdict, and timing in a trace if the adapter calls a local judge. Do not use geometric distance as a quality criterion inside this callback.

Wire a new adapter explicitly into each generation/refinement path the installation uses and expose its validation state in the run record. Adding a refinement callback does not automatically screen the scoring-only generator or every diffusion candidate. Mechanical word/token limits and artifact checks also need to match the profile. Add focused tests for an accepted example, a substantive domain failure, and adapter failure; then inspect real generated text. Preserve the distinction between a heuristic pass and human review.

## Data needed to resume

Keep the named profile, corpus, pinned model identities, rendered prompts, and validation notes with the installation. The persistent cache defaults to `.cache/point2text` relative to the launch directory. Preserve its map artifacts and `runs/` records; the host library is `library.sqlite3` inside that cache. The exact cache path and launch directory belong in the handoff command.

The UI's **Saved paragraphs** are copies of verified run candidates. **Show on map** adds an overlay; it does not add corpus records or refit PCA. Saved paragraphs from other maps remain in the library and are not mixed into the active map. A new embedding contract or corpus needs a new map and fresh targets, while the old map's data remains available for resuming its original setup.
