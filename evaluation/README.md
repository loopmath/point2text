# Canonical refinement fixtures

`epsilon_witnesses.jsonl` contains ten original technical paragraphs, held out from
the map corpus and prompt retrieval. Each has 60–100 words and fewer than 256
MiniLM tokens. Their projected coordinates provide known attainable targets under
the mechanical output policy. The fixtures are public; this is a development
test set, not evidence of generalization to unseen scientific writing.

Run a local repair diagnostic:

```sh
uv run --extra encoder python scripts/evaluate_canonical_refinement.py \
  --mode local-repair --case hidden-ml --output .cache/local-repair.json
```

The harness changes one word using a fixed mutation table and gives the optimizer
the changed paragraph and target. It does not give the proposer the original word
as a special candidate. Recovering the witness demonstrates local repair only.

Run target-to-text generation on an Apple Silicon Mac:

```sh
uv run --extra encoder --extra diffusion python scripts/evaluate_canonical_refinement.py \
  --mode generation --case hidden-ml --output .cache/witness-generation.json
```

Generation receives only the target; the witness is excluded from its prompt,
retrieval corpus, and initialization. Reports include every selected case, full
precision distances, budgets, stop reasons, text, and repeated epsilon checks.
Numerical accuracy and readable, scientifically sound text require separate
evaluation. Neither mode establishes feasibility for arbitrary clicked points.
