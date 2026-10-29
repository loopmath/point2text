# Initial demo validation — 29 October 2026

These are observed development checks, not a research benchmark or a promise of performance on other machines.

## Environment

- Apple M4 Max, 128 GiB unified memory; Python 3.12.12.
- CPU sentence encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, float32, mean pooling, L2 normalization, 256-token limit including special tokens. Over-limit text is rejected before scoring.
- Local generator: Ollama `gemma4:12b`, installed digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
- Corpus: 100 original conceptual paragraphs across ten topics; the real tokenizer measured a maximum of 62 input tokens.
- Frozen map ID: `6b6f4e24edfccb558b78535986cc0611d5ce23ab9dda441c992b896977b0aa2d`. PC1/PC2 retain approximately 6.11% and 5.62% of embedding variance. The plot is a limited view of semantic space.

## Checks and observations

| Check | Observed result |
| --- | --- |
| Unit/API suite | 30 passed; mocked generation and numerical fixtures, no model downloads |
| Source and wheel builds | Both artifacts built successfully |
| Independent wheel environment | Imported outside the checkout; CLI, all 100 corpus records, HTML/JS/CSS resources present |
| Browser interaction | Real map loaded, target selected, three candidates generated and plotted, closest selected |
| Real three-candidate search | 11.70 seconds, warm local model; candidates 90, 91, and 90 words |
| Measured distances | 0.05324, 0.08512, and 0.14634 unwhitened PCA units |
| Independent canonical re-scoring | All three texts re-embedded through the installed wheel; coordinates/distances matched recorded values exactly in this environment |
| Cancellation | Stopped another search at about 5.44 seconds; one verified paragraph remained available and generation controls recovered |

The three-candidate target was `(0.03782757211018523, -0.020649042373064936)`, the plotted location of the corpus's distribution-shift paragraph. Nearby context was available to generation. This is an interactive demonstration, not a hidden-witness evaluation. No tolerance threshold or blind quality assessment was applied. Cold startup, peak memory, smaller Macs, and diffusion guidance have not been benchmarked.

## Reproduction

Follow the root README to start the app. Run the fixture suite with `uv run --extra dev pytest`. Run `python scripts/smoke_local.py` for an optional real-model API smoke check. That script chooses the first corpus record's position, requests a new candidate, and reports measured distance/time; it is a plumbing check rather than reproduction of the particular three-candidate target above.

Every search stores a local JSON record with exact text, target, map and model identities, candidate seed, measured coordinates/distances, and elapsed time. Runtime versions, hardware, model implementation, and sampling behavior can change generated text even with the same seed.
