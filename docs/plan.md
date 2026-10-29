# Demo first, research behind stable interfaces

The first milestone is a local Mac click-to-text demo for technical/research paragraphs. Other Mac tiers will be measured separately. This records the implementation direction, not a claim that every research extension is proven.

## First complete loop

```text
original technical corpus → canonical encoder → saved PCA map
                                                 ↓ click
nearby context → local paragraph candidates → canonical re-embedding
                                                 ↓
                          actual point, distance, text, run record
```

Independent candidates are selected by actual PCA distance. This provides a useful interface and baseline before diffusion-specific guidance. Exact matches at arbitrary map points are not promised.

## Orthogonal workflows

| Workflow | First deliverable | Stable boundary | Later work |
| --- | --- | --- | --- |
| Corpus and map | Original technical corpus, frozen encoder, cached PCA | `MapBundle.payload`, `nearest`, `score`; map/encoder IDs | User corpora, multiple encoders, reachability |
| Local generation | Configurable Ollama model, cancellation, model digest | `TextGenerator.generate` produces decoded text | Native MLX and diffusion runtimes |
| Search | Best verified independent candidate | `BaselineSearch.run`, `Candidate` | Revision, remasking, guidance, trajectory refinement |
| Interface | Target selection, neighbors, measured candidate points | Map JSON and run API | Comparison, controls, large-map rendering |
| Integration and package | CLI, source layout, wheel assets, tests, CI, run artifacts | Ordinary install and one launch command | Hardware matrix, releases, job scheduling |
| Evaluation | Correctness fixtures and real local smoke runs | Actual-text scoring and recorded budgets | Hidden witnesses, quality review, paired time budgets |

The first implementation uses three parallel component workers plus an integration owner: map/corpus, generation/search, and interface. Shared contracts and dependency files have one owner. The requested agent configuration is GPT-6.1-sol with xhigh thinking.

## Contracts

`TextEncoder.identity` identifies the encoder/preprocessing configuration; `encode(texts)` returns canonical vectors. `MapBundle` owns fitted arrays and map identity. `score(text, target)` re-embeds real text and returns its point and Euclidean PCA distance. The UI never fits or changes projection axes.

`TextGenerator.generate(examples, seed, cancel_event)` supplies a new paragraph. It does not know about PCA or decide success. `BaselineSearch` chooses context, asks for proposals, scores actual text, and publishes verified `Candidate` values. A future strategy can keep that result contract while exposing sampler-specific hooks privately.

The API owns jobs, polling, cancellation, validation, and persistence. The browser receives text, coordinates, and scalar diagnostics. One local generation worker runs at a time; history is bounded in memory and completed records persist as JSON.

## Next stages

1. Make the complete demo reproducible from a clean checkout and installed wheel. Verify real scoring/generation, target/result plotting, and cancellation.
2. Improve usefulness: corpus quality, paragraphs, target selection, latency, and saved experiments. Record map IDs and budgets.
3. Add optional diffusion runtime. Prove ordinary execution, pre-commit sampler intervention, hard-token encoder parity, input gradients, and combined execution. Check the exact mlx-vlm implementation; a generic callback is insufficient.
4. Compare scoring-only and guided search with matched model, prompt information, output constraints, device, and elapsed-time budget. Measure final-text error, quality, diversity, and soft-to-hard disagreement. Keep residual policy fixed.
5. Evaluate residual priors, other encoders/projections, trajectory refinement, acceleration, and smaller Macs separately. A new encoder rebuilds the map; coordinates are not portable across encoders.

Residual completion is off the demo's critical path. Start directly in map space; later evaluation checks full-space semantics. Unit-norm feasibility and support diagnostics do not prove natural-text reachability.

## Evidence

The supplied [research brief](research/source-brief.html) and three scoped reviews are retained as [research notes](research/README.md). They distinguish literature from proposed adaptations, predate the demo, and are not benchmark reports. Current installation and behavior belong in the root README.
