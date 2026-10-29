# Contributing

Use Python 3.12 and `uv sync --extra encoder --extra dev` for the reference demo environment. Use just `uv sync --extra dev` for fixture tests without the encoder stack. Run `uv run --extra dev pytest` and `uv build` before submitting a pull request. Update the lockfile when changing dependencies. No model download or live Ollama daemon should be required by the default test suite.

The first milestone is a useful local click-to-text demo for technical and research paragraphs. Improve this path before expanding the research surface. Describe observable behavior and relevant validation in pull requests; state hardware/model for performance claims.

## Component ownership

- `corpus.py`, `encoder.py`, `geometry.py`, and `assets/`: corpus provenance, canonical text scoring, immutable map artifacts.
- `generation.py`: model transport, decoding, cancellation, runtime identity.
- `search.py`: strategy and candidate verification; no HTTP or browser dependencies.
- `server.py`: configuration, validation, bounded job lifecycle, run persistence.
- `static/`: interaction; consume coordinates without refitting the map.
- `contracts.py`: backend-independent types. Discuss interface changes before changing multiple components.

Use separate branches or worktrees for parallel work and agree on contracts first. Keep dependency changes and shared contracts under one integration owner.

## Scientific invariants

Accepted candidates must be decoded to the exact text shown in the UI and scored through the ordinary encoder tokenizer, pooling, normalization, and fixed projection. Never substitute a soft-token surrogate point for the real result. Encoder/preprocessing changes create a new map identity; saved clicks are map-specific.

Keep model weights frozen for the baseline. Add gradient access as a separate optional capability. Keep model-specific tensors inside runtime adapters; exchange candidate text and scalar diagnostics with the web layer.

The demo corpus is original project material. External corpora need documented source and redistribution terms. Do not commit model weights, access tokens, machine paths, generated runs, or user corpora. Keep bundled examples small enough to ship in a wheel.

Research changes should compare actual-text quality/error against elapsed time, use appropriate held-out targets, and separate evidence from hypotheses. A small soft loss or a working runtime hook alone does not establish useful guidance.

Contributions use the repository's MIT license. Model and dependency licenses remain separate.

## Technical writing

Use the shortest clear explanation that preserves the technical content. Remove repetition and process detail; never add sentences to fill a page budget. Prefer a compact equation, algorithm, or measured table when it explains more directly. Page limits are ceilings, not targets, and readable type should not be reduced to compensate for excess prose.
