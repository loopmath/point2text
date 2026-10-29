---
name: point2text-use-case
description: Adapt the point2text repository to a user's writing task, corpus, and embedding model at setup time, then validate and hand off the local app. Use for use-case onboarding or changing a semantic map's setup.
---

# Adapt point2text

Locate the point2text checkout; repository paths here are relative to that checkout, including when this skill was copied elsewhere. Read `docs/adapting.md` for the current schema and supported encoder graph; read `docs/agent-onboarding.md` when extending domain quality checks or handing off the installation. Inspect current CLI help and implementation before issuing commands.

## Establish the setup

Use the user's existing answers. Ask only for missing choices that affect the result: the writing task and language, desired length/style, corpus source or permission to author one, chosen embedding checkpoint, and available local generator/runtime. A different embedding model is an ordinary setup choice; do not silently substitute the demo model. Fiction, support replies, and other domains need their own prompts and acceptance criteria.

Create a named TOML profile and representative JSONL corpus using `examples/profiles/` and the profile guide. Keep private source data and generated records out of public commits. Pin an actual model revision; a placeholder or a moving branch is not a reproducible handoff. Record pooling, normalization, input prompt, token limit, and tokenizer behavior through the encoder identity.

The embedding choice is fixed while this map is used. Changing the encoder, its input contract, or the corpus creates a new map. Never reuse old target coordinates or refit the displayed map with saved generations. Distances and epsilon thresholds need interpretation within the chosen map; demo thresholds do not transfer automatically.

## Check the actual capability

Use `load_profile(path)` and the resulting `build_encoder()`, `load_records()`, `build_map(cache_dir)`, and `prompt_policy` interfaces. Validate corpus IDs, text lengths with the chosen tokenizer, finite embeddings, and the map build before starting a generation run.

The bundled differentiable adapter supports an audited Transformer → attention-mask mean pooling graph with specific normalization and token-layout settings. Inspect `encoder.gradient_capability()` and adapter checks; a loadable SentenceTransformer checkpoint alone does not establish guidance support. For unsupported graphs or prompts, use the scoring baseline or a supported no-gradient measured-text path. Another `TextEncoder` can support scoring, but needs a separate adapter and validation for gradients. Report the selected mode and reason; do not bypass support checks.

Render `prompt_policy.describe(examples)` for real neighboring records and show the actual prompts in the handoff. Check generation, diffusion, and repair paths used by this setup for leftover technical-domain wording. Keep example text as data rather than instructions. A scientific-claim screen cannot certify fiction or a different domain; supply a relevant quality adapter before enabling refinement, or keep the scoring baseline and record that domain quality remains unvalidated.

## Validate and hand off

Run focused existing checks plus a real local smoke run when the required models are available. Try at least three representative targets covering distinct corpus regions or a sparse boundary. Preserve decoded outputs, target and measured coordinates, map ID, encoder/generator identities, seeds, settings, elapsed work, and failure reasons. If runtime or model access prevents a run, state the missing evidence instead of fabricating results.

Review actual text against the user's domain criteria separately from 2D distance. Closeness does not prove coherence, factuality, style, or reachability. Report both observed text quality and geometric error; do not invent benchmark claims.

Launch the local app with its profile and persistent cache. Use a tailnet address when the user requested access from another device. In the handoff, give the exact command/URL, profile and corpus paths, pinned identities, rendered prompts, validation observations, limitations, and the cache/run/library locations needed to resume. Keep saved paragraphs scoped to their original map; they are overlays on frozen PCA.
