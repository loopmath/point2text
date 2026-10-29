# Target-blind sentence paraphrase banks

`SentenceParaphraseBank` is a separate source of whole-sentence edits. It does
not change the existing atomic proposer's 1–8-word restriction. A fixed bank can
provide larger syntactic moves for measured two-dimensional edit combinations;
no improvement in accuracy, direction coverage, or quality is assumed before
measuring the actual resulting paragraphs.

```python
from point2text.diffusion.sentence_bank import (
    SentenceParaphraseBank, OllamaSentenceParaphraseProvider,
)

provider = OllamaSentenceParaphraseProvider(
    model="an-already-installed-local-model", max_variants=4,
    max_calls=16, max_total_seconds=120, should_stop=should_stop,
)
proposer = SentenceParaphraseBank(provider=provider, should_stop=should_stop)
proposals = proposer.propose(paragraph, target, max_proposals=8)
```

Construction performs no network access, model download, or generation. The
optional provider uses loopback Ollama only, verifies the installed local model
digest, sends the full paragraph and exact selected sentence as source data, and
requests a small JSON list of complete replacement sentences. It never receives
target coordinates, encoder state, or reward feedback. Temperature and seed are
fixed. Calls, elapsed time, cache capacity, source size, context size, output
size, and output tokens are bounded. Streaming cancellation includes HTTP
prefill; a temporary cancellation does not permanently cache an empty result.
An independent full-paragraph quality policy remains required.

Provider diagnostics retain a bounded receipt per exact paragraph/span under
`diagnostics["sources"]`: source and sentence hashes, span, status, elapsed time,
raw model response, raw variants, accepted variants, per-variant checks and
rejection counts. Malformed JSON and partial streamed responses remain visible
on error. Each raw response is capped at `max_tokens * 32` UTF-8 bytes; receipt
count is capped at `max_cached_sources`. A retry keeps up to eight previous
status labels, while storing only the latest raw response. Cache hits preserve
the completed/error receipt. Temporary stops do not negative-cache variants;
clearing a local stop permits a fresh bounded attempt. Budget refusals are
recorded without issuing model calls. Aggregate token/call/time counters and
accepted/raw variant counts remain available for comparison.

`propose(text,target,max_proposals=...)` pages through fixed source-order variants
without repeats. The target is validated but does not affect generation or
ordering. `all_proposals(text)` does not advance the page; `hotspots(...)` returns
whole-sentence character spans. These methods perform no encoder queries. A
completed sentence's variants are frozen. Interrupted construction resumes at
the unfinished sentence. Source capacity is a hard cap, with no silent eviction
or regeneration; use a fresh instance for a new serialized search run.

Each proposal changes exactly one complete sentence at an original character
span. Provenance includes SHA256 hashes of the entire source paragraph and
source sentence, `source_spans`, `replacements`, provider identity, target-blind
status, and supplied review metadata. Applying the recorded replacement at the
recorded span must reconstruct the complete proposed paragraph exactly. Repeated
identical sentences are distinguished by offsets rather than fuzzy anchors.
`MeasuredCombinationPlanner.record` can consume these atoms after separate
canonical measurement, and compose nonoverlapping sentences. Combined changes
still require a fresh canonical score and contextual quality review.

## Supplied JSON banks

Pass a path or mapping as `reviewed_bank`. Example:

```json
{
  "schema": "point2text-sentence-bank-v1",
  "sources": [{
    "source_sha256": "SHA256_OF_EXACT_PARAGRAPH_UTF8",
    "text": "EXACT PARAGRAPH",
    "sentences": [{
      "span": [0, 42],
      "original": "EXACT SOURCE SENTENCE",
      "variants": [{
        "text": "COMPLETE REPLACEMENT SENTENCE",
        "review": {"reviewer": "REVIEW_ID", "accepted": true}
      }]
    }]
  }]
}
```

The optional paragraph `text` must match its hash. At use time the paragraph
hash, sentence span, and original sentence must match the actual source exactly;
otherwise an entry is unused or rejected. Plain string variants are also
accepted. Supplied review metadata is reported as provenance, never promoted
to a semantic certificate. JSON input size and source/sentence/variant counts
have explicit bounds.

## Conservative checks and limits

Variants must be one complete plain sentence, without model artifacts or extra
sentences. Mechanical screening preserves literal numeric values and their
unit pairings, capitalized entity tokens, negation contractions, and the
existing conservative counters for technical referents, modality, scope,
quantity, and conspicuous claim strength. It rejects obvious substitutions
such as measurements becoming estimates or a possibility becoming certainty.
The checks deliberately overreject some legitimate rewrites, including certain
inflection changes, and do not infer entailment, resolve pronouns, verify causal
direction, or understand every entity or scientific unit. Reordered claims can
pass literal counters while changing meaning. Nothing here certifies language,
grammar, factuality, scientific equivalence, or originality.

The literal signal lists and sentence/entity heuristics are **English-oriented**.
An explicit content-word exception covers only these sentence-initial title-case
words: **Dark, Observations, Measurements, Model, Models, Data, Results, Maps,
Analysis, Experiments, Researchers**. Every such exempted lexical word must keep
the identical casefolded occurrence count across the entire original and
replacement sentence. Only its initial title-case occurrence is discounted;
interior capitalized tokens, proper names, acronyms, and mixed-case gene symbols
remain case-sensitive. Thus “Dark matter … observations” can become
“Observations … dark matter”; removing, replacing, or duplicating either
exempted referent is rejected. Alice, Apple, DNA, and NASA are not exempted.

This narrow rule addresses observed false positives; it does not identify
entities semantically. Common words can also be names, and unlisted technical
openings such as “Calibration” or “Drift” still cause conservative false negatives
when their capitalization moves. Inspect raw receipts and independent reviews
before changing the list. Numeric, unit, modal, negation, and technical-reference
guards are unchanged. These checks do not establish semantic equivalence or
comparable preservation for other languages.

Full-paragraph mechanical limits, canonical re-embedding, contextual quality
review against the fixed reference, and the existing epsilon confirmation rules
remain mandatory. Only independently accepted real text may become an
incumbent. The bank has no encoder and cannot declare a target reached.

Tests use static fixtures and mock HTTP transport, including reconstruction,
nonoverlapping measured combinations, numeric/unit/modal drift, unchanged old
atomic guards, exact hashes, paging/capacity, repeated sentences, cancellation,
and provider call limits. They make no real model or GPU calls.
