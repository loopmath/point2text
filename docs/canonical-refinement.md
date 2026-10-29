# Canonical feedback refinement

Automatic fresh-generation restarts and budget allocation are described in [fresh basins](canonical-restarts.md). The browser exposes restart count and candidates per restart; the CLI also offers experimental whole-sentence repair.

This experimental path keeps the map and both models frozen. It optimizes
`distance = ||PCA(canonical_encode(text)) - target||`, using ordinary decoded text
for acceptance. The soft cross-tokenizer guidance loss proposes directions; it
does not certify target accuracy.

## Run

On an Apple Silicon Mac:

```sh
uv sync --python 3.12 --extra encoder --extra diffusion --extra dev
uv run point2text-diffusion --refine --serve
```

Open `http://127.0.0.1:8479`. Add `--tailnet` to bind directly to the machine's
Tailscale IPv4 address. The browser exposes tolerance and time budget; completion
can mean the budget expired or refinement stalled above tolerance. Existing
baseline and independent diffusion commands keep their behavior.

```sh
uv run point2text-diffusion --refine --target-record t01-03 \
  --epsilon 0.0001 --refinement-seconds 120 --max-hard-queries 1024 \
  --output .cache/refinement.json
```

## Search loop

1. Generate coarse diffusion candidates. Every three passes, decode a clean
   argmax draft and score it canonically if it satisfies mechanical constraints.
   Retain the best text independently of the sampler's current canvas.
2. At its **actual MiniLM token IDs**, differentiate the two PCA coordinates with
   respect to input vectors. Rank whole-word edits using their predicted 2D
   coordinate changes. Search a small beam and combinations of two disjoint edits
   to allow coordinate changes to cancel. These are local linear predictions;
   all candidate text is decoded, re-tokenized and scored normally.
3. In native CLI/browser mode, restrict word candidates to a frozen DiffusionGemma
   vocabulary screen: corrupt one generator token, keep surrounding context, and
   run one denoising forward. Only supported whole-word, single-generator-token
   replacements are allowed. Intersect the top 32 native tokens with curated
   grammatical word families and protect listed technical collocations. Cache the
   result, probe at most four spans per text, and disable unscreened
   insertions/deletions/phrase changes. This screen is not
   a sequence likelihood or proof of grammatical or scientific correctness.
4. Every other round, or when discrete edits stall, repair a selected one-to-four
   word span with diffusion. The remaining generator tokens are locked in sampling,
   re-noising, self-conditioning, and final decoding. Repairs may explore worse
   text internally; only measured improvements update the published incumbent.
5. Stop on confirmed tolerance, cancellation, a query/time/round limit, or four
   rounds without meaningful improvement. Always retain the best verified text.

Refinement uses a 192-token canvas to reduce sentence cutoffs. Default limits are
120 seconds, 1,024 scored texts, 20 rounds, beam width four,
64 proposals per round, and eight passes per native repair. Timing includes model
work inside the search. Expiry is checked between native passes and contextual
word probes; a running forward cannot be interrupted. CPU proposal analysis and
batched encoding also finish their current operation before stopping.

## Verification and interfaces

`CanonicalVerifier` requires 60–100 whitespace words, at most 256 canonical
encoder tokens, finite coordinates, complete sentence punctuation, no obvious
model headers/special tokens, and no detected repetitive draft artifacts.
It deduplicates normalized text and canonical encoder token sequences.
These are mechanical checks. An optional external `quality_callback(text)` can
reject additional candidates; the bundled demo has no coherence certificate.

Two encoder queries are reserved for epsilon confirmation. Success requires two
fresh singleton canonical encodings of the real text at unsquared distance
`<= epsilon`; the reported result uses the worse check. At `epsilon=1e-4`, the
equivalent squared loss is `1e-8`. Batch scores, rounded labels, and surrogate
losses cannot declare success.

The package separates proposal contracts, verification, controller, canonical
edit proposer, native runtime, and replacement screen. The controller supports
CPU local repair from `initial_text`; the native runtime supports passive draft
observers and protected-span repair independently. See the
[evaluation harness](../evaluation/README.md).

Run records include incumbent history, rejections, confirmation checks, native
traces, contextual proposal costs, hard encoder queries, and explicit stop reason.
The query budget counts scored texts, including failed attempts, not gradients or
generator forwards. Those costs are reported separately.

## Initial development probes

On the development M4 Max, using the pinned models and fixed map:

| Case | Result | Scored texts | Search time |
| --- | --- | ---: | ---: |
| Hidden ML target, generation from coordinates | Best coarse draft `0.061270` → `0.008909`; stalled | 146 | 21.92 s |
| ML witness with one word changed | Recovered witness; two fresh checks at `0`; epsilon confirmed | 67 | 0.14 s |
| Systems witness with one word changed | `0.000674`; stalled; meaning degraded | 184 | 0.44 s |

These are three development cases before the final draft/lexical hardening, not
a success-rate benchmark. The local-repair
cases used curated lexical edits without a native contextual screen. The failed
systems case changed “connection fails” to “relationship fails,” illustrating
that low distance can reward a bad edit. The generation case used native
screening, but its text has not undergone scientific quality review. Full
[generation](experiments/canonical-witness-generation.json) and
[local-repair](experiments/canonical-local-repair.json) records retain failures,
texts, histories, and budget information. The first generation record predates
the additional contextual-screen cost counters; native work is included in its
wall time and proposer time.

A subsequent hosted request exposed repetitive incomplete drafts and a bad native
substitution (“ensure stability” → “ensure model”). The current defaults reject
incomplete/repetitive drafts and require both native screening and curated word
families. These guards address observed failures; they do not certify meaning.

With those guards, a 192-token hosted request produced a complete SGD paragraph
at `0.015393` but a top-eight replacement screen allowed no useful edit. Expanding
that screen to 32 while keeping the grammatical-family intersection reduced the
same paragraph to `0.004157` in a separate local refinement probe (nine scored
texts, 28 native screening forwards). This motivated the current top-32 default.
The same final hosted profile then reproduced `0.015393 → 0.004157` end to end
in 26.76 seconds, with 11 canonical text queries and 32 native screening forwards.
It stopped as `stalled`; see the [full hosted record](experiments/canonical-hosted-refinement.json).
This remains above epsilon and is a single development case. The
[record](experiments/canonical-screened-local.json) contains its initial text and
full-precision target. Save that initial text as UTF-8 and reproduce the local
method with `--refine --discrete-only --initial-text-file FILE --target X Y`.
This mode still uses native vocabulary screening, but omits span repairs.

## Limits

The optimization can exploit a two-dimensional embedding while damaging meaning.
Contextual vocabulary constraints reduce unrestricted substitutions but do not
resolve that objective mismatch. A known witness proves one allowed paragraph
exists at its target; recovering it from a nearby paragraph is a local diagnostic,
not evidence of novel generation. Arbitrary clicked targets may have no acceptable
paragraph within tolerance. A stalled run is not an impossibility proof.
