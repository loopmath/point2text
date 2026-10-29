# Optional local semantic quality screen

`TechnicalQualityGate` is a heuristic scientific-text reviewer using an already
installed local Ollama model. It provides a callable `quality_callback` for the
canonical verifier. It does not establish scientific truth or certify text
quality. Canonical distance and grammatical surface checks remain separate.

```python
from point2text.diffusion.quality import TechnicalQualityGate

gate = TechnicalQualityGate(
    model="gemma4:12b",
    reference_text=initial_paragraph,
    should_stop=lambda: cancel.is_set() or time.monotonic() >= deadline,
)
status = gate.precheck()  # Inventory only: no weight loading or model pulling.
report = gate(candidate_paragraph)
# report: accepted, reasons, dimensions, model_identity, policy_digest, cost
```

The synchronous interface belongs in a worker. `evaluate(text, reference=...)`
can provide a different reference for one evaluation; otherwise it uses
`reference_text`. Assigning `gate.reference_text` after coarse generation is
supported. Keep that reference fixed during refinement so successive edits
cannot redefine the baseline and accumulate scientific drift.

The model evaluates grammar, coherence, and technical plausibility. When a
reference is supplied, it also evaluates preservation of all scientific claims,
qualifications, causal direction, entities, quantities, units, and uncertainties,
plus no deterioration in clarity, precision, or scientific meaning. Without a
reference those last two dimensions must explicitly be `not_applicable`.
Acceptance requires every applicable dimension to be `pass`. `fail`,
`uncertain`, missing dimensions, unexpected fields, invalid types, incorrect
applicability, duplicate JSON keys, truncated output, model errors, cancellation,
and deadlines reject the candidate. Reasons and each dimension's short
assessment are available to the caller.

Before model work, numeric literals must match the reference as a multiset.
This conservative rule can reject equivalent numeric reformattings; it does not
prove that the same numbers retain the same meaning. The model must still
review units, associations, comparisons, and the rest of the text. Candidate
and reference paragraphs are JSON-encoded quoted data in the user prompt;
reviewer instructions, schema, and acceptance rules are fixed separately.
No coordinates, target distances, gradients, or optimization rewards are passed
to the judge.

The default judge is `gemma4:12b`, with seed zero, temperature zero, an 8192-token
context, and a 768-token response budget. A conservative UTF-8 input-byte bound
reserves room for the complete response and prompt template rather than relying
on silent context truncation. There are separate text, request-byte,
response-byte, cache-entry, trace-entry, and 90-second wall-clock limits.
Cancellation/deadline checks poll during inventory requests and model prefill.
Only loopback Ollama addresses are allowed, redirects are refused, cloud model
names and explicitly remote inventory entries are rejected, and the adapter
never pulls models, changes settings, or tries a fallback model.

The first successful inventory lookup freezes the installed model digest.
Every evaluation rechecks that digest, including cache hits; a changed tag
rejects until a new gate is created. Cache keys include the full policy, model
identity and digest, complete candidate, and complete reference. Valid negative
judgments are cached along with positive judgments; transport and parsing
failures are not. Returned reports and traces are copies so callers cannot
mutate cached acceptance. `gate.trace` records model/policy identity, accept and
reject counts, cache hits, errors, elapsed time, dispatched judge requests, and
server-reported token and duration costs. These costs describe the judge only.

The fake-transport suite validates policy plumbing and failure behavior. A small
real-model calibration can reveal obvious failures but cannot validate the
judge's accuracy across scientific subjects. Human review and testing across
several target texts remain necessary when interpreting refinement results.

The recorded [local calibration](experiments/semantic-quality-calibration.json)
used exactly two judge calls against installed `gemma4:12b`: an authored gradient
descent paragraph passed, and a fluent version reversing three scientific claims
failed plausibility, claim preservation, and no deterioration. A separate changed
sample count failed the numeric precheck without model work. The two judge calls
took about 31 seconds together. This is a sanity check, not a measured accuracy
or robustness result; the judge even marked coherence as passing while its short
assessment mentioned a contradiction in the corrupted paragraph.
