# Six-page system technical report

[Read the complete six-page PDF](point2text-system-report.pdf) or [the Markdown source](report.md).

The report covers the fixed map and baseline, actual uniform-state DiffusionGemma sampler, sparse gradient bridge, canonical feedback, protected repair, measured edit combinations, paraphrases, scientific quality gate, profiles and browser/storage. It separates measured improvements, negative results and verified interfaces; it does not claim an epsilon or text-quality guarantee.

Rebuild from the repository root:

```sh
uv run --extra paper python docs/system-report/build_report.py
```

The renderer requires exactly six explicit source pages with two columns each, rejects overflow, and checks the output's page count and extracted text. Its fonts are the standard PDF Times/Helvetica/Courier families; equations are rendered at 350 DPI. No model weights, model calls or private files are needed to build the PDF.

The ten-target development panel records four numerical reductions and zero successes at the research tolerance of 1e-4. Independent blinded review found four clear quality false passes despite all ten final texts passing the local judge. A separate constructed-bank local diagnostic recovered four exact three-edit witnesses out of four eligible cases; six fixtures lacked bank coverage. Independent blinded review found the four edits preserved grammar and claims. This demonstrates local feasibility, not novel target-only generation. Full receipts are linked in the report.

A later six-anchor paired test returned identical texts with and without measured combinations; two final gains passed independent quality review. [Current evidence](../optimization-progress.md) preserves the rejections and uncertain cases.

Development measurements are linked in the report. They are not an equal-budget ablation or generalization benchmark. Quality-gated evaluation and ordinary mechanical/canonical scoring are distinguished.
