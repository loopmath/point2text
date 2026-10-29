# Six-page system technical report

[Read the complete six-page PDF](point2text-system-report.pdf) or [the Markdown source](report.md).

The report covers the fixed map and baseline, actual uniform-state DiffusionGemma sampler, sparse gradient bridge, canonical feedback, protected repair, measured edit combinations, paraphrases, scientific quality gate, profiles and browser/storage. It separates measured improvements, negative results and verified interfaces; it does not claim an epsilon or text-quality guarantee.

Rebuild from the repository root:

```sh
uv run --extra paper python docs/system-report/build_report.py
```

The renderer requires exactly six explicit source pages with two columns each, rejects overflow, and checks the output's page count and extracted text. Its fonts are the standard PDF Times/Helvetica/Courier families; equations are rendered at 350 DPI. No model weights, model calls or private files are needed to build the PDF.

The ten-target development panel records four numerical reductions and zero successes at the research tolerance of 1e-4. Independent blinded review found four clear quality false passes despite all ten final texts passing the local judge. Full receipts are linked in the report.

Development measurements are linked in the report. They are not an equal-budget ablation or generalization benchmark. Quality-gated evaluation and ordinary mechanical/canonical scoring are distinguished.
