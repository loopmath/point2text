# Six-page system technical report

[Read the complete six-page PDF](point2text-system-report.pdf) or [the Markdown source](report.md).

Updated for implementation snapshot **0fdfde3** on 29 October 2026. The report covers the fixed map and baseline, uniform-state DiffusionGemma sampler, top-k gradient bridge, shared-budget restarts, protected word/sentence repairs, measured edit combinations, phrase and sentence banks, quality policies, profiles and browser/storage.

The current adaptive six-click panel improved retained quality-reviewed text on five targets. Best distance is **0.009275**; **zero targets reached 1e-4**. The native-only, optional hybrid and offline-review stages are distinguished. The default browser's mechanical checks do not certify scientific quality.

Rebuild from the repository root:

```sh
uv run --extra paper python docs/system-report/build_report.py
```

The renderer requires exactly six explicit source pages with two columns each, rejects overflow, and checks the output's page count and extracted text. Its fonts are the standard PDF Times/Helvetica/Courier families; equations are rendered at 350 DPI. No model weights, model calls or private files are needed to build the PDF.

The report also preserves historical quality failures and separates constructed local witness recoveries from actual target-to-text generation. A fixed-bank replay admitted five additional variants without improving any final distance. [Current evidence](../optimization-progress.md) links all requests, texts, hashes, rejection records and blinded reviews. These development results are not an equal-compute causal comparison or a held-out generalization benchmark.
