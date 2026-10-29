# Diffusion technical report

[Read the two-page PDF](point2text-diffusion-report.pdf) or the [Markdown source](paper.md).

The report describes the implemented native sampler hook, frozen encoder surrogate, update-count hyperparameter, canonical verification, observed probes, and current compute costs. It reports both an improving and a worsening development probe; no general quality or efficiency advantage is claimed.

Sections 2-3 spell out the selected logits, input-vector bridge, gradient path, and native/inner loops. They distinguish the existing heuristic from the proposed [closed-loop path to canonical distance 1e-4](../optimization-to-epsilon.md).

Rebuild from the repository root:

```bash
uv run --extra paper python docs/paper/build.py \
  --source docs/diffusion-paper/paper.md \
  --output docs/diffusion-paper/point2text-diffusion-report.pdf
```

The same renderer builds the concise baseline report. Both builds require exactly two pages and preserve readable typography.
