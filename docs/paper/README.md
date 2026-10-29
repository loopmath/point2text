# Two-page technical report

[Read the PDF](point2text-technical-report.pdf) or the [Markdown source](paper.md).

This conference-style report explains the deployed generate-and-verify baseline: exact encoder preprocessing, frozen PCA coordinates, neighborhood prompting, candidate ranking, runtime boundaries, engineering observations, limitations, and the diffusion extension's acceptance gates. It is not an official NeurIPS submission or a claim of research benchmark performance.

The two-page layout is built from readable Markdown and a ReportLab renderer. Mathematical expressions use Matplotlib's STIX math fonts. No model downloads or live server are needed to rebuild the report.

```bash
uv run --extra paper python docs/paper/build.py
```

The build fails unless the result is exactly two pages. `observations.json` retains the original numerical development observations. Experimental diffusion implementation details live in the separate diffusion branch and its component documentation.
