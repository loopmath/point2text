# Four-page technical report

[Read the PDF](point2text-technical-report.pdf) or the [Markdown source](paper.md).

This conference-style report explains the deployed generate-and-verify baseline: exact encoder preprocessing, frozen PCA coordinates, neighborhood prompting, candidate ranking, runtime boundaries, engineering observations, limitations, and the diffusion extension's acceptance gates. It is not an official NeurIPS submission or a claim of research benchmark performance.

The fixed four-page layout is built from readable Markdown, a small observation snapshot, and a ReportLab renderer. Mathematical expressions are rendered with Matplotlib's STIX math fonts; the scatter diagram uses vector PDF primitives. No model downloads or live server are needed to rebuild the report.

```bash
uv run --extra paper python docs/paper/build.py
```

The build fails unless the result is exactly four pages. `observations.json` contains only the original demo corpus's projected coordinates and numerical observations from the documented development run. Current experimental diffusion implementation details live in the separate diffusion branch and its component documentation.
