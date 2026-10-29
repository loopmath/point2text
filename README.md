# point2text

**Experimental diffusion:** a working local diffusion path is included in `main`. See [diffusion setup, algorithm, and observed probes](docs/diffusion.md) and the [parallel implementation plan](docs/diffusion-plan.md). The default `point2text` command remains the scoring-only baseline; `point2text-diffusion` runs the new path.

Read the [two-page diffusion paper](docs/diffusion-paper/point2text-diffusion-report.pdf) for the implemented method and [efficiency audit](docs/diffusion-efficiency.md) for measured work and remaining optimization gates.

The new [canonical refinement prototype](docs/canonical-refinement.md) adds draft feedback, protected-span diffusion repair, and canonical-token discrete search. Run `point2text-diffusion --refine --serve` on port 8479. Its tolerance is a stopping objective, not a guarantee of a readable paragraph at every point. The [optimization plan](docs/optimization-to-epsilon.md) records the motivation and feasibility limits.

**Pick a point. Generate a paragraph. See where it lands.**

A local research tool for generating technical text toward a selected position on a fixed semantic map. The first demo embeds a bundled technical corpus, fits a two-dimensional PCA map, generates several paragraphs with a local Ollama model, and plots each paragraph's **actual decoded-text embedding**.

This is a scoring-only baseline. It does not yet implement diffusion guidance, guarantee arbitrary target reachability, or establish research benchmark results. The encoder and generator weights remain frozen.

## Quick start

Requires Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and a running [Ollama](https://ollama.com/) installation with an instruction model. The reference demo targets Apple silicon; CPU sentence encoding and an Ollama endpoint are portable interfaces.

```bash
git clone https://github.com/loopmath/point2text.git
cd point2text
uv sync --python 3.12 --extra encoder
ollama list
uv run --extra encoder point2text --model YOUR_INSTALLED_MODEL
```

Open **http://127.0.0.1:8765**. For example, the development Mac uses `--model gemma4:12b`. You can use a different locally installed instruction model; the app never pulls a generator automatically. If Ollama is not running, start the desktop app or `ollama serve`. Install a model separately with `ollama pull MODEL_NAME`.

On first launch, the app downloads the pinned [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) encoder from Hugging Face, embeds the corpus on CPU, and caches the map. Generation and scoring then run locally. The browser shows a loading state until the map is ready. No API key is required.

To install from source without uv, inside a Python 3.11–3.13 virtual environment:

```bash
python -m pip install '.[encoder]'
point2text --model YOUR_INSTALLED_MODEL
```

The wheel includes the corpus and web interface. Running from the repository directory is not required after installation.

## Using the demo

1. Explore the technical corpus and its neighboring paragraphs.
2. Click a target on the map, or move the target with the keyboard.
3. Generate up to six candidates. Neighboring corpus paragraphs provide topic context to the local model.
4. Compare the target with each paragraph's measured point. The lowest Euclidean PCA distance is selected as the best candidate.

Changing the target cancels the current search. Cancellation keeps any already verified results in the run record. Sparse areas and positions between topics may be difficult to reach; the app reports remaining distance instead of declaring a guaranteed match.

The bundled corpus consists of original, curated conceptual paragraphs for a usable first demo. It is not a collection of published research abstracts or an independent evaluation set. Neighbor-conditioned generation on it is a demonstration, not held-out reconstruction performance. PCA also discards embedding dimensions; map proximity alone does not establish full semantic equivalence.

## Configuration and run records

| Setting | Default | Purpose |
| --- | --- | --- |
| `--model` / `POINT2TEXT_MODEL` | `gemma4:12b` | Installed Ollama instruction model |
| `--port` | `8765` | Local HTTP port |
| `--host` | `127.0.0.1` | Bind address |
| `--tailnet` | off | Bind to the active Tailscale IPv4 address instead of loopback |
| `POINT2TEXT_OLLAMA_URL` | `http://127.0.0.1:11434` | Generator endpoint |
| `POINT2TEXT_CACHE` | `.cache/point2text` under the launch directory | Map artifacts and run records |

Map artifacts include the corpus/encoder identity and fitted projection. Completed, cancelled, and failed searches write JSON under `POINT2TEXT_CACHE/runs/`, including target, map ID, actual candidate text and coordinates, model identity/digest, seed, elapsed time, and strategy. These records support inspection; exact token replay across runtime versions is not guaranteed. Model weights and run records are not committed.

API documentation is at `/docs`. One local worker runs generation at a time; the map stays immutable during searches. The server is designed as a local application, without hosted multi-user identity or scheduling.

## Access from another device over Tailscale

Connect the host and your other device to the same tailnet, then run:

```bash
uv run --extra encoder point2text --tailnet --port 8477 --model YOUR_INSTALLED_MODEL
tailscale ip -4
```

Open `http://TAILSCALE_IPV4:8477` from the other device. This binds directly to the tailnet interface and does not change existing Tailscale Serve or Funnel routes. Tailnet access rules determine who can reach it; the app has no additional account/login layer. See [background hosting](docs/tailscale.md) to keep the demo available while away from the host.

## Development

```bash
uv sync --python 3.12 --extra encoder --extra dev
uv run --extra dev pytest
uv build
```

Unit and API tests use fixtures and mocked generators; they do not download models or require Ollama. CI runs checks on Linux and macOS and verifies packaged assets outside the checkout. Real-model smoke tests run separately.

The `encoder` extra contains the reference sentence encoder's PyTorch stack. Core APIs, alternative adapters, and fixture tests can be installed without it using `uv sync --extra dev` or `pip install .`.

With the demo running, `python scripts/smoke_local.py` requests one real candidate and checks the full API loop. See the [initial validation record](docs/validation.md) for observed results and their limits.

The package separates corpus/encoder/map, generation, search, and web delivery. See [architecture and parallel workflows](docs/plan.md), [contributing](CONTRIBUTING.md), and [research notes](docs/research/README.md). It includes a native diffusion adapter and an approximate gradient bridge; residual priors, broader adapters, and rigorous effectiveness evaluation remain follow-up work.

For the underlying mathematics and implemented algorithm, read the [two-page technical report](docs/paper/point2text-technical-report.pdf), with [reproducible source](docs/paper/README.md).

## License

Project code, documentation, and the original bundled corpus are [MIT licensed](LICENSE). Downloaded model weights and third-party dependencies retain their own licenses; they are not redistributed here.
