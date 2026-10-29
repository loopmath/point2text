# Native diffusion runtime

`point2text.diffusion.runtime.MLXDiffusionRuntime` implements the shared
`DiffusionRuntime` protocol. It runs real DiffusionGemma on Apple Silicon;
there is no autoregressive fallback in this class.

The audited first stack is Python 3.12, `mlx-vlm==0.7.11`, `mlx==0.32.4`,
and checkpoint `mlx-community/diffusiongemma-26B-A4B-it-4bit` at revision
`a7a81407613811e8ba63af92ac0d852b809e191f`. Runtime dependencies stay optional.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'mlx-vlm==0.7.11'
```

```python
from point2text.diffusion.contracts import DiffusionRequest
from point2text.diffusion.runtime import MLXDiffusionRuntime

runtime = MLXDiffusionRuntime()
result = runtime.generate(DiffusionRequest(
    "Write one concise research paragraph about sparse expert routing.",
    seed=42, max_tokens=128, steps=32,
))
print(result.text)
print(result.trace)
```

The tokenizer property loads tokenizer files on first use. Frozen model weights
load on the first generation. A shared lock serializes native generation because
MLX's RNG is global. Each request gets a fresh native hybrid prompt KV cache.

## Hook seam

The hook receives `DiffusionStep` with native MLX arrays:

- `logits`: `[1, canvas_length, vocabulary_size]` after the native linear
  temperature schedule, detached from the generator.
- `token_ids`, `editable_mask`: `[1, canvas_length]`.
- `step`: zero-based denoising pass; `num_steps`: requested pass budget.
- `tokenizer`: actual HF tokenizer; `context_ids`: formatted prompt IDs.

Return finite MLX logits with the original shape. Every argmax, categorical
draw, confidence/entropy calculation, acceptance decision, early stopping test,
and self-conditioning update consumes those returned logits. Hooks must not
consume MLX's global RNG. An identity hook should preserve seeded token IDs.
Hook validation and tracing add some overhead.

The default is the checkpoint's entropy-bound sampler, for which all canvas
positions can be re-edited. The optional confidence-threshold sampler marks only
unrevealed positions editable. Its commit/final argmax behavior follows the
audited native implementation. A guidance adapter must honor editable masks.

The adapter retains native helper functions rather than copying model internals.
Its explicit single-canvas loop preserves dynamic prompt caching, decoder masks,
linear temperature shaping, random initialization/renoising, native confidence
or entropy acceptance, stopping, and self-conditioning. It intentionally does
not enable compilation, multimodal input, prefix-cache reuse, or multi-canvas
generation. Request lengths are constrained to 1–256 tokens; the first demo uses
128 tokens and 32 denoising passes. Cancellation is checked between passes and
returns no text from an unfinished canvas.

The public [mlx-vlm source](https://github.com/Blaizzy/mlx-vlm/blob/217548e24d0d1586b4981c3b50d2f1d3a8e64975/mlx_vlm/generate/diffusion.py)
matches the installed 0.7.11 diffusion module (SHA-256
`76e88e4faa8823feb278e7078b714244c6a570a5d3d990cc4ad1863e1a75c8f8`).
The adapter rejects other mlx-vlm versions until their internals are audited.
The [checkpoint configuration](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit/blob/a7a81407613811e8ba63af92ac0d852b809e191f/config.json)
records the released sampling settings and mixed quantization.

## Verification

`tests/test_diffusion_runtime.py` exercises real tiny MLX tensors with a fake
model: identity parity, native sampler interception, post-hook self-conditioning,
native tensor shapes, invalid hook rejection, and cancellation. It skips when
MLX/MLX-VLM are unavailable and does not download a checkpoint.

Passing runtime tests demonstrates correct plumbing. Steering effectiveness
requires measuring decoded, re-tokenized text with the real frozen sentence
encoder and fixed map; lower soft loss alone is insufficient.

The opt-in `tests/smoke_diffusion_runtime.py` loads the real checkpoint and
compares unguided generation, an identity hook, the unmodified native mlx-vlm
sampler, and a controlled one-position intervention. It writes a JSON evidence
file under ignored `artifacts/` by default.

On 29 October 2026, this smoke test passed on the local M4 Max / 128 GiB:
identity token parity, native sampler token parity, interception of real
categorical sampler inputs, and changed final tokens under intervention. The
128-token request produced an actual `[1, 128, 262144]` canvas. The first
standalone model load plus unguided generation took 13.43 seconds, with 14
denoising passes, 1.15 seconds prompt prefill, and 17.48 GB peak MLX allocation.
These are one-machine observations, not a cross-hardware benchmark. Structured
leading Gemma channel metadata is removed before returning paragraph text;
ordinary content is decoded from its actual generated IDs.
