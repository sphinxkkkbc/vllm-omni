# AuK-Flash (experimental)

Two independent `LLM_GENERATION` stages: Stage 0 runs the upstream Transformers
Qwen2.5-Omni Thinker once (`use_cache=False`) and the official AuK VAE encoder.
Stage 1 owns unpadded request-local flow state and executes one Flux2Edit step
per scheduler invocation, followed by a per-request Euler update. The four-step
Flash schedule and CFG-off recipe are fixed. Final VAE decode returns 24 kHz audio.

The full-payload connector and generation runner are reused unchanged. The thin
model-local `AukGenerationScheduler` inherits `OmniGenerationScheduler`, strips
the internal completion tensor and rearms an incomplete work item without
appending tokens. It does not implement a new queue or token sampler. Both stages
use separate engine execution loops, with `async_chunk: false` and
`async_scheduling: false`. Inter-stage transport is still asynchronous. The
default deployment uses two GPUs so new Qwen encodes cannot occupy the flow GPU.
TP/PP greater than one and AuK Base/CFG are rejected in this first version.

## Files and reuse

- `auk_condition.py`: full condition schema, trained layer fusion and Stage 0.
  Reuses upstream Thinker, processor and official VAE, without a Qwen fork.
- `auk_flow.py`: request state, padding materialization, logical RoPE positions,
  single-step graph input buffers and Stage 1. Reuses official Flux2Edit blocks,
  VAE and vLLM `CUDAGraphWrapper`. Mirrors the diffusion step-state/denoise/update
  split without importing its engine or changing a runner interface.
- `configuration_auk.py`, `pipeline.py`, `scheduler.py`, `prompt_utils.py`:
  configuration, topology, lifecycle adapter and shared online/offline prompts.

IndexTTS 2.5 [PR #5957](https://github.com/vllm-project/vllm-omni/pull/5957)
provides the reference for full-payload transfer, model-local length-aware CFM
batching and list-valued output splitting. Its full-loop CFM execution is not
used: AuK returns to scheduling after each flow step to allow new admission.

## Install and prepare

Install vLLM-Omni and the official [AuK source](https://github.com/Tencent-Hunyuan/AuK).
AuK's package pins an older PyTorch stack: install its source with `--no-deps`
into the existing vLLM environment, then provide `omegaconf`, `torchdiffeq`,
`x-transformers`, `qwen-omni-utils` and the Qwen2.5-Omni-capable Transformers
version appropriate for that environment. This combination still requires GPU
validation; do not replace the vLLM PyTorch stack with AuK's pinned version.

Download the official Flash, VAE and Qwen checkpoints. Create serving metadata
in the local Flash directory (supply the actual release YAML and weight names):

```bash
python examples/offline_inference/text_to_speech/auk/prepare_config.py \
  --model /path/to/AuK-Flash --config RELEASE.yaml \
  --checkpoint RELEASE.safetensors --qwen-path /path/to/Qwen2.5-Omni-3B
```

The command refuses to overwrite an existing `config.json`. If one exists,
merge the AuK-specific fields into it manually. Official weights are unchanged.

## Offline

```bash
python examples/offline_inference/text_to_speech/auk/end2end.py \
  --model /path/to/AuK-Flash --duration-seconds 3 --output auk.wav
```

Both clients support `--ref-audio reference.wav` and `--seed`. The initial
speech API requires explicit `duration_seconds`; no duration predictor is invented.

## Online

```bash
vllm serve /path/to/AuK-Flash --omni --deploy-config vllm_omni/deploy/auk.yaml
python examples/online_serving/text_to_speech/auk/speech_client.py \
  --model /path/to/AuK-Flash --duration-seconds 3 --output auk.wav
```

`instructions` supplies a voice/style instruction through the speech API.
`voice=default` is supported; use `ref_audio` for reference conditioning.

## Padding and CUDA graphs

`bucket_granularity` defaults provisionally to 32 and is configurable in
`config.json`; it is not a model constraint or a benchmark-selected optimum.
Semantic, reference and target each have their own roundup dimension. Real
lengths determine RoPE positions across `[text | ref | target]`. The official
positional convolution masks each convolution and embeds ref/target separately.

To opt into startup capture, set `graph_buckets` to a finite list of physical
`[batch, semantic, reference, target]` shapes, for example `[[1, 128, 128, 160]]`.
The descriptor also includes every tensor's dtype/device. Captures contain only
Flux2Edit; Euler updates and request lifecycle run outside the graph. All runtime
inputs, including masks and conditions, are copied into static buffers. No step
index is part of the key. Unconfigured shapes fall back to eager. Graphs are
not captured on live requests. Benchmark representative lengths and concurrency
before selecting bucket granularity and startup capture shapes.

## Tests

```bash
pytest -q tests/model_executor/models/auk
AUK_MODEL=/path/to/AuK-Flash pytest -q tests/e2e/offline_inference/test_auk.py
AUK_MODEL=/path/to/AuK-Flash pytest -q tests/e2e/online_serving/test_auk.py
```

Unit tests include real official small Flux2Edit blocks with nonzero output
weights, exact-vs-roundup parity, torchdiffeq four-step parity, fusion, reference
encoding contract, payload serialization, different-step batching, lifecycle,
and online/offline prompt parity. CUDA graph parity is CUDA-gated. The E2E
suites require two H100-class GPUs and a prepared release; they check complete
finite audio and expected duration under concurrent/different-length requests.
They must be run before claiming production readiness or full-checkpoint parity.
