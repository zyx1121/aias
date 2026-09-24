```
 █████╗ ██╗ █████╗ ███████╗
██╔══██╗██║██╔══██╗██╔════╝
███████║██║███████║███████╗
██╔══██║██║██╔══██║╚════██║
██║  ██║██║██║  ██║███████║
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝
```

# aias

> One installer turns a Windows PC with an NVIDIA GPU into a model host your agent drives by itself.

[![CI](https://github.com/zyx1121/aias/actions/workflows/ci.yml/badge.svg)](https://github.com/zyx1121/aias/actions) &nbsp;[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)

Running a local model on Windows used to mean an afternoon of WSL networking, Docker setup and GPU flags, then remembering all of it next time. aias does that setup once, keeps it running, and hands the controls to your agent over MCP. You ask for a model; the agent pulls it, loads it, and gives you an OpenAI compatible URL. It can also transcribe a recording with Whisper and tell who spoke when with NVIDIA's Nemotron 3 Diarization.

```
> "load Qwen3 0.6B on vLLM and tell me when it answers"
  ⚡ model_up { engine: "vllm", model: "Qwen/Qwen3-0.6B" }
  ⚡ job_status { job_id: "3f9a1c2e" }
✓ Qwen/Qwen3-0.6B is up at http://127.0.0.1:8000/v1
```

## What it does

- **Installs** WSL, Docker, the NVIDIA container toolkit, Ollama and vLLM into one dedicated WSL distro
- **Diarizes** audio from a URL into RTTM with a NeMo engine, built the first time you load it
- **Transcribes** audio from a URL into timestamped segments with Whisper on vLLM
- **Serves** an MCP server on `http://127.0.0.1:11400/mcp` that starts with Windows
- **Removes** all of it from Settings > Apps, including images and models

## Quickstart

1. Download `aias-setup-<version>.exe` from the latest CI run and run it. It needs admin rights, and pulls about 31 GB of engine images.
2. Add the MCP server to your agent:

```bash
claude mcp add --transport http aias http://127.0.0.1:11400/mcp
```

Any other MCP client:

```jsonc
{
  "mcpServers": {
    "aias": { "type": "http", "url": "http://127.0.0.1:11400/mcp" }
  }
}
```

> [!NOTE]
> Needs Windows 11 (or Windows 10 22H2) on x64, an NVIDIA GPU with a current driver, and about 60 GB of free disk.
> On a PC without WSL, setup enables it, asks for one restart, and continues after you log in.

## Tools

| Tool | Description |
|------|-------------|
| `status` | Each model that is up (base URL, VRAM and its source, pinned, last used, jobs), the VRAM budget, running jobs |
| `model_list` | Models already on disk, per engine, with size, the VRAM each needs, and whether it is loaded |
| `model_pull` | Download an Ollama model (`qwen3:8b`) or a Hugging Face repo for vLLM (`Qwen/Qwen3-0.6B`) or nemo (`nvidia/Nemotron-3-Diarization`); returns a job |
| `model_up` | Load a model next to the others if it fits the VRAM budget; otherwise return a plan (`evict: "auto"` carries it out, `dry_run` only shows it); `pin` keeps a model from eviction; returns a job |
| `model_down` | Stop one model (`model`), or every engine and job when called without one |
| `diarize` | Label who spoke when in an audio URL, up to 8 speakers; needs nemo up; reserves its extra VRAM per file; returns a job |
| `transcribe` | Speech to text with segment timestamps from an audio URL; needs Whisper up on vLLM; returns a job |
| `job_status` | Progress of a pull, up, diarize or transcribe job; waits up to 120 s for it to finish; a done diarize or transcribe job carries its output |
| `logs` | Recent log lines of an engine |

## Examples

| You say | The agent runs |
|---------|-------------|
| "What models do I have?" | `model_list {}` |
| "Get qwen3:8b and load it" | `model_pull { engine: "ollama", model: "qwen3:8b" }`, then `model_up` |
| "Free the GPU" | `model_down {}` |
| "Unload qwen3:8b, keep the rest" | `model_down { model: "qwen3:8b" }` |
| "Would Llama 8B fit next to Whisper?" | `model_up { engine: "ollama", model: "llama3.1:8b", dry_run: true }` |
| "Who speaks when in this recording?" | `model_up { engine: "nemo", model: "nvidia/Nemotron-3-Diarization" }`, then `diarize { audio_url: "https://..." }` |
| "Transcribe this recording" | `model_up { engine: "vllm", model: "openai/whisper-large-v3" }`, then `transcribe { audio_url: "https://..." }` |

## Sharing the GPU

Several models stay up at once: one Ollama server with any number of models, one vLLM model and one nemo model. aias keeps a ledger of what each holds and admits a new model only when

- the ledger plus the new need fits the budget: card memory − 1024 MiB for the desktop − 512 MiB margin (8704 MiB on a 10 GB card; `AIAS_DESKTOP_RESERVE_MIB` and `AIAS_SAFETY_MIB` override them), and
- nvidia-smi shows at least the need + 512 MiB free.

The ledger decides, not nvidia-smi: under WSL an overcommitted card does not fail, the driver quietly moves memory to system RAM and everything slows down.

| Need of | Comes from, first that exists |
|---------|-------------------------------|
| any model | `vram_mib` passed to `model_up` (1 to the card size); it wins over a measurement, so a wrong one can be overridden |
| any model | a measurement: nvidia-smi before and after a start with no other job running and nothing stopped for it, kept in the `aias-state` volume |
| vLLM | `gpu_memory_utilization` x card: 0.4 for Whisper, 0.8 otherwise |
| Ollama | what `/api/ps` reports once loaded, + 250 MiB; before that the model file + its KV cache at the 32k context (layers, KV heads and head size from `/api/show`) + 250 MiB, or twice the file if the metadata is missing |
| nemo | 1300 MiB, plus a per-file reservation during `diarize`: the highest measured MiB per minute of audio (about 39), or 43 before the first run |

vLLM normally sizes its KV cache from whatever is free on the card, so next to another engine it would take a different amount each time. aias passes `--kv-cache-memory` instead (450 MB for Whisper; for other models the budget left after the weights of the current revision and 1.5 GiB of overhead), so an instance takes what it was budgeted for. If that KV cache cannot hold one `max_model_len` sequence (layers, KV heads and head size from the model's `config.json`), `model_up` refuses and says which `vram_mib` would.

When a model does not fit, `model_up` changes nothing and returns `{refused, fits, need_mib, free_mib, evict}`. `evict` is the fewest models to stop, never pinned ones or ones running a job. Among sets of that size, it compares the most recently used member of each and picks the set where that one was used longest ago (then the next most recent, and so on), so the models used last are the last to go; pass `evict: "auto"` to stop exactly those. The job plans again once it holds the placement lock and stops the whole set or nothing. If even that is not enough, `evict` is empty and the reason says so. vLLM and nemo hold one model each, so asking for a different one replaces it (unless it is pinned or busy).

Every 10 s aias compares nvidia-smi with the ledger. If more is in use than the ledger and the desktop reserve explain, `status` shows `pressure: true` and new models are refused until it clears; nothing running is stopped. `over_budget` means clients loaded more into Ollama directly than the budget allows, and `cpu_offload` on an Ollama model means Ollama itself put part of it in system memory.

## Audio input

`diarize` and `transcribe` take an `audio_url`. The MCP server downloads it, decodes it to 16 kHz mono, and only then hands it to the engine. ffmpeg runs as `nobody` with `no_new_privs` and file size and memory limits (`prlimit` and `setpriv`), so a hostile file cannot reach the Docker socket the MCP server holds. Limits:

- http or https, any format ffmpeg reads, up to 2 GB and 10 minutes of download
- up to 2 hours of audio, so offline diarization fits in 10 GB of GPU memory
- the host, and every redirect, must resolve to a public address: private, loopback and link-local addresses are refused

## Diarization

The nemo engine runs [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), a 100M parameter streaming Sortformer that labels up to 8 speakers, overlaps included. The first `model_up` with `engine: "nemo"` builds its image (about 11 GB, 5 minutes); later starts reuse the build cache and take about 30 s, and an upgrade that changed the engine rebuilds it. nemo loads only Hugging Face repos under `nvidia/`, because a `.nemo` archive can carry code.

`diarize` returns a job. The finished job's `result` holds:

| Field | Meaning |
|-------|---------|
| `speakers` | Each speaker's label, total seconds and segment count |
| `rttm` | One `SPEAKER audio 1 <start> <duration> ... <speaker>` line per segment |
| `audio_s`, `elapsed_s`, `gpu_peak_mib` | Audio length, processing time, peak GPU memory |

`mode` picks the streaming latency from the model card. `offline` is the most accurate and the fastest on a whole file; the others simulate a live stream.

| Mode | Latency | 10 min of audio on an RTX 3080 |
|------|---------|-------------------------------|
| `offline` | 30.4 s | 1.3 s |
| `low` | 1.04 s | 34 s |
| `verylow` | 0.64 s | 45 s |
| `ultralow` | 0.32 s | 106 s |

Peak GPU memory in offline mode grows with length: 0.75 GB for 10 minutes, 3.5 GB for 87 minutes (11 s to process).

## Speech to text

`transcribe` runs a Whisper model on the vllm engine; [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) is the one tested. `model_up` notices a Whisper model and uses `max_model_len` 448 (Whisper's decoder limit) and `gpu_memory_utilization` 0.4 unless you pass them: that is about 4.2 GB, and 0.37 fails to start. It takes about 100 s to load.

| Argument | Meaning |
|----------|---------|
| `audio_url` | The recording; see [Audio input](#audio-input) |
| `language` | ISO 639-1 code, `zh` by default; a tag like `zh-TW` or `zh_CN` is cut to `zh` |
| `traditional` | Convert simplified Chinese characters to traditional (OpenCC `s2tw`), on by default; applies only to `zh` |

Whisper drifts from traditional to simplified Chinese after about 30 seconds, hence `traditional`. It converts characters only; it does not swap words such as 軟件 for 軟體, so the text stays what was said.

vLLM splits long audio into clips of up to 30 s by itself. The finished job's `result` holds `text` (the whole transcript), `segments` (each with `start`, `end` in seconds and `text`), `audio_s` and `elapsed_s`.

## How it works

Setup imports Ubuntu 24.04 as a WSL distro named `aias` under `C:\ProgramData\aias`, and a scheduled task keeps it running from boot. Inside it, Docker Compose runs the MCP server permanently and starts Ollama, vLLM and nemo on demand, side by side within the VRAM budget. The MCP server fetches and decodes audio itself, so the engines never see a URL. Every port is published on 127.0.0.1 only, and the MCP server rejects requests whose Host header is not local, so nothing is reachable from the network.

| Endpoint | Serves |
|----------|--------|
| `http://127.0.0.1:11400/mcp` | MCP server |
| `http://127.0.0.1:11434/v1` | Ollama, OpenAI compatible |
| `http://127.0.0.1:8000/v1` | vLLM, OpenAI compatible |

nemo has no port of its own: only the MCP server talks to it.

## Limitations

- NVIDIA only: vLLM and the container toolkit need CUDA.
- One vLLM model at a time (plus one nemo model and any number of Ollama models).
- Models loaded into Ollama by other clients are tracked but not admitted: they can push the ledger past the budget (`over_budget`).
- `diarize` and `transcribe` take a public URL, not a local file or a LAN address: upload the recording somewhere reachable from the internet first.
- `transcribe` has no speaker labels; match its segments against a `diarize` RTTM by time.
- Docker must not run in another WSL distro at the same time, because all WSL2 distros share one network namespace. Setup checks and stops if it does.
- Unsigned installer: Windows SmartScreen asks before it runs.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

[MIT](LICENSE) · one GPU, zero YAML to remember
