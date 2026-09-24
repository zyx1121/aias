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

Running a local model on Windows used to mean an afternoon of WSL networking, Docker setup and GPU flags, then remembering all of it next time. aias does that setup once, keeps it running, and hands the controls to your agent over MCP. You ask for a model; the agent pulls it, loads it, and gives you an OpenAI compatible URL. It can also tell who spoke when in a recording, with NVIDIA's Nemotron 3 Diarization.

```
> "load Qwen3 0.6B on vLLM and tell me when it answers"
  ⚡ model_up { engine: "vllm", model: "Qwen/Qwen3-0.6B" }
  ⚡ job_status { job_id: "3f9a1c2e" }
✓ Qwen/Qwen3-0.6B is up at http://127.0.0.1:8000/v1
```

## What it does

- **Installs** WSL, Docker, the NVIDIA container toolkit, Ollama and vLLM into one dedicated WSL distro
- **Diarizes** audio from a URL into RTTM with a NeMo engine, built the first time you load it
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
| `status` | The engine and model that are up, their base URL, GPU memory, running jobs |
| `model_list` | Models already on disk, per engine, with size |
| `model_pull` | Download an Ollama model (`qwen3:8b`) or a Hugging Face repo for vLLM (`Qwen/Qwen3-0.6B`) or nemo (`nvidia/Nemotron-3-Diarization`); returns a job |
| `model_up` | Load a model and stop any other engine; returns a job |
| `model_down` | Stop every engine and free the GPU; cancels a running job |
| `diarize` | Label who spoke when in an audio URL, up to 8 speakers; needs nemo up; returns a job |
| `job_status` | Progress of a pull, up or diarize job; waits up to 120 s for it to finish; a done diarize job carries the RTTM |
| `logs` | Recent log lines of an engine |

## Examples

| You say | The agent runs |
|---------|-------------|
| "What models do I have?" | `model_list {}` |
| "Get qwen3:8b and load it" | `model_pull { engine: "ollama", model: "qwen3:8b" }`, then `model_up` |
| "Free the GPU" | `model_down {}` |
| "Who speaks when in this recording?" | `model_up { engine: "nemo", model: "nvidia/Nemotron-3-Diarization" }`, then `diarize { audio_url: "https://..." }` |

## Diarization

The nemo engine runs [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), a 100M parameter streaming Sortformer that labels up to 8 speakers, overlaps included. The first `model_up` with `engine: "nemo"` builds its image (about 11 GB, 5 minutes); later starts reuse the build cache and take about 30 s, and an upgrade that changed the engine rebuilds it. nemo loads only Hugging Face repos under `nvidia/`, because a `.nemo` archive can carry code.

`diarize` downloads `audio_url` on the server, resamples it to 16 kHz mono, and returns a job. Limits:

- http or https, any format ffmpeg reads, up to 2 GB and 10 minutes of download
- up to 2 hours of audio, so offline mode fits in 10 GB of GPU memory
- the host, and every redirect, must resolve to a public address: private, loopback and link-local addresses are refused

The finished job's `result` holds:

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

## How it works

Setup imports Ubuntu 24.04 as a WSL distro named `aias` under `C:\ProgramData\aias`, and a scheduled task keeps it running from boot. Inside it, Docker Compose runs the MCP server permanently and starts Ollama, vLLM or nemo on demand, one at a time. Every port is published on 127.0.0.1 only, and the MCP server rejects requests whose Host header is not local, so nothing is reachable from the network.

| Endpoint | Serves |
|----------|--------|
| `http://127.0.0.1:11400/mcp` | MCP server |
| `http://127.0.0.1:11434/v1` | Ollama, OpenAI compatible |
| `http://127.0.0.1:8000/v1` | vLLM, OpenAI compatible |

nemo has no port of its own: only the MCP server talks to it.

## Limitations

- NVIDIA only: vLLM and the container toolkit need CUDA.
- One model at a time: a consumer GPU cannot hold two. The diarization model counts as one.
- `diarize` takes a public URL, not a local file or a LAN address: upload the recording somewhere reachable from the internet first.
- Docker must not run in another WSL distro at the same time, because all WSL2 distros share one network namespace. Setup checks and stops if it does.
- Unsigned installer: Windows SmartScreen asks before it runs.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

[MIT](LICENSE) · one GPU, zero YAML to remember
