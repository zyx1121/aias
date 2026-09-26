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

Running a local model on Windows used to mean an afternoon of WSL networking, Docker setup and GPU flags, then remembering all of it next time. aias does that setup once, keeps it running, and hands the controls to your agent over MCP. You ask for a model; the agent pulls it, loads it, and gives you an OpenAI compatible URL. It can also transcribe a recording with Whisper, tell who spoke when with NVIDIA's Nemotron 3 Diarization, and make sound effects and music from a text prompt with Meta's AudioGen and MusicGen.

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
- **Generates** sound effects and music from a text prompt (AudioGen, MusicGen), built the first time you load it
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
| `model_pull` | Download an Ollama model (`qwen3:8b`) or a Hugging Face repo for vLLM (`Qwen/Qwen3-0.6B`), nemo (`nvidia/Nemotron-3-Diarization`) or audio (`facebook/audiogen-medium`, `facebook/musicgen-medium`); returns a job |
| `model_up` | Load a model next to the others if it fits the VRAM budget; otherwise return a plan (`evict: "auto"` carries it out, `dry_run` only shows it); `pin` keeps a model from eviction; returns a job |
| `model_down` | Stop one model (`model`), or every engine and job when called without one |
| `diarize` | Label who spoke when in an audio URL, up to 8 speakers; needs nemo up; reserves its extra VRAM per file; returns a job |
| `transcribe` | Speech to text with segment timestamps from an audio URL; needs Whisper up on vLLM; `diarize: true` also labels speakers (needs nemo up too); returns a job |
| `generate_audio` | Sound effects or music from a text prompt, 0.5 to 30 s; needs audio up; reserves its extra VRAM per clip; returns a job whose result has a URL to the WAV |
| `job_status` | Progress of a pull, up, diarize, transcribe or generate job; waits up to 120 s for it to finish; a done diarize, transcribe or generate job carries its output |
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
| "Who said what in this meeting?" | `model_up` Whisper and nemo, then `transcribe { audio_url: "https://...", diarize: true }` |
| "Make a 20 second lo-fi beat" | `model_up { engine: "audio", model: "facebook/musicgen-medium" }`, then `generate_audio { prompt: "lo-fi hip hop beat with soft piano", duration_s: 20 }` |
| "Make a 5 second sound of a door creaking" | `model_up { engine: "audio", model: "facebook/audiogen-medium" }`, then `generate_audio { prompt: "a wooden door creaking open", duration_s: 5 }` |

## Sharing the GPU

Several models stay up at once: one Ollama server with any number of models, up to 10 vLLM models (each in its own container on its own port, 8000 to 8009, lowest free first, so a lone model is still on 8000), one nemo model and one audio model. aias keeps a ledger of what each holds and admits a new model only when

- the ledger plus the new need fits the budget: card memory − 1024 MiB for the desktop − 512 MiB margin (8704 MiB on a 10 GB card; `AIAS_DESKTOP_RESERVE_MIB` and `AIAS_SAFETY_MIB` override them), and
- nvidia-smi shows at least the need + 512 MiB free.

The ledger decides, not nvidia-smi: under WSL an overcommitted card does not fail, the driver quietly moves memory to system RAM and everything slows down.

| Need of | Comes from, first that exists |
|---------|-------------------------------|
| any model | `vram_mib` passed to `model_up` (1 to the card size); it wins over a measurement, so a wrong one can be overridden |
| any model | a measurement: nvidia-smi before and after a start with no other job running and nothing stopped for it, kept in the `aias-state` volume |
| vLLM | `gpu_memory_utilization` x card: 0.4 for Whisper, 0.8 otherwise |
| Ollama | what `/api/ps` reports once loaded, + 250 MiB; before that the model file + its KV cache at the 32k context (layers, KV heads and head size from `/api/show`) + 250 MiB, or twice the file if the metadata is missing |
| nemo | 1300 MiB, plus a per-file reservation during `diarize`: the highest MiB per minute measured on a file of 10 minutes or more (39.2 on king), or 40 before such a run, and at least 128 MiB |
| audio | per model, measured on king: 5570 MiB for AudioGen, 4740 for MusicGen, after the warmup clip each engine start runs; plus a per-clip reservation during `generate_audio` that grows with the clip (AudioGen 76 MiB per second up to 10 s and 40 past it, MusicGen 100 per second), at least 128 MiB, scaled up by the highest measured-to-estimate ratio of a clip of 5 s or more |

vLLM normally sizes its KV cache from whatever is free on the card, so next to another engine it would take a different amount each time. aias passes `--kv-cache-memory` instead (450 MB for Whisper; for other models the budget left after the weights of the current revision and 1.5 GiB of overhead), so an instance takes what it was budgeted for. If that KV cache cannot hold one `max_model_len` sequence (layers, KV heads and head size from the model's `config.json`), `model_up` refuses and says which `vram_mib` would.

When a model does not fit, `model_up` changes nothing and returns `{refused, fits, need_mib, free_mib, evict}`. `evict` is the fewest models to stop, never pinned ones or ones running a job. Among sets of that size, it compares the most recently used member of each and picks the set where that one was used longest ago (then the next most recent, and so on), so the models used last are the last to go; pass `evict: "auto"` to stop exactly those. The job plans again once it holds the placement lock and stops the whole set or nothing. The lock covers only that decision: the model is booked (listed, not ready) and the lock released before it starts, so another model or a diarization can be admitted while a vLLM model spends a minute or two loading; a start that fails or is cancelled gives its booking back. If even that is not enough, `evict` is empty and the reason says so. nemo and audio hold one model each, so asking for a different one replaces it (unless it is pinned or busy); vLLM models do not replace each other.

Every 10 s aias compares nvidia-smi with the ledger. If more is in use than the ledger and the desktop reserve explain, `status` shows `pressure: true` and new models are refused until it clears; nothing running is stopped. `over_budget` means clients loaded more into Ollama directly than the budget allows, and `cpu_offload` on an Ollama model means Ollama itself put part of it in system memory.

## Audio input

`diarize` and `transcribe` take an `audio_url`. The MCP server downloads it, decodes it to 16 kHz mono, and only then hands it to the engine. The decoding runs in a separate `decoder` container, one `compose run` per file: no network (`network_mode: none`), a read-only root with a small tmpfs, no capabilities, `no-new-privileges`, uid 65534, memory and process limits, no GPU and no Docker socket; inside it `prlimit` caps the output file and address space. It sees the shared `audio-work` volume, where only its own job directory is open to it. A hostile file therefore cannot reach the Docker socket the MCP server holds, or the network. Limits:

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

With several Whisper models up, `model` picks one; left out, the one used most recently. The result's `model` says which ran.

### Transcript with speakers

`transcribe` with `diarize: true` decodes the audio once and sends the same WAV to Whisper and to nemo (offline mode) at the same time. Both models must already be up; the error names the `model_up` that is missing. The diarization's GPU reservation follows the same rule as `diarize` (refused with a plan unless `evict: "auto"`).

Each Whisper segment gets the speaker who talks the most during it:

| Field | Meaning |
|-------|---------|
| `speaker` | Label from the diarization (`speaker_0`, ...), or null if nobody was labelled during the segment |
| `speaker_confidence` | Share of the segment that speaker talks, 0 to 1 |
| `speaker_uncertain` | True when that share is under 0.3, or a second speaker talks 0.3 or more of it (the segment straddles a change) |

How long a file fits depends on what else is up: the diarization reservation must fit in (budget − models up) ÷ MiB per minute. On a 10 GB card with Whisper (4187 MiB) and nemo (843 MiB) up, that is (8704 − 4187 − 843) ÷ 39.2 ≈ 93 minutes (92 at the default 40 MiB per minute, before a long file has been measured); nemo alone takes the full 2 hours. `status` shows the current figure as `max_audio_minutes` on the nemo entry: the longest file (rounded down to 0.1 minute) that the reservation check would admit right now without evicting anything, computed by that same check, so it is 0 under memory pressure or when not even the 128 MiB minimum fits. While nemo is running a file, the next one waits for it, and `max_audio_minutes_next` gives the figure once that file's reservation is released. A refused run gives the whole minutes that fit.

Segments are not split: Whisper cuts on pauses, and a split at a guessed word boundary would be less reliable than the flag. The result also holds `speakers` (seconds from the diarization and segments per speaker), `turns` (adjacent segments of one speaker merged, with `start`, `end`, `text`), the `rttm`, and `transcribe_s` / `diarize_s` next to `elapsed_s`.

## Audio generation

The audio engine turns an English description into audio with one of two models, one at a time (asking for the other replaces it). Both run on Meta's [AudioCraft](https://github.com/facebookresearch/audiocraft):

| Model | Makes | Output | Longest clip | License of the output |
|-------|-------|--------|--------------|-----------------------|
| [facebook/audiogen-medium](https://huggingface.co/facebook/audiogen-medium) | Sound effects | 16 kHz mono | 30 s | CC BY-NC 4.0 |
| [facebook/musicgen-medium](https://huggingface.co/facebook/musicgen-medium) | Music | 32 kHz mono | 30 s | CC BY-NC 4.0 |

`model_pull` fetches each with the text encoder it loads (t5-large for AudioGen, t5-base for MusicGen). The first `model_up` with `engine: "audio"` builds the image (10 GB, 5 minutes); later starts reuse the build cache. audio loads only these two repos, because AudioCraft reads its weights with pickle, which can carry code.

| Argument | Meaning |
|----------|---------|
| `prompt` | What it should sound like, in English: `"dog barking in the distance, light rain"`; up to 500 characters |
| `duration_s` | 0.5 to 30 s, 5 by default. AudioGen is trained on 10 s clips and extends a longer one 5 s at a time, which drifts more; MusicGen is trained on 30 s clips |
| `seed` | Makes a run repeatable; left out, a random one, which the result gives |
| `cfg_coef` | How closely it follows the prompt, 0 to 10, 3 by default |

One clip is generated at a time; the next waits. The finished job's `result` holds `url` (`http://127.0.0.1:11400/files/<name>.wav`, the same port as the MCP server, so it also works through an SSH tunnel to it that keeps port 11400 on the local side, as `/mcp` does), `audio_s`, `sample_rate`, `channels`, `seed`, `elapsed_s`, `gpu_peak_mib`, `license` and `expires`. Files are kept 24 hours and at most 1 GB, oldest out first. `status` gives the model's `makes`, `max_duration_s`, `sample_rate` and `license`.

Measured on an RTX 3080 (10 GB):

| Model | Start (cached, with warmup) | Idle VRAM | 5 s clip | 10 s clip | 30 s clip |
|-------|----------------|-----------|----------|-----------|-----------|
| AudioGen | 27 s | 5570 MiB | 14 s, +370 MiB | 31 s, +732 MiB | 101 s, +1488 MiB |
| MusicGen | 25 s | 4740 MiB | 10 to 14 s, +428 MiB | 21 to 25 s, +972 MiB | 90 to 95 s, +2718 MiB |

Each clip's extra VRAM is reserved while it runs, then given back. The same seed gives the same file.

Neither fits next to Whisper (4187 MiB) on a 10 GB card: `model_up` refuses with Whisper in `evict`, as for any other model. An 8 GB card (6656 MiB budget) holds either one with clips up to about 18 s (AudioGen) or 19 s (MusicGen); a 6 GB card cannot load them.

## How it works

Setup imports Ubuntu 24.04 as a WSL distro named `aias` under `C:\ProgramData\aias`, and a scheduled task keeps it running from boot. Inside it, Docker Compose runs the MCP server permanently and starts Ollama, vLLM, nemo and audio on demand, side by side within the VRAM budget. The MCP server fetches and decodes audio itself, so the engines never see a URL. Every port is published on 127.0.0.1 only, and the MCP server rejects requests whose Host header is not local, so nothing is reachable from the network.

| Endpoint | Serves |
|----------|--------|
| `http://127.0.0.1:11400/mcp` | MCP server |
| `http://127.0.0.1:11400/files/<name>.wav` | Sound files `generate_audio` made |
| `http://127.0.0.1:11434/v1` | Ollama, OpenAI compatible |
| `http://127.0.0.1:8000/v1` to `:8009/v1` | vLLM, OpenAI compatible, one port per model; `status` gives each model's `base_url` |

nemo and audio have no port of their own: only the MCP server talks to them. Each vLLM model runs in its own container made with `docker compose run`, named `aias-vllm-<port>` and labelled `aias.engine=vllm`, `aias.model`, `aias.port` and what the ledger booked for it; `compose stop` does not reach such containers, so aias finds, stops and (after a restart of the MCP server) re-adopts them by these labels. If a port is taken by something outside aias, the next free one is used. The single `aias-vllm-1` container of earlier versions is stopped when found running, and removed by `model_down {}`.

## Limitations

- NVIDIA only: vLLM and the container toolkit need CUDA.
- Up to 10 vLLM models, one nemo model, one audio model and any number of Ollama models at a time, as far as the budget allows.
- The audio engine needs a GPU of compute capability 9.0 or lower (RTX 20 to 40 series and older): AudioCraft pins torch 2.1, which has no kernels for RTX 50 series cards, so `model_up` refuses there.
- `generate_audio` takes English prompts and makes no speech; what it makes is for non-commercial use (the models' CC BY-NC 4.0 license).
- Models loaded into Ollama by other clients are tracked but not admitted: they can push the ledger past the budget (`over_budget`).
- `diarize` and `transcribe` take a public URL, not a local file or a LAN address: upload the recording somewhere reachable from the internet first.
- `transcribe` has no speaker labels; match its segments against a `diarize` RTTM by time.
- Docker must not run in another WSL distro at the same time, because all WSL2 distros share one network namespace. Setup checks and stops if it does.
- Unsigned installer: Windows SmartScreen asks before it runs.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

[MIT](LICENSE) · one GPU, zero YAML to remember
