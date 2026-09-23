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

Running a local model on Windows used to mean an afternoon of WSL networking, Docker setup and GPU flags, then remembering all of it next time. aias does that setup once, keeps it running, and hands the controls to your agent over MCP. You ask for a model; the agent pulls it, loads it, and gives you an OpenAI compatible URL.

```
> "load Qwen3 0.6B on vLLM and tell me when it answers"
  ⚡ model_up { engine: "vllm", model: "Qwen/Qwen3-0.6B" }
  ⚡ job_status { job_id: "3f9a1c2e" }
✓ Qwen/Qwen3-0.6B is up at http://127.0.0.1:8000/v1
```

## What it does

- **Installs** WSL, Docker, the NVIDIA container toolkit, Ollama and vLLM into one dedicated WSL distro
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
| `model_pull` | Download an Ollama model (`qwen3:8b`) or a Hugging Face repo for vLLM (`Qwen/Qwen3-0.6B`); returns a job |
| `model_up` | Load a model and stop any other engine; returns a job |
| `model_down` | Stop every engine and free the GPU |
| `job_status` | Progress of a pull or up job; waits up to 120 s for it to finish |
| `logs` | Recent log lines of an engine |

## Examples

| You say | The agent runs |
|---------|-------------|
| "What models do I have?" | `model_list {}` |
| "Get qwen3:8b and load it" | `model_pull { engine: "ollama", model: "qwen3:8b" }`, then `model_up` |
| "Free the GPU" | `model_down {}` |

## How it works

Setup imports Ubuntu 24.04 as a WSL distro named `aias` under `C:\ProgramData\aias`, and a scheduled task keeps it running from boot. Inside it, Docker Compose runs the MCP server permanently and starts Ollama or vLLM on demand, one at a time. Every port is published on 127.0.0.1 only, and the MCP server rejects requests whose Host header is not local, so nothing is reachable from the network.

| Endpoint | Serves |
|----------|--------|
| `http://127.0.0.1:11400/mcp` | MCP server |
| `http://127.0.0.1:11434/v1` | Ollama, OpenAI compatible |
| `http://127.0.0.1:8000/v1` | vLLM, OpenAI compatible |

## Limitations

- NVIDIA only: vLLM and the container toolkit need CUDA.
- One model at a time: a consumer GPU cannot hold two.
- Docker must not run in another WSL distro at the same time, because all WSL2 distros share one network namespace. Setup checks and stops if it does.
- Unsigned installer: Windows SmartScreen asks before it runs.

## Contributing

Issues and PRs welcome: start with [CONTRIBUTING.md](https://github.com/zyx1121/.github/blob/main/CONTRIBUTING.md).

## License

[MIT](LICENSE) · one GPU, zero YAML to remember
