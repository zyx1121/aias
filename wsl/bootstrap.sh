#!/usr/bin/env bash
# Provision the aias distro: Docker, the NVIDIA container toolkit, the stack.
# Run as root inside the distro with systemd up: bootstrap.sh <app dir as a WSL path>
set -euo pipefail

SRC="${1:?usage: bootstrap.sh <app dir>}"
DEST=/opt/aias
export DEBIAN_FRONTEND=noninteractive

step() { printf '\n==> %s\n' "$*"; }

step "Installing Docker"
apt-get update -q
apt-get install -y -q docker.io docker-compose-v2 docker-buildx curl gpg ca-certificates

step "Installing the NVIDIA container toolkit"
keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "$keyring"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -q
apt-get install -y -q nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker

step "Installing the stack to $DEST"
install -d "$DEST"
rm -rf "$DEST/mcp" "$DEST/nemo"
cp "$SRC/wsl/compose.yaml" "$DEST/compose.yaml"
cp -r "$SRC/mcp" "$DEST/mcp"
# Built by the MCP server on the first `model_up engine=nemo`, not here.
cp -r "$SRC/nemo" "$DEST/nemo"
compose=(docker compose -f "$DEST/compose.yaml")

step "Building and starting the MCP server"
"${compose[@]}" build mcp
"${compose[@]}" up -d mcp

step "Checking GPU access from a container"
"${compose[@]}" exec -T mcp nvidia-smi -L

step "Pulling engine images (about 31 GB, the slow part)"
"${compose[@]}" pull ollama vllm

step "Waiting for the MCP server"
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11400/health >/dev/null; then
    echo "aias is up: http://127.0.0.1:11400/mcp"
    exit 0
  fi
  sleep 2
done
echo "MCP server did not answer on :11400" >&2
"${compose[@]}" logs --tail 50 mcp >&2
exit 1
