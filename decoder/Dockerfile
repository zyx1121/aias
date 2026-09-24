# Decodes untrusted audio, away from the MCP server and its Docker socket.
# compose.yaml runs it with no network, a read-only root, no capabilities,
# no_new_privs, as nobody, with memory and process limits.
FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY decode.sh /decode.sh
ENTRYPOINT ["/decode.sh"]
