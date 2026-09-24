FROM docker:29-cli AS docker

FROM python:3.12-slim
# ffmpeg decodes every audio URL to 16 kHz mono WAV before an engine sees it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
# buildx lets compose build the nemo image on its first model_up.
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
RUN pip install --no-cache-dir mcp==2.2.0 huggingface_hub==1.32.0 httpx opencc==1.4.2
WORKDIR /app
COPY server.py audio.py vram.py ./
EXPOSE 11400
CMD ["python", "server.py"]
