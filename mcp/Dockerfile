FROM docker:29-cli AS docker

FROM python:3.12-slim
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
RUN pip install --no-cache-dir mcp==2.2.0 huggingface_hub==1.32.0 httpx
WORKDIR /app
COPY server.py .
EXPOSE 11400
CMD ["python", "server.py"]
