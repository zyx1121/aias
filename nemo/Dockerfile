FROM python:3.12-slim-bookworm
# gcc and the libc headers stay in the image: NeMo's flex_attention makes
# Triton compile kernels on the first request.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.18
# NeMo 3.0.0 on PyPI rejects the model's RoPE attention, so pin a main commit
# that loads nvidia/Nemotron-3-Diarization.
ARG NEMO_SHA=cf724ac337d1ebc7d0dda1e23fb80916f52927a5
RUN uv pip install --system --no-cache Cython packaging \
 && uv pip install --system --no-cache \
      torch==2.14.0 \
      "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_SHA}" \
      starlette==1.7.0 uvicorn==0.53.0 httpx==0.28.1
ENV TRITON_CACHE_DIR=/cache/triton TORCHINDUCTOR_CACHE_DIR=/cache/inductor
WORKDIR /app
COPY server.py .
EXPOSE 8100
CMD ["python", "server.py"]
