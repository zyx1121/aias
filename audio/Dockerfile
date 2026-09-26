FROM python:3.11-slim-bookworm
# ffmpeg for torchaudio and av; libsndfile for soundfile.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.18
# AudioCraft 1.3.0 pins torch 2.1.0 and pulls in training and demo packages
# (gradio, demucs, torchtext, ...). Install it without dependencies and add
# only what AudioGen inference imports, pinned: the image is built on the
# user's machine at the first model_up, so a new upstream release must not break it.
# av 12 instead of AudioCraft's pinned 11.0.0: 11.0.0 has no wheel here and
# builds from source; AudioGen only uses av.open to read audio. torchmetrics
# because audiocraft.models imports the solvers, which import the metrics.
# spacy 3.7.5, since 3.7.6 is yanked on PyPI.
RUN uv pip install --system --no-cache \
      torch==2.1.0 torchaudio==2.1.0 xformers==0.0.22.post7 numpy==1.26.4 \
      transformers==4.40.2 huggingface_hub==0.23.5 sentencepiece==0.2.0 \
      einops==0.8.2 flashy==0.0.2 hydra-core==1.3.7 hydra-colorlog==1.2.0 julius==0.2.8 num2words==0.5.14 \
      spacy==3.7.5 librosa==0.11.0 soundfile==0.14.0 av==12.3.0 \
      torchmetrics==1.9.0 \
      starlette==1.7.0 uvicorn==0.53.0 \
 && uv pip install --system --no-cache --no-deps audiocraft==1.3.0
WORKDIR /app
COPY server.py .
EXPOSE 8200
CMD ["python", "server.py"]
