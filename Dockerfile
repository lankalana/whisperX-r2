FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS builder
WORKDIR /build
RUN apt-get update && apt-get install --no-install-recommends -y \
	python3.12-venv \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv venv
COPY pyproject.toml /build/pyproject.toml
RUN /build/venv/bin/python -m pip install --no-cache-dir --no-compile .

# Install whisperx separately to allow layer caching
COPY ./whisperx /build/whisperx
RUN /build/venv/bin/python -m pip install --no-cache-dir --no-compile --no-deps .

# Pre-compile the virtual environment to speed up startup time. Increases build size but reduces runtime overhead.
RUN /build/venv/bin/python -m compileall /build/venv

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS whisperx
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y \
	python3.12 libpython3.12 ffmpeg \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/venv /opt/whisperx-venv

ENV HF_HOME=/app/.cache/hf
ENV TORCHINDUCTOR_CACHE_DIR=/app/.cache/torch
ENV TRITON_CACHE_DIR=/app/.cache/triton

ENTRYPOINT ["/opt/whisperx-venv/bin/python", "-m", "whisperx"]
