ARG BASE_IMAGE=dustynv/pytorch:2.7-r36.4.0-cu128-24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/speaches-ai/speaches"
LABEL org.opencontainers.image.licenses="MIT"
# `ffmpeg` is installed because without it `gradio` won't work with mp3(possible others as well) files
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# "ubuntu" is the default user on ubuntu images with UID=1000. This user is used for two reasons:
#   1. It's generally a good practice to run containers as non-root users. See https://www.docker.com/blog/understanding-the-docker-user-instruction/
#   2. Docker Spaces on HuggingFace don't support running containers as root. See https://huggingface.co/docs/hub/en/spaces-sdks-docker#permissions
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH

RUN mkdir -p $HOME/miniconda3 && curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o $HOME/miniconda3/miniconda.sh && \
    bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3 && \
    rm $HOME/miniconda3/miniconda.sh && \
    echo "export PATH=$HOME/miniconda3/bin:\$PATH" >> $HOME/.bashrc

ENV PATH=$HOME/miniconda3/bin:$PATH
# Workaround to install torchcodec dependency with aarch64
# NOTE: Central repo doesn't support aarch64
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n speaches-env python=3.12
RUN conda run -n speaches-env conda install conda-forge::torchcodec==0.7.0 conda-forge::uv conda-forge::uvicorn
#COPY --chown=ubuntu pyproject.toml ./
#RUN p2c yaml --python 3.12 > environment.yml
#RUN conda env update -n speaches-env -f environment.yml
WORKDIR $HOME/speaches
COPY --chown=ubuntu . .
#RUN conda run -n speaches-env uv lock
#RUN conda run -n speaches-env uv sync --frozen --compile-bytecode --no-install-project --extra ui
RUN conda run -n speaches-env uv pip install .[ui]
RUN conda list -n speaches-env
RUN echo "source activate speaches-env" >> $HOME/.bashrc
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
#COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv
#COPY --chown=ubuntu pyproject.toml uv.lock ./
# NOTE: per https://docs.astral.sh/uv/guides/install-python, `uv` will automatically install the necessary python version
# https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
# TODO: figure out if `/home/ubuntu/.cache/uv` should be used instead of `/root/.cache/uv`
#RUN --mount=type=cache,target=/root/.cache/uv \
#    --mount=type=bind,source=uv.lock,target=uv.lock \
#    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#    uv sync --frozen --compile-bytecode --no-install-project --extra ui
#COPY --chown=ubuntu . .
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv export --frozen --all-groups --format requirements.txt > requirements.txt
#RUN grep -v '^-e ' requirements.txt > /tmp/requirements.txt

# Creating a directory for the cache to avoid the following error:
# PermissionError: [Errno 13] Permission denied: '/home/ubuntu/.cache/huggingface/hub'
# This error occurs because the volume is mounted as root and the `ubuntu` user doesn't have permission to write to it. Pre-creating the directory solves this issue.
RUN mkdir -p $HOME/.cache/huggingface/hub

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
#ENV PATH="$HOME/speaches/.venv/bin:$PATH"
#ENV PATH="/opt/conda/envs/rapids-env/bin:$PATH"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubenablehftransfer
# NOTE: I've disabled this because it doesn't inside of Docker container. I couldn't pinpoint the exact reason. This doesn't happen when running the server locally.
# RuntimeError: An error occurred while downloading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#donottrack
# https://www.reddit.com/r/StableDiffusion/comments/1f6asvd/gradio_sends_ip_address_telemetry_by_default/
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 8000
CMD ["conda", "run", "--no-capture-output", "-n", "speaches-env", "uvicorn", "--factory", "speaches.main:create_app"]
