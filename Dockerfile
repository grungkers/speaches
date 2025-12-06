ARG BASE_IMAGE=torchcodec:r36.4.tegra-aarch64-cp312-cu126-24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/speaches-ai/speaches"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR $HOME/speaches
COPY --chown=ubuntu . .
RUN uv pip install .[ui] --no-upgrade
RUN uv pip list -n speaches-env

RUN mkdir -p $HOME/.cache/huggingface/hub

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "--factory", "speaches.main:create_app"]
