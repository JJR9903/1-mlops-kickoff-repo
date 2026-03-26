FROM continuumio/miniconda3:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONPATH=/app
COPY conda-lock.yml .

RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    /opt/conda/envs/mlops/bin/pip install nicegui wandb python-dotenv && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

ENV PATH=/opt/conda/envs/mlops/bin:$PATH

COPY . .

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8050}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8050}"]
