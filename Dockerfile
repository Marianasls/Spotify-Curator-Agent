# Builder (Compilação e Wheels)
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Gera wheels isoladas para evitar recompilação no container final
RUN pip install --no-cache-dir --user -r requirements.txt


# Runtime (Imagem Final Enxuta)
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Apenas dependências de runtime do sistema (curl para o healthcheck e libpq se usar postgres nativo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root antes de copiar arquivos
RUN useradd -m -u 1001 appuser

# Copia pacotes Python instalados no estágio anterior
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copia apenas o código necessário
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser web/ ./web/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Flags de hardening e produção para o Streamlit
ENTRYPOINT ["streamlit", "run", "web/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]