# The web app. Build: docker build -t etf-app .
# data/ is not in the image: it's mounted at /app/data at runtime (PDFs aren't in git).
FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies first: this layer is rebuilt only when uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# the project itself, installed in editable mode, so paths.ROOT is /app
COPY README.md manifest.csv ./
COPY src ./src
RUN uv sync --locked --no-dev

# download the embedding model now, not on the first question
RUN python -c "from etf_assistant.retrieval.embeddings import embed_query; embed_query('warm up')"

RUN useradd --uid 1000 --no-create-home app && chown -R app /app/models
USER app

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["web"]
