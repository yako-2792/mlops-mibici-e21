FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv && \
    uv sync --no-dev

COPY mibici_e21/ ./mibici_e21/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mibici_e21.api:app", "--host", "0.0.0.0", "--port", "8000"]