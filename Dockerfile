FROM python:3.11.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY rules ./rules
COPY prompts ./prompts
RUN pip install --no-cache-dir uv==0.12.19 && uv sync --frozen --no-dev --extra server
RUN useradd --create-home aster && mkdir /app/data && chown aster /app/data
USER aster
EXPOSE 8000
CMD ["sh", "-c", ".venv/bin/uvicorn aster_gym.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-proxy-headers"]
