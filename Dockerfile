FROM python:3.14-alpine AS base

WORKDIR /app

FROM base AS builder

RUN apk update && apk add build-base
RUN pip install poetry==2.2.1

# Install split into two steps (the dependencies and the sources)
# in order to leverage the Docker caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root \
  --no-directory --only main

COPY . ./
RUN poetry install --no-interaction --no-ansi --no-cache --only main


FROM base AS server

RUN adduser -u 1001 --disabled-password --gecos "" appuser
USER appuser

# Copy the sources and virtual env. No poetry.
COPY --chown=appuser --from=builder /app .

ENV PATH="/app/.venv/bin:$PATH"

ENV DEEPEVAL_TELEMETRY_OPT_OUT=1

EXPOSE 5000
CMD ["uvicorn", "aidial_admin_evaluation_metrics.main:app", "--host", "0.0.0.0", "--port", "5000"]
