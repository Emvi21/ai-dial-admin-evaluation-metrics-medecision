# AI DIAL Admin Evaluation Metrics

## Overview

This project implements a metrics calculation service for the AI DIAL evaluation framework.

The service provides an API for evaluating AI model outputs against ground truth data using multiple metrics. The list of available metrics is provided dynamically through the API.

### API Documentation

Interactive API documentation is available when the service is running:
- **Swagger UI**: http://localhost:5001/docs
- **ReDoc**: http://localhost:5001/redoc
- **OpenAPI Schema**: http://localhost:5001/openapi.json

The static version of the API documentation is available in the [docs/api](./docs/api) folder:
- **Swagger UI**: [docs/api/swagger-ui.html](./docs/api/swagger-ui.html)
- **ReDoc**: [docs/api/redoc.html](./docs/api/redoc.html)
- **OpenAPI Schema**: [docs/api/openapi.json](./docs/api/openapi.json)

## Developer environment

This project uses [Python>=3.14](https://www.python.org/downloads/) and [Poetry>=2.2.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

### IDE configuration

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).

Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

### Make on Windows

As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

```sh
winget install GnuWin32.Make
```

For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

### Environment Variables

Copy `.env.example` to `.env` and customize it for your environment:

|Variable|Default|Description|
|---|---|---|
|LOG_LEVEL|INFO|Log level. Use `DEBUG` for development and `INFO` in production. Allowed values: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.|
|LOG_LEVEL_OVERRIDE|{}|Override log level for specific modules (JSON dict, e.g., `{"uvicorn.access":"INFO"}`).|
|DIAL_URL|*Required*|Base URL of the DIAL API endpoint (e.g., `https://your-dial-instance.example.com`). Used for all LLM and embeddings calls made by LLM-based metrics.|
|DIAL_API_KEY|*Empty*|API key for authenticating all requests to the DIAL instance. Used for all LLM and embeddings calls made by LLM-based metrics. Required for any LLM-based or embeddings-based metric to work. Without it, only metrics that do not call LLMs (e.g., `exact_match`, `regex_match`, `deepdiff`) will work.|
|EVAL__APP__MAX_CONCURRENT_EVALUATIONS|10|Maximum number of evaluation requests processed concurrently. Requests beyond this limit are queued.|
|EVAL__APP__MAX_QUEUE_BACKLOG|100|Maximum number of evaluation requests that can wait in the queue. Requests arriving when the queue is full are rejected with an error.|
|EVAL__METRICS__COMMON__SUPPORTED_MODELS|`gemini-2.5-flash-lite`, `gpt-5-nano-2025-08-07`, `anthropic.claude-haiku-4-5-20251001-v1:0`, `gemini-3.1-flash-lite-preview`, `gpt-5-mini-2025-08-07`, `gpt-5.4-mini-2026-03-17`|Default list of DIAL LLM deployment names available across all LLM-based metric groups. Applied to each metric group unless overridden by that group's own setting. This list of the models will be displayed to the user to choose from during the metrics configuration in the DIAL Admin Panel.|
|EVAL__METRICS__COMMON__DEFAULT_MODEL|`gemini-3.1-flash-lite-preview`|Default LLM deployment selected when no model is specified in a metric request. Applied to each metric group unless overridden by that group's own setting. Must be present in the supported models list. This model will be displayed to the user as a default value during the metrics configuration in the DIAL Admin Panel.|
|EVAL__METRICS__DEEPEVAL__SUPPORTED_MODELS|*from common*|Overrides the supported models list for DeepEval metrics only.|
|EVAL__METRICS__DEEPEVAL__DEFAULT_MODEL|*from common*|Overrides the default model for DeepEval metrics only.|
|EVAL__METRICS__DEEPEVAL__VERBOSE_MODE|false|When enabled, DeepEval prints intermediate evaluation steps to stdout (useful for debugging). Does not affect the API response content.|
|EVAL__METRICS__AIDIAL_RAG_EVAL__SUPPORTED_MODELS|*from common*|Overrides the supported models list for aidial-rag-eval metrics only.|
|EVAL__METRICS__AIDIAL_RAG_EVAL__DEFAULT_MODEL|*from common*|Overrides the default model for aidial-rag-eval metrics only.|
|EVAL__METRICS__AIDIAL_RAG_EVAL__MAX_CONCURRENCY|8|Maximum number of parallel LLM calls made internally by aidial-rag-eval metrics during a single evaluation.|
|EVAL__METRICS__RAGAS__SUPPORTED_MODELS|*from common*|Overrides the supported models list for Ragas metrics only.|
|EVAL__METRICS__RAGAS__DEFAULT_MODEL|*from common*|Overrides the default model for Ragas metrics only.|
|EVAL__METRICS__RAGAS__EMBEDDINGS_MODEL|`text-embedding-ada-002`|DIAL deployment name of the embeddings model used by Ragas metrics that require semantic similarity (e.g., `ragas.answer_relevancy`). Must be an embeddings deployment available on the configured DIAL instance.|

OpenTelemetry configuration is supported via standard environment variables. See [OpenTelemetry documentation](https://opentelemetry.io/docs/languages/sdk-configuration/general/) for available options.

## Run

Run the development server locally:

```sh
make serve
```

Run the development server in Docker:

```sh
make docker_serve
```

Open `localhost:5001/docs` to make sure the server is up and running.

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests:

```sh
make test
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```

## Build docs

To build the docs:

```sh
make docs
```
