PORT ?= 5001
IMAGE_NAME ?= ai-dial-admin-evaluation-metrics
PLATFORM ?= linux/amd64
POETRY ?= poetry
DOCKER ?= docker
ARGS ?=

.PHONY: all install build serve install_nox clean lint format test docker_build docker_serve docs help

all: build

install:
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_admin_evaluation_metrics.main:app" --host "0.0.0.0" --port $(PORT) --reload --env-file ./.env

install_nox:
	$(POETRY) install --only nox


clean: install_nox
	$(POETRY) run nox -s clean
	$(POETRY) env remove --all

lint: install_nox
	$(POETRY) run nox -s lint

format: install_nox
	$(POETRY) run nox -s format

test: install_nox
	$(POETRY) run -- nox -s test -- $(ARGS)

docs: install
	$(POETRY) run python scripts/generate_docs.py \
		--openapi docs/api/openapi.json \
		--swagger docs/api/swagger-ui.html \
		--redoc docs/api/redoc.html


docker_build:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .

docker_serve: docker_build
	$(DOCKER) run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo '-- DOCS --'
	@echo 'docs                         - generate API documentation from OpenAPI schema'
