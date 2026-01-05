SHELL := /bin/bash

setup-infra:
	docker compose -f infra/docker-compose.yml up -d

setup-index:
	uv run infra/opensearch/scripts/cli.py initialize-index-config $(model)

bulk-dataset:
	uv run infra/opensearch/scripts/cli.py bulk-dataset $(dataset) --bulk-size ${bulk_size}

cleanup:
	uv run infra/opensearch/scripts/cli.py deregister-models

teardown:
	docker compose -f infra/docker-compose.yml down --volumes

lint:
	uv run ruff check .
	uv run pyrefly check .
