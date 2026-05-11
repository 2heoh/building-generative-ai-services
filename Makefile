.PHONY: help setup venv install run dev client test test-unit test-integration test-v lint clean

# Extra arguments passed after make target (e.g. make test tests/integration/test_minimum_functionality.py)
TEST_PATH := $(filter-out help setup venv install run dev client test test-unit test-integration test-v lint clean,$(MAKECMDGOALS))

# Prevent make from treating extra arguments as file targets
$(TEST_PATH):
	@:

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv .venv

setup: venv install ## Full setup: create venv and install dependencies

install: ## Install dependencies
	.venv/bin/pip install -r requirements.txt

run: ## Run production server
	.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

dev: ## Run dev server with auto-reload
	.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

client: ## Run Streamlit client
	.venv/bin/streamlit run client.py

test: ## Run tests (optional: specify path, e.g. make test tests/integration/test_llm.py)
	.venv/bin/python -m pytest -s -v $(TEST_PATH)

test-unit: ## Run unit tests only
	.venv/bin/python -m pytest -v -m unit

test-integration: ## Run integration tests only
	.venv/bin/python -m pytest -v -m integration

test-v: ## Run tests with extra verbosity
	.venv/bin/python -m pytest -vv

lint: ## Run linter (ruff)
	.venv/bin/ruff check .

clean: ## Remove cache and bytecode files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete