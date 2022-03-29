.PHONY: clean clean-test clean-pyc clean-build dev venv help requirements-dev.txt
.DEFAULT_GOAL := help
-include .env

help:
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev:  ## Setup dev environment
	poetry install
	.venv/bin/pre-commit install

format: dev  ## Scan and format all files with pre-commit
	.venv/bin/pre-commit run --all-files

dvc: dev ## Install DVC and setup remote storage
	@test -s .env || { echo ".env does not exist! Exiting..."; exit 1; }
	poetry add dvc
	.venv/bin/dvc init
	gsutil mb -p $(GCP_PROJECT) gs://$(TM_PROJECT)
	.venv/bin/dvc remote add -d storage gs://$(TM_PROJECT)
	cat .templates/dvc/.pre-commit-config.yaml >> .pre-commit-config.yaml

test:  ## Run all tests
	.venv/bin/pytest -v

minimal:  ## Delete files to create minimally functional repository
	./ci/minimal.sh

jupyter: dev  ## Install jupyter and related config files
	sed -i -e '/repos:/r .templates/jupyter/.pre-commit-config.yaml' .pre-commit-config.yaml
	cp .templates/jupyter/.flake8_nb .
	poetry add ipykernel --dev
