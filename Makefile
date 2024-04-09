#* Variables
SHELL := /usr/bin/env bash
PYTHON ?= python3



#* Installation
.PHONY: project-init
project-init: poetry-install tools-install mypy-install

.PHONY: poetry-install
poetry-install:
	poetry install -n

.PHONY: poetry-lock-update
poetry-lock-update:
	poetry lock --no-update

.PHONY: poetry-export
poetry-export:
	poetry lock -n && poetry export --without-hashes > requirements.txt

.PHONY: poetry-export-dev
poetry-export-dev:
	poetry lock -n && poetry export --with dev --without-hashes > requirements.dev.txt

.PHONY: tools-install
tools-install:
	poetry run pre-commit install --hook-type prepare-commit-msg --hook-type pre-commit
	poetry run nbdime config-git --enable





#* Linting
.PHONY: mypy-install
mypy-install:
	poetry run mypy --install-types --non-interactive ./

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E "(.ipynb_checkpoints$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean-all
clean-all: pycache-remove build-remove 

#* Service targets
.PHONY: grep-todos
grep-todos:
	git grep -EIn "TODO|FIXME|XXX"
