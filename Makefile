PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip

.PHONY: install test fetch-data process-data train reproduce

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

test:
	$(VENV_PYTHON) -m pytest -q

fetch-data:
	$(VENV_PYTHON) scripts/data/fetch_data.py

process-data:
	$(VENV_PYTHON) scripts/data/process_data.py

train:
	$(VENV_PYTHON) scripts/training/train_logistic.py
	$(VENV_PYTHON) scripts/training/train_random_forest.py

reproduce: process-data train
	$(VENV_PYTHON) scripts/training/generate_results_csv.py
	@echo "Reproducible pipeline complete. Check results/ and data/images/ outputs."
