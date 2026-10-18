#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-mibici-e21
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
HF ?= false
PUSH_TO_HF = $(if $(filter true,$(HF)),True,False)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements clean lint format test create_environment data check-hf ingest-stations ingest-trips ingest-trips-range

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

requirements: ## Install dependencies with uv
	uv sync

clean: ## Delete compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint: ## Check formatting and style with Ruff
	uv run ruff format --check
	uv run ruff check

format: ## Format the code with Ruff
	uv run ruff check --fix
	uv run ruff format

test: ## Run the unit tests with pytest
	uv run pytest tests/ -v

create_environment: ## Create the virtual environment with uv
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

data: requirements ## Run the dataset script
	uv run python mibici_e21/dataset.py

check-hf:
	@case "$(HF)" in \
		true|false) ;; \
		*) echo "Error: HF must be true or false. Example: make ingest-stations HF=true"; exit 1 ;; \
	esac

ingest-stations: check-hf ## Ingest the station catalog. Usage: make ingest-stations HF=true
	uv run python -c "from mibici_e21.ingestion.pipeline import IngestionPipeline; IngestionPipeline().ingest_stations(push_to_hf=$(PUSH_TO_HF))"

ingest-trips: check-hf ## Ingest one year of trips. Usage: make ingest-trips YEAR=2024 HF=true
	@if [ -z "$(YEAR)" ]; then echo "Error: YEAR is required. Example: make ingest-trips YEAR=2024 HF=true"; exit 1; fi
	uv run python -c "from mibici_e21.ingestion.pipeline import IngestionPipeline; IngestionPipeline().ingest_trips_year($(YEAR), push_to_hf=$(PUSH_TO_HF))"

ingest-trips-range: check-hf ## Ingest a range of trip years. Usage: make ingest-trips-range START=2015 END=2024 HF=true
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then echo "Error: START and END are required. Example: make ingest-trips-range START=2015 END=2024 HF=false"; exit 1; fi
	uv run python -c "from mibici_e21.ingestion.pipeline import IngestionPipeline; IngestionPipeline().ingest_trips_range($(START), $(END), push_to_hf=$(PUSH_TO_HF))"

download-data: ## Download the Processed (.parquet) directly from Hugging Face Hub
	uv run python -c "from mibici_e21.dataset import download_dataset; download_dataset()"

clean-temp: ## Clean temporary CSV files in data/temp/
	uv run python -c "from mibici_e21.dataset import clean_temp_files; clean_temp_files()"

clean-processed: ## Clean all processed Parquet files in data/processed/
	uv run python -c "from mibici_e21.dataset import clean_processed_files; clean_processed_files()"

clean-all: ## Clean both temporary and processed files
	uv run python -c "from mibici_e21.dataset import clean_all_data_files; clean_all_data_files()"
