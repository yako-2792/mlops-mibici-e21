#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_VERSION = 3.11
ENV_MANAGER ?= uv
MLFLOW_RUN = uv run mlflow run . --env-manager $(ENV_MANAGER)
TRACKING_SETTING = $(shell uv run --quiet python -c "from mibici_e21.config import TRACKING_SETTINGS; print(TRACKING_SETTINGS.$(1))")
ifndef MLFLOW_TRACKING_URI
MLFLOW_TRACKING_URI := $(call TRACKING_SETTING,TRACKING_URI)
endif
ifndef MLFLOW_EXPERIMENT_NAME
MLFLOW_EXPERIMENT_NAME := $(call TRACKING_SETTING,EXPERIMENT_NAME)
endif
export MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME
HF ?= false
PUSH_TO_HF = $(if $(filter true,$(HF)),True,False)
SAVE_CSV ?= false

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements clean lint format test create_environment check-hf ingest-stations ingest-trips ingest-trips-range download-data materialize-yearly-features materialize-features-range mlflow-server ingest train register predict clean-temp clean-raw clean-interim clean-processed clean-external clean-all

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-32s\033[0m %s\n", $$1, $$2}'

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

# Every uv.lock in this list that exists is audited.
AUDITED_PROJECTS = . validation

.PHONY: security-checks
security-checks: ## Scan the code with bandit, audit locked dependencies, and test the repo hygiene
	uv run bandit -r mibici_e21 --quiet
	@for project in $(AUDITED_PROJECTS); do \
		if [ -f $$project/uv.lock ]; then \
			echo "pip-audit $$project/uv.lock"; \
			uv export --project $$project --frozen --no-emit-project --format requirements.txt --quiet \
				| uv run pip-audit --strict --disable-pip --requirement /dev/stdin || exit 1; \
		fi; \
	done
	uv run pytest -v tests/test_security.py

check: security-checks

create_environment: ## Create the virtual environment with uv
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

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

clean-temp: ## Delete the contents of data/temp/
	uv run python -c "from mibici_e21.dataset import clean_temp_files; clean_temp_files()"

clean-raw: ## Delete the contents of data/raw/
	uv run python -c "from mibici_e21.dataset import clean_raw_files; clean_raw_files()"

clean-interim: ## Delete the contents of data/interim/
	uv run python -c "from mibici_e21.dataset import clean_interim_files; clean_interim_files()"

clean-processed: ## Delete the contents of data/processed/
	uv run python -c "from mibici_e21.dataset import clean_processed_files; clean_processed_files()"

clean-external: ## Delete the contents of data/external/
	uv run python -c "from mibici_e21.dataset import clean_external_files; clean_external_files()"

clean-all: ## Delete the contents of every directory under data/
	uv run python -c "from mibici_e21.dataset import clean_all_data_files; clean_all_data_files()"

materialize-yearly-features: ## Materialize one year of gold features. Usage: make materialize-yearly-features YEAR=2023
	@if [ -z "$(YEAR)" ]; then echo "Error: YEAR is required. Example: make materialize-yearly-features YEAR=2023"; exit 1; fi
	uv run python -c "from mibici_e21.features import materialize_yearly_trip_features; materialize_yearly_trip_features($(YEAR))"

materialize-features-range: ## Materialize a range of gold features. Usage: make materialize-features-range START=2015 END=2024
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then echo "Error: START and END are required. Example: make materialize-features-range START=2015 END=2024"; exit 1; fi
	uv run python -c "from mibici_e21.features import materialize_features_range; materialize_features_range($(START), $(END))"

mlflow-server: ## Start the local MLflow tracking server with a SQLite backend on port 5000
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000

ingest: ## Download the training year, build its gold files, and log the data manifest
	$(MLFLOW_RUN) -e ingest

train: ## Train through the MLflow Project. Usage: make train Q=p1 MODEL=rf PARAMS="-P n_estimators=200 -P max_depth=14"
	@if [ -z "$(Q)" ] || [ -z "$(MODEL)" ]; then echo "Error: Q and MODEL are required. Example: make train Q=p1 MODEL=baseline"; exit 1; fi
	$(MLFLOW_RUN) -e train -P question=$(Q) -P model=$(MODEL) $(PARAMS)

register: ## Promote the best candidate of a question to champion. Usage: make register Q=p1
	@if [ -z "$(Q)" ]; then echo "Error: Q is required. Example: make register Q=p1"; exit 1; fi
	$(MLFLOW_RUN) -e register -P question=$(Q) $(PARAMS)

predict: ## Score a year with the champion models. Usage: make predict YEAR=2024 SAVE_CSV=true
	@if [ -z "$(YEAR)" ]; then echo "Error: YEAR is required. Example: make predict YEAR=2024 SAVE_CSV=true"; exit 1; fi
	@case "$(SAVE_CSV)" in \
		true|false) ;; \
		*) echo "Error: SAVE_CSV must be true or false. Example: make predict YEAR=2024 SAVE_CSV=true"; exit 1 ;; \
	esac
	uv run python -m mibici_e21.modeling.predict --year $(YEAR) $(if $(filter true,$(SAVE_CSV)),--csv)
