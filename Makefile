PYTHON ?= .venv/bin/python
MICROMAMBA ?= /shared/home/rdelprete/bin/micromamba
MICROMAMBA_PREFIX ?=
PYTHONPATH ?= src
RELEASE_NAME ?= smoke_v1
SMOKE_OUTPUT_DIR ?= outputs/moe/smoke_$(shell date -u +%Y%m%d_%H%M%S)
ROUTERSET_DIR ?= routerset
REBUILT_MANIFEST ?= $(ROUTERSET_DIR)/multilabel_dataset/manifest_moe_train.jsonl
TRAIN_OUTPUT_DIR ?= outputs/moe/prepare_$(shell date -u +%Y%m%d_%H%M%S)
RUNTIME_ROOT ?= $(TRAIN_OUTPUT_DIR)/runtime
ACCELERATOR ?= cpu
DEVICES ?= 1
PRECISION ?=

ifeq ($(strip $(MICROMAMBA_PREFIX)),)
RUNNER = PYTHONPATH=$(PYTHONPATH) $(PYTHON)
else
RUNNER = PYTHONPATH=$(PYTHONPATH) $(MICROMAMBA) run -p $(MICROMAMBA_PREFIX) python
endif

.PHONY: clean-cache routerset-download routerset-materialize routerset-materialize-clean routerset-audit routerset-raw-audit train-prepare full-train smoketest smoketest-preflight
MATERIALIZED_DATASET_DIR ?= outputs/routerset/materialized_256
MATERIALIZED_CLEAN_DATASET_DIR ?= outputs/routerset/materialized_256_clean

clean-cache:
	rm -rf $(HOME)/.cache/* $(HOME)/.mamba/pkgs $(HOME)/.local/share/mamba/pkgs 2>/dev/null || true
	$(MICROMAMBA) clean --all -y || true
	find /tmp /var/tmp -maxdepth 1 -user "$$(id -u)" \
		\( -name 'pip-*' -o -name 'build-env-*' -o -name 'build-reqs-*' -o -name 'uv-*' -o -name 'pytest-of-*' -o -name 'pymp-*' -o -name '.tmp*' -o -name 'tmp.*' -o -name 'wandb-*' -o -name 'torchinductor_*' -o -name 'hf-*' -o -name 'huggingface*' \) \
		-exec rm -rf {} +
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} +

routerset-download:
	$(RUNNER) scripts/download_routerset.py --local-dir $(ROUTERSET_DIR)

routerset-materialize:
	$(RUNNER) scripts/materialize_routerset_dataset.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--output-dir $(MATERIALIZED_DATASET_DIR)

routerset-materialize-clean:
	$(RUNNER) scripts/materialize_routerset_dataset.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--output-dir $(MATERIALIZED_CLEAN_DATASET_DIR) \
		--clean

routerset-audit:
	$(RUNNER) scripts/audit_routerset_dataset.py \
		--dataset-root $(MATERIALIZED_DATASET_DIR)

routerset-raw-audit:
	$(RUNNER) scripts/audit_raw_routerset_dataset.py \
		--routerset-dir $(ROUTERSET_DIR)

train-prepare:
	$(RUNNER) scripts/train_moe_switcher.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--prepare-only \
		--release-name $(RELEASE_NAME) \
		--output-dir $(TRAIN_OUTPUT_DIR) \
		--runtime-root $(RUNTIME_ROOT) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		$(if $(PRECISION),--precision $(PRECISION),) \
		--rebuild-splits \
		--rebuilt-manifest-out $(REBUILT_MANIFEST)

full-train:
	$(RUNNER) scripts/full_train_moe.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--output-dir $(TRAIN_OUTPUT_DIR) \
		--release-name $(RELEASE_NAME) \
		--runtime-root $(RUNTIME_ROOT) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		$(if $(PRECISION),--precision $(PRECISION),) \
		--rebuild-splits \
		--rebuilt-manifest-out $(REBUILT_MANIFEST)

smoketest:
	$(RUNNER) scripts/smoke_test_moe.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--release-name $(RELEASE_NAME) \
		--output-dir $(SMOKE_OUTPUT_DIR) \
		--runtime-root $(RUNTIME_ROOT) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		$(if $(PRECISION),--precision $(PRECISION),) \
		--rebuilt-manifest-out $(REBUILT_MANIFEST)

smoketest-preflight:
	$(RUNNER) scripts/train_moe_switcher.py \
		--routerset-dir $(ROUTERSET_DIR) \
		--preflight-only \
		--release-name $(RELEASE_NAME) \
		--output-dir $(SMOKE_OUTPUT_DIR) \
		--rebuild-splits \
		--rebuilt-manifest-out $(REBUILT_MANIFEST)
