# ==============================================================================
# IEEE Sensors Journal (Feb 2026)
# "Multistage Nonuniformity Correction Pipeline for Single-Frame Infrared Images
#  Based on Hybrid High-Order Directional and Low-Rank Prior Information"
# ==============================================================================

SHELL := /bin/bash
VENV := .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python3),$(VENV)/bin/python3,python3)
PIP := $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip3)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
export MPLCONFIGDIR := /tmp/matplotlib_cache
export PYTHONPATH := .

.PHONY: all help setup data test mod01 mod02 mod03 mod04 mod05 mod06 clean \
        interactive demo_interactive play01 play02 play03 play04 play05 play06 \
        web seed_uploads

help:
	@echo "IEEE Sensors Journal NUC Project Management"
	@echo "--------------------------------------------"
	@echo "make setup             : Initialize virtual environment and install dependencies"
	@echo "make data              : Harvest 100% authentic real IR benchmark scenes into data/clean_scenes/"
	@echo "make test              : Execute full test suite via pytest"
	@echo "make mod01             : Run Mod01: Directional Priors & Nonuniformity Simulator"
	@echo "make mod02             : Run Mod02: Stage 1 ADMM High-Order Directional Destriping"
	@echo "make mod03             : Run Mod03: Stage 2 Patch Low-Rank Prior Recovery"
	@echo "make mod04             : Run Mod04: Baselines & Benchmark Comparison"
	@echo "make mod05             : Run Mod05: Ablation Studies & Noise Stress Tests"
	@echo "make mod06             : Run Mod06: Runtime & Multi-Core Scaling Benchmarks"
	@echo "make all               : Run full pipeline end-to-end"
	@echo "make clean             : Remove Python build caches and temporary files"
	@echo "--------------------------------------------"
	@echo "Secondary Interactive & Viva Demonstration:"
	@echo "make interactive       : Launch unified interactive TUI laboratory hub"
	@echo "make demo_interactive  : Run automated non-blocking demo across all 6 interactive modules"
	@echo "make play01            : Launch Module 1 interactive noise & prior explorer"
	@echo "make play02            : Launch Module 2 interactive ADMM hyperparameter tuner"
	@echo "make play03            : Launch Module 3 interactive patch count & rank explorer"
	@echo "make play04            : Launch Module 4 interactive head-to-head algorithm race"
	@echo "make play05            : Launch Module 5 interactive 'What-If' noise stress tester"
	@echo "make play06            : Launch Module 6 interactive multi-core scaling benchmark"

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "[SETUP] Creating virtual environment at $(VENV)..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "[SETUP] Installing pinned dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt
	@echo "[SETUP] Environment ready. Python interpreter: $$($(PYTHON) --version)"

data:
	@echo "[DATA] Harvesting authentic real thermal scenes in data/clean_scenes/..."
	$(PYTHON) common/fetch_real_datasets.py

test:
	@echo "[TEST] Running pytest across common and modules 01-06..."
	$(PYTEST) -v

mod01:
	@echo "[MOD01] Priors and Nonuniformity Simulator..."
	$(PYTHON) -m mod01_priors_and_noise_simulator.src.run_mod01
	$(PYTEST) mod01_priors_and_noise_simulator/tests/ -v

mod02:
	@echo "[MOD02] Stage 1 ADMM High-Order Destriping..."
	$(PYTHON) -m mod02_stage1_admm_destriping.src.run_mod02
	$(PYTEST) mod02_stage1_admm_destriping/tests/ -v

mod03:
	@echo "[MOD03] Stage 2 Patch Low-Rank Recovery..."
	$(PYTHON) -m mod03_stage2_patch_lowrank.src.run_mod03
	$(PYTEST) mod03_stage2_patch_lowrank/tests/ -v

mod04:
	@echo "[MOD04] Baselines and Quantitative Benchmarks..."
	$(PYTHON) -m mod04_baselines_and_benchmarks.src.run_mod04
	$(PYTEST) mod04_baselines_and_benchmarks/tests/ -v

mod05:
	@echo "[MOD05] Ablation Studies and Stress Tests..."
	$(PYTHON) -m mod05_ablation_and_stress_tests.src.run_mod05
	$(PYTEST) mod05_ablation_and_stress_tests/tests/ -v

mod06:
	@echo "[MOD06] Runtime Profiling and Scaling..."
	$(PYTHON) -m mod06_runtime_and_parfor_scaling.src.run_mod06
	$(PYTEST) mod06_runtime_and_parfor_scaling/tests/ -v

all: setup data test mod01 mod02 mod03 mod04 mod05 mod06
	@echo "======================================================================"
	@echo "  All stages and validation checks completed successfully!"
	@echo "======================================================================"

# ==============================================================================
# Secondary Interactive Layer Targets (Strictly isolated from primary benchmarks)
# ==============================================================================

interactive:
	@echo "[INTERACTIVE] Launching Unified Laboratory Hub..."
	$(PYTHON) scripts/interactive_hub.py

demo_interactive:
	@echo "[INTERACTIVE DEMO] Executing non-blocking demo verification suite..."
	$(PYTHON) scripts/interactive_hub.py --demo

play01:
	@echo "[PLAY01] Module 1 Interactive Noise & Prior Explorer..."
	$(PYTHON) -m mod01_priors_and_noise_simulator.src.interactive_mod01

play02:
	@echo "[PLAY02] Module 2 Interactive ADMM Hyperparameter Tuner..."
	$(PYTHON) -m mod02_stage1_admm_destriping.src.interactive_mod02

play03:
	@echo "[PLAY03] Module 3 Interactive Patch Low-Rank SVD Explorer..."
	$(PYTHON) -m mod03_stage2_patch_lowrank.src.interactive_mod03

play04:
	@echo "[PLAY04] Module 4 Interactive Head-to-Head Algorithm Race..."
	$(PYTHON) -m mod04_baselines_and_benchmarks.src.interactive_mod04

play05:
	@echo "[PLAY05] Module 5 Interactive What-If Noise Stress Tester..."
	$(PYTHON) -m mod05_ablation_and_stress_tests.src.interactive_mod05

play06:
	@echo "[PLAY06] Module 6 Interactive Live Multi-Core Scaling Benchmark..."
	$(PYTHON) -m mod06_runtime_and_parfor_scaling.src.interactive_mod06

web:
	@echo "[WEB] Launching MATLAB-Style Web Workbench on http://127.0.0.1:8080..."
	$(PYTHON) -m web_workbench.server --port 8080

seed_uploads:
	@echo "[SEED] Pre-seeding 6 canonical benchmark runs into uploaded_inputs/ & uploaded_outputs/..."
	$(PYTHON) -m web_workbench.server --seed-examples

clean:
	@echo "[CLEAN] Removing build artifacts and bytecode caches..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
