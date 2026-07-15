# GraphCast Weather Prediction — CHAM Server Deployment Guide

GraphCast is a machine-learning weather prediction system built on Graph Neural Networks. This project fine-tunes a pre-trained GraphCast model on regional ERA5 reanalysis data (Nagpur, India) and provides a production API for weather forecasting. The entire codebase is self-contained — no external dependencies outside this repository are required.

---

## Scripts in This Repository

| Script / Module | Location | Description |
|---|---|---|
| `era5_downloader.py` | `data_collection/` | Downloads ERA5 reanalysis data from the Copernicus CDS API |
| `imd_downloader.py` | `data_collection/` | Downloads IMD ground-truth rainfall data |
| `pipeline.py` | `compression/` | PCA-based data compression pipeline (5 stages) |
| `validator.py` | `compression/` | Validates compression quality per variable |
| `validation_experiment.py` | `compression/` | A/B training experiment comparing original vs compressed data |
| `test_data_pipeline.py` | root | Verifies coordinate alignment, forcings, and normalization |
| `validate_training_inputs.py` | root | Checks dimensions, pressure levels, and NaN counts |
| `run_unseen_workflow.py` | root | Master orchestrator for architecture check, validation, and training |
| `benchmark_hardware.py` | root | Benchmarks CPU/GPU performance (optional) |
| `run_pipeline_after_download.py` | root | Automated end-to-end pipeline: compress → train → validate |
| `app.py` | `production_pipeline/` | FastAPI server with 16 routes for forecasting and dashboard |
| `training.py` | `production_pipeline/` | Fine-tuning loop with checkpoint I/O |
| `inference.py` | `production_pipeline/` | JIT-compiled forward pass and autoregressive rollout |
| `preprocessing.py` | `production_pipeline/` | Coordinate alignment and variable standardization |
| `normalization.py` | `production_pipeline/` | Loads Google normalization stats and wraps predictor |
| `progressive_upgrade.py` | `production_pipeline/` | Year-by-year progressive training orchestrator |
| `unseen_validation.py` | `analysis/` | Computes RMSE, MAE, correlation, CSI on unseen years |
| `statistical_report.py` | `analysis/` | Generates statistical analysis reports |
| `sensitivity_analysis.py` | `analysis/` | Variable sensitivity analysis |
| `setup.py` | root | Makes the local `graphcast/` package pip-installable |
| `environment.yml` | root | Conda environment specification with all dependencies |

---

## Execution Sequence

> **All commands must be run from the `scripts-graphcast/` root directory.**

---

### Step 1 — Create the Environment and Install Dependencies

```bash
conda env create -f environment.yml
conda activate graphcast
pip install -e .
```

**Purpose:** Creates a conda environment with all required packages (JAX, Haiku, xarray, netCDF4, etc.) and installs the local `graphcast/` package in editable mode so all imports resolve correctly.

**Why needed:** Without this step, no Python script in the project will run. The `pip install -e .` command registers the internal `graphcast/` folder as an importable package.

**Expected output:**
```
Solving environment: done
Preparing transaction: done
Successfully installed graphcast-0.1
```

---

### Step 2 — Test CDS API Connection

```bash
python data_collection/era5_downloader.py --test --region nagpur
```

**Purpose:** Verifies that the Copernicus Climate Data Store (CDS) API credentials are configured correctly and the server is reachable. This sends a small test request without downloading any data.

**Why needed:** ERA5 data download requires a valid CDS API key stored in `~/.cdsapirc`. Running this test first prevents wasting time on a full download that would fail due to missing or invalid credentials.

**Expected output:**
```
✅ CDS API test PASSED — connection successful
Region: nagpur (lat: 17-25, lon: 75-82)
```

---

### Step 3 — Download ERA5 Data

```bash
python data_collection/era5_downloader.py --year 2015 --region nagpur
```

**Purpose:** Downloads one full year of ERA5 reanalysis data (6-hourly timesteps) from the Copernicus CDS API for the specified region. Downloads pressure-level variables (temperature, wind, humidity, geopotential, etc.) and single-level variables (2m temperature, surface pressure, precipitation, etc.).

**Why needed:** ERA5 data is the training input for GraphCast. The model learns to predict future atmospheric states from past states. Each year produces 24 monthly NetCDF files (~700 MB total for Nagpur region).

**Expected output:**
```
Downloading ERA5 data for 2015, region: nagpur
  ✓ 2015-01 pressure levels ... saved
  ✓ 2015-01 single levels ... saved
  ...
  ✓ 2015-12 single levels ... saved
Download complete: 24 files saved to data/ERA5/raw/2015/
```

> Repeat this step for each year you want to train on (e.g., 2015, 2016).

---

### Step 4 — Compress Downloaded Data

```bash
python compression/pipeline.py data/ERA5/raw/2015/ --year 2015
```

**Purpose:** Runs a 5-stage compression pipeline on the raw ERA5 data:
1. **Missing value imputation** — fills NaN gaps
2. **Outlier detection** — flags physically impossible values using z-scores and physical bounds
3. **PCA compression** — reduces data dimensionality while preserving 99%+ variance
4. **Quality validation** — checks mean error, std error, and Pearson correlation for all 15 variables
5. **Summary report** — logs compression ratio and pass/fail status

**Why needed:** Raw ERA5 data is too large for efficient training. PCA compression reduces storage by ~140x while introducing less than 0.05% error for most variables. The validator ensures compression quality meets thresholds before proceeding to training.

**Expected output:**
```
Stage 1: Missing values — 0 NaNs found
Stage 2: Outlier detection — 175,916 values flagged
Stage 3: PCA compression — 140.3x ratio (703 MB → 5.1 MB)
Stage 4: Validation — 15/15 variables PASS
Stage 5: Summary complete
Pipeline finished: EXIT 0
```

---

### Step 5 — Verify Pipeline and Training Inputs

```bash
python test_data_pipeline.py
python validate_training_inputs.py
```

**Purpose:**
- `test_data_pipeline.py` — Loads the checkpoint template dataset and verifies that coordinate alignment, forcing variable computation (TISR, year/day progress), and Z-score normalization all work correctly end-to-end.
- `validate_training_inputs.py` — Checks that the processed data has the correct dimensions, expected pressure levels (13 levels), proper variable names, and zero NaN values.

**Why needed:** These are pre-training sanity checks. If coordinate alignment or normalization is broken, training will either crash with dimension errors or produce garbage predictions. Running these two scripts catches issues before committing to a long training run.

**Expected output:**
```
# test_data_pipeline.py
✅ Coordinate alignment ... PASSED
✅ Forcing variables    ... PASSED
✅ Normalization        ... PASSED
🎉 All verification steps passed successfully!

# validate_training_inputs.py
✅ Dimensions check     ... PASSED
✅ Pressure levels      ... PASSED (13 levels)
✅ NaN check            ... PASSED (0 NaNs)
🎉 Validation Successful: All checks passed!
```

---

### Step 6 — Fine-Tune the Model

```bash
python run_unseen_workflow.py --stage 3 --year 2016 --use-simulation --epochs 20
```

**Purpose:** Fine-tunes the pre-trained GraphCast model on the specified year's data. Stage 3 of the workflow:
1. Loads the base model from `checkpoints/fine_tuned_model.nc`
2. Loads normalization statistics from `checkpoints/mean_by_level.nc`, `stddev_by_level.nc`, and `diffs_stddev_by_level.nc`
3. Prepares training pairs (2 input timesteps → 1 target timestep)
4. Runs the training loop for the specified number of epochs using Adam optimizer
5. Saves the fine-tuned checkpoint to `checkpoints/model_<year>.nc`

**Why needed:** The base model is pre-trained on global ERA5 data. Fine-tuning on regional data (Nagpur) adapts the model to local weather patterns, improving forecast accuracy for the target region.

**Expected output:**
```
Stage 3: Training
Loading base model ... 70,739 parameters
Loading normalization stats ... done
Preparing training data for 2016 ...
Epoch  1/20 — loss: 0.8423 — 38.2s
Epoch  2/20 — loss: 0.7891 — 37.8s
...
Epoch 20/20 — loss: 0.3215 — 37.5s
Checkpoint saved: checkpoints/model_2016.nc
```

> **Note:** Use `--use-simulation` flag when real ERA5 data is not yet downloaded; it generates synthetic training data from the template dataset. Remove this flag when using actual downloaded data.

---

### Step 7 — Validate the Fine-Tuned Model

```bash
python run_unseen_workflow.py --stage 2 --year 2016 --checkpoint checkpoints/model_2016.nc
```

**Purpose:** Evaluates the fine-tuned model on unseen data. Runs a forward pass on validation timesteps and computes error metrics:
- **RMSE** (Root Mean Square Error) per variable
- **MAE** (Mean Absolute Error) per variable
- **Pearson correlation** per variable
- **CSI** (Critical Success Index) for precipitation events

**Why needed:** Training loss alone does not confirm that the model generalizes well. This step measures actual forecast accuracy against held-out data to ensure the fine-tuned model produces meaningful predictions.

**Expected output:**
```
Stage 2: Validation
Loading checkpoint: checkpoints/model_2016.nc
Running inference on validation set ...
Results:
  temperature      — RMSE: 2.59 K,  correlation: 0.98
  geopotential     — RMSE: 362 m²/s², correlation: 0.99
  u_component_wind — RMSE: 5.21 m/s, correlation: 0.95
  ...
Metrics saved to: logs/validation_2016.csv
```

---

### Step 8 — Start the Production API Server

```bash
python -m uvicorn production_pipeline.app:app --host 0.0.0.0 --port 8000
```

**Purpose:** Launches a FastAPI web server with 16 API routes providing:
- Weather forecast generation (single-step and multi-step rollout)
- Model status and health monitoring
- Interactive dashboard for visualization
- Training progress tracking

**Why needed:** The API server is the production interface for consuming forecasts. It wraps the trained model behind HTTP endpoints, enabling integration with dashboards, mobile apps, or downstream services.

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     16 routes registered
INFO:     Application startup complete
```

> Access the dashboard at `http://<server-ip>:8000` in a browser.

---

### Step 9 (Optional) — Benchmark Hardware Performance

```bash
python benchmark_hardware.py
```

**Purpose:** Profiles the server hardware by running JAX matrix operations and a full 72-hour forecast rollout. Reports CPU/GPU performance, memory usage, and estimated inference times.

**Why needed:** Useful for initial server setup to confirm JAX detects the GPU correctly and to establish baseline performance numbers for capacity planning.

**Expected output:**
```
Hardware: CPU (GPU not detected)
JAX matrix multiply (1000x1000): 0.045s
Full 72h rollout: 143.75s
Memory peak: 2.1 GB
```

---

### Step 10 (Optional) — Automated End-to-End Pipeline

```bash
python run_pipeline_after_download.py
```

**Purpose:** Automatically runs the full pipeline in sequence: compression → pre-training verification → training → validation. This is a convenience wrapper that chains Steps 4 through 7 into a single command.

**Why needed:** When deploying for multiple years, this script eliminates the need to manually run each step. It monitors the `data/ERA5/raw/` directory and triggers processing as soon as new data is detected.

**Expected output:**
```
Detected data for 2015 in data/ERA5/raw/2015/
Running compression pipeline ... 15/15 PASS
Running pre-training checks  ... PASSED
Running fine-tuning (20 epochs) ... checkpoint saved
Running validation ... metrics saved
Pipeline complete: EXIT 0
```
