# TROUBLESHOOTING GUIDE

## 1. Conda Environment Creation Failure

**Error**

The original environment.yml could not be recreated on CHAMP because the required CUDA and cuDNN versions were unavailable.

```
ResolvePackageNotFound:
cudnn=8.6.0
```

**Root Cause**

The GraphCast repository was developed for CUDA 11.x, while CHAMP provides CUDA 12.1 and cuDNN 9.1.

**Solution**

- Switched from Conda to a Python virtual environment (venv).
- Installed all dependencies using a manually created requirements.txt.

## 2. Cartopy Installation Failure

**Error**

```
fatal error: geos_c.h: No such file or directory
```

**Root Cause**

CHAMP does not provide the GEOS development library required by Cartopy.

**Solution**

- Searched the repository.
- Confirmed Cartopy was only used for visualization.
- Removed Cartopy from the installation requirements.

## 3. JAX–Haiku Compatibility Error

**Error**

```
AttributeError:
module 'jax' has no attribute 'linear_util'
```

**Root Cause**

The repository used an older version of Haiku that depended on deprecated JAX APIs.

**Solution**

Updated:

- JAX → 0.4.38
- JAXLIB → 0.4.38
- DM-Haiku → 0.0.16

## 4. Chex Compatibility Error

**Error**

```
AttributeError:
module 'jax.core' has no attribute 'Shape'
```

**Root Cause**

Older Chex version was incompatible with newer JAX.

**Solution**

Updated Chex to:

```
0.1.88
```

## 5. Xarray DataTree Error

**Error**

```
AttributeError:
module 'xarray' has no attribute 'DataTree'
```

**Root Cause**

The installed Xarray version was older than the repository expected.

**Solution**

Updated Xarray to:

```
2024.10.0
```

## 6. Dataset Time Decoding Failure

**Error**

```
AssertionError

result.dtype == timedelta64[ns]

while executing
test_data_pipeline.py
```

**Root Cause**

The sample ERA5 NetCDF dataset stores the time coordinate in integer hours.

Newer Xarray versions decode time differently than the original environment.

**Solution**

Replaced

```python
xr.open_dataset(..., engine="scipy")
```

with

```python
xr.open_dataset(
    ...,
    engine="netcdf4",
    decode_times=False
)
```

and manually converted the time coordinate using

```python
pd.to_timedelta(...)
```

## 7. Training Input Validation Failure

**Error**

Exactly the same time decoding error occurred in

```
validate_training_inputs.py
```

**Root Cause**

The same dataset loading method was used.

**Solution**

Applied the identical compatibility patch used in `test_data_pipeline.py`.

## 8. Optax Compatibility Error

**Error**

```
AttributeError:
jax.numpy has no attribute DeviceArray
```

**Root Cause**

Optax 0.1.4 still referenced deprecated JAX APIs.

**Solution**

Updated

```
Optax
0.1.4
   ↓
0.2.0
```

using

```
pip install --no-deps optax==0.2.0
```

to preserve the existing JAX installation.

## 9. Successful GraphCast Validation

Successfully completed

```
python test_data_pipeline.py
```

Verified

- Dataset loading
- Coordinate alignment
- GraphCast forcing generation
- Input extraction
- Target extraction
- Data normalization
- Variable consistency
- Time coordinate handling

## 10. Successful Training Input Validation

Successfully completed

```
python validate_training_inputs.py
```

Verified

- Dataset dimensions
- Pressure levels
- Variable availability
- Missing value checks
- Input consistency
- Target consistency
- Training readiness

## 11. Successful GraphCast Architecture Verification

Successfully executed

```
python run_unseen_workflow.py --stage 1
```

Verified

- GraphCast checkpoint loading
- Model configuration loading
- Graph Neural Network architecture
- Parameter count
- Forecast pipeline
- Autoregressive rollout logic
- Complete GraphCast initialization

---

## Document 2: Today's Workflow (Updated)

Your numbering has a duplicate "12". Here's the corrected continuation.

### 12) Compatibility Adjustments (If Required)

To ensure compatibility between the GraphCast repository and the CHAMP software environment, the following updates were applied:

- Updated JAX to v0.4.38
- Updated JAXLIB to v0.4.38
- Updated DM-Haiku to v0.0.16
- Updated Chex to v0.1.88
- Updated Optax to v0.2.0
- Updated Xarray to v2024.10.0
- Applied compatibility patches in test_data_pipeline.py and validate_training_inputs.py to correctly interpret dataset time coordinates using the NetCDF4 backend and manual pandas Timedelta conversion.

### 13) Validated Training Inputs

Executed the training input validation script to verify that the processed ERA5 dataset satisfies all GraphCast training requirements.

Command used:

```
python validate_training_inputs.py
```

The validation successfully verified:

- Dataset dimensions
- Pressure level consistency
- Variable availability
- Missing value checks
- Input tensor integrity
- Target tensor integrity
- GraphCast forcing variables
- Training readiness

The validation completed successfully, confirming that the processed ERA5 dataset is fully compatible with the GraphCast training pipeline.

### 14) Verified GraphCast Model Architecture

Executed the GraphCast workflow in Stage 1 to validate successful checkpoint loading and model initialization.

Command used:

```
python run_unseen_workflow.py --stage 1
```

The workflow successfully verified:

- GraphCast checkpoint loading
- Model architecture configuration
- Graph Neural Network parameters
- Forecast variable configuration
- Data preprocessing workflow
- GraphCast autoregressive rollout logic
- End-to-end model initialization

The successful completion of Stage 1 confirmed that the GraphCast software stack, model checkpoint, and execution environment are fully operational on the CHAMP HPC system.
