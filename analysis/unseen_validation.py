"""
unseen_validation.py
====================
Runs inference on unseen ERA5 datasets (e.g. 2016-2024) using a pre-trained
GraphCast_small checkpoint and computes evaluation metrics:
- Temperature, wind, pressure: RMSE, MAE, Correlation, Bias
- Precipitation: CSI (Critical Success Index), POD (Probability of Detection), FAR (False Alarm Ratio)

Saves tabular reports to: logs/validation_metrics_{year}.csv
Saves visual scorecard to: logs/validation_scorecard_{year}.png
"""

import os
import sys
import argparse
import logging
import dataclasses
import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from graphcast import data_utils
from graphcast import graphcast
from production_pipeline import preprocessing
from production_pipeline import normalization
from production_pipeline import training
from production_pipeline import inference
from production_pipeline.progressive_upgrade import generate_simulated_dataset, check_downloads_complete

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/unseen_validation.log", mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger("UnseenValidation")

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    is_precipitation: bool = False,
    precip_threshold: float = 0.1
) -> dict:
    """Computes standard continuous metrics and precipitation contingency metrics if applicable."""
    act_flat = actual.flatten()
    pred_flat = predicted.flatten()
    
    # Remove NaNs
    mask = ~(np.isnan(act_flat) | np.isnan(pred_flat))
    act_clean = act_flat[mask]
    pred_clean = pred_flat[mask]
    
    results = {
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "corr": np.nan,
        "pod": np.nan,
        "far": np.nan,
        "csi": np.nan
    }
    
    if len(act_clean) == 0:
        return results
        
    # Continuous metrics
    results["rmse"] = float(np.sqrt(np.mean((pred_clean - act_clean) ** 2)))
    results["mae"] = float(np.mean(np.abs(pred_clean - act_clean)))
    results["bias"] = float(np.mean(pred_clean - act_clean))
    
    if len(act_clean) > 1:
        std_act = np.std(act_clean)
        std_pred = np.std(pred_clean)
        if std_act > 1e-8 and std_pred > 1e-8:
            results["corr"] = float(np.corrcoef(act_clean, pred_clean)[0, 1])
        else:
            results["corr"] = 0.0
    else:
        results["corr"] = 0.0
        
    # Precipitation metrics
    if is_precipitation:
        hits = np.sum((pred_clean >= precip_threshold) & (act_clean >= precip_threshold))
        false_alarms = np.sum((pred_clean >= precip_threshold) & (act_clean < precip_threshold))
        misses = np.sum((pred_clean < precip_threshold) & (act_clean >= precip_threshold))
        
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        csi = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else 0.0
        
        results["pod"] = float(pod)
        results["far"] = float(far)
        results["csi"] = float(csi)
        
    return results


def run_unseen_validation(
    year: int,
    checkpoint_path: str,
    use_simulation: bool = False,
    eval_days: int = 30,
    eval_interval_hours: int = 24,
    lead_steps: int = 4,
    precip_threshold: float = 0.1
) -> pd.DataFrame:
    """Executes the validation rollout loop over the specified year and calculates stats."""
    log.info("=" * 60)
    log.info(f"Starting GraphCast Validation for Year {year}")
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Evaluation window: {eval_days} days | Interval: {eval_interval_hours}h")
    log.info(f"Forecast rollout length: {lead_steps} steps ({lead_steps * 6} hours)")
    log.info("=" * 60)
    
    # 1. Load Pre-trained model
    ckpt = training.load_pretrained_checkpoint(checkpoint_path)
    
    # 2. Load Stats
    norm_stats = normalization.load_google_stats(
        "checkpoints/diffs_stddev_by_level.nc",
        "checkpoints/mean_by_level.nc",
        "checkpoints/stddev_by_level.nc"
    )
    
    # 3. Load or generate the year dataset
    ds = None
    if use_simulation:
        log.info(f"Generating simulated dataset for evaluation year {year}...")
        ds = generate_simulated_dataset(year, "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc")
    else:
        # Check if raw downloads exist
        if not check_downloads_complete(year):
            log.info(f"Raw downloads for {year} are incomplete. Initiating download...")
            from data_collection.era5_downloader import ERA5Downloader
            downloader = ERA5Downloader(region=[25.0, 74.0, 17.0, 85.0]) # Nagpur
            downloader.download_year(year)
            
        raw_path = f"data/ERA5/raw/{year}"
        if os.path.exists(raw_path):
            from compression.pipeline import load_era5_year
            ds = load_era5_year(raw_path)
        else:
            log.warning(f"Raw path {raw_path} not found. Falling back to simulation mode.")
            ds = generate_simulated_dataset(year, "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc")

    # 4. Align Coordinates and Preprocess
    ds = preprocessing.align_coordinates(ds)
    ds = preprocessing.standardize_variables(ds)
    ds = preprocessing.add_graphcast_forcings(ds)
    
    # JIT Compile the predictor
    jitted_forward = inference.build_jitted_forward(
        model_config=ckpt.model_config,
        task_config=ckpt.task_config,
        norm_stats=norm_stats,
        params=ckpt.params,
        state={}
    )
    
    # Select initialization times
    # In GraphCast, we need:
    # - 12 hours of history (T-6h, T0) as inputs
    # - lead_steps into the future as targets/forcings
    # The minimum required window size is 12h + (lead_steps * 6h)
    min_window = pd.Timedelta("12h") + lead_steps * pd.Timedelta("6h")
    
    # Extract available times and determine indexing
    total_timesteps = len(ds.time)
    
    # Calculate step interval (each step is 6 hours)
    step_size = max(1, int(eval_interval_hours / 6))
    
    # We need index 0 for history (T-6h), index 1 for T0, so validation initializations start at index 1.
    # We must stop early enough to allow lead_steps into the future.
    start_idx = 1
    end_idx = total_timesteps - lead_steps
    
    if start_idx >= end_idx:
        log.error("Dataset time coordinates range is too short for the requested lead steps!")
        return pd.DataFrame()
        
    eval_indices = list(range(start_idx, end_idx, step_size))
    
    # Limit number of eval days to save execution time
    max_eval_points = min(len(eval_indices), max(1, (eval_days * 24) // eval_interval_hours))
    eval_indices = eval_indices[:max_eval_points]
    
    log.info(f"Identified {len(eval_indices)} forecast initialization steps for validation.")
    
    # Store comparisons: metrics per variable per lead step
    eval_records = []
    
    # Relative time offsets for slicing [-6h, 0h, 6h, 12h, ... L*6h]
    relative_times = [pd.Timedelta(hours=h) for h in range(-6, lead_steps * 6 + 1, 6)]
    
    for i, idx in enumerate(eval_indices):
        # Extract absolute init time for logging
        if "datetime" in ds.coords:
            dt_vals = ds["datetime"].values
            if dt_vals.ndim > 1:
                t_init = pd.to_datetime(dt_vals[0, idx])
            else:
                t_init = pd.to_datetime(dt_vals[idx])
        else:
            t_init = pd.to_datetime(ds["time"].values[idx])
            
        log.info(f"[{i+1}/{len(eval_indices)}] Forecasting from initialization: {t_init}")
        
        # Slice dataset for this sequence by index
        seq_ds = ds.isel(time=slice(idx - 1, idx + lead_steps + 1))
        
        # Reset the time dimension coordinate to relative timedeltas starting at -6h
        # (This aligns with what data_utils.extract_inputs_targets_forcings expects)
        seq_ds = seq_ds.assign_coords(time=relative_times)
        
        # Extract inputs, targets, and forcings
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            seq_ds,
            target_lead_times=slice("6h", f"{lead_steps * 6}h"),
            **dataclasses.asdict(ckpt.task_config)
        )
        
        targets_template = targets.isel(time=[0])
        
        # Run autoregressive rollout
        predictions = inference.recursive_prediction_loop(
            jitted_forward_fn=jitted_forward,
            inputs=inputs,
            forcings=forcings,
            targets_template=targets_template,
            lead_steps=lead_steps
        )
        
        # Accumulate records for each step
        for step in range(lead_steps):
            pred_step = predictions.isel(time=[step])
            target_step = targets.isel(time=[step])
            lead_hours = (step + 1) * 6
            
            for var in ckpt.task_config.target_variables:
                if var not in pred_step.data_vars or var not in target_step.data_vars:
                    continue
                
                # Check if it has a level dimension (3D variable)
                if "level" in target_step[var].dims:
                    # Compute per level
                    for lvl in target_step.level.values:
                        act_vals = target_step[var].sel(level=lvl).values
                        pred_vals = pred_step[var].sel(level=lvl).values
                        if var == "total_precipitation_6hr":
                            act_vals = act_vals * 1000.0
                            pred_vals = pred_vals * 1000.0
                        metrics = compute_metrics(
                            actual=act_vals,
                            predicted=pred_vals,
                            is_precipitation=(var == "total_precipitation_6hr"),
                            precip_threshold=precip_threshold
                        )
                        eval_records.append({
                            "init_time": t_init,
                            "variable": var,
                            "level": int(lvl),
                            "step": step + 1,
                            "lead_hours": lead_hours,
                            **metrics
                        })
                else:
                    # 2D Surface variable
                    act_vals = target_step[var].values
                    pred_vals = pred_step[var].values
                    if var == "total_precipitation_6hr":
                        act_vals = act_vals * 1000.0
                        pred_vals = pred_vals * 1000.0
                    metrics = compute_metrics(
                        actual=act_vals,
                        predicted=pred_vals,
                        is_precipitation=(var == "total_precipitation_6hr"),
                        precip_threshold=precip_threshold
                    )
                    eval_records.append({
                        "init_time": t_init,
                        "variable": var,
                        "level": "surface",
                        "step": step + 1,
                        "lead_hours": lead_hours,
                        **metrics
                    })
                    
    # Combine results and aggregate across initialization times
    df_raw = pd.DataFrame(eval_records)
    if df_raw.empty:
        log.error("No validation records generated!")
        return df_raw
        
    # Group by variable, level, step, and lead_hours to compute average metrics
    df_summary = df_raw.groupby(["variable", "level", "step", "lead_hours"]).mean(numeric_only=True).reset_index()
    
    # Save metrics to CSV
    csv_out_path = f"logs/validation_metrics_{year}.csv"
    df_summary.to_csv(csv_out_path, index=False)
    log.info(f"Validation summary report exported to: {csv_out_path}")
    
    # Generate the scorecard plots
    plot_validation_scorecards(df_summary, year)
    
    return df_summary


def plot_validation_scorecards(df: pd.DataFrame, year: int):
    """Generates visual scorecards showing metric performance over prediction lead time."""
    log.info("Generating validation scorecard visualizations...")
    
    surface_vars = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "total_precipitation_6hr"]
    available_vars = [v for v in surface_vars if v in df["variable"].values]
    
    if not available_vars:
        log.warning("No surface variables available for scorecard plotting.")
        return
        
    fig, axes = plt.subplots(len(available_vars), 2, figsize=(14, 3 * len(available_vars)), squeeze=False)
    
    for idx, var in enumerate(available_vars):
        var_df = df[(df["variable"] == var) & (df["level"] == "surface")].sort_values("lead_hours")
        if var_df.empty:
            # Fallback to level average if level dimension was present
            var_df = df[df["variable"] == var].groupby("lead_hours").mean().reset_index()
            
        lead_times = var_df["lead_hours"].values
        
        # Panel 1: RMSE (or CSI for precipitation)
        ax_left = axes[idx, 0]
        if var == "total_precipitation_6hr":
            ax_left.plot(lead_times, var_df["csi"].values, marker='o', color='forestgreen', linewidth=2, label="CSI")
            ax_left.plot(lead_times, var_df["pod"].values, marker='s', color='dodgerblue', linestyle='--', label="POD (Recall)")
            ax_left.plot(lead_times, var_df["far"].values, marker='^', color='crimson', linestyle=':', label="FAR")
            ax_left.set_title(f"Precipitation Event Metrics (Threshold >= 0.1mm)")
            ax_left.set_ylabel("Score (0 to 1)")
            ax_left.set_ylim(-0.05, 1.05)
            ax_left.legend()
        else:
            ax_left.plot(lead_times, var_df["rmse"].values, marker='o', color='royalblue', linewidth=2, label="RMSE")
            ax_left.plot(lead_times, np.abs(var_df["bias"].values), marker='x', color='orange', linestyle='--', label="|Bias|")
            ax_left.set_title(f"{var} - Forecast Error Growth")
            ax_left.set_ylabel("Error Value")
            ax_left.legend()
            
        ax_left.set_xlabel("Lead Time (Hours)")
        ax_left.grid(True, linestyle=':', alpha=0.6)
        
        # Panel 2: Correlation Coefficient
        ax_right = axes[idx, 1]
        ax_right.plot(lead_times, var_df["corr"].values, marker='o', color='purple', linewidth=2, label="Correlation")
        ax_right.set_title(f"{var} - Spatial Correlation")
        ax_right.set_ylabel("Pearson R")
        ax_right.set_xlabel("Lead Time (Hours)")
        ax_right.set_ylim(-0.1, 1.1)
        ax_right.grid(True, linestyle=':', alpha=0.6)
        ax_right.legend()
        
    plt.tight_layout()
    plot_out_path = f"logs/validation_scorecard_{year}.png"
    plt.savefig(plot_out_path, dpi=150)
    plt.close()
    log.info(f"Validation scorecard saved to: {plot_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCast Unseen Year Validation Script")
    parser.add_argument("--year", type=int, default=2016, help="Unseen year to evaluate (default: 2016)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fine_tuned_model.nc", help="Model checkpoint path")
    parser.add_argument("--use-simulation", action="store_true", help="Bypass downloader and run validation on high-fidelity synthetic variations")
    parser.add_argument("--eval-days", type=int, default=15, help="Number of days to evaluate (default: 15 to maintain quick execution)")
    parser.add_argument("--eval-interval-hours", type=int, default=24, help="Frequency of forecast initializations in hours (default: 24h)")
    parser.add_argument("--lead-steps", type=int, default=4, help="Rollout step iterations (6h each, default: 4 steps = 24h)")
    parser.add_argument("--precip-threshold", type=float, default=0.1, help="Precipitation binary classification threshold (default: 0.1)")
    args = parser.parse_args()
    
    run_unseen_validation(
        year=args.year,
        checkpoint_path=args.checkpoint,
        use_simulation=args.use_simulation,
        eval_days=args.eval_days,
        eval_interval_hours=args.eval_interval_hours,
        lead_steps=args.lead_steps,
        precip_threshold=args.precip_threshold
    )
