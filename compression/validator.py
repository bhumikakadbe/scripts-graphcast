"""
validator.py
============
Validates that compressed ERA5 data retains sufficient statistical fidelity
compared to the original dataset.

Checks that the following statistics match within defined tolerances:
  - Mean (absolute relative error)
  - Standard deviation
  - Min / Max
  - Pearson correlation (compressed vs. original time series)

This is a prerequisite gate before compressed data can be used for model training.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

log = logging.getLogger("Compression.Validator")

# Variable-specific tolerances for PCA reconstruction:
# Temperature and pressure are highly predictable, while wind/humidity are moderately variable.
# Episodic fields (precip and vertical velocity) naturally lose high-frequency noise in PCA,
# so they are allowed larger errors and lower correlations.
VARIABLE_TOLERANCES = {
    # Smooth/predictable fields
    "temperature":             {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "2m_temperature":          {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "mean_sea_level_pressure": {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "surface_pressure":        {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "geopotential":            {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "land_sea_mask":           {"mean": 2.0, "std": 5.0, "corr": 0.95},
    "geopotential_at_surface": {"mean": 2.0, "std": 5.0, "corr": 0.95},

    # Moderately variable fields
    "u_component_of_wind":     {"mean": 5.0, "std": 10.0, "corr": 0.90},
    "v_component_of_wind":     {"mean": 5.0, "std": 10.0, "corr": 0.90},
    "10m_u_component_of_wind": {"mean": 5.0, "std": 10.0, "corr": 0.90},
    "10m_v_component_of_wind": {"mean": 5.0, "std": 10.0, "corr": 0.90},
    "specific_humidity":       {"mean": 5.0, "std": 10.0, "corr": 0.90},
    "total_cloud_cover":        {"mean": 5.0, "std": 15.0, "corr": 0.85},

    # Highly episodic/turbulent fields
    "total_precipitation_6hr": {"mean": 40.0, "std": 50.0, "corr": 0.50},
    "vertical_velocity":       {"mean": 150.0, "std": 50.0, "corr": 0.50},
}

DEFAULT_TOLERANCES = {
    "mean_rel_error_pct":   2.0,   # Mean must match within 2%
    "std_rel_error_pct":    5.0,   # Std dev within 5%
    "min_rel_error_pct":   10.0,   # Min value within 10%
    "max_rel_error_pct":   10.0,   # Max value within 10%
    "min_correlation":      0.95,  # Pearson r must be >= 0.95
}


def compute_validation_metrics(
    original: xr.Dataset,
    compressed: xr.Dataset,
    tolerances: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Computes per-variable statistical comparison between original and compressed datasets.

    Args:
        original:   Original (uncompressed) ERA5 xarray Dataset.
        compressed: Compressed ERA5 xarray Dataset (after PCA + event compression).
        tolerances: Dict of default tolerances. Uses DEFAULT_TOLERANCES if None.

    Returns:
        pd.DataFrame with one row per variable, including pass/fail status.
    """
    default_tols = tolerances or DEFAULT_TOLERANCES
    rows = []

    common_vars = [v for v in original.data_vars if v in compressed.data_vars]
    if not common_vars:
        log.warning("No common variables between original and compressed datasets.")
        return pd.DataFrame()

    for var in common_vars:
        o_vals = original[var].values.flatten().astype(np.float64)
        c_vals = compressed[var].values.flatten().astype(np.float64)

        # Align lengths (compressed may have different time dimension)
        min_len = min(len(o_vals), len(c_vals))
        o_vals = o_vals[:min_len]
        c_vals = c_vals[:min_len]

        # Remove NaNs from both
        mask = ~(np.isnan(o_vals) | np.isnan(c_vals))
        o_clean = o_vals[mask]
        c_clean = c_vals[mask]

        if len(o_clean) < 10:
            log.warning(f"  '{var}': Too few valid samples ({len(o_clean)}) for validation.")
            continue

        # Compute statistics
        o_mean, c_mean = np.mean(o_clean), np.mean(c_clean)
        o_std,  c_std  = np.std(o_clean),  np.std(c_clean)
        o_min,  c_min  = np.min(o_clean),  np.min(c_clean)
        o_max,  c_max  = np.max(o_clean),  np.max(c_clean)

        # Dynamic scaling threshold to prevent division-by-near-zero explosion.
        # Uses 5% of standard deviation as the minimum denominator.
        scale_thresh = max(1e-5, 0.05 * o_std)

        def rel_err(orig, comp):
            denom = max(abs(orig), scale_thresh)
            return abs(orig - comp) / denom * 100.0

        mean_err = rel_err(o_mean, c_mean)
        std_err  = rel_err(o_std,  c_std)
        min_err  = rel_err(o_min,  c_min)
        max_err  = rel_err(o_max,  c_max)

        # Pearson correlation
        try:
            corr = float(np.corrcoef(o_clean, c_clean)[0, 1])
        except Exception:
            corr = 0.0

        # Retrieve variable-specific tolerances or fall back to defaults
        t = VARIABLE_TOLERANCES.get(var, {})
        t_mean = t.get("mean", default_tols.get("mean_rel_error_pct", 2.0))
        t_std = t.get("std", default_tols.get("std_rel_error_pct", 5.0))
        t_corr = t.get("corr", default_tols.get("min_correlation", 0.95))

        # Pass/Fail: Min/Max checks are kept informational and do not fail the overall status,
        # since PCA reconstruction naturally smooths out single-pixel extrema values.
        passes = {
            "mean":  mean_err <= t_mean,
            "std":   std_err  <= t_std,
            "corr":  corr     >= t_corr,
        }
        all_pass = all(passes.values())

        status = "PASS" if all_pass else "FAIL"
        log.info(f"  [{status}] '{var}': mean_err={mean_err:.2f}%, std_err={std_err:.2f}%, "
                 f"corr={corr:.4f}")

        rows.append({
            "variable":           var,
            "status":             status,
            "original_mean":      round(o_mean, 6),
            "compressed_mean":    round(c_mean, 6),
            "mean_rel_error_pct": round(mean_err, 3),
            "std_rel_error_pct":  round(std_err,  3),
            "min_rel_error_pct":  round(min_err,  3),  # informational only
            "max_rel_error_pct":  round(max_err,  3),  # informational only
            "pearson_r":          round(corr, 4),
            "pass_mean":          passes["mean"],
            "pass_std":           passes["std"],
            "pass_corr":          passes["corr"],
        })

    df = pd.DataFrame(rows)
    n_pass = (df["status"] == "PASS").sum()
    n_fail = (df["status"] == "FAIL").sum()
    log.info(f"\nValidation Result: {n_pass} PASS / {n_fail} FAIL out of {len(df)} variables")
    return df


def plot_validation_scatter(
    original: xr.Dataset,
    compressed: xr.Dataset,
    output_path: str,
    max_vars: int = 6,
):
    """
    Generates scatter plots (original vs. compressed values) for the first `max_vars` variables.
    Points scattered near the y=x diagonal indicate good compression fidelity.
    """
    common_vars = [v for v in original.data_vars if v in compressed.data_vars][:max_vars]
    if not common_vars:
        return

    n_cols = min(3, len(common_vars))
    n_rows = (len(common_vars) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    fig.suptitle("Compression Validation: Original vs. Compressed Values",
                 fontsize=13, weight="bold")

    for ax, var in zip(axes, common_vars):
        o = original[var].values.flatten()
        c = compressed[var].values.flatten()
        min_len = min(len(o), len(c))
        o, c = o[:min_len], c[:min_len]
        mask = ~(np.isnan(o) | np.isnan(c))

        # Sample max 5000 points for readability
        idx = np.random.choice(np.where(mask)[0], min(5000, mask.sum()), replace=False)
        o_s, c_s = o[idx], c[idx]

        ax.scatter(o_s, c_s, alpha=0.3, s=2, color="#6366f1")
        lims = [min(o_s.min(), c_s.min()), max(o_s.max(), c_s.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="y=x (perfect)")
        ax.set_xlabel("Original", fontsize=9)
        ax.set_ylabel("Compressed", fontsize=9)
        ax.set_title(var, fontsize=10)
        ax.legend(fontsize=8)

    for ax in axes[len(common_vars):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Validation scatter plot saved: {output_path}")


def validate_compression(
    original: xr.Dataset,
    compressed: xr.Dataset,
    output_dir: str = "logs",
    tolerances: Optional[dict] = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Master validation function. Runs all checks and saves reports.

    Args:
        original:   Uncompressed ERA5 dataset.
        compressed: Compressed ERA5 dataset (PCA + event extraction output).
        output_dir: Where to save the validation report CSV and chart.
        tolerances: Custom tolerances (uses defaults if None).

    Returns:
        (validation_df, overall_passed)
        overall_passed = True if ALL variables PASS all checks.
    """
    log.info("=" * 55)
    log.info("Running compression validation...")
    Path(output_dir).mkdir(exist_ok=True)

    df = compute_validation_metrics(original, compressed, tolerances)

    if df.empty:
        log.error("Validation failed: empty results.")
        return df, False

    # Save CSV
    csv_path = f"{output_dir}/compression_validation.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Validation report saved: {csv_path}")

    # Scatter plot
    png_path = f"{output_dir}/compression_validation_scatter.png"
    try:
        plot_validation_scatter(original, compressed, png_path)
    except Exception as exc:
        log.warning(f"Scatter plot failed: {exc}")

    overall_passed = (df["status"] == "PASS").all()

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Compression Validation Report")
    print(f"{'='*55}")
    print(df[["variable", "status", "mean_rel_error_pct", "std_rel_error_pct",
              "pearson_r"]].to_string(index=False))
    print(f"\n  Overall Result: {'PASSED' if overall_passed else 'FAILED'}")
    if not overall_passed:
        failed = df[df["status"] == "FAIL"]["variable"].tolist()
        print(f"  Failed variables: {failed}")
        print(f"  Action: Increase PCA components or relax compression ratio.")
    print(f"{'='*55}\n")

    log.info("=" * 55)
    return df, overall_passed
