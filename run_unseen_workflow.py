"""
run_unseen_workflow.py
======================
Master coordinator script to use, validate, and fine-tune GraphCast_small beyond 2015.

Usage:
    # Stage 1: Print model structure, parameters, and input details
    python run_unseen_workflow.py --stage 1

    # Stage 2: Validate the pretrained model on year 2016 (using simulation to run instantly)
    python run_unseen_workflow.py --stage 2 --year 2016 --use-simulation --eval-days 5

    # Stage 3: Fine-tune the model on year 2016 (runs training and saves model_2016.nc)
    python run_unseen_workflow.py --stage 3 --year 2016 --use-simulation --epochs 1
"""

import os
import sys
import argparse
import logging
import dataclasses
import xarray as xr
import jax
import pandas as pd

from production_pipeline import training
from production_pipeline import preprocessing
from production_pipeline import progressive_upgrade
from graphcast import graphcast
from analysis.unseen_validation import run_unseen_validation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("WorkflowOrchestrator")


def run_stage_1(checkpoint_path: str):
    """Stage 1: Understand how data is loaded, how inference works, and how the model predicts weather."""
    log.info("\n" + "=" * 60)
    log.info("  STAGE 1: Understanding GraphCast Architecture & Data Flow")
    log.info("=" * 60)
    
    # 1. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint not found at: {checkpoint_path}")
        return
        
    ckpt = training.load_pretrained_checkpoint(checkpoint_path)
    
    # 2. Count parameters
    parameter_leaves = jax.tree_util.tree_leaves(ckpt.params)
    parameter_count = sum(p.size for p in parameter_leaves)
    
    # 3. Model configurations
    log.info("\n[Model Architecture Details]")
    log.info(f"  Checkpoint File:         {checkpoint_path}")
    log.info(f"  Model GNN Latent Size:   {ckpt.model_config.latent_size}")
    log.info(f"  GNN Message Steps:       {ckpt.model_config.gnn_msg_steps}")
    log.info(f"  GNN Hidden Layers:       {ckpt.model_config.hidden_layers}")
    log.info(f"  Total Parameter Count:   {parameter_count:,} weights")
    
    # 4. Task configurations
    log.info("\n[Task and Variables Configuration]")
    log.info(f"  Input Duration:          {ckpt.task_config.input_duration} (T-6h and T0)")
    log.info(f"  Pressure Levels ({len(ckpt.task_config.pressure_levels)} levels): {list(ckpt.task_config.pressure_levels)}")
    log.info(f"  Forcing Variables ({len(ckpt.task_config.forcing_variables)} vars): {list(ckpt.task_config.forcing_variables)}")
    log.info(f"  Target variables to predict ({len(ckpt.task_config.target_variables)} vars):")
    for var in ckpt.task_config.target_variables:
        log.info(f"    - {var}")
        
    # 5. Explain Data Loading & Processing Flow
    log.info("\n[Data Pipeline Flow Analysis]")
    log.info("  1. Coordinates Alignment:")
    log.info("     - Latitude is sorted descending (North-to-South) matching GraphCast expectation.")
    log.info("     - Longitude is standardized to range [0, 360) and sorted ascending.")
    log.info("     - Pressure levels are sorted ascending.")
    log.info("  2. Atmospheric Derived Forcings:")
    log.info("     - Derived variables added: year_progress (sin/cos representation of season) and day_progress.")
    log.info("     - incident solar radiation (TISR) is calculated dynamically for each lat-lon coordinate.")
    log.info("  3. Standard Z-Score Normalization:")
    log.info("     - Inputs are standardized: z = (x - mean) / stddev using Google precomputed stats.")
    log.info("     - Target variables represent atmospheric residual differences: diff = Target - Last_Input.")
    log.info("     - Residuals are normalized: norm_diff = diff / diffs_stddev.")
    log.info("  4. Autoregressive Rollout Prediction:")
    log.info("     - Prediction starts with 2 steps of history [-6h, 0h].")
    log.info("     - The GNN outputs the standardized residual for step 1 (+6h).")
    log.info("     - The prediction is denormalized and added to T0 to form the actual state at +6h.")
    log.info("     - The input window slides: T-6h is dropped, T0 becomes T-6h, and the prediction becomes T0.")
    log.info("     - The loop repeats recursively for the requested forecast lead times.")
    
    log.info("\nStage 1 explanation completed successfully.\n")


def run_stage_2(year: int, checkpoint_path: str, use_simulation: bool, eval_days: int, lead_steps: int):
    """Stage 2: Validate the model on unseen post-2015 data."""
    log.info("\n" + "=" * 60)
    log.info(f"  STAGE 2: Evaluating Model Generalization on Unseen {year}")
    log.info("=" * 60)
    
    metrics_df = run_unseen_validation(
        year=year,
        checkpoint_path=checkpoint_path,
        use_simulation=use_simulation,
        eval_days=eval_days,
        eval_interval_hours=24,
        lead_steps=lead_steps
    )
    
    if not metrics_df.empty:
        log.info(f"\n[Validation Scores Summary for Year {year} (Average over lead times)]")
        # Print main surface variables metrics
        surface_metrics = metrics_df[metrics_df["level"] == "surface"]
        print(surface_metrics[["variable", "step", "lead_hours", "rmse", "mae", "bias", "corr", "csi"]].to_string(index=False))
        
        log.info(f"\nTabular metrics saved to logs/validation_metrics_{year}.csv")
        log.info(f"Scorecard plots saved to logs/validation_scorecard_{year}.png")
    else:
        log.error("Validation failed to compute metrics!")


def run_stage_3(year: int, use_simulation: bool, epochs: int, base_checkpoint: str):
    """Stage 3: Fine-tune the model extending it beyond 2015."""
    log.info("\n" + "=" * 60)
    log.info(f"  STAGE 3: Fine-Tuning GraphCast Model on Year {year}")
    log.info("=" * 60)
    
    # Run the progressive upgrade training loop
    # We will copy the base checkpoint to checkpoints/fine_tuned_model.nc first if requested
    if os.path.exists(base_checkpoint) and base_checkpoint != "checkpoints/fine_tuned_model.nc":
        import shutil
        shutil.copy(base_checkpoint, "checkpoints/fine_tuned_model.nc")
        log.info(f"Copied {base_checkpoint} to checkpoints/fine_tuned_model.nc as training base.")
        
    generated_ckpts = progressive_upgrade.run_progressive_upgrade_flow(
        start_year=year,
        end_year=year,
        epochs_per_year=epochs,
        use_simulation=use_simulation
    )
    
    if generated_ckpts:
        saved_ckpt = f"checkpoints/model_{year}.nc"
        log.info(f"\n🎉 Stage 3 Training Complete. Fine-tuned model saved to: {saved_ckpt}")
        
        # Proactively run a quick validation on the fine-tuned model to evaluate performance!
        log.info("\nEvaluating the newly fine-tuned model's performance...")
        run_stage_2(
            year=year,
            checkpoint_path=saved_ckpt,
            use_simulation=use_simulation,
            eval_days=5,
            lead_steps=4
        )
    else:
        log.error("Training cycle aborted or failed!")


def main():
    parser = argparse.ArgumentParser(
        description="GraphCast Small Post-2015 Lifecycle (Stages 1, 2, 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3],
                        help="Execution Stage: 1 (Understand), 2 (Validate), 3 (Fine-tune)")
    parser.add_argument("--year", type=int, default=2016, help="Target year for validation or fine-tuning")
    parser.add_argument("--use-simulation", action="store_true", default=False,
                        help="Use high-fidelity simulated variation datasets to bypass CDS download queue")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fine_tuned_model.nc",
                        help="Model checkpoint path (default: checkpoints/fine_tuned_model.nc)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (Stage 3)")
    parser.add_argument("--eval-days", type=int, default=15, help="Number of days to validate (Stage 2)")
    parser.add_argument("--lead-steps", type=int, default=4, help="Rollout steps for forecast (6h each, default: 4 = 24h)")
    args = parser.parse_args()
    
    if args.stage == 1:
        run_stage_1(args.checkpoint)
    elif args.stage == 2:
        run_stage_2(
            year=args.year,
            checkpoint_path=args.checkpoint,
            use_simulation=args.use_simulation,
            eval_days=args.eval_days,
            lead_steps=args.lead_steps
        )
    elif args.stage == 3:
        run_stage_3(
            year=args.year,
            use_simulation=args.use_simulation,
            epochs=args.epochs,
            base_checkpoint=args.checkpoint
        )


if __name__ == "__main__":
    main()
