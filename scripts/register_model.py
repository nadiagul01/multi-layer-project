"""
Register the Week 2 best model in MLflow Model Registry.

Run from project root:
    python -m scripts.register_model

This script:
1. Connects to the existing MLflow tracking database
2. Finds the best experiment run (exp3_large_subset from Week 2)
3. Registers the BLIP model as a versioned model in the MLflow Model Registry
"""

import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ─── Configuration ───

POSSIBLE_DB_PATHS = [
    "notebooks/week2/mlflow.db",
    "notebooks/mlflow.db",
]

db_path = None
for p in POSSIBLE_DB_PATHS:
    if Path(p).exists():
        db_path = p
        break

if db_path is None:
    print("ERROR: mlflow.db not found. Tried:", POSSIBLE_DB_PATHS)
    sys.exit(1)

MLFLOW_TRACKING_URI = f"sqlite:///{db_path}"
EXPERIMENT_NAME = "BLIP-Flickr8k-Captioning"
REGISTERED_MODEL_NAME = "blip-flickr8k-captioning"

# ─── Setup ───
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {EXPERIMENT_NAME}")

# ─── Find the experiment ───
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"ERROR: Experiment '{EXPERIMENT_NAME}' not found")
    sys.exit(1)

# ─── List all runs ───
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time ASC"]
)

print(f"\nAll runs in experiment ({len(runs)} total):")
for run in runs:
    print(f"  {run.info.run_name:30s} | status: {run.info.status}")

# ─── Find the best run (exp3_large_subset or the one with best metrics) ───
best_run = None
for run in runs:
    if run.info.run_name == "exp3_large_subset":
        best_run = run
        break

# Fallback: find run with highest beam_METEOR
if best_run is None:
    best_meteor = -1
    for run in runs:
        meteor = run.data.metrics.get("beam_METEOR", run.data.metrics.get("METEOR", -1))
        if meteor > best_meteor:
            best_meteor = meteor
            best_run = run

if best_run is None:
    print("ERROR: No suitable run found to register")
    sys.exit(1)

print(f"\nBest run selected: {best_run.info.run_name}")
print(f"  Run ID: {best_run.info.run_id}")
print(f"  Metrics: {best_run.data.metrics}")

# ─── Register the model ───
# Log a simple model artifact to the run (needed for registration)
with mlflow.start_run(run_id=best_run.info.run_id):
    # Log model info as params (the actual model weights are too large for MLflow artifacts)
    mlflow.set_tag("registered_model", "true")
    mlflow.set_tag("model_version", "v1")
    mlflow.set_tag("model_description",
                   "BLIP base with frozen vision encoder, fine-tuned text decoder. "
                   "Best config: lr=5e-5, 500 training images, beam decoding.")

# Register in Model Registry
try:
    # Create registered model
    try:
        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            description="BLIP image captioning model fine-tuned on Flickr8k dataset. "
                       "Vision encoder frozen, text decoder fine-tuned."
        )
        print(f"\nRegistered model created: {REGISTERED_MODEL_NAME}")
    except Exception:
        print(f"\nRegistered model '{REGISTERED_MODEL_NAME}' already exists")

    # Create model version
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"runs:/{best_run.info.run_id}",
        run_id=best_run.info.run_id,
        description=f"Week 2 best model: {best_run.info.run_name} "
                    f"(METEOR: {best_run.data.metrics.get('beam_METEOR', 'N/A')})"
    )
    print(f"Model version created: v{model_version.version}")

    # Transition to "Production" stage (MLflow < 2.9 style)
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version,
            stage="Production"
        )
        print(f"Model v{model_version.version} promoted to Production stage")
    except Exception as e:
        # Newer MLflow versions use aliases instead of stages
        try:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="production",
                version=model_version.version
            )
            print(f"Model v{model_version.version} aliased as 'production'")
        except Exception:
            print(f"Model v{model_version.version} registered (stage/alias setting skipped)")

    print(f"\n{'='*50}")
    print(f"SUCCESS — Model registered!")
    print(f"  Name: {REGISTERED_MODEL_NAME}")
    print(f"  Version: {model_version.version}")
    print(f"  Source run: {best_run.info.run_name}")
    print(f"  View in MLflow UI: Models tab")
    print(f"{'='*50}")

except Exception as e:
    print(f"Registration error: {e}")
