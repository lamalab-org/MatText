"""
MODNet Training Script for Materials Property Prediction

This script trains MODNet models for 3 different material properties
(dielectric, gvrh, perovskites) across multiple alpha values for
total_energy targets.

Featurization is cached per property to avoid redundant computation.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import logging
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from modnet.preprocessing import MODData
from modnet.models import MODNetModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("normalized_data")
OUTPUT_DIR = Path("modnet_outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"

# Properties to train on
PROPERTIES = ["dielectric", "gvrh", "perovskites"]

# Alpha values for total_energy targets
ALPHA_VALUES = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]

# Training parameters
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "lr": 0.001,
    "val_fraction": 0.1,
    "n_feat": 100,  # Number of features to select
    "verbose": 1
}


def load_dataset(file_path: Path) -> List[Dict]:
    """Load dataset from JSON file (one JSON object per line)."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def cif_to_structure(cif_string: str) -> Structure:
    """Convert CIF string to pymatgen Structure."""
    return Structure.from_str(cif_string, fmt="cif")


def create_base_moddata(
    data: List[Dict],
    structure_key: str = "cif_p1"
) -> tuple:
    """
    Create structures and IDs from dataset entries.
    Returns structures and ids separately so targets can be added later.
    """
    structures = []
    ids = []

    for entry in data:
        try:
            structure = cif_to_structure(entry[structure_key])
            structures.append(structure)
            ids.append(entry.get("mbid", f"entry_{len(ids)}"))
        except Exception as e:
            logger.warning(f"Failed to process entry {entry.get('mbid', 'unknown')}: {e}")

    logger.info(f"Successfully loaded {len(structures)} structures")
    return structures, ids


def featurize_structures(
    structures: List[Structure],
    ids: List[str],
    cache_path: Optional[Path] = None,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Featurize structures and optionally cache the result.
    This is the slowest step, so caching is important.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached featurized data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Featurizing {len(structures)} structures (this may take a while)...")

    # Create a temporary MODData just for featurization
    # Use dummy targets since we only need features
    dummy_targets = [0.0] * len(structures)
    mod_data = MODData(
        materials=structures,
        targets=dummy_targets,
        target_names=["dummy"],
        structure_ids=ids
    )

    mod_data.featurize(n_jobs=n_jobs)
    df_featurized = mod_data.get_featurized_df()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(df_featurized, f)
        logger.info(f"Cached featurized data to {cache_path}")

    return df_featurized


def create_moddata_with_targets(
    df_featurized: pd.DataFrame,
    data: List[Dict],
    target_name: str,
    ids: List[str]
) -> MODData:
    """
    Create MODData with pre-featurized data and specific targets.
    """
    # Extract targets for entries that were successfully featurized
    id_to_target = {}
    for entry in data:
        mbid = entry.get("mbid", "")
        if mbid in df_featurized.index:
            id_to_target[mbid] = entry[target_name]

    # Align targets with featurized data
    valid_ids = [idx for idx in df_featurized.index if idx in id_to_target]
    targets = np.array([[id_to_target[idx]] for idx in valid_ids])  # 2D array

    df_targets = pd.DataFrame(
        {target_name: [id_to_target[idx] for idx in valid_ids]},
        index=valid_ids
    )

    # Create MODData with pre-featurized data and targets
    mod_data = MODData(
        targets=targets,
        target_names=[target_name],
        structure_ids=valid_ids,
        df_featurized=df_featurized.loc[valid_ids]
    )
    mod_data.df_targets = df_targets

    return mod_data


def train_model(
    train_data: MODData,
    target_name: str,
    config: Dict
) -> MODNetModel:
    """Train a MODNet model."""
    model = MODNetModel(
        targets=[[target_name]],
        weights={target_name: 1.0},
        n_feat=config["n_feat"],
        num_neurons=([256], [128], [64], [32]),
        act="elu"
    )

    model.fit(
        training_data=train_data,
        val_fraction=config["val_fraction"],
        lr=config["lr"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=config["verbose"]
    )

    return model


def evaluate_model(
    model: MODNetModel,
    test_data: MODData,
    target_name: str
) -> Dict:
    """Evaluate model on test data."""
    # Get predictions (disable remap_out_of_bounds to avoid MODNet 0.4.5 bug)
    predictions_df = model.predict(test_data, remap_out_of_bounds=False)
    predictions = predictions_df[target_name].values
    actuals = test_data.df_targets[target_name].values

    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_samples": len(actuals),
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist()
    }


def run_property_training(property_name: str) -> Dict:
    """
    Run training for all alpha values of a single property.
    Featurizes once and reuses for all targets.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing property: {property_name}")
    logger.info(f"{'='*60}")

    # Load datasets
    train_file = DATA_DIR / f"train_{property_name}_0.json"
    test_file = DATA_DIR / f"test_{property_name}_0.json"

    logger.info(f"Loading training data from {train_file}")
    train_entries = load_dataset(train_file)

    logger.info(f"Loading test data from {test_file}")
    test_entries = load_dataset(test_file)

    # Create structures
    logger.info("Creating structures from CIF...")
    train_structures, train_ids = create_base_moddata(train_entries)
    test_structures, test_ids = create_base_moddata(test_entries)

    # Featurize (with caching)
    train_cache = CACHE_DIR / f"train_{property_name}_features.pkl"
    test_cache = CACHE_DIR / f"test_{property_name}_features.pkl"

    train_features = featurize_structures(train_structures, train_ids, train_cache)
    test_features = featurize_structures(test_structures, test_ids, test_cache)

    results = {}

    for alpha in ALPHA_VALUES:
        target_name = f"total_energy_alpha_{alpha}"
        logger.info(f"\n{'-'*40}")
        logger.info(f"Training for {target_name}")
        logger.info(f"{'-'*40}")

        try:
            # Create MODData with specific targets
            train_moddata = create_moddata_with_targets(
                train_features, train_entries, target_name, train_ids
            )
            test_moddata = create_moddata_with_targets(
                test_features, test_entries, target_name, test_ids
            )

            # Feature selection
            logger.info("Performing feature selection...")
            train_moddata.feature_selection(n=TRAINING_CONFIG["n_feat"])

            # Train model
            logger.info("Training model...")
            model = train_model(train_moddata, target_name, TRAINING_CONFIG)

            # Save checkpoint
            checkpoint_path = CHECKPOINT_DIR / property_name / f"model_alpha_{alpha}"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            logger.info(f"Saved model to {checkpoint_path}")

            # Evaluate
            logger.info("Evaluating on test set...")
            eval_results = evaluate_model(model, test_moddata, target_name)

            logger.info(f"Results: MAE={eval_results['mae']:.6f}, R²={eval_results['r2']:.6f}")

            results[f"alpha_{alpha}"] = {
                "property": property_name,
                "target": target_name,
                "alpha": alpha,
                "metrics": {
                    "mae": eval_results["mae"],
                    "mse": eval_results["mse"],
                    "rmse": eval_results["rmse"],
                    "r2": eval_results["r2"]
                },
                "n_train": len(train_entries),
                "n_test": eval_results["n_samples"],
                "checkpoint_path": str(checkpoint_path),
                "predictions": eval_results["predictions"],
                "actuals": eval_results["actuals"]
            }

        except Exception as e:
            logger.error(f"Failed to train {target_name}: {e}")
            import traceback
            traceback.print_exc()
            results[f"alpha_{alpha}"] = {"error": str(e)}

        # Save intermediate results
        results_file = RESULTS_DIR / f"results_{property_name}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    """Main entry point."""
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    all_results = {}

    for property_name in PROPERTIES:
        try:
            results = run_property_training(property_name)
            all_results[property_name] = results
        except Exception as e:
            logger.error(f"Failed property {property_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[property_name] = {"error": str(e)}

    # Save all results
    final_results_file = RESULTS_DIR / "all_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"All results saved to {final_results_file}")
    logger.info(f"Checkpoints saved to {CHECKPOINT_DIR}")
    logger.info(f"{'='*60}")

    # Print summary
    print("\n\nSUMMARY OF RESULTS")
    print("="*60)
    for prop in PROPERTIES:
        if prop in all_results:
            print(f"\n{prop.upper()}:")
            prop_results = all_results[prop]
            if "error" in prop_results:
                print(f"  ERROR: {prop_results['error']}")
            else:
                for alpha in ALPHA_VALUES:
                    key = f"alpha_{alpha}"
                    if key in prop_results:
                        res = prop_results[key]
                        if "error" in res:
                            print(f"  alpha={alpha}: ERROR - {res['error']}")
                        else:
                            print(f"  alpha={alpha}: MAE={res['metrics']['mae']:.4f}, R²={res['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
