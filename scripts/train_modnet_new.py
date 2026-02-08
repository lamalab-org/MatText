"""
MODNet Training Script for NEW Material Property Datasets

Trains on: bandgap, form_energy, jdft2d, phonons
Data source: HuggingFace dataset jablonkagroup/MatText-hypo_pot
Output directory: modnet_outputs_new/ (separate from original)
"""

import json
import logging
import multiprocessing
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset as hf_load_dataset
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from pymatgen.core import Structure
from sklearn.feature_selection import mutual_info_regression

# Suppress TensorFlow warnings and configure Keras compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # Use Keras 2 for MODNet compatibility
warnings.filterwarnings('ignore')



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - NEW datasets
HF_DATASET_NAME = "jablonkagroup/MatText-hypo_pot"
OUTPUT_DIR = Path("modnet_outputs_new")  # Separate output directory
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"

# NEW properties to train on (these are the subset names in the HF dataset)
PROPERTIES =["phonons"] #["bandgap", "form_energy", "jdft2d", "phonons"]

# Alpha values for total_energy targets
ALPHA_VALUES = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]

# Get number of available CPU cores for parallelization
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free

# Training parameters
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "lr": 0.001,
    "val_fraction": 0.1,
    "n_feat": 100,
    "verbose": 1,
    "n_jobs": N_JOBS  # For featurization parallelization
}


def load_dataset_from_hf(property_name: str, split: str = "train") -> list[dict]:
    """Load dataset from HuggingFace for a specific property and split.

    Args:
        property_name: Name of the property (subset) to load
        split: Either "train" or "test"

    Returns:
        List of dictionaries containing the data
    """
    logger.info(f"Loading {split} split for property '{property_name}' from HuggingFace...")

    # Load the specific subset from HF
    dataset = hf_load_dataset(HF_DATASET_NAME, name=property_name, split=split)

    # Convert to list of dictionaries
    data = [dict(example) for example in dataset]

    logger.info(f"Loaded {len(data)} examples from HF dataset")
    return data


def cif_to_structure(cif_string: str) -> Structure:
    """Convert CIF string to pymatgen Structure."""
    return Structure.from_str(cif_string, fmt="cif")


def create_base_moddata(
    data: list[dict],
    structure_key: str = "cif_p1"
) -> tuple:
    """Create structures and IDs from dataset entries."""
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
    structures: list[Structure],
    ids: list[str],
    cache_path: Path | None = None,
    n_jobs: int | None = None
) -> pd.DataFrame:
    """Featurize structures and optionally cache the result.

    Args:
        structures: List of pymatgen Structure objects
        ids: List of structure IDs
        cache_path: Optional path to cache the featurized data
        n_jobs: Number of parallel jobs for featurization (default: use TRAINING_CONFIG)

    Returns:
        DataFrame with featurized data
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached featurized data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if n_jobs is None:
        n_jobs = TRAINING_CONFIG["n_jobs"]

    logger.info(f"Featurizing {len(structures)} structures using {n_jobs} parallel jobs...")
    logger.info("This may take a while depending on dataset size...")

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


def fast_feature_selection(
    df_features: pd.DataFrame,
    targets: np.ndarray,
    n_features: int = 100
) -> list[str]:
    """Fast feature selection using only MI with target."""
    logger.info(f"Fast feature selection: selecting top {n_features} features by MI with target...")

    df_clean = df_features.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)

    mi_scores = mutual_info_regression(df_clean.values, targets.ravel(), random_state=42)

    feature_ranking = pd.Series(mi_scores, index=df_clean.columns)
    feature_ranking = feature_ranking.sort_values(ascending=False)

    selected_features = feature_ranking.head(n_features).index.tolist()

    logger.info(f"Selected {len(selected_features)} features")
    return selected_features


def create_moddata_for_training(
    df_featurized: pd.DataFrame,
    data: list[dict],
    target_name: str,
    selected_features: list[str]
) -> MODData:
    """Create MODData with selected features and targets for training."""
    # Extract targets for entries that were successfully featurized
    id_to_target = {}
    for entry in data:
        mbid = entry.get("mbid", "")
        if mbid in df_featurized.index:
            id_to_target[mbid] = entry[target_name]

    # Align targets with featurized data
    valid_ids = [idx for idx in df_featurized.index if idx in id_to_target]
    targets = np.array([[id_to_target[idx]] for idx in valid_ids])

    # Select only the features we want
    df_selected = df_featurized.loc[valid_ids, selected_features]

    # Create MODData
    mod_data = MODData(
        targets=targets,
        target_names=[target_name],
        structure_ids=valid_ids,
        df_featurized=df_selected
    )

    # Set optimal features (skip MODNet's feature selection)
    mod_data.optimal_features = selected_features

    return mod_data


def train_model(
    train_data: MODData,
    target_name: str,
    config: dict
) -> MODNetModel:
    """Train a MODNet model."""
    n_feat = min(config["n_feat"], len(train_data.optimal_features))

    model = MODNetModel(
        targets=[[target_name]],
        weights={target_name: 1.0},
        n_feat=n_feat,
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
) -> dict:
    """Evaluate model on test data."""
    # Get predictions (disable remap_out_of_bounds to avoid MODNet 0.4.5 bug)
    predictions_df = model.predict(test_data, remap_out_of_bounds=False)

    # Handle column name - MODNet may use different naming conventions
    if target_name in predictions_df.columns:
        predictions = predictions_df[target_name].values
    else:
        # Use first column if target name not found
        predictions = predictions_df.iloc[:, 0].values

    # Get actuals - similar handling
    if target_name in test_data.df_targets.columns:
        actuals = test_data.df_targets[target_name].values
    else:
        actuals = test_data.df_targets.iloc[:, 0].values

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


def run_property_training(property_name: str) -> dict:
    """Run training for all alpha values of a single property."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing property: {property_name}")
    logger.info(f"{'='*60}")

    # Load data from HuggingFace
    logger.info(f"Loading training data from HuggingFace dataset: {HF_DATASET_NAME}")
    train_entries = load_dataset_from_hf(property_name, split="train")

    logger.info(f"Loading test data from HuggingFace dataset: {HF_DATASET_NAME}")
    test_entries = load_dataset_from_hf(property_name, split="test")

    logger.info("Creating structures from CIF...")
    train_structures, train_ids = create_base_moddata(train_entries)
    test_structures, test_ids = create_base_moddata(test_entries)

    train_cache = CACHE_DIR / f"train_{property_name}_features.pkl"
    test_cache = CACHE_DIR / f"test_{property_name}_features.pkl"

    logger.info(f"Featurizing training set with {TRAINING_CONFIG['n_jobs']} parallel jobs...")
    train_features = featurize_structures(
        train_structures, train_ids, train_cache, n_jobs=TRAINING_CONFIG["n_jobs"]
    )

    logger.info(f"Featurizing test set with {TRAINING_CONFIG['n_jobs']} parallel jobs...")
    test_features = featurize_structures(
        test_structures, test_ids, test_cache, n_jobs=TRAINING_CONFIG["n_jobs"]
    )

    results = {}

    for alpha in ALPHA_VALUES:
        target_name = f"total_energy_alpha_{alpha}"
        logger.info(f"\n{'-'*40}")
        logger.info(f"Training for {target_name}")
        logger.info(f"{'-'*40}")

        try:
            id_to_target = {}
            for entry in train_entries:
                mbid = entry.get("mbid", "")
                if mbid in train_features.index:
                    id_to_target[mbid] = entry[target_name]

            valid_ids = [idx for idx in train_features.index if idx in id_to_target]
            train_targets = np.array([id_to_target[idx] for idx in valid_ids])

            selected_features = fast_feature_selection(
                train_features.loc[valid_ids],
                train_targets,
                n_features=TRAINING_CONFIG["n_feat"]
            )

            train_moddata = create_moddata_for_training(
                train_features, train_entries, target_name, selected_features
            )
            test_moddata = create_moddata_for_training(
                test_features, test_entries, target_name, selected_features
            )

            logger.info("Training model...")
            model = train_model(train_moddata, target_name, TRAINING_CONFIG)

            checkpoint_path = CHECKPOINT_DIR / property_name / f"model_alpha_{alpha}"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            logger.info(f"Saved model to {checkpoint_path}")

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
                "selected_features": selected_features,
                "predictions": eval_results["predictions"],
                "actuals": eval_results["actuals"]
            }

        except Exception as e:
            logger.error(f"Failed to train {target_name}: {e}")
            import traceback
            traceback.print_exc()
            results[f"alpha_{alpha}"] = {"error": str(e)}

        results_file = RESULTS_DIR / f"results_{property_name}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info("MODNet Training Script - HuggingFace Dataset")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {HF_DATASET_NAME}")
    logger.info(f"Properties to train: {PROPERTIES}")
    logger.info(f"Parallel jobs for featurization: {TRAINING_CONFIG['n_jobs']}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*60}\n")

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

    final_results_file = RESULTS_DIR / "all_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"All results saved to {final_results_file}")
    logger.info(f"Checkpoints saved to {CHECKPOINT_DIR}")
    logger.info(f"{'='*60}")

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
