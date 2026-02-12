"""
coGN Training Script for NEW Material Property Datasets

Trains on: bandgap, form_energy, jdft2d, phonons
Data source: HuggingFace dataset jablonkagroup/MatText-hypo_pot
Output directory: cogn_outputs_new/

Compatible with: kgcnn==3.1.0, Python 3.11

Installation:
    micromamba create -n kgcnn_py311 python=3.11 -y
    micromamba run -n kgcnn_py311 pip install kgcnn==3.1.0
"""

import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from datasets import load_dataset as hf_load_dataset
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler

# Import coGN model from kgcnn (v3.1.0)
from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices
from kgcnn.graph.base import GraphDict

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GPU Configuration
# ============================================================================

def configure_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

            # Optional: Set visible devices (can be controlled via CUDA_VISIBLE_DEVICES)
            # tf.config.set_visible_devices(gpus[0:1], 'GPU')  # Use only first GPU

            # Enable mixed precision training for faster training on modern GPUs
            # This can provide 2-3x speedup on V100, A100, etc.
            use_mixed_precision = os.environ.get('USE_MIXED_PRECISION', 'true').lower() == 'true'
            if use_mixed_precision:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Mixed precision training enabled (float16)")
                except Exception as e:
                    logger.warning(f"Could not enable mixed precision: {e}")

            # Log GPU details
            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                logger.info(f"GPU {i}: {gpu_details.get('device_name', 'Unknown')}")

            return True
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            return False
    else:
        logger.info("No GPU found, using CPU")

        # Configure CPU for optimal performance
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Auto
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Auto

        return False

# Configure GPU on import
GPU_AVAILABLE = configure_gpu()

# Configuration - NEW datasets
HF_DATASET_NAME = "jablonkagroup/MatText-hypo_pot"
# Use local path or cluster path depending on where script runs
if Path("/data/alamparan").exists():
    OUTPUT_DIR = Path("/data/alamparan/COGN")
else:
    OUTPUT_DIR = Path.cwd().parent / "cogn_outputs_local"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"
GRAPH_CACHE_DIR = OUTPUT_DIR / "graph_cache"

# Properties to train on
# Use single property for local testing, full list for cluster
if Path("/data/alamparan").exists():
    PROPERTIES =  ["dielectric","gvrh","jdft2d", "phonons", "bandgap", "formenergy"]
    ALPHA_VALUES = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]
else:
    # Full training locally
    PROPERTIES = ["dielectric","gvrh","jdft2d", "phonons", "bandgap", "formenergy"]
    ALPHA_VALUES = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]

# Crystal preprocessing parameters
N_NEIGHBORS = 24  # Number of nearest neighbors for graph construction

# Training parameters
# Use fewer epochs for local testing, full training on cluster
if Path("/data/alamparan").exists():
    NUM_EPOCHS = 800
else:
    NUM_EPOCHS = 100  # Test with fewer epochs locally

TRAINING_CONFIG = {
    "epochs": NUM_EPOCHS,
    "batch_size": 128 if GPU_AVAILABLE else 64,  # Larger batch for GPU
    "lr_start": 0.0005 if GPU_AVAILABLE else 0.0001,  # Higher LR for GPU with mixed precision
    "lr_stop": 1e-5 if GPU_AVAILABLE else 1e-6,
    "val_fraction": 0.1,
    "use_scaler": True,
    "verbose": 1,
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

    dataset = hf_load_dataset(HF_DATASET_NAME, name=property_name, split=split)
    data = [dict(example) for example in dataset]

    logger.info(f"Loaded {len(data)} examples from HF dataset")
    return data


def cif_to_structure(cif_string: str) -> Structure:
    """Convert CIF string to pymatgen Structure."""
    return Structure.from_str(cif_string, fmt="cif")


def preprocess_crystals_to_graphs(
    data: list[dict],
    preprocessor,
    structure_key: str = "cif_p1"
) -> list:
    """Convert crystal structures to graph representations.

    Args:
        data: List of dataset entries with CIF strings
        preprocessor: Crystal preprocessor for graph construction
        structure_key: Key in data dict containing CIF string

    Returns:
        List of NetworkX graphs with metadata
    """
    logger.info(f"Converting {len(data)} crystals to graphs...")

    graphs = []
    valid_ids = []

    for entry in data:
        try:
            # Convert CIF to structure
            structure = cif_to_structure(entry[structure_key])

            # Add metadata to structure
            mbid = entry.get("mbid", f"entry_{len(valid_ids)}")

            # Preprocess structure to graph (returns NetworkX graph)
            graph = preprocessor(structure)

            # Add metadata to graph using NetworkX graph attributes
            graph.graph['dataset_id'] = mbid

            graphs.append(graph)
            valid_ids.append(mbid)

        except Exception as e:
            logger.warning(f"Failed to process entry {entry.get('mbid', 'unknown')}: {e}")

    logger.info(f"Successfully converted {len(graphs)} structures to graphs")

    return graphs


def save_graphs_to_pickle(graphs: list, filepath: Path):
    """Save graph list to pickle file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(graphs, f)
    logger.info(f"Saved graph data to {filepath}")


def load_graphs_from_pickle(filepath: Path) -> list:
    """Load graph list from pickle file."""
    logger.info(f"Loading cached graph data from {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_input_tensors(inputs, graphs):
    """Convert list of NetworkX graphs to input tensors for coGN model.

    Args:
        inputs: Model input layers
        graphs: List of NetworkX MultiDiGraph objects

    Returns:
        Dictionary of input tensors
    """
    input_names = [inp.name.split(':')[0] for inp in inputs]
    input_tensors = {}

    # Collect all attributes from NetworkX graphs
    all_edge_attrs = {}
    all_node_attrs = {}
    all_graph_attrs = {}
    edge_indices_list = []
    line_graph_edge_indices_list = []

    for graph in graphs:
        # Get edge indices from NetworkX graph structure
        edges = list(graph.edges())
        if len(edges) > 0:
            edge_idx = np.array(edges)
            edge_indices_list.append(edge_idx[:, [1, 0]])  # Reverse for coGN

            # Line graph edge indices for angles
            if 'line_graph_edge_indices' in input_names:
                angle_idx = get_angle_indices(edge_idx, edge_pairing='kj')[2].reshape(-1, 2)
                line_graph_edge_indices_list.append(angle_idx)

            # Collect edge attributes
            edge_data_list = [edge_data for u, v, edge_data in graph.edges(data=True)]
            if edge_data_list:
                for key in edge_data_list[0].keys():
                    if key not in all_edge_attrs:
                        all_edge_attrs[key] = []
                    all_edge_attrs[key].append(np.array([ed[key] for ed in edge_data_list]))

        # Collect node attributes
        node_data_list = [node_data for node, node_data in graph.nodes(data=True)]
        if node_data_list:
            for key in node_data_list[0].keys():
                if key not in all_node_attrs:
                    all_node_attrs[key] = []
                all_node_attrs[key].append(np.array([nd[key] for nd in node_data_list]))

        # Collect graph-level attributes
        for key, val in graph.graph.items():
            if key != 'dataset_id':
                if key not in all_graph_attrs:
                    all_graph_attrs[key] = []
                all_graph_attrs[key].append(val)

    # Build ragged tensors for node attributes
    for key, vals in all_node_attrs.items():
        if key in input_names:
            concatenated = np.concatenate(vals)
            row_lengths = [len(v) for v in vals]
            input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                concatenated.astype(np.float32) if concatenated.dtype == np.float64 else concatenated,
                row_lengths
            )

    # Build ragged tensors for edge attributes
    for key, vals in all_edge_attrs.items():
        if key in input_names and len(vals) > 0:
            concatenated = np.concatenate(vals)
            row_lengths = [len(v) for v in vals]
            input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                concatenated.astype(np.float32) if concatenated.dtype == np.float64 else concatenated,
                row_lengths
            )

    # Build tensors for graph-level attributes
    for key, vals in all_graph_attrs.items():
        if key in input_names:
            arr = np.array(vals)
            input_tensors[key] = tf.constant(arr.astype(np.float32) if arr.dtype == np.float64 else arr)

    # Edge indices
    if edge_indices_list:
        input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
            np.concatenate(edge_indices_list).astype(np.int32),
            [len(e) for e in edge_indices_list]
        )

    # Line graph edge indices
    if 'line_graph_edge_indices' in input_names and line_graph_edge_indices_list:
        input_tensors['line_graph_edge_indices'] = tf.RaggedTensor.from_row_lengths(
            np.concatenate(line_graph_edge_indices_list).astype(np.int32),
            [len(l) for l in line_graph_edge_indices_list]
        )

    return input_tensors


def get_id_index_mapping(graphs):
    """Create mapping from dataset IDs to indices in graph list."""
    index_mapping = {
        graph.graph['dataset_id']: i for i, graph in enumerate(graphs)
    }
    return index_mapping


def get_graphs_by_ids(id_index_mapping, graphs, mbids):
    """Extract graphs by their dataset IDs."""
    idxs = [id_index_mapping[mbid] for mbid in mbids]
    return [graphs[i] for i in idxs]


def get_lr_scheduler(dataset_size, batch_size, epochs, lr_start=0.0005, lr_stop=1e-5):
    """Create polynomial decay learning rate scheduler."""
    steps_per_epoch = dataset_size / batch_size
    num_steps = epochs * steps_per_epoch
    scheduler = ks.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr_start,
        decay_steps=num_steps,
        end_learning_rate=lr_stop
    )
    return scheduler


def train_model(
    model,
    x_train,
    y_train,
    config: dict
) -> tuple:
    """Train coGN model.

    Args:
        model: Compiled Keras model
        x_train: Training input tensors
        y_train: Training targets
        config: Training configuration

    Returns:
        Tuple of (model, history)
    """
    start = datetime.now()
    history = model.fit(
        x_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=config["verbose"]
    )
    duration = (datetime.now() - start).total_seconds()

    logger.info(f"Training completed in {duration:.2f} seconds")

    return model, history


def evaluate_model(
    model,
    x_test,
    y_test_actual,
    use_scaler: bool = False,
    scaler=None
) -> dict:
    """Evaluate model on test data.

    Args:
        model: Trained model
        x_test: Test input tensors
        y_test_actual: Actual test targets
        use_scaler: Whether scaling was used
        scaler: StandardScaler object if scaling was used

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.predict(x_test)

    # Inverse transform if scaler was used
    if use_scaler and scaler is not None:
        predictions = scaler.inverse_transform(predictions)

    predictions = predictions.flatten()
    actuals = y_test_actual.flatten()

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


def run_property_training(property_name: str, preprocessor) -> dict:
    """Run training for all alpha values of a single property.

    Args:
        property_name: Name of the property to train on
        preprocessor: Crystal preprocessor for graph construction

    Returns:
        Dictionary of results for all alpha values
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing property: {property_name}")
    logger.info(f"{'='*60}")

    # Load data from HuggingFace
    logger.info(f"Loading training data from HuggingFace dataset: {HF_DATASET_NAME}")
    train_entries = load_dataset_from_hf(property_name, split="train")

    logger.info(f"Loading test data from HuggingFace dataset: {HF_DATASET_NAME}")
    test_entries = load_dataset_from_hf(property_name, split="test")

    # Cache paths for preprocessed graphs
    train_graph_cache = GRAPH_CACHE_DIR / f"{property_name}_train_graphs.pkl"
    test_graph_cache = GRAPH_CACHE_DIR / f"{property_name}_test_graphs.pkl"

    # Preprocess crystals to graphs or load from cache
    if train_graph_cache.exists():
        train_graphs = load_graphs_from_pickle(train_graph_cache)
        train_id_mapping = get_id_index_mapping(train_graphs)
    else:
        logger.info("Preprocessing training crystals to graphs...")
        train_graphs = preprocess_crystals_to_graphs(train_entries, preprocessor)
        save_graphs_to_pickle(train_graphs, train_graph_cache)
        train_id_mapping = get_id_index_mapping(train_graphs)

    if test_graph_cache.exists():
        test_graphs = load_graphs_from_pickle(test_graph_cache)
        test_id_mapping = get_id_index_mapping(test_graphs)
    else:
        logger.info("Preprocessing test crystals to graphs...")
        test_graphs = preprocess_crystals_to_graphs(test_entries, preprocessor)
        save_graphs_to_pickle(test_graphs, test_graph_cache)
        test_id_mapping = get_id_index_mapping(test_graphs)

    results = {}

    for alpha in ALPHA_VALUES:
        target_name = f"total_energy_alpha_{alpha}"
        logger.info(f"\n{'-'*40}")
        logger.info(f"Training for {target_name}")
        logger.info(f"{'-'*40}")

        try:
            # Create coGN model
            model = make_model(**model_default)

            # Debug: Print model inputs
            logger.info(f"Model expects {len(model.inputs)} inputs:")
            for inp in model.inputs:
                logger.info(f"  - {inp.name}: shape={inp.shape}, dtype={inp.dtype}")

            # Get training targets
            train_mbids = []
            train_targets = []
            for entry in train_entries:
                mbid = entry.get("mbid", "")
                if mbid in train_id_mapping:
                    train_mbids.append(mbid)
                    train_targets.append(entry[target_name])

            y_all = np.array(train_targets).reshape(-1, 1)

            # Debug: Check for NaN/Inf in targets
            logger.info(f"Target stats - min: {y_all.min():.4f}, max: {y_all.max():.4f}, mean: {y_all.mean():.4f}")
            if np.isnan(y_all).any():
                logger.error(f"NaN values found in targets!")
                nan_count = np.isnan(y_all).sum()
                logger.error(f"Number of NaN targets: {nan_count}/{len(y_all)}")
                # Remove NaN samples
                valid_mask = ~np.isnan(y_all).flatten()
                y_all = y_all[valid_mask]
                train_mbids = [mbids for mbids, valid in zip(train_mbids, valid_mask) if valid]
                logger.info(f"Filtered to {len(y_all)} valid samples")

            # Manual train/validation split (10%)
            n_samples = len(train_mbids)
            n_val = int(n_samples * 0.1)
            indices = np.random.RandomState(42).permutation(n_samples)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]

            train_mbids_split = [train_mbids[i] for i in train_idx]
            val_mbids_split = [train_mbids[i] for i in val_idx]
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]

            logger.info(f"Split: {len(train_mbids_split)} training, {len(val_mbids_split)} validation")

            # Get training graphs and convert to tensors
            train_graphs_subset = get_graphs_by_ids(
                train_id_mapping, train_graphs, train_mbids_split
            )
            x_train = get_input_tensors(model.inputs, train_graphs_subset)

            # Get validation graphs and convert to tensors
            val_graphs_subset = get_graphs_by_ids(
                train_id_mapping, train_graphs, val_mbids_split
            )
            x_val = get_input_tensors(model.inputs, val_graphs_subset)

            # Debug: Print what tensors were created
            logger.info(f"Created {len(x_train)} input tensors:")
            for key, tensor in x_train.items():
                logger.info(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

            # Debug: Check for NaN/Inf in input tensors
            for key, tensor in x_train.items():
                tensor_vals = tensor.values if hasattr(tensor, 'values') else tensor
                vals_np = tensor_vals.numpy()
                if np.isnan(vals_np).any():
                    logger.warning(f"NaN found in input tensor: {key} ({np.isnan(vals_np).sum()} values)")
                if np.isinf(vals_np).any():
                    logger.warning(f"Inf found in input tensor: {key} ({np.isinf(vals_np).sum()} values)")
                # Check for extreme values
                if len(vals_np) > 0:
                    val_min, val_max = vals_np.min(), vals_np.max()
                    if abs(val_max) > 1e6 or abs(val_min) > 1e6:
                        logger.warning(f"Extreme values in {key}: min={val_min:.2e}, max={val_max:.2e}")

            # Apply scaling with outlier clipping
            scaler = None
            if TRAINING_CONFIG["use_scaler"]:
                # Clip extreme outliers before scaling (beyond 5 std devs)
                mean_y = y_train.mean()
                std_y = y_train.std()
                clip_min = mean_y - 5 * std_y
                clip_max = mean_y + 5 * std_y

                y_train_clipped = np.clip(y_train, clip_min, clip_max)
                y_val_clipped = np.clip(y_val, clip_min, clip_max)

                clipped_train = (y_train != y_train_clipped).sum()
                if clipped_train > 0:
                    logger.warning(f"Clipped {clipped_train} extreme outliers in training targets")

                scaler = StandardScaler()
                y_train = scaler.fit_transform(y_train_clipped)
                y_val = scaler.transform(y_val_clipped)
                logger.info(f"Scaled target stats - train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
                logger.info(f"Scaled target stats - val mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")

            # Compile model with VERY aggressive gradient clipping to prevent NaN
            loss = ks.losses.MeanAbsoluteError()
            metrics = [ks.losses.MeanAbsoluteError()]  # Remove MSE which can explode

            # Use constant low learning rate instead of schedule
            optimizer = ks.optimizers.legacy.Adam(  # Use legacy optimizer for stability
                learning_rate=0.00005,  # Very low learning rate
                clipnorm=0.5,  # VERY aggressive gradient norm clipping
                clipvalue=0.1,  # Clip individual gradient values
                epsilon=1e-7  # Increase epsilon for numerical stability
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Setup callbacks
            nan_callback = ks.callbacks.TerminateOnNaN()

            # Create checkpoint directory before training
            checkpoint_path = CHECKPOINT_DIR / property_name / f"model_alpha_{alpha}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Add model checkpoint callback to save best model
            checkpoint_callback = ks.callbacks.ModelCheckpoint(
                str(checkpoint_path / "best_weights.h5"),
                monitor='val_loss',  # Monitor validation loss
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )

            # Early stopping if no improvement
            early_stop = ks.callbacks.EarlyStopping(
                monitor='val_loss',  # Monitor validation loss
                patience=50,
                restore_best_weights=True,
                verbose=1
            )

            # Train model
            logger.info(f"Training model with {len(y_train)} training samples and {len(y_val)} validation samples...")
            start = datetime.now()
            history = model.fit(
                x_train, y_train,
                batch_size=TRAINING_CONFIG["batch_size"],
                epochs=TRAINING_CONFIG["epochs"],
                verbose=TRAINING_CONFIG["verbose"],
                validation_data=(x_val, y_val),  # Pass validation data explicitly
                callbacks=[nan_callback, checkpoint_callback, early_stop]
            )
            duration = (datetime.now() - start).total_seconds()
            logger.info(f"Training completed in {duration:.2f} seconds")
            logger.info(f"Final train loss: {history.history['loss'][-1]:.6f}, val loss: {history.history['val_loss'][-1]:.6f}")
            logger.info(f"Best model saved to {checkpoint_path / 'best_weights.h5'}")

            # Get test targets
            test_mbids = []
            test_targets = []
            for entry in test_entries:
                mbid = entry.get("mbid", "")
                if mbid in test_id_mapping:
                    test_mbids.append(mbid)
                    test_targets.append(entry[target_name])

            y_test = np.array(test_targets).reshape(-1, 1)

            # Get test graphs and convert to tensors
            test_graphs_subset = get_graphs_by_ids(
                test_id_mapping, test_graphs, test_mbids
            )
            x_test = get_input_tensors(model.inputs, test_graphs_subset)

            # Evaluate
            logger.info("Evaluating on test set...")
            eval_results = evaluate_model(
                model, x_test, y_test,
                use_scaler=TRAINING_CONFIG["use_scaler"],
                scaler=scaler
            )

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
                "n_train": len(train_mbids),
                "n_test": eval_results["n_samples"],
                "checkpoint_path": str(checkpoint_path),
                "history": {k: [float(v) for v in vals]
                           for k, vals in history.history.items()},
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
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    GRAPH_CACHE_DIR.mkdir(exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info("coGN Training Script - HuggingFace Dataset")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {HF_DATASET_NAME}")
    logger.info(f"Properties to train: {PROPERTIES}")
    logger.info(f"Number of neighbors for graph construction: {N_NEIGHBORS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*60}\n")

    # Initialize crystal preprocessor
    preprocessor = KNNAsymmetricUnitCell(N_NEIGHBORS)
    logger.info(f"Using preprocessor: {preprocessor.__class__.__name__} with {N_NEIGHBORS} neighbors")

    all_results = {}

    for property_name in PROPERTIES:
        try:
            results = run_property_training(property_name, preprocessor)
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
