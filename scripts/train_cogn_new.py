"""
coGN Training Script for NEW Material Property Datasets

Trains on: bandgap, form_energy, jdft2d, phonons
Data source: HuggingFace dataset jablonkagroup/MatText-hypo_pot
Output directory: cogn_outputs_new/
"""

import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from datasets import load_dataset as hf_load_dataset
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler

# Import coGN model from kgcnn
from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices
from graphlist import GraphList, HDFGraphList

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

# Configuration - NEW datasets
HF_DATASET_NAME = "jablonkagroup/MatText-hypo_pot"
OUTPUT_DIR = Path("/data/alamparan/COGN")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"
GRAPH_CACHE_DIR = OUTPUT_DIR / "graph_cache"

# Properties to train on
PROPERTIES =  ["dielectric","gvrh","jdft2d", "phonons", "bandgap", "formenergy"]

# Alpha values for total_energy targets
ALPHA_VALUES = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]

# Crystal preprocessing parameters
N_NEIGHBORS = 24  # Number of nearest neighbors for graph construction

# Training parameters
TRAINING_CONFIG = {
    "epochs": 800,
    "batch_size": 64,
    "lr_start": 0.0005,
    "lr_stop": 1e-5,
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
) -> GraphList:
    """Convert crystal structures to graph representations.

    Args:
        data: List of dataset entries with CIF strings
        preprocessor: Crystal preprocessor for graph construction
        structure_key: Key in data dict containing CIF string

    Returns:
        GraphList containing preprocessed crystal graphs
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
            setattr(structure, "dataset_id", mbid.encode())

            # Preprocess structure to graph
            graph = preprocessor(structure)

            # Transfer metadata to graph
            setattr(graph, "dataset_id", mbid.encode())

            graphs.append(graph)
            valid_ids.append(mbid)

        except Exception as e:
            logger.warning(f"Failed to process entry {entry.get('mbid', 'unknown')}: {e}")

    logger.info(f"Successfully converted {len(graphs)} structures to graphs")

    # Create GraphList
    graphlist = GraphList.from_nx_graphs(
        graphs,
        node_attribute_names=preprocessor.node_attributes,
        edge_attribute_names=preprocessor.edge_attributes,
        graph_attribute_names=preprocessor.graph_attributes + ["dataset_id"]
    )

    return graphlist


def save_graphlist_to_hdf(graphlist: GraphList, filepath: Path):
    """Save GraphList to HDF5 file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(filepath), "w") as f:
        HDFGraphList.from_graphlist(f, graphlist)
    logger.info(f"Saved graph data to {filepath}")


def load_graphlist_from_hdf(filepath: Path) -> GraphList:
    """Load GraphList from HDF5 file."""
    logger.info(f"Loading cached graph data from {filepath}")
    with h5py.File(str(filepath), "r") as f:
        return HDFGraphList(f)


def get_input_tensors(inputs, graphlist):
    """Convert GraphList to input tensors for coGN model.

    Args:
        inputs: Model input layers
        graphlist: GraphList data structure

    Returns:
        Dictionary of input tensors
    """
    input_names = [input.name for input in inputs]
    input_tensors = {}

    # Edge attributes
    for input_name in graphlist.edge_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                graphlist.edge_attributes[input_name],
                graphlist.num_edges
            )

    # Node attributes
    for input_name in graphlist.node_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                graphlist.node_attributes[input_name],
                graphlist.num_nodes
            )

    # Graph attributes
    for input_name in graphlist.graph_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.convert_to_tensor(
                graphlist.graph_attributes[input_name]
            )

    # Edge indices (reverse order for coGN)
    input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
        graphlist.edge_indices[:][:, [1, 0]],
        graphlist.num_edges
    )

    # Line graph edge indices for angle interactions
    if 'line_graph_edge_indices' in input_names:
        graphs_line_graph_edge_indices = []
        for g in graphlist:
            line_graph_edge_indices = get_angle_indices(
                g.edge_indices, edge_pairing='kj'
            )[2].reshape(-1, 2)
            graphs_line_graph_edge_indices.append(line_graph_edge_indices)

        line_graph_edge_indices = tf.RaggedTensor.from_row_lengths(
            np.concatenate(graphs_line_graph_edge_indices),
            [len(l) for l in graphs_line_graph_edge_indices]
        )
        input_tensors['line_graph_edge_indices'] = line_graph_edge_indices

    return input_tensors


def get_id_index_mapping(graphlist):
    """Create mapping from dataset IDs to indices in graphlist."""
    index_mapping = {
        id_.decode(): i for i, id_ in
        enumerate(graphlist.graph_attributes['dataset_id'][:])
    }
    return index_mapping


def get_graphs_by_ids(id_index_mapping, graphlist, mbids):
    """Extract graphs by their dataset IDs."""
    idxs = [id_index_mapping[mbid] for mbid in mbids]
    return graphlist[idxs]


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
    train_graph_cache = GRAPH_CACHE_DIR / f"{property_name}_train_graphs.h5"
    test_graph_cache = GRAPH_CACHE_DIR / f"{property_name}_test_graphs.h5"

    # Preprocess crystals to graphs or load from cache
    if train_graph_cache.exists():
        with h5py.File(str(train_graph_cache), "r") as f:
            train_graphs = HDFGraphList(f)
            train_id_mapping = get_id_index_mapping(train_graphs)
    else:
        logger.info("Preprocessing training crystals to graphs...")
        train_graphlist = preprocess_crystals_to_graphs(train_entries, preprocessor)
        save_graphlist_to_hdf(train_graphlist, train_graph_cache)
        with h5py.File(str(train_graph_cache), "r") as f:
            train_graphs = HDFGraphList(f)
            train_id_mapping = get_id_index_mapping(train_graphs)

    if test_graph_cache.exists():
        with h5py.File(str(test_graph_cache), "r") as f:
            test_graphs = HDFGraphList(f)
            test_id_mapping = get_id_index_mapping(test_graphs)
    else:
        logger.info("Preprocessing test crystals to graphs...")
        test_graphlist = preprocess_crystals_to_graphs(test_entries, preprocessor)
        save_graphlist_to_hdf(test_graphlist, test_graph_cache)
        with h5py.File(str(test_graph_cache), "r") as f:
            test_graphs = HDFGraphList(f)
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

            # Get training targets
            train_mbids = []
            train_targets = []
            for entry in train_entries:
                mbid = entry.get("mbid", "")
                if mbid in train_id_mapping:
                    train_mbids.append(mbid)
                    train_targets.append(entry[target_name])

            y_train = np.array(train_targets).reshape(-1, 1)

            # Get training graphs and convert to tensors
            train_graphs_subset = get_graphs_by_ids(
                train_id_mapping, train_graphs, train_mbids
            )
            x_train = get_input_tensors(model.inputs, train_graphs_subset)

            # Apply scaling
            scaler = None
            if TRAINING_CONFIG["use_scaler"]:
                scaler = StandardScaler()
                y_train = scaler.fit_transform(y_train)

            # Compile model
            loss = ks.losses.MeanAbsoluteError()
            metrics = [ks.losses.MeanAbsoluteError(), ks.losses.MeanSquaredError()]

            scheduler = get_lr_scheduler(
                y_train.shape[0],
                TRAINING_CONFIG["batch_size"],
                TRAINING_CONFIG["epochs"],
                lr_start=TRAINING_CONFIG["lr_start"],
                lr_stop=TRAINING_CONFIG["lr_stop"]
            )
            optimizer = ks.optimizers.Adam(learning_rate=scheduler)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Train model
            logger.info("Training model...")
            model, history = train_model(model, x_train, y_train, TRAINING_CONFIG)

            # Save model
            checkpoint_path = CHECKPOINT_DIR / property_name / f"model_alpha_{alpha}"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(checkpoint_path / "weights.h5"))
            logger.info(f"Saved model to {checkpoint_path}")

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
