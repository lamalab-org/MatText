"""
Script to identify which specific materials cause NaN during training.
Tests each batch individually to find problematic samples.
"""

import json
import logging
import os
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from datasets import load_dataset as hf_load_dataset
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler

from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HF_DATASET_NAME = "jablonkagroup/MatText-hypo_pot"
TEST_PROPERTY = "dielectric"
TEST_ALPHA = "0.2"
N_NEIGHBORS = 24
BATCH_SIZE = 64


def cif_to_structure(cif_string: str) -> Structure:
    """Convert CIF string to pymatgen Structure."""
    return Structure.from_str(cif_string, fmt="cif")


def preprocess_crystal_to_graph(entry, preprocessor):
    """Convert single crystal to graph."""
    try:
        structure = cif_to_structure(entry['cif_p1'])
        graph = preprocessor(structure)
        graph.graph['dataset_id'] = entry.get('mbid', 'unknown')
        return graph, entry.get('mbid', 'unknown'), None
    except Exception as e:
        return None, entry.get('mbid', 'unknown'), str(e)


def convert_graphs_to_tensors(inputs, graphs):
    """Convert graphs to tensors."""
    input_names = [inp.name.split(':')[0] for inp in inputs]
    input_tensors = {}

    all_edge_attrs = {}
    all_node_attrs = {}
    all_graph_attrs = {}
    edge_indices_list = []

    for graph in graphs:
        edges = list(graph.edges())
        if len(edges) > 0:
            edge_idx = np.array(edges)
            edge_indices_list.append(edge_idx[:, [1, 0]])

            edge_data_list = [edge_data for u, v, edge_data in graph.edges(data=True)]
            if edge_data_list:
                for key in edge_data_list[0].keys():
                    if key not in all_edge_attrs:
                        all_edge_attrs[key] = []
                    all_edge_attrs[key].append(np.array([ed[key] for ed in edge_data_list]))

        node_data_list = [node_data for node, node_data in graph.nodes(data=True)]
        if node_data_list:
            for key in node_data_list[0].keys():
                if key not in all_node_attrs:
                    all_node_attrs[key] = []
                all_node_attrs[key].append(np.array([nd[key] for nd in node_data_list]))

        for key, val in graph.graph.items():
            if key != 'dataset_id':
                if key not in all_graph_attrs:
                    all_graph_attrs[key] = []
                all_graph_attrs[key].append(val)

    for key, vals in all_node_attrs.items():
        if key in input_names:
            concatenated = np.concatenate(vals)
            input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                concatenated.astype(np.float32) if concatenated.dtype == np.float64 else concatenated,
                [len(v) for v in vals]
            )

    for key, vals in all_edge_attrs.items():
        if key in input_names and len(vals) > 0:
            concatenated = np.concatenate(vals)
            input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                concatenated.astype(np.float32) if concatenated.dtype == np.float64 else concatenated,
                [len(v) for v in vals]
            )

    for key, vals in all_graph_attrs.items():
        if key in input_names:
            arr = np.array(vals)
            input_tensors[key] = tf.constant(arr.astype(np.float32) if arr.dtype == np.float64 else arr)

    if edge_indices_list:
        input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
            np.concatenate(edge_indices_list).astype(np.int32),
            [len(e) for e in edge_indices_list]
        )

    return input_tensors


def test_batch(model, graphs, targets, mbids, batch_name):
    """Test training on a single batch."""
    try:
        # Convert to tensors
        x_batch = convert_graphs_to_tensors(model.inputs, graphs)
        y_batch = np.array(targets).reshape(-1, 1)

        # Check inputs for NaN/Inf
        for key, tensor in x_batch.items():
            tensor_vals = tensor.values if hasattr(tensor, 'values') else tensor
            vals_np = tensor_vals.numpy()
            if np.isnan(vals_np).any():
                return False, f"NaN in input tensor {key}", mbids
            if np.isinf(vals_np).any():
                return False, f"Inf in input tensor {key}", mbids

        # Check targets
        if np.isnan(y_batch).any() or np.isinf(y_batch).any():
            return False, "NaN/Inf in targets", mbids

        # Normalize
        if y_batch.std() > 0:
            y_batch = (y_batch - y_batch.mean()) / y_batch.std()

        # Try prediction (forward pass)
        try:
            predictions = model.predict(x_batch, verbose=0)
            if np.isnan(predictions).any() or np.isinf(predictions).any():
                return False, "NaN/Inf in predictions", mbids
        except Exception as e:
            return False, f"Prediction failed: {e}", mbids

        # Try one training step
        history = model.fit(x_batch, y_batch, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if np.isnan(loss) or np.isinf(loss):
            return False, f"NaN/Inf loss after training: {loss}", mbids

        if loss > 100:
            return False, f"Extremely high loss: {loss}", mbids

        return True, f"OK (loss: {loss:.4f})", mbids

    except Exception as e:
        return False, f"Exception: {str(e)}", mbids


def main():
    """Find problematic materials."""
    logger.info("Loading dataset...")
    train_data = hf_load_dataset(HF_DATASET_NAME, name=TEST_PROPERTY, split="train")
    train_entries = [dict(x) for x in train_data]

    logger.info(f"Loaded {len(train_entries)} training samples")

    # Preprocess all to graphs
    logger.info("Preprocessing crystals to graphs...")
    preprocessor = KNNAsymmetricUnitCell(N_NEIGHBORS)

    graphs = []
    mbids = []
    targets = []
    failed_preprocessing = []

    target_name = f'total_energy_alpha_{TEST_ALPHA}'

    for i, entry in enumerate(train_entries):
        if i % 100 == 0:
            logger.info(f"Processing {i}/{len(train_entries)}...")

        graph, mbid, error = preprocess_crystal_to_graph(entry, preprocessor)

        if graph is not None:
            graphs.append(graph)
            mbids.append(mbid)
            targets.append(entry[target_name])
        else:
            failed_preprocessing.append((mbid, error))

    logger.info(f"Successfully preprocessed {len(graphs)} graphs")
    if failed_preprocessing:
        logger.warning(f"Failed to preprocess {len(failed_preprocessing)} samples")

    # Create model
    logger.info("Creating model...")
    model = make_model(**model_default)

    # Compile with very conservative settings
    optimizer = ks.optimizers.legacy.Adam(learning_rate=0.00001, clipnorm=0.5, clipvalue=0.1)
    loss = ks.losses.MeanAbsoluteError()
    model.compile(optimizer=optimizer, loss=loss)

    # Test each batch
    logger.info(f"\nTesting batches of size {BATCH_SIZE}...")
    logger.info("="*80)

    problematic_batches = []
    good_batches = []

    for batch_idx in range(0, len(graphs), BATCH_SIZE):
        end_idx = min(batch_idx + BATCH_SIZE, len(graphs))
        batch_graphs = graphs[batch_idx:end_idx]
        batch_targets = targets[batch_idx:end_idx]
        batch_mbids = mbids[batch_idx:end_idx]

        batch_name = f"Batch {batch_idx//BATCH_SIZE + 1} (samples {batch_idx}-{end_idx})"

        success, message, mbids_in_batch = test_batch(
            model, batch_graphs, batch_targets, batch_mbids, batch_name
        )

        if success:
            logger.info(f"✅ {batch_name}: {message}")
            good_batches.append({
                'batch_idx': batch_idx,
                'size': len(batch_graphs),
                'mbids': mbids_in_batch,
                'status': 'OK'
            })
        else:
            logger.error(f"❌ {batch_name}: {message}")
            problematic_batches.append({
                'batch_idx': batch_idx,
                'size': len(batch_graphs),
                'mbids': mbids_in_batch,
                'error': message,
                'targets': [float(t) for t in batch_targets],
                'target_stats': {
                    'min': float(np.min(batch_targets)),
                    'max': float(np.max(batch_targets)),
                    'mean': float(np.mean(batch_targets)),
                    'std': float(np.std(batch_targets))
                }
            })

    # Summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total batches tested: {len(good_batches) + len(problematic_batches)}")
    logger.info(f"Good batches: {len(good_batches)}")
    logger.info(f"Problematic batches: {len(problematic_batches)}")

    if problematic_batches:
        logger.info(f"\n{'='*80}")
        logger.info("PROBLEMATIC BATCHES DETAILS:")
        logger.info("="*80)

        for pb in problematic_batches:
            logger.info(f"\nBatch starting at index {pb['batch_idx']}:")
            logger.info(f"  Error: {pb['error']}")
            logger.info(f"  Samples: {pb['size']}")
            logger.info(f"  Target stats: min={pb['target_stats']['min']:.4f}, "
                       f"max={pb['target_stats']['max']:.4f}, "
                       f"mean={pb['target_stats']['mean']:.4f}, "
                       f"std={pb['target_stats']['std']:.4f}")
            logger.info(f"  MBIDs: {pb['mbids'][:5]}...")  # Show first 5

        # Save details
        output_file = Path("problematic_materials.json")
        with open(output_file, 'w') as f:
            json.dump({
                'failed_preprocessing': failed_preprocessing,
                'problematic_batches': problematic_batches,
                'summary': {
                    'total_samples': len(train_entries),
                    'successful_graphs': len(graphs),
                    'failed_preprocessing': len(failed_preprocessing),
                    'good_batches': len(good_batches),
                    'problematic_batches': len(problematic_batches)
                }
            }, f, indent=2)

        logger.info(f"\n✅ Saved detailed analysis to {output_file}")
    else:
        logger.info("\n✅ All batches passed! No problematic materials found.")


if __name__ == "__main__":
    main()
