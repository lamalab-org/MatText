"""
Find the EXACT sample(s) causing NaN in predictions.
Tests samples one-by-one within problematic batches.
"""

import json
import logging
import os
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from datasets import load_dataset as hf_load_dataset
from pymatgen.core import Structure

from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_DATASET_NAME = "jablonkagroup/MatText-hypo_pot"
TEST_PROPERTY = "dielectric"
N_NEIGHBORS = 24


def test_single_sample(model, graph, target, mbid):
    """Test a single sample."""
    try:
        # Convert to tensors
        input_tensors = {}

        # Node attributes
        node_data_list = [node_data for node, node_data in graph.nodes(data=True)]
        if node_data_list:
            for key in node_data_list[0].keys():
                vals = np.array([nd[key] for nd in node_data_list])
                input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                    vals.astype(np.float32) if vals.dtype == np.float64 else vals,
                    [len(vals)]
                )

        # Edge indices
        edges = list(graph.edges())
        if edges:
            edge_idx = np.array(edges)[:, [1, 0]]
            input_tensors['edge_indices'] = tf.RaggedTensor.from_row_lengths(
                edge_idx.astype(np.int32), [len(edge_idx)]
            )

            # Edge attributes
            edge_data_list = [edge_data for u, v, edge_data in graph.edges(data=True)]
            if edge_data_list:
                for key in edge_data_list[0].keys():
                    vals = np.array([ed[key] for ed in edge_data_list])
                    input_tensors[key] = tf.RaggedTensor.from_row_lengths(
                        vals.astype(np.float32) if vals.dtype == np.float64 else vals,
                        [len(vals)]
                    )

        # Check for NaN/Inf in inputs
        for key, tensor in input_tensors.items():
            tensor_vals = tensor.values if hasattr(tensor, 'values') else tensor
            vals_np = tensor_vals.numpy()

            if np.isnan(vals_np).any():
                return False, f"NaN in input {key}", None
            if np.isinf(vals_np).any():
                return False, f"Inf in input {key}", None

            # Check for extreme values
            if len(vals_np) > 0 and vals_np.dtype in [np.float32, np.float64]:
                val_max = np.abs(vals_np).max()
                if val_max > 1e10:
                    return False, f"Extreme value in {key}: {val_max:.2e}", None

        # Try prediction
        pred = model.predict(input_tensors, verbose=0)

        if np.isnan(pred).any():
            return False, "NaN in prediction", input_tensors
        if np.isinf(pred).any():
            return False, "Inf in prediction", input_tensors
        if np.abs(pred).max() > 1e10:
            return False, f"Extreme prediction: {pred[0][0]:.2e}", input_tensors

        return True, f"OK (pred: {pred[0][0]:.4f})", None

    except Exception as e:
        return False, f"Exception: {str(e)}", None


def analyze_problematic_features(graph, mbid):
    """Analyze features of a problematic graph."""
    info = {
        'mbid': mbid,
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'node_features': {},
        'edge_features': {},
        'graph_features': {}
    }

    # Node features
    node_data_list = [node_data for node, node_data in graph.nodes(data=True)]
    if node_data_list:
        for key in node_data_list[0].keys():
            vals = np.array([nd[key] for nd in node_data_list])
            info['node_features'][key] = {
                'shape': vals.shape,
                'dtype': str(vals.dtype),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'mean': float(np.mean(vals)),
                'has_nan': bool(np.isnan(vals).any()),
                'has_inf': bool(np.isinf(vals).any())
            }

    # Edge features
    if graph.number_of_edges() > 0:
        edge_data_list = [edge_data for u, v, edge_data in graph.edges(data=True)]
        if edge_data_list:
            for key in edge_data_list[0].keys():
                vals = np.array([ed[key] for ed in edge_data_list])
                info['edge_features'][key] = {
                    'shape': vals.shape,
                    'dtype': str(vals.dtype),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'mean': float(np.mean(vals)),
                    'has_nan': bool(np.isnan(vals).any()),
                    'has_inf': bool(np.isinf(vals).any())
                }

    return info


def main():
    """Find exact problematic samples."""
    logger.info("Loading problematic batches info...")

    with open('problematic_materials.json', 'r') as f:
        data = json.load(f)

    problematic_batches = data['problematic_batches']
    logger.info(f"Found {len(problematic_batches)} problematic batches")

    # Load dataset
    logger.info("Loading dataset...")
    train_data = hf_load_dataset(HF_DATASET_NAME, name=TEST_PROPERTY, split="train")
    train_entries = [dict(x) for x in train_data]

    # Create preprocessor and model
    logger.info("Creating model...")
    preprocessor = KNNAsymmetricUnitCell(N_NEIGHBORS)
    model = make_model(**model_default)

    # Test samples in first problematic batch individually
    first_prob_batch = problematic_batches[0]
    batch_start = first_prob_batch['batch_idx']
    batch_mbids = first_prob_batch['mbids']

    logger.info(f"\nTesting samples individually in Batch 1 (indices {batch_start}-{batch_start+64})...")
    logger.info("="*80)

    bad_samples = []
    good_samples = []

    for i, mbid in enumerate(batch_mbids):
        # Find entry
        entry = next((e for e in train_entries if e.get('mbid') == mbid), None)
        if not entry:
            logger.warning(f"Could not find entry for {mbid}")
            continue

        # Preprocess
        try:
            structure = Structure.from_str(entry['cif_p1'], fmt="cif")
            graph = preprocessor(structure)
            graph.graph['dataset_id'] = mbid
        except Exception as e:
            logger.error(f"❌ {mbid}: Preprocessing failed - {e}")
            bad_samples.append({'mbid': mbid, 'error': f'Preprocessing: {e}'})
            continue

        # Test
        target = entry[f'total_energy_alpha_0.2']
        success, message, tensors = test_single_sample(model, graph, target, mbid)

        if success:
            logger.info(f"✅ {mbid}: {message}")
            good_samples.append(mbid)
        else:
            logger.error(f"❌ {mbid}: {message}")

            # Analyze features
            feature_info = analyze_problematic_features(graph, mbid)
            bad_samples.append({
                'mbid': mbid,
                'error': message,
                'target': float(target),
                'features': feature_info
            })

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DETAILED ANALYSIS")
    logger.info("="*80)
    logger.info(f"Good samples: {len(good_samples)}/{len(batch_mbids)}")
    logger.info(f"Bad samples: {len(bad_samples)}/{len(batch_mbids)}")

    if bad_samples:
        logger.info(f"\nBAD SAMPLES DETAILS:")
        logger.info("="*80)

        for sample in bad_samples[:10]:  # Show first 10
            logger.info(f"\n{sample['mbid']}:")
            logger.info(f"  Error: {sample['error']}")
            if 'features' in sample:
                features = sample['features']
                logger.info(f"  Nodes: {features['num_nodes']}, Edges: {features['num_edges']}")

                # Show problematic features
                logger.info(f"  Node features:")
                for feat_name, feat_info in features['node_features'].items():
                    if feat_info['has_nan'] or feat_info['has_inf']:
                        logger.info(f"    ❌ {feat_name}: has_nan={feat_info['has_nan']}, has_inf={feat_info['has_inf']}")
                    elif abs(feat_info['max']) > 1e6 or abs(feat_info['min']) > 1e6:
                        logger.info(f"    ⚠️  {feat_name}: range=[{feat_info['min']:.2e}, {feat_info['max']:.2e}]")

                logger.info(f"  Edge features:")
                for feat_name, feat_info in features['edge_features'].items():
                    if feat_info['has_nan'] or feat_info['has_inf']:
                        logger.info(f"    ❌ {feat_name}: has_nan={feat_info['has_nan']}, has_inf={feat_info['has_inf']}")
                    elif abs(feat_info['max']) > 1e6 or abs(feat_info['min']) > 1e6:
                        logger.info(f"    ⚠️  {feat_name}: range=[{feat_info['min']:.2e}, {feat_info['max']:.2e}]")

        # Save detailed report
        with open('bad_samples_detailed.json', 'w') as f:
            json.dump(bad_samples, f, indent=2)

        logger.info(f"\n✅ Saved detailed analysis to bad_samples_detailed.json")


if __name__ == "__main__":
    main()
