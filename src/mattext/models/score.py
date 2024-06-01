from matbench.data_ops import load
import numpy as np

MATTEXT_MATBENCH = {
    "kvrh": "matbench_log_kvrh",
    "gvrh": "matbench_log_gvrh",
    "perovskites": "matbench_perovskites",
}


def fold_key_namer(fold_key):
    return f"fold_{fold_key}"


def mattext_score():
    pass


def load_true_scores(dataset, mbids):
    df = load(MATTEXT_MATBENCH[dataset])
    scores = []
    for mbid in mbids:
        # Get the score for the mbid
        score = df.loc[mbid]["log10(K_VRH)"]
        return scores.append(score)
