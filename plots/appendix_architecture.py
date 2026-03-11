import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from commons import MODEL_STYLES, NON_MATTEXT_REPS, save_figure
from huggingface_hub import hf_hub_download
from lama_aesthetics import ONE_COL_WIDTH, TWO_COL_WIDTH

lama_aesthetics.get_style("main")

# --- Figure size constants ---
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH / 1.618


def range_frame(ax, x, y, pad=0.15):
    y_min, y_max = np.array(y).min(), np.array(y).max()
    x_min, x_max = np.array(x).min(), np.array(x).max()
    ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(x_min, x_max)
    ax.spines["left"].set_bounds(y_min, y_max)


def download_data(
    repo_id: str = "n0w0f/cliff-contribution-analysis_2",
    force_download: bool = True,
) -> pd.DataFrame:
    print(f"Downloading data from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        force_download=force_download,
    )
    table = pq.read_table(file_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def calculate_contributions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["geom_contrib"] = (
        df["error_geom_0.0"] + df["error_geom_0.2"] + df["error_geom_0.4"]
    ) - 3 * df["error_geom_0.5"]
    df["comp_contrib"] = (
        df["error_geom_1.0"] + df["error_geom_0.8"] + df["error_geom_0.6"]
    ) - 3 * df["error_geom_0.5"]
    return df


def compute_per_dataset_values(df: pd.DataFrame) -> dict:
    """
    Returns:
        {property: {model_key: {"comp": float, "geom": float}}}
    where model_key is one of "MatText", "CoGN", "MODNet".
    """
    result = {}
    properties = df[df["representation"] == "cogn_rep"]["property"].unique()

    for prop in properties:
        prop_df = df[df["property"] == prop]

        cogn_rows = prop_df[prop_df["representation"] == "cogn_rep"]
        modnet_rows = prop_df[prop_df["representation"] == "modnet"]
        mattext_rows = prop_df[~prop_df["representation"].isin(NON_MATTEXT_REPS)]

        if len(cogn_rows) == 0 or len(mattext_rows) == 0:
            continue

        result[prop] = {
            "MatText": {
                "comp": mattext_rows["comp_contrib"].mean(),
                "geom": mattext_rows["geom_contrib"].mean(),
            },
            "CoGN": {
                "comp": cogn_rows.iloc[0]["comp_contrib"],
                "geom": cogn_rows.iloc[0]["geom_contrib"],
            },
        }
        if len(modnet_rows) > 0:
            result[prop]["MODNet"] = {
                "comp": modnet_rows.iloc[0]["comp_contrib"],
                "geom": modnet_rows.iloc[0]["geom_contrib"],
            }

    return result


def compute_averaged_values(per_dataset: dict) -> dict:
    """Average comp and geom contributions across all datasets for each model.

    Returns: {model_key: {"comp": float, "geom": float, "comp_std": float, "geom_std": float}}
    """
    accum = {m: {"comp": [], "geom": []} for m in ["MatText", "CoGN", "MODNet"]}

    for _prop, models in per_dataset.items():
        for model_key, vals in models.items():
            accum[model_key]["comp"].append(vals["comp"])
            accum[model_key]["geom"].append(vals["geom"])

    averaged = {}
    for model_key, data in accum.items():
        if len(data["comp"]) == 0:
            continue
        averaged[model_key] = {
            "comp": np.mean(data["comp"]),
            "geom": np.mean(data["geom"]),
            "comp_std": np.std(data["comp"]),
            "geom_std": np.std(data["geom"]),
        }
    return averaged


def plot_panel_c(averaged: dict, output_file: str):
    """Single dumbbell plot (panel C) with one dumbbell per model type,
    averaged across all datasets.

    Error bars show std across datasets.
    """
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))

    x_positions = [0, 1]
    offset_step = 0.04
    offsets = {"MatText": offset_step, "CoGN": 0.0, "MODNet": -offset_step}

    all_y = []

    for model_key, vals in averaged.items():
        style = MODEL_STYLES[model_key]
        off = offsets[model_key]

        comp = vals["comp"]
        geom = vals["geom"]
        comp_std = vals["comp_std"]
        geom_std = vals["geom_std"]
        all_y.extend([comp, geom])

        # Connecting line
        ax.plot(
            [0, 1],
            [comp + off, geom + off],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            alpha=0.85,
            zorder=3,
        )

        # Error bars + markers
        for x_pos, y_val, _y_std in [(0, comp, comp_std), (1, geom, geom_std)]:
            ax.errorbar(
                x_pos,
                y_val + off,
                # yerr=y_std,
                fmt=style["marker"],
                color=style["color"],
                markersize=7,
                capsize=3,
                linewidth=1.2,
                alpha=0.85,
                zorder=4,
                label=style["label"] if x_pos == 0 else "_nolegend_",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Composition", "Geometry"])
    ax.set_ylabel("GCMG")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=1)

    range_frame(ax, np.array(x_positions), np.array(all_y), pad=0.2)

    ax.legend(frameon=False, fontsize=8, loc="best")
    plt.tight_layout()

    png_path, pdf_path = save_figure(fig, output_file)
    print(f"Saved panel C: {png_path} and {pdf_path}")
    plt.close(fig)


def plot_appendix(per_dataset: dict, output_file: str, single_row: bool = False):
    """
    Appendix figure: one subplot per dataset, dumbbells colored by model type
    (MatText=red, CoGN=blue, MODNet=grey).

    Args:
        single_row: if True, all subplots in one row; otherwise 2-row layout.
    """
    properties = sorted(per_dataset.keys())
    n_props = len(properties)

    if single_row:
        n_cols = n_props
        n_rows = 1
    else:
        n_cols = min(3, n_props)
        n_rows = int(np.ceil(n_props / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            TWO_COL_WIDTH if not single_row else TWO_COL_WIDTH * (n_props / 3),
            ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * n_rows,
        ),
        sharey=False,
    )
    axes = np.array(axes).flatten()

    offset_step = 0.04
    offsets = {"MatText": offset_step, "CoGN": 0.0, "MODNet": -offset_step}

    # Store all_y per subplot so we can sync limits after the loop
    all_y_per_idx = {}

    for idx, prop in enumerate(properties):
        ax = axes[idx]
        models = per_dataset[prop]
        all_y = []

        for model_key, vals in models.items():
            style = MODEL_STYLES[model_key]
            off = offsets[model_key]
            comp, geom = vals["comp"], vals["geom"]
            all_y.extend([comp, geom])

            ax.plot(
                [0, 1],
                [comp + off, geom + off],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                alpha=0.85,
                zorder=3,
            )
            for x_pos, y_val in [(0, comp), (1, geom)]:
                ax.scatter(
                    x_pos,
                    y_val + off,
                    s=50,
                    color=style["color"],
                    alpha=0.85,
                    marker=style["marker"],
                    zorder=4,
                    label=style["label"] if (x_pos == 0 and idx == 0) else "_nolegend_",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Composition", "Geometry"], fontsize=8)
        # Only label the leftmost subplot in each row
        if idx % n_cols == 0:
            ax.set_ylabel("Contribution to \n CC-Cliff", fontsize=9)
        else:
            ax.set_ylabel("")
        ax.set_title(f"Dataset {idx + 1}", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=1)

        if all_y:
            range_frame(ax, np.array([0, 1]), np.array(all_y), pad=0.2)
            all_y_per_idx[idx] = all_y

    # Sync y limits for datasets 2+ so they are comparable;
    # Dataset 1 (idx=0) keeps its own range to avoid clipping
    shared_indices = [i for i in all_y_per_idx if i > 0]
    if shared_indices:
        shared_vals = [v for i in shared_indices for v in all_y_per_idx[i]]
        pad = 0.2
        s_min, s_max = np.min(shared_vals), np.max(shared_vals)
        y_lo = s_min - pad * (s_max - s_min)
        y_hi = s_max + pad * (s_max - s_min)
        for i in shared_indices:
            ax = axes[i]
            ax.set_ylim(y_lo, y_hi)
            ax.spines["left"].set_bounds(s_min, s_max)

    for idx in range(n_props, len(axes)):
        axes[idx].set_visible(False)

    # Single legend on first axis
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    plt.tight_layout()
    png_path, pdf_path = save_figure(fig, output_file)
    print(f"Saved appendix figure: {png_path} and {pdf_path}")
    plt.close(fig)


def main():
    df = download_data(force_download=True)
    df = calculate_contributions(df)

    per_dataset = compute_per_dataset_values(df)
    print(f"\nDatasets found: {list(per_dataset.keys())}")

    averaged = compute_averaged_values(per_dataset)
    print("\nAveraged contributions across datasets:")
    for model, vals in averaged.items():
        print(
            f"  {model}: comp={vals['comp']:.3f} ± {vals['comp_std']:.3f}, "
            f"geom={vals['geom']:.3f} ± {vals['geom_std']:.3f}"
        )

    plot_panel_c(averaged, "appendix_architecture_panel_c")
    plot_appendix(per_dataset, "appendix_architecture_2row", single_row=False)
    plot_appendix(per_dataset, "appendix_architecture_1row", single_row=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
