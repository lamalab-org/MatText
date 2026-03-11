import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from commons import (
    GROUP_STYLES,
    PROPERTY_DISPLAY_MAPPING,
    REPRESENTATION_MAPPING,
    get_representation_group_and_color,
    range_frame,
    save_figure,
)
from datasets import load_dataset
from lama_aesthetics import TWO_COL_WIDTH
from loguru import logger

lama_aesthetics.get_style("main")


def hf_to_nested_dict(repo_id, size_filter=None):
    """Load HF dataset and convert to nested dict: {rep: {task: {"rmse": {...}}}}.

    If size_filter is given (e.g. "30k"), only rows matching that size are included.
    """
    ds = load_dataset(repo_id, split="train")
    result = {}

    # HF task name -> matbench property key
    TASK_REVERSE_MAP = {
        "gvrh": "matbench_log_gvrh",
        "kvrh": "matbench_log_kvrh",
        "perovskites": "matbench_perovskites",
        "dielectric": "matbench_dielectric",
    }

    for row in ds:
        if size_filter and row["size"] != size_filter:
            continue

        rep = row["representation"]
        task = TASK_REVERSE_MAP.get(row["task"], row["task"])

        result.setdefault(rep, {})

        mean_rmse = row["mean_rmse"]
        std_rmse = row["std_rmse"]
        mean_mae = row["mean_mae"]
        std_mae = row["std_mae"]

        result[rep][task] = {
            "rmse": {
                "mean": mean_rmse,
                "std": std_rmse,
                "min": mean_rmse - std_rmse,
                "max": mean_rmse + std_rmse,
            },
            "mae": {
                "mean": mean_mae,
                "std": std_mae,
                "min": mean_mae - std_mae,
                "max": mean_mae + std_mae,
            },
        }
    return result


def plot_30k_bar_results(data_30k, properties, output_filepath):
    """Create bar plots for 30K results across three properties."""

    num_properties = len(properties)
    GOLDEN_RATIO = 1.618

    subplot_width = TWO_COL_WIDTH / num_properties
    subplot_height = subplot_width / GOLDEN_RATIO

    fig, axes = plt.subplots(
        1, num_properties, figsize=(TWO_COL_WIDTH, subplot_height), sharey=False
    )

    if num_properties == 1:
        axes = [axes]

    # Get all available representations in the data
    available_representations = list(data_30k.keys())

    # Sort representations by group for visual organization
    def get_group_order(rep):
        group, _ = get_representation_group_and_color(rep)
        group_order = {"compositional": 0, "local": 1, "geometric": 2, "unknown": 3}
        return group_order.get(group, 3)

    available_representations.sort(key=lambda x: (get_group_order(x), x))
    logger.info(f"Representations will be ordered as: {available_representations}")

    for i, prop in enumerate(properties):
        ax = axes[i]

        rmse_means = []
        rmse_stds = []
        colors = []
        labels = []

        for rep in available_representations:
            try:
                if rep in data_30k and prop in data_30k[rep]:
                    rmse_stats = data_30k[rep][prop]["rmse"]
                    rmse_means.append(rmse_stats["mean"])
                    rmse_stds.append(rmse_stats["std"])

                    _, color = get_representation_group_and_color(rep)
                    colors.append(color)
                    labels.append(REPRESENTATION_MAPPING.get(rep, rep))
                else:
                    continue
            except KeyError:
                continue

        if not rmse_means:
            logger.warning(f"No data found for property {prop}")
            continue

        logger.info(f"Property {prop}: Found {len(rmse_means)} representations")

        x_pos = np.arange(len(rmse_means))

        ax.bar(
            x_pos,
            rmse_means,
            yerr=rmse_stds,
            color=colors,
            alpha=0.8,
            capsize=0,
        )

        ax.set_title(PROPERTY_DISPLAY_MAPPING.get(prop, prop), fontsize=10, pad=10)
        ax.set_ylabel("RMSE" if i == 0 else "", fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        range_frame(ax, x_pos, rmse_means)

        ax.grid(False)
        ax.set_axisbelow(True)

    # Create custom legend based on groups
    legend_elements = []
    for group_name, group_info in GROUP_STYLES.items():
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=group_info["color"],
                alpha=0.8,
                label=group_name.capitalize(),
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    # Force rotation after layout
    for i in range(num_properties):
        ax = axes[i] if num_properties > 1 else axes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_horizontalalignment("center")

    png_path, pdf_path = save_figure(fig, output_filepath.replace(".png", ""))
    logger.success(f"Bar plot saved to {png_path} and {pdf_path}")


if __name__ == "__main__":
    bert_repo = "n0w0f/MatText_bert_scaleup_results"

    properties = ["matbench_log_gvrh", "matbench_log_kvrh", "matbench_perovskites"]

    logger.info("Loading 30k data from HF...")
    data_30k = hf_to_nested_dict(bert_repo, size_filter="30k")
    logger.info(f"Found representations: {list(data_30k.keys())}")

    if data_30k:
        plot_30k_bar_results(data_30k, properties, "figure_5")
    else:
        logger.warning("No data available to plot.")
