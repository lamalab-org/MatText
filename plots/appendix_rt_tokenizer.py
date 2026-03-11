import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from commons import (
    DATA_SIZE_MAPPING,
    GROUP_STYLES,
    PROPERTY_DISPLAY_MAPPING,
    REPRESENTATION_MAPPING,
    get_representation_group_and_color,
    save_figure,
)
from datasets import load_dataset
from lama_aesthetics import TWO_COL_WIDTH
from scipy.constants import golden

lama_aesthetics.get_style("main")

# --- Constants ---
GOLDEN_RATIO = golden


def range_frame(ax, x, y, pad=0.1):
    """Apply range frame styling similar to the provided script."""
    y_min, y_max = np.array(y).min(), np.array(y).max()
    x_min, x_max = np.array(x).min(), np.array(x).max()
    ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + 2 * pad * (x_max - x_min))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_bounds(y_min, y_max)


def load_data_from_hf():
    """Load RT and Normal tokenizer data from HuggingFace Hub.

    Returns nested dicts keyed as data[size][rep][prop] = {mae: {mean, std, ...}, ...}
    """
    df = load_dataset("n0w0f/MatText-rt-scaleup", split="train").to_pandas()

    METRICS = ["mae", "rmse", "mape", "max_error"]
    STATS = ["mean", "std", "min", "max"]

    def _build_nested(sub_df):
        nested = {}
        for _, row in sub_df.iterrows():
            size = row["data_size"]
            rep = row["representation"]
            prop = row["property"]
            metrics = {m: {s: row[f"{m}_{s}"] for s in STATS} for m in METRICS}
            nested.setdefault(size, {}).setdefault(rep, {})[prop] = metrics
        return nested

    rt_data = _build_nested(df[df["tokenizer"] == "rt"])
    normal_data = _build_nested(df[df["tokenizer"] == "normal"])
    return rt_data, normal_data


def calculate_percentage_improvement(
    rt_data, normal_data, data_sizes, representations, properties
):
    """
    Calculate percentage improvement: ((Normal_RMSE - RT_RMSE) / Normal_RMSE) * 100
    Positive values indicate improvement (RT is better).
    """
    results = {}

    for size in data_sizes:
        results[size] = {}
        for prop in properties:
            results[size][prop] = {}

            for rep in representations:
                try:
                    # Get RT tokenizer RMSE
                    if (
                        size in rt_data
                        and rep in rt_data[size]
                        and prop in rt_data[size][rep]
                    ):
                        rt_rmse = rt_data[size][rep][prop]["mae"]["mean"]
                    else:
                        results[size][prop][rep] = None
                        continue

                    # Get Normal tokenizer mae
                    if (
                        size in normal_data
                        and rep in normal_data[size]
                        and prop in normal_data[size][rep]
                    ):
                        normal_rmse = normal_data[size][rep][prop]["mae"]["mean"]
                    else:
                        results[size][prop][rep] = None
                        continue

                    # Calculate percentage improvement (positive = RT better)
                    if normal_rmse != 0:
                        pct_improvement = ((normal_rmse - rt_rmse) / normal_rmse) * 100
                        results[size][prop][rep] = pct_improvement
                        print(
                            f"{size}-{prop}-{rep}: RT={rt_rmse:.4f}, Normal={normal_rmse:.4f}, Improvement={pct_improvement:.1f}%"
                        )
                    else:
                        results[size][prop][rep] = None

                except Exception as e:
                    print(f"Error calculating for {size}-{prop}-{rep}: {e}")
                    results[size][prop][rep] = None

    return results


def create_percentage_improvement_plot():
    """Create comparison plot showing percentage improvements using group-based
    coloring."""

    # Load data from HuggingFace Hub
    print("Loading data from HuggingFace Hub (n0w0f/MatText-rt-scaleup)...")
    rt_data, normal_data = load_data_from_hf()

    # Define what to compare
    data_sizes = ["30k", "100k", "300k", "2m"]
    properties = ["matbench_log_gvrh", "matbench_log_kvrh", "matbench_perovskites"]

    # Get common representations from both datasets
    rt_reps = set()
    normal_reps = set()

    for size in data_sizes:
        if size in rt_data:
            rt_reps.update(rt_data[size].keys())
        if size in normal_data:
            normal_reps.update(normal_data[size].keys())

    # Find common representations
    common_reps = list(rt_reps.intersection(normal_reps))
    if not common_reps:
        print("No common representations found between datasets!")
        print(f"RT reps: {list(rt_reps)}")
        print(f"Normal reps: {list(normal_reps)}")
        return

    print(f"Common representations found: {common_reps}")

    # Sort representations by group for better visual organization
    def get_group_order(rep):
        group, _ = get_representation_group_and_color(rep)
        group_order = {"compositional": 0, "local": 1, "geometric": 2, "unknown": 3}
        return group_order.get(group, 3)

    # Sort by group first, then alphabetically within group for consistency
    common_reps.sort(key=lambda x: (get_group_order(x), x))
    print(f"Representations will be ordered as: {common_reps}")

    # Calculate percentage improvements
    print("\nCalculating percentage improvements...")
    pct_improvements = calculate_percentage_improvement(
        rt_data, normal_data, data_sizes, common_reps, properties
    )

    # Create the plot: 4 rows (data sizes) x 3 columns (properties)
    num_rows = len(data_sizes)
    num_cols = len(properties)

    # Calculate figure size
    subplot_width = TWO_COL_WIDTH / num_cols
    _subplot_height = subplot_width / GOLDEN_RATIO
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH / golden

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(TWO_COL_WIDTH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),
        sharex=True,
        sharey=False,
    )

    # Ensure axes is 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, size in enumerate(data_sizes):
        for j, prop in enumerate(properties):
            ax = axes[i, j]

            # Collect data for this subplot
            rep_names = []
            pct_values = []
            bar_colors = []

            for rep in common_reps:
                pct_improvement = pct_improvements[size][prop].get(rep)
                if pct_improvement is not None:
                    rep_names.append(REPRESENTATION_MAPPING.get(rep, rep))
                    pct_values.append(pct_improvement)

                    # Get color based on group (regardless of positive/negative)
                    _, color = get_representation_group_and_color(rep)
                    bar_colors.append(color)

            if not pct_values:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                title = f"{DATA_SIZE_MAPPING[size]} - {PROPERTY_DISPLAY_MAPPING[prop]}"
                ax.set_title(title, fontsize=10, pad=10)
                continue

            # Create bar plot
            x_pos = np.arange(len(rep_names))
            bars = ax.bar(x_pos, pct_values, color=bar_colors, alpha=0.8)

            # Add horizontal line at y=0
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

            # Customize subplot
            if i == 0:
                title = f"{PROPERTY_DISPLAY_MAPPING[prop]}"
                ax.set_title(title, fontsize=10, pad=10)

            # Set y-label only for leftmost column
            if j == 0:
                ax.set_ylabel(f"{DATA_SIZE_MAPPING[size]}\n% Δ MAE", fontsize=9)

            # Set x-axis
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rep_names, fontsize=8)

            # Force rotation using setp for better control
            # plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

            # Apply rangeframe styling
            range_frame(ax, x_pos, pct_values)

            # Turn off grid (matching the provided script style)
            ax.grid(False)
            ax.set_axisbelow(True)

            # Add value labels on bars
            for bar, val in zip(bars, pct_values, strict=False):
                height = bar.get_height()
                label_y = height + (1 if height >= 0 else -1)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                    fontweight="bold",
                )

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

    # Add legend
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=9,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Force rotation after layout adjustments
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                tick.set_horizontalalignment("center")

    # Save plot
    png_path, pdf_path = save_figure(fig, "appendix_rt_tokenizer")
    print(f"\nComparison plot saved to: {png_path} and {pdf_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    all_improvements = []
    all_degradations = []
    total_comparisons = 0

    for size in data_sizes:
        for prop in properties:
            for rep in common_reps:
                pct_improvement = pct_improvements[size][prop].get(rep)
                if pct_improvement is not None:
                    total_comparisons += 1
                    if pct_improvement > 0:
                        all_improvements.append(pct_improvement)
                    else:
                        all_degradations.append(abs(pct_improvement))

    improvement_count = len(all_improvements)
    degradation_count = len(all_degradations)

    print(f"Total comparisons: {total_comparisons}")
    print(
        f"RT improvements: {improvement_count} ({100 * improvement_count / total_comparisons:.1f}%)"
    )
    print(
        f"RT degradations: {degradation_count} ({100 * degradation_count / total_comparisons:.1f}%)"
    )

    if all_improvements:
        print("\nImprovement stats (positive values):")
        print(f"  Average improvement: {np.mean(all_improvements):.1f}%")
        print(f"  Max improvement: {np.max(all_improvements):.1f}%")
        print(f"  Min improvement: {np.min(all_improvements):.1f}%")

    if all_degradations:
        print("\nDegradation stats (negative values):")
        print(f"  Average degradation: {np.mean(all_degradations):.1f}%")
        print(f"  Max degradation: {np.max(all_degradations):.1f}%")
        print(f"  Min degradation: {np.min(all_degradations):.1f}%")


if __name__ == "__main__":
    create_percentage_improvement_plot()
