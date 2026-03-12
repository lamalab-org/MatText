import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from commons import (
    PROPERTY_DISPLAY_MAPPING,
    PROPERTY_KEY_IN_MODEL_JSON,
    REPRESENTATION_MAPPING,
    TASK_REVERSE_MAP,
    save_figure,
)
from datasets import load_dataset
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.constants import golden

lama_aesthetics.get_style("main")

# --- Constants ---
TWO_COL_WIDTH_INCH = 5.5
GOLDEN_RATIO = golden

# Boolean flag to control error band display
SHOW_SINGLE_ERROR_BAND = True

SELECTED_REPRESENTATIONS = {
    "compositional_data": ["composition", "atom_sequences_plusplus"],
    "geometric_data": ["crystal_text_llm", "cif_symmetrized"],
    "local_data": ["local_env", "slices", "robocrys"],
    "compositional_model": ["composition", "atom_sequences_plusplus"],
    "geometric_model": ["crystal_text_llm", "cif_symmetrized"],
    "local_model": ["local_env", "slices", "robocrys"],
}

# Per-representation color overrides (takes precedence over group color)
REP_COLOR_OVERRIDES = {
    "robocrys": "#6c757d",  # grey, distinct from local group red for readability
}

GROUP_STYLES = {
    "compositional": {
        "color": "#e76f51",
        "data_members": SELECTED_REPRESENTATIONS["compositional_data"],
        "model_members": SELECTED_REPRESENTATIONS["compositional_model"],
        "linestyles": ["-", "dotted", "--", "-."],
    },
    "local": {
        "color": "#c1121f",
        "data_members": SELECTED_REPRESENTATIONS["local_data"],
        "model_members": SELECTED_REPRESENTATIONS["local_model"],
        "linestyles": ["-", "dotted", "dotted"],
    },
    "geometric": {
        "color": "#79155B",
        "data_members": SELECTED_REPRESENTATIONS["geometric_data"],
        "model_members": SELECTED_REPRESENTATIONS["geometric_model"],
        "linestyles": ["-", "dotted", "--", "-.", ":"],
    },
}


# --- Data Loading from HuggingFace ---
def hf_to_data_scaling_dict(repo_id):
    """Load BERT HF dataset and convert to nested dict for data scaling plots.

    Returns: {size: {representation: {matbench_task: {"rmse": {"mean", "std", "min", "max"}}}}}
    """
    ds = load_dataset(repo_id, split="train")
    result = {}
    for row in ds:
        size = row["size"]
        rep = row["representation"]
        task = TASK_REVERSE_MAP.get(row["task"], row["task"])

        result.setdefault(size, {}).setdefault(rep, {})

        mean_rmse = row["mean_rmse"]
        std_rmse = row["std_rmse"]
        mean_mae = row["mean_mae"]
        std_mae = row["std_mae"]

        result[size][rep][task] = {
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


def hf_to_model_scaling_dict(repo_id):
    """Load LLaMA HF dataset and convert to nested dict for model scaling
    plots.

    Returns: {model_size: {representation: {matbench_task: {"rmse": {"mean", "std", "min", "max"}}}}}
    """
    ds = load_dataset(repo_id, split="train")
    result = {}
    for row in ds:
        size = row["model_size"].upper()  # "7b" -> "7B"
        rep = row["representation"]
        task = TASK_REVERSE_MAP.get(row["property"], row["property"])

        result.setdefault(size, {}).setdefault(rep, {})

        mean_rmse = row["mean_rmse"]
        std_rmse = row["std_rmse"]

        result[size][rep][task] = {
            "rmse": {
                "mean": mean_rmse,
                "std": std_rmse,
                "min": mean_rmse - std_rmse,
                "max": mean_rmse + std_rmse,
            },
        }
    return result


# --- Tick Formatters ---
def format_log_ticks_for_data_axis(value, pos):
    """Custom formatter for log scale x-axis for dataset sizes."""
    if value == 3e4:
        return r"$10^4$"
    if value == 3e5:
        return r"$10^5$"
    if value == 2e6:
        return r"$10^6$"
    return ""


# --- Plotting Functions ---
def plot_dataset_scaling_relative_change(ax, data, property_name, show_xlabel=False):
    scales_numeric = np.array([30000, 100000, 300000, 2000000])
    scale_labels_for_data_access = ["30k", "100k", "300k", "2m"]

    ax.set_xscale("log")
    all_percentage_changes = []

    for _group_name, group_style in GROUP_STYLES.items():
        for member_idx, member_key in enumerate(group_style["data_members"]):
            rmse_mean_vals, all_fold_rmse_mins, all_fold_rmse_maxs = [], [], []
            for scale_l in scale_labels_for_data_access:
                try:
                    stats = data[scale_l][member_key][property_name]["rmse"]
                    rmse_mean_vals.append(stats["mean"])
                    all_fold_rmse_mins.append(stats["min"])
                    all_fold_rmse_maxs.append(stats["max"])
                except KeyError:
                    rmse_mean_vals.append(np.nan)
                    all_fold_rmse_mins.append(np.nan)
                    all_fold_rmse_maxs.append(np.nan)

            if not rmse_mean_vals or np.isnan(rmse_mean_vals[0]):
                continue

            base_rmse_30k = rmse_mean_vals[0]
            pct_change_mean = (
                (np.array(rmse_mean_vals) - base_rmse_30k) / base_rmse_30k * 100
            )

            line_style_key = group_style["linestyles"][
                member_idx % len(group_style["linestyles"])
            ]
            actual_linestyle = {
                "dashed": "--",
                "dotted": ":",
                "-": "-",
                "-.": "-.",
            }.get(line_style_key, line_style_key)

            plot_color = REP_COLOR_OVERRIDES.get(member_key, group_style["color"])
            ax.plot(
                scales_numeric,
                pct_change_mean,
                label=REPRESENTATION_MAPPING.get(member_key, member_key),
                color=plot_color,
                linestyle=actual_linestyle,
                linewidth=1.5,
                marker="o",
                markersize=4,
                clip_on=False,
            )

            valid_mins = [m for m in all_fold_rmse_mins if not np.isnan(m)]
            valid_maxs = [m for m in all_fold_rmse_maxs if not np.isnan(m)]

            if valid_mins and valid_maxs:
                if SHOW_SINGLE_ERROR_BAND:
                    for min_val in valid_mins:
                        all_percentage_changes.append(
                            (min_val - base_rmse_30k) / base_rmse_30k * 100
                        )
                    for max_val in valid_maxs:
                        all_percentage_changes.append(
                            (max_val - base_rmse_30k) / base_rmse_30k * 100
                        )
                else:
                    global_min_rmse = np.min(valid_mins)
                    global_max_rmse = np.max(valid_maxs)
                    lower_band_pct = (
                        (global_min_rmse - base_rmse_30k) / base_rmse_30k * 100
                    )
                    upper_band_pct = (
                        (global_max_rmse - base_rmse_30k) / base_rmse_30k * 100
                    )
                    ax.fill_between(
                        scales_numeric,
                        [lower_band_pct] * len(scales_numeric),
                        [upper_band_pct] * len(scales_numeric),
                        color=group_style["color"],
                        alpha=0.08,
                    )

    if SHOW_SINGLE_ERROR_BAND and all_percentage_changes:
        error_band_min = np.min(all_percentage_changes)
        error_band_max = np.max(all_percentage_changes)
        ax.fill_between(
            scales_numeric,
            [error_band_min] * len(scales_numeric),
            [error_band_max] * len(scales_numeric),
            color="#d62728",
            alpha=0.10,
            label="Error Range (All Reps)",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.spines[["left", "bottom"]].set_position(("outward", 5))
    ax.grid(False)
    ax.set_xlim(scales_numeric[0] * 0.9, scales_numeric[-1] * 1.1)

    ax.set_xticks(scales_numeric)

    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(format_log_ticks_for_data_axis))
        ax.xaxis.set_minor_formatter(NullFormatter())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    else:
        ax.set_xticklabels([])


def plot_model_scaling_relative_change(ax, data, property_name, show_xlabel=False):
    model_sizes_numeric = np.array([7, 13, 70])
    model_labels_for_data_access = ["7B", "13B", "70B"]
    short_prop_key = PROPERTY_KEY_IN_MODEL_JSON[property_name]

    ax.set_xscale("log")
    all_percentage_changes = []

    for _group_name, group_style in GROUP_STYLES.items():
        for member_idx, member_key in enumerate(group_style["model_members"]):
            rmse_mean_vals, all_fold_rmse_mins, all_fold_rmse_maxs = [], [], []
            for size_l in model_labels_for_data_access:
                try:
                    stats = data[size_l][member_key][short_prop_key]["rmse"]
                    rmse_mean_vals.append(stats["mean"])
                    all_fold_rmse_mins.append(stats["min"])
                    all_fold_rmse_maxs.append(stats["max"])
                except KeyError:
                    rmse_mean_vals.append(np.nan)
                    all_fold_rmse_mins.append(np.nan)
                    all_fold_rmse_maxs.append(np.nan)

            if not rmse_mean_vals or np.isnan(rmse_mean_vals[0]):
                continue

            base_rmse_7b = rmse_mean_vals[0]
            pct_change_mean = (
                (np.array(rmse_mean_vals) - base_rmse_7b) / base_rmse_7b * 100
            )

            line_style_key = group_style["linestyles"][
                member_idx % len(group_style["linestyles"])
            ]
            actual_linestyle = {
                "dashed": "--",
                "dotted": ":",
                "-": "-",
                "-.": "-.",
            }.get(line_style_key, line_style_key)

            plot_color = REP_COLOR_OVERRIDES.get(member_key, group_style["color"])
            ax.plot(
                model_sizes_numeric,
                pct_change_mean,
                label=REPRESENTATION_MAPPING.get(member_key, member_key),
                color=plot_color,
                linestyle=actual_linestyle,
                linewidth=1.5,
                marker="o",
                markersize=4,
                clip_on=False,
            )

            valid_mins = [m for m in all_fold_rmse_mins if not np.isnan(m)]
            valid_maxs = [m for m in all_fold_rmse_maxs if not np.isnan(m)]

            if valid_mins and valid_maxs:
                if SHOW_SINGLE_ERROR_BAND:
                    for min_val in valid_mins:
                        all_percentage_changes.append(
                            (min_val - base_rmse_7b) / base_rmse_7b * 100
                        )
                    for max_val in valid_maxs:
                        all_percentage_changes.append(
                            (max_val - base_rmse_7b) / base_rmse_7b * 100
                        )
                else:
                    global_min_rmse = np.min(valid_mins)
                    global_max_rmse = np.max(valid_maxs)
                    lower_band_pct = (
                        (global_min_rmse - base_rmse_7b) / base_rmse_7b * 100
                    )
                    upper_band_pct = (
                        (global_max_rmse - base_rmse_7b) / base_rmse_7b * 100
                    )
                    ax.fill_between(
                        model_sizes_numeric,
                        [lower_band_pct] * len(model_sizes_numeric),
                        [upper_band_pct] * len(model_sizes_numeric),
                        color=group_style["color"],
                        alpha=0.08,
                    )

    if SHOW_SINGLE_ERROR_BAND and all_percentage_changes:
        error_band_min = np.min(all_percentage_changes)
        error_band_max = np.max(all_percentage_changes)
        ax.fill_between(
            model_sizes_numeric,
            [error_band_min] * len(model_sizes_numeric),
            [error_band_max] * len(model_sizes_numeric),
            color="#d62728",
            alpha=0.10,
            label="Error Range (All Reps)",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.spines[["left", "bottom"]].set_position(("outward", 5))
    ax.grid(False)
    ax.set_xlim(model_sizes_numeric[0] * 0.85, model_sizes_numeric[-1] * 1.15)

    ax.set_xticks(model_sizes_numeric)
    if show_xlabel:
        ax.set_xticklabels([f"{size}B" for size in model_sizes_numeric])
    else:
        ax.set_xticklabels([])


def create_combined_relative_scaling_plot(
    data_size_data, model_size_data, output_filepath
):
    properties = ["matbench_log_gvrh", "matbench_log_kvrh", "matbench_perovskites"]
    num_properties = len(properties)
    num_scaling_types = 2

    subplot_width = TWO_COL_WIDTH_INCH / num_properties
    subplot_height = subplot_width / GOLDEN_RATIO
    fig_height = subplot_height * num_scaling_types
    fig_height_with_padding = fig_height * 1.5

    fig, axs = plt.subplots(
        num_scaling_types,
        num_properties,
        figsize=(TWO_COL_WIDTH_INCH, fig_height_with_padding),
        sharey="row",
    )

    for j, prop_key in enumerate(properties):
        # Dataset Scaling (Row 0)
        ax_data = axs[0, j]
        plot_dataset_scaling_relative_change(
            ax_data, data_size_data, prop_key, show_xlabel=True
        )
        ax_data.set_title(PROPERTY_DISPLAY_MAPPING[prop_key], fontsize=10)
        if j == 0:
            axs[0, 0].set_ylabel("Dataset Scaling\n(% Change RMSE)", fontsize=9)

        # Model Scaling (Row 1)
        ax_model = axs[1, j]
        plot_model_scaling_relative_change(
            ax_model, model_size_data, prop_key, show_xlabel=True
        )
        if j == 0:
            axs[1, 0].set_ylabel("Model Scaling\n(% Change RMSE)", fontsize=9)

    # Consolidate legend
    unique_line_items = {}
    for r_idx in range(num_scaling_types):
        temp_handles, temp_labels = axs[r_idx, 0].get_legend_handles_labels()
        for h, lbl in zip(temp_handles, temp_labels, strict=False):
            if lbl != "Error Range (All Reps)" and lbl not in unique_line_items:
                unique_line_items[lbl] = h

    sorted_labels = [
        REPRESENTATION_MAPPING[k]
        for k in REPRESENTATION_MAPPING
        if REPRESENTATION_MAPPING[k] in unique_line_items
    ]
    for label in unique_line_items.keys():
        if label not in sorted_labels:
            sorted_labels.append(label)

    final_handles = [
        unique_line_items[lbl] for lbl in sorted_labels if lbl in unique_line_items
    ]
    final_labels = [lbl for lbl in sorted_labels if lbl in unique_line_items]

    if SHOW_SINGLE_ERROR_BAND:
        for r_idx in range(num_scaling_types):
            for c_idx in range(num_properties):
                h_temp, l_temp = axs[r_idx, c_idx].get_legend_handles_labels()
                for handle_err, label_err in zip(h_temp, l_temp, strict=False):
                    if label_err == "Error Range (All Reps)":
                        if "Error Range (All Reps)" not in final_labels:
                            final_handles.append(handle_err)
                            final_labels.append(label_err)
                        break

    # Deduplicate
    final_unique = {}
    for h, lbl in zip(final_handles, final_labels, strict=False):
        if lbl not in final_unique:
            final_unique[lbl] = h
    final_handles = list(final_unique.values())
    final_labels = list(final_unique.keys())

    num_legend_cols = min(
        len(final_labels), 3 if len(final_labels) > 4 else len(final_labels)
    )

    legend_y_anchor = -0.05
    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        ncol=num_legend_cols + 1,
        frameon=False,
        fontsize=8,
    )

    bottom_margin = 0.25 if num_legend_cols > 3 else 0.20
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=bottom_margin, hspace=0.45, wspace=0.25
    )

    png_path, pdf_path = save_figure(fig, output_filepath)
    print(f"Combined plot saved to {png_path} and {pdf_path}")


if __name__ == "__main__":
    bert_repo = "n0w0f/MatText_bert_scaleup_results"
    llama_repo = "n0w0f/MatText_llama_scaleup_results"
    output_filename = "figure_6"

    print("Loading BERT data scaling results from HF...")
    data_size_data = hf_to_data_scaling_dict(bert_repo)
    print(f" Sizes: {list(data_size_data.keys())}")

    print("Loading LLaMA model scaling results from HF...")
    model_size_data = hf_to_model_scaling_dict(llama_repo)
    print(f" Model sizes: {list(model_size_data.keys())}")

    # Sanity check
    all_rep_keys_needed = set()
    for group_style_dict in GROUP_STYLES.values():
        all_rep_keys_needed.update(group_style_dict.get("data_members", []))
        all_rep_keys_needed.update(group_style_dict.get("model_members", []))

    missing_in_mapping = [
        key for key in all_rep_keys_needed if key not in REPRESENTATION_MAPPING
    ]
    if missing_in_mapping:
        print(f"ERROR: Keys missing in REPRESENTATION_MAPPING: {missing_in_mapping}")
    else:
        create_combined_relative_scaling_plot(
            data_size_data, model_size_data, output_filename
        )
