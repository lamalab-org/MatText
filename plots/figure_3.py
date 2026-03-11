import lama_aesthetics
import matplotlib.pyplot as plt
from commons import range_frame, save_figure
from datasets import load_dataset
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger

lama_aesthetics.get_style("main")

# Updated group styles with n-gram handling
GROUP_STYLES = {
    "compositional": {
        "color": "#79155B",
        "members": {
            "composition": {"label": "semantic", "style": "-"},
            "composition_ngram": {"label": "semantic n-Gram", "style": ":"},
        },
    },
    "geometric": {
        "color": "#c1121f",
        "members": {
            "cif_p1": {"label": "semantic + spatial", "style": "-"},
            "cif_p1_ngram": {"label": "semantic + spatial n-Gram", "style": "--"},
        },
    },
    "local": {
        "color": "#c1121f",
        "members": {},
    },
}

map_properties_to_titles = {
    "dielectric": "Dielectric",
    "kvrh": "KVRH",
    "perovskites": "Perovskites",
}


def compute_gcmg(alpha_loss_data_dict):
    """Compute Geometry-Composition Modeling Gap (GCMG).

    GCMG = [ (loss@0 - loss@0.5) +  (loss@0.2 -loss@0.5) + (loss@0.4 -loss@0.5) ] -
           [ (loss@1 - loss@0.5) +  (loss@0.8 -loss@0.5) + (loss@0.6 -loss@0.5) ]

    Args:
        alpha_loss_data_dict (dict): A dictionary where keys are alpha strings (e.g., "0", "0.2")
                                     and values are dictionaries containing "eval_loss".

    Returns:
        float or None: The computed GCMG value, or None if required data is missing.
    """
    required_alphas = ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"]
    losses_at_alpha = {}

    for req_alpha in required_alphas:
        if (
            req_alpha in alpha_loss_data_dict
            and "eval_loss" in alpha_loss_data_dict[req_alpha]
        ):
            losses_at_alpha[req_alpha] = alpha_loss_data_dict[req_alpha]["eval_loss"]
        else:
            return None

    l0, l02, l04, l05, l06, l08, l1 = (
        losses_at_alpha["0"],
        losses_at_alpha["0.2"],
        losses_at_alpha["0.4"],
        losses_at_alpha["0.5"],
        losses_at_alpha["0.6"],
        losses_at_alpha["0.8"],
        losses_at_alpha["1"],
    )

    geometry_contribution = (l0 - l05) + (l02 - l05) + (l04 - l05)
    composition_contribution = (l1 - l05) + (l08 - l05) + (l06 - l05)

    gcmg = geometry_contribution - composition_contribution
    return gcmg


def load_data_from_hf():
    """Load binning data from HuggingFace Hub and reconstruct the nested
    dict."""
    df = load_dataset("n0w0f/MatText-ngram-binning", split="train").to_pandas()

    bins = sorted(df["bin"].unique().tolist())
    properties = sorted(df["property"].unique().tolist())

    # Reconstruct nested structure: data[bin_str][rep][prop][alpha_str] = {eval_loss: ...}
    nested = {}
    for _, row in df.iterrows():
        b = str(int(row["bin"]))
        rep = row["representation"]
        prop = row["property"]
        a = row["alpha"]
        alpha = str(int(a)) if a == int(a) else str(a)
        nested.setdefault(b, {}).setdefault(rep, {}).setdefault(prop, {})[alpha] = {
            "eval_loss": row["eval_loss"]
        }

    return {"metadata": {"bins": bins, "properties": properties}, "data": nested}


def create_gcmg_plot():
    """Create the GCMG plot with grouped representations."""
    data = load_data_from_hf()

    # Extract metadata
    bins = data.get("metadata", {}).get("bins", [])
    properties = data.get("metadata", {}).get("properties", [])

    if not bins or not properties:
        logger.error("'bins' or 'properties' missing from metadata in the JSON file.")
        return

    property_titles = ["Dataset 1", "Dataset 2", "Dataset 3"]
    if len(properties) != len(property_titles):
        logger.warning(
            f"Mismatch between number of properties in JSON ({len(properties)}) and predefined titles ({len(property_titles)}). Using JSON properties."
        )
        property_titles = [p.replace("_", " ").title() for p in properties]

    # Compute GCMG values
    gcmg_values = {prop: {} for prop in properties}
    for bin_num_int in bins:
        bin_num_str = str(bin_num_int)
        bin_data = data.get("data", {}).get(bin_num_str, {})
        if not bin_data:
            continue

        for _group_name, group_style in GROUP_STYLES.items():
            for rep_name in group_style["members"].keys():
                if rep_name in bin_data:
                    for _prop_idx, prop_name in enumerate(properties):
                        if prop_name in bin_data[rep_name]:
                            gcmg_val = compute_gcmg(bin_data[rep_name][prop_name])
                            if gcmg_val is not None:
                                if rep_name not in gcmg_values[prop_name]:
                                    gcmg_values[prop_name][rep_name] = []
                                gcmg_values[prop_name][rep_name].append(gcmg_val)

    # Create plots
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_HEIGHT
    fig, axs = plt.subplots(
        1,
        len(properties),
        figsize=(TWO_COL_WIDTH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH),
        sharey=False,
    )
    if len(properties) == 1:
        axs = [axs]

    for idx, (prop_name, title) in enumerate(
        zip(properties, property_titles, strict=False)
    ):
        ax = axs[idx]
        all_y_values_for_prop = []
        plotted_something = False

        # Plot each group
        for _group_name, group_style in GROUP_STYLES.items():
            for rep_name, rep_style in group_style["members"].items():
                if prop_name in gcmg_values and rep_name in gcmg_values[prop_name]:
                    y_values = gcmg_values[prop_name][rep_name]

                    if y_values and len(y_values) == len(bins):
                        all_y_values_for_prop.extend(y_values)
                        ax.plot(
                            bins,
                            y_values,
                            marker="o",
                            markersize=4,
                            linestyle=rep_style["style"],
                            color=group_style["color"],
                            label=rep_style["label"],
                            linewidth=1.5,
                        )
                        plotted_something = True

        if idx == 1:
            ax.set_xlabel("Number of Bins")
        if idx == 0:
            ax.set_ylabel("SSA")
        ax.set_title(f"{title}")
        ax.set_xscale("log")

        if plotted_something and all_y_values_for_prop:
            logger.debug(f"Y values for {prop_name}: {all_y_values_for_prop}")
            range_frame(ax, bins, all_y_values_for_prop)
        elif not plotted_something:
            ax.text(
                0.5,
                0.5,
                "No data to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )
            range_frame(ax, bins if bins else [1, 1000], [0, 1])

    # Handle legend
    handles, labels = [], []
    for ax_ in axs[::-1]:
        h, lbl = ax_.get_legend_handles_labels()
        if h:
            handles, labels = h, lbl
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.15),
        )
    else:
        logger.warning("No legend items to display.")

    plt.tight_layout()

    png_path, pdf_path = save_figure(fig, "figure_3")
    logger.success(f"Plots saved as {png_path} and {pdf_path}")


if __name__ == "__main__":
    create_gcmg_plot()
