import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from commons import DEFAULT_COLOR, PROPERTY_COLORS, save_figure
from huggingface_hub import hf_hub_download
from lama_aesthetics import ONE_COL_WIDTH

lama_aesthetics.get_style("main")

# Figure size constants
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH / 1.618


def range_frame(ax, x, y, pad=0.1):
    """Apply range frame to axes."""
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
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
    """Download data from HuggingFace."""
    print(f"Downloading data from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        force_download=force_download,
    )
    table = pq.read_table(file_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows")
    return df


def calculate_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate geometry and composition contributions."""
    df = df.copy()
    df["geom_contrib"] = (
        df["error_geom_0.0"] + df["error_geom_0.2"] + df["error_geom_0.4"]
    ) - 3 * df["error_geom_0.5"]

    df["comp_contrib"] = (
        df["error_geom_1.0"] + df["error_geom_0.8"] + df["error_geom_0.6"]
    ) - 3 * df["error_geom_0.5"]

    return df


def prepare_comparison_data(df: pd.DataFrame) -> dict:
    """Prepare comparison data with mean of all MatText reps vs cogn_rep and
    modnet.

    Non-MatText reps (cogn_rep, modnet) are excluded from the mean.
    Returns dict: {property: {'mean_all': {'comp': x, 'geom': y},
                               'cogn_rep': {'comp': x, 'geom': y},
                               'modnet': {'comp': x, 'geom': y} (if available)}}
    """
    NON_MATTEXT_REPS = {"cogn_rep", "modnet"}

    comparison = {}
    cogn_properties = df[df["representation"] == "cogn_rep"]["property"].unique()

    for prop in cogn_properties:
        prop_data = df[df["property"] == prop].copy()

        # Get cogn_rep data
        cogn_data = prop_data[prop_data["representation"] == "cogn_rep"].iloc[0]

        # Get mean of all MatText representations (exclude non-MatText)
        mattext_reps = prop_data[~prop_data["representation"].isin(NON_MATTEXT_REPS)]

        # Only include if there are MatText representations
        if len(mattext_reps) > 0:
            mean_comp = mattext_reps["comp_contrib"].mean()
            mean_geom = mattext_reps["geom_contrib"].mean()
            cogn_comp = cogn_data["comp_contrib"]
            cogn_geom = cogn_data["geom_contrib"]

            comparison[prop] = {
                "mean_all": {"comp": mean_comp, "geom": mean_geom},
                "cogn_rep": {"comp": cogn_comp, "geom": cogn_geom},
            }

            # Add modnet if available for this property
            modnet_rows = prop_data[prop_data["representation"] == "modnet"]
            if len(modnet_rows) > 0:
                modnet_data = modnet_rows.iloc[0]
                comparison[prop]["modnet"] = {
                    "comp": modnet_data["comp_contrib"],
                    "geom": modnet_data["geom_contrib"],
                }

    return comparison


def plot_comparison_dumbbells(comparison_data: dict, output_file: str):
    """Create subplot dumbbell plots comparing mean_all vs cogn_rep vs modnet.

    One subplot per property with up to three dumbbells each. Solid line
    for mean_all, dashed line for cogn_rep, dotted line for modnet.
    """
    if not comparison_data:
        print("No data to plot.")
        return

    # Sort properties by mean difference (geom - comp)
    sorted_properties = sorted(
        comparison_data.keys(),
        key=lambda p: (
            comparison_data[p]["mean_all"]["geom"]
            - comparison_data[p]["mean_all"]["comp"]
        ),
        reverse=True,
    )

    n_properties = len(sorted_properties)

    # Calculate grid dimensions
    n_cols = min(3, n_properties)
    n_rows = int(np.ceil(n_properties / n_cols))

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            ONE_COL_WIDTH * n_cols,
            ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * n_rows,
        ),
        sharey=False,
    )

    # Handle single subplot case
    if n_properties == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x_positions = [0, 1]  # 0 for composition, 1 for geometry

    # Property name mapping
    property_map = {
        "dielectric": "Dielectric",
        "phonons": "Phonons",
        "gvrh": "GVRH",
        "form_energy": "Form Energy",
        "bandgap": "Bandgap",
        "perovskites": "Perovskites",
    }

    # Offset for the two dumbbells
    offset = 0.01

    for idx, prop_name in enumerate(sorted_properties):
        ax = axes[idx]
        prop_data = comparison_data[prop_name]
        color = PROPERTY_COLORS.get(prop_name, DEFAULT_COLOR)

        all_y_values = []

        # Plot mean_all dumbbell (SOLID, slightly above)
        mean_comp = prop_data["mean_all"]["comp"]
        mean_geom = prop_data["mean_all"]["geom"]
        all_y_values.extend([mean_comp, mean_geom])

        # Line for mean_all (SOLID)
        ax.plot(
            [0, 1],
            [mean_comp + offset, mean_geom + offset],
            color=color,
            alpha=0.8,
            linewidth=2.5,
            linestyle="-",
        )
        # Points for mean_all
        ax.scatter(
            0,
            mean_comp + offset,
            s=80,
            color=color,
            alpha=0.8,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
        ax.scatter(
            1,
            mean_geom + offset,
            s=80,
            color=color,
            alpha=0.8,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )

        # Plot cogn_rep dumbbell (DASHED, slightly below)
        cogn_comp = prop_data["cogn_rep"]["comp"]
        cogn_geom = prop_data["cogn_rep"]["geom"]
        all_y_values.extend([cogn_comp, cogn_geom])

        # Line for cogn_rep (DASHED)
        ax.plot(
            [0, 1],
            [cogn_comp - offset, cogn_geom - offset],
            color=color,
            alpha=0.6,
            linewidth=2,
            linestyle="--",
        )
        # Points for cogn_rep
        ax.scatter(
            0,
            cogn_comp - offset,
            s=60,
            color=color,
            alpha=0.6,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
        ax.scatter(
            1,
            cogn_geom - offset,
            s=60,
            color=color,
            alpha=0.6,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

        # Plot modnet dumbbell (DOTTED, further below) if available
        if "modnet" in prop_data:
            modnet_comp = prop_data["modnet"]["comp"]
            modnet_geom = prop_data["modnet"]["geom"]
            all_y_values.extend([modnet_comp, modnet_geom])

            # Line for modnet (DOTTED)
            ax.plot(
                [0, 1],
                [modnet_comp - 2 * offset, modnet_geom - 2 * offset],
                color=color,
                alpha=0.4,
                linewidth=2,
                linestyle=":",
            )
            # Points for modnet
            ax.scatter(
                0,
                modnet_comp - 2 * offset,
                s=50,
                color=color,
                alpha=0.4,
                marker="s",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
            ax.scatter(
                1,
                modnet_geom - 2 * offset,
                s=50,
                color=color,
                alpha=0.4,
                marker="s",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

        # Apply range frame
        range_frame(ax, np.array(x_positions), np.array(all_y_values), pad=0.1)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set x-ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Composition", "Geometry"], fontsize=8)

        # Set axis labels
        ax.set_ylabel("GCMG", fontsize=10)

        # Set title with property name
        ax.set_title(
            property_map.get(prop_name, prop_name.replace("_", " ").title()),
            fontsize=11,
            fontweight="bold",
            color=color,
        )

        ax.grid(False)

    # Hide empty subplots
    for idx in range(n_properties, len(axes)):
        axes[idx].set_visible(False)

    # Create legend (only on first subplot)
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="-",
            linewidth=2.5,
            alpha=0.8,
            label="MatText",
        ),
        plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label="CoGN",
        ),
        plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle=":",
            linewidth=2,
            alpha=0.4,
            label="MODNet",
        ),
    ]

    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=False)

    plt.tight_layout()

    # Save plot
    png_path, pdf_path = save_figure(fig, output_file)
    print(f"✓ Saved: {png_path} and {pdf_path}")
    plt.close(fig)


def main():
    print("=" * 80)
    print("COGN Rep vs Mean All Reps - Dumbbell Comparison")
    print("=" * 80)

    # Download and process data
    df = download_data(force_download=True)
    print("\nCalculating contributions...")
    df = calculate_contributions(df)

    print("Preparing comparison data...")
    comparison_data = prepare_comparison_data(df)

    print(f"\nProperties to plot: {len(comparison_data)}")
    for prop in comparison_data.keys():
        print(f"  - {prop}")

    # Create plot
    print("\nCreating plot...")
    plot_comparison_dumbbells(comparison_data, "appendix_panel_comparison")

    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
