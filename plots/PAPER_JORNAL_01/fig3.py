import os  # For saving the plot

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

# Imports for custom scale
import matplotlib.scale as mscale
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.constants import golden

# Keep your existing constants
ONE_COL_WIDTH_INCH = 2.75
TWO_COL_WIDTH_INCH = 5.5
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

plt.style.use("lamalab.mplstyle")


# --- Custom Scale Definition ---
class SquishScale(mscale.ScaleBase):
    name = "squish"

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.cutoff = kwargs.pop("cutoff", 0.6)
        self.compression = kwargs.pop("compression", 0.3)
        if self.compression <= 0 or self.compression > 1:
            raise ValueError(
                "Compression factor must be between 0 (exclusive) and 1 (inclusive)"
            )

    def get_transform(self):
        return self.SquishTransform(self.cutoff, self.compression)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.AutoLocator())
        axis.set_major_formatter(mticker.ScalarFormatter())
        axis.set_minor_locator(mticker.NullLocator())

    class SquishTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, cutoff, compression):
            super().__init__()
            self.cutoff = cutoff
            self.compression = compression

        def transform_non_affine(self, a):
            res = np.copy(a).astype(float)
            mask = res > self.cutoff
            res[mask] = self.cutoff + (res[mask] - self.cutoff) * self.compression
            return res

        def inverted(self):
            return SquishScale.InvertedSquishTransform(self.cutoff, self.compression)

    class InvertedSquishTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, cutoff, compression):
            super().__init__()
            self.cutoff = cutoff
            self.compression = compression

        def transform_non_affine(self, a):
            res = np.copy(a).astype(float)
            mask = res > self.cutoff
            res[mask] = self.cutoff + (res[mask] - self.cutoff) / self.compression
            return res


mscale.register_scale(SquishScale)
# --- End Custom Scale Definition ---


# Define range_frame function
def range_frame(ax, x, y_data_range, pad=0.1):  # y_data_range is conceptual [0,1]
    # For ylim, consider the requested data range (typically 0 to 1 for normalized MAE)
    effective_y_min = y_data_range[0]
    effective_y_max = y_data_range[1]

    ax.set_ylim(
        effective_y_min - pad * (effective_y_max - effective_y_min),
        effective_y_max + pad * (effective_y_max - effective_y_min),
    )

    filtered_x = [val for val in x if val is not None]
    if filtered_x:
        x_min, x_max = np.min(filtered_x), np.max(filtered_x)
    else:
        x_min, x_max = 0, 1

    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))

    # Using user's preference for spines
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))  # Show bottom spine
    ax.spines["left"].set_bounds(effective_y_min, effective_y_max)
    if filtered_x:  # Only set bounds if x_min, x_max are valid
        ax.spines["bottom"].set_bounds(x_min, x_max)
    else:
        ax.spines["bottom"].set_visible(False)  # Hide if no x data

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# Define the data
data = {
    "COGN": [0.0269, 0.3088, 0.017, 0.1559, 0.0535, 0.0689],
    "DimeNet": [0.0376, None, 0.0235, 0.1993, 0.0572, 0.0792],
    "SchNet": [0.0342, 0.3277, 0.0218, 0.2352, 0.059, 0.0796],
    "MatText-Best": [0.067748, 0.41169, 0.175, 0.42, 0.086, 0.09922],
    "LLM-prop": [None, None, None, 0.241, None, None],
    "Robocrys": [0.165, None, None, 0.221, 0.107, 0.132],
    #'RF-SCM': [0.2355, 0.4196, 0.1165, 0.3452, 0.082, 0.104],
    "CrabNet": [0.4065, 0.3234, 0.0862, 0.2655, 0.0758, 0.1014],
    #'Dummy': [0.5660,0.8088, 1.0059,1.3272,0.2897,0.2931]
}

# Initial properties list (will be reordered)
initial_properties = [
    "Perovskite\nFE",
    "Refractive\nIndex",
    "Formation\nEnergy",
    "Band\ngap",
    "Bulk\nmodulus",
    "Sheer\nmodulus",
]

# Define model types
model_types = {
    "COGN": "GNN",
    "DimeNet": "GNN",
    "SchNet": "GNN",
    "MatText-Best": "LLM",
    "Robocrys": "LLM",
    "CrabNet": "LLM",
    "LLM-prop": "LLM",
}

# Convert to DataFrame for easier manipulation
# Corrected: use 'data' instead of 'raw_data'
df = pd.DataFrame(data, index=initial_properties).T
df["model_type"] = df.index.map(model_types)


# Function to normalize data per property
def normalize_property(series):
    valid = series.dropna()
    if len(valid) <= 1:
        return series
    min_val = valid.min()
    max_val = valid.max()
    range_val = max_val - min_val
    if range_val == 0:
        return series
    return (series - min_val) / range_val


# Create a new DataFrame for normalized values
normalized_df = pd.DataFrame(index=df.index)
normalized_df["model_type"] = df["model_type"]

# Normalize each property
for prop in initial_properties:
    normalized_df[prop] = normalize_property(df[prop])

# --- Determine property order based on LLM minimum scaled MAE ---
llm_min_maes = {}
llm_models_df = normalized_df[normalized_df["model_type"] == "LLM"]

for prop in initial_properties:
    if not llm_models_df.empty and prop in llm_models_df.columns:
        min_mae_for_prop = llm_models_df[prop].dropna().min()
        if pd.notna(min_mae_for_prop):
            llm_min_maes[prop] = min_mae_for_prop
        else:
            llm_min_maes[prop] = float("inf")  # If no LLM data for a prop, put it last
    else:
        llm_min_maes[prop] = float("inf")

# Sort properties by the minimum LLM MAE
sorted_properties_tuples = sorted(llm_min_maes.items(), key=lambda item: item[1])
properties = [
    item[0] for item in sorted_properties_tuples
]  # This is the new ordered list


fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH))

# Apply the custom Y scale
ax.set_yscale("squish", cutoff=0.6, compression=0.3)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Explicit ticks for clarity

all_y_values_for_range_frame = []  # Collect all y_pos for range_frame min/max

# Plot each model type with different colors
for model_type, color in [("GNN", "#2563eb"), ("LLM", "#c1121f"), ("Other", "#6b7280")]:
    type_models = normalized_df[normalized_df["model_type"] == model_type].index

    for model in type_models:
        # Use the new 'properties' order for accessing data
        model_data = normalized_df.loc[model, properties].copy()

        for i, prop in enumerate(
            properties
        ):  # Iterate using the new sorted 'properties'
            if pd.notna(model_data[prop]):
                x_pos = i
                y_pos = model_data[prop]
                all_y_values_for_range_frame.append(y_pos)

                ax.scatter(
                    x_pos,
                    y_pos,
                    color=color,
                    s=80,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=1,
                    zorder=3,
                )

                # Add model name label with outline for contrast and GRAY color
                text_label = model.replace("-Best", "")

                text = ax.text(
                    x_pos + 0.1,
                    y_pos,
                    model.replace("-Best", ""),
                    fontsize=6,
                    va="center",
                    ha="left",
                    zorder=4,
                )
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=2, foreground="white")]
                )
                # text = ax.text(x_pos + 0.1, y_pos, text_label,
                #                fontsize=6, va='center', ha='left', zorder=4, color='dimgray') # Changed to dimgray
                # text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white')])

# Apply range frame to the plot
x_values = list(range(len(properties)))
# Pass the conceptual [0,1] range for y as the data is normalized to this.
range_frame(
    ax, x_values, [0.0, 1.0], pad=0.07
)  # Reduced padding for y, increased for x for labels

# Set the axis labels
# ax.set_xlabel('Material Properties', fontsize=7, fontweight='bold')
ax.set_ylabel(
    "Scaled MAE", fontsize=9, fontweight="bold"
)  # Changed "lowest" to "worst"

# Set the x-ticks to property names (using the new sorted 'properties' list)
ax.set_xticks(range(len(properties)))
ax.set_xticklabels(
    [p.replace(" ", "\n") for p in properties], fontsize=7
)  # Use the sorted properties
ax.tick_params(axis="x", which="major", pad=7)  # Add padding for x-tick labels

# Add vertical lines for property separation
for i in range(len(properties)):
    ax.axvline(
        x=i, color="#d1d5db", alpha=0.5, linestyle="-", ymin=0.05, ymax=0.95, zorder=0
    )

# Custom legend (kept commented as per your script, but this version is from previous solution)
legend_elements = [
    plt.scatter([], [], c="#2563eb", s=60, label="Graph"),
    plt.scatter([], [], c="#c1121f", s=60, label="LLM"),  # Matched color with plot
]
ax.legend(
    handles=legend_elements,
    loc="lower left",
    bbox_to_anchor=(-0.15, -0.25),  # Adjust anchor as needed
    fontsize=7,
    frameon=False,
)

# Adjust layout and save
plt.tight_layout()  # Add bottom margin for legend
output_dir = "fig3"  # Define an output directory
os.makedirs(output_dir, exist_ok=True)
plot_filename = os.path.join(output_dir, "llm_gnn_wall_yaxis_scaled_ordered.pdf")
plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
plt.savefig(
    plot_filename.replace(".pdf", ".png"), format="png", dpi=300, bbox_inches="tight"
)
plt.show()
