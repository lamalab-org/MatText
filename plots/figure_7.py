import lama_aesthetics
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

# Imports for custom scale
import matplotlib.scale as mscale
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from commons import save_figure
from datasets import load_dataset
from lama_aesthetics import ONE_COL_WIDTH, TWO_COL_WIDTH

lama_aesthetics.get_style("main")

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH / 1.618
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH / 1.618


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


# Load data from HuggingFace Hub
_hf_col_map = {
    "Perovskite_FE": "Perovskite\nFE",
    "Refractive_Index": "Refractive\nIndex",
    "Formation_Energy": "Formation\nEnergy",
    "Band_gap": "Band\ngap",
    "Bulk_modulus": "Bulk\nmodulus",
    "Sheer_modulus": "Sheer\nmodulus",
}

# Initial properties list (will be reordered)
initial_properties = list(_hf_col_map.values())

df = (
    load_dataset("n0w0f/gnn_llm_wall", split="train")
    .to_pandas()
    .set_index("model")
    .rename(columns=_hf_col_map)
)


# Function to normalize data per property
def normalize_property(series):
    valid = series.dropna()
    # If there is 0 or 1 valid value, return a constant normalized value (e.g., 0.5)
    if len(valid) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float).where(
            ~series.isna(), other=pd.NA
        )
    min_val = valid.min()
    max_val = valid.max()
    range_val = max_val - min_val
    # If all valid values are identical, map them to a constant normalized value
    if range_val == 0:
        return pd.Series(0.5, index=series.index, dtype=float).where(
            ~series.isna(), other=pd.NA
        )
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


fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH))

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

# Apply range frame to the plot
x_values = list(range(len(properties)))
# Pass the conceptual [0,1] range for y as the data is normalized to this.
range_frame(
    ax, x_values, [0.0, 1.0], pad=0.07
)  # Reduced padding for y, increased for x for labels

ax.set_ylabel("Scaled MAE", fontsize=9, fontweight="bold")

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

legend_elements = [
    plt.scatter([], [], c="#2563eb", s=60, label="Graph"),
    plt.scatter([], [], c="#c1121f", s=60, label="LLM"),
    plt.scatter([], [], c="#6b7280", s=60, label="Other"),
]
ax.legend(
    handles=legend_elements,
    loc="lower left",
    bbox_to_anchor=(-0.15, -0.25),
    fontsize=7,
    frameon=False,
)

plt.tight_layout()
save_figure(fig, "figure_7")
