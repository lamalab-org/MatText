import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects


# Define range_frame function
def range_frame(ax, x, y, pad=0.1):
    y_min, y_max = np.array(y).min(), np.array(y).max()
    filtered_x = [val for val in x if val is not None]
    if filtered_x:
        x_min, x_max = np.min(filtered_x), np.max(filtered_x)
    else:
        x_min, x_max = 0, 1  # Default values or handle accordingly

    ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ax.spines["left"].set_position(("outward", 5))  # Reduced from 10 to 5
    ax.spines["bottom"].set_visible(
        False
    )  # Hide bottom spine since we're using vertical lines
    ax.spines["left"].set_bounds(y_min, y_max)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# Define the data
data = {
    "COGN": [0.0269, 0.3088, 0.017, 0.1559, 0.0535, 0.0689],
    #'CGCNN': [0.0452, 0.5988, 0.0337, 0.2972, 0.0712, 0.0895],
    "DimeNet": [0.0376, None, 0.0235, 0.1993, 0.0572, 0.0792],
    "SchNet": [0.0342, 0.3277, 0.0218, 0.2352, 0.059, 0.0796],
    "MatText-Best": [0.067748, 0.41169, 0.175, 0.42, 0.086, 0.09922],
    "LLM-prop": [None, None, None, 0.241, None, None],
    "Robocrys": [0.165, None, None, 0.221, 0.107, 0.132],
    # 'RF-SCM': [0.2355, 0.4196, 0.1165, 0.3452, 0.082, 0.104],
    "CrabNet": [0.4065, 0.3234, 0.0862, 0.2655, 0.0758, 0.1014],
    #'Dummy': [0.5660, 0.8088, 1.0059, 1.3272, 0.2897, 0.2931]
    "MODNet": [0.0908, 0.2711, 0.0448, 0.2199, 0.0548, 0.0731],
}

properties = [
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
    "RF-SCM": "Other",
    "CrabNet": "LLM",
    "LLM-prop": "LLM",
    "Dummy": "Other",
}

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, index=properties).T
df["model_type"] = df.index.map(model_types)


# Function to normalize data per property
def normalize_property(series):
    """Normalize a property series so that min value is 0 and max value is 1"""
    valid = series.dropna()
    if len(valid) <= 1:
        return series  # Can't normalize single value
    min_val = valid.min()
    max_val = valid.max()
    range_val = max_val - min_val
    if range_val == 0:
        return series  # Can't normalize if all values are the same
    return (series - min_val) / range_val


# Create a new DataFrame for normalized values
normalized_df = pd.DataFrame(index=df.index)
normalized_df["model_type"] = df["model_type"]

# Create a new DataFrame for normalized values (for visualization)
normalized_size_df = pd.DataFrame(index=df.index)
normalized_size_df["model_type"] = df["model_type"]

# Normalize each property
for prop in properties:
    normalized_df[prop] = normalize_property(df[prop])
    # For point sizes, normalize directly from original error values (1 = worst, 0 = best)
    normalized_size_df[prop] = normalize_property(df[prop])


fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH))

# Create dictionaries to store GNN min and LLM min for each property
gnn_mins = {}
llm_mins = {}

# Collect all y values for range_frame
all_y_values = []

# Draw vertical lines for each property


# Function to scale point size based on normalized error (1 = biggest point)
def scale_point_size(normalized_value, min_size=40, max_size=250):
    """Scale point size based on normalized value - 1 (worst) = largest point"""
    return min_size + normalized_value * (max_size - min_size)


# Plot each model type with different colors
for model_type, color in [("GNN", "#702963"), ("LLM", "#c1121f"), ("Other", "#6b7280")]:
    # Get models of this type
    type_models = normalized_df[normalized_df["model_type"] == model_type].index

    for model in type_models:
        model_data = normalized_df.loc[model, properties].copy()
        size_data = normalized_size_df.loc[model, properties].copy()  # For point sizing

        # Plot each data point for this model
        for i, prop in enumerate(properties):
            if pd.notna(model_data[prop]):
                x_pos = i
                y_pos = model_data[prop]
                all_y_values.append(y_pos)  # Collect y values for range_frame

                # Store minimum values for GNN and LLM models
                if model_type == "GNN" and (
                    prop not in gnn_mins or y_pos < gnn_mins[prop][1]
                ):
                    gnn_mins[prop] = (x_pos, y_pos, model)
                elif model_type == "LLM" and (
                    prop not in llm_mins or y_pos < llm_mins[prop][1]
                ):
                    llm_mins[prop] = (x_pos, y_pos, model)

                # Calculate point size based on normalized error (bigger error = bigger point)
                norm_error = size_data[prop]
                # point_size = scale_point_size(norm_error)

                # Plot the point
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

                # Add model name label with outline for contrast
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
range_frame(ax, x_values, all_y_values, pad=0.1)  # Reduced padding for tighter fit

# Set the axis labels
# ax.set_xlabel('Material Properties', fontsize=7, fontweight='bold')
ax.set_ylabel("Scaled MAE", fontsize=9, fontweight="bold")

# Set the x-ticks to property names
ax.set_xticks(range(len(properties)))
ax.set_xticklabels(properties, fontsize=7)
ax.tick_params(axis="x", which="both", bottom=False)  # Remove x-axis ticks

for i in range(len(properties)):
    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    ax.axvline(
        x=i, color="#d1d5db", alpha=0.3, linestyle="-", ymin=0.1, ymax=0.9, zorder=0
    )


# Create a custom legend for model types
legend_elements = [
    plt.scatter([], [], c="#702963", s=80, label="Graph"),
    plt.scatter([], [], c="#c1121f", s=80, label="LLM"),
]
ax.legend(
    handles=legend_elements, loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=7
)


# Adjust layout and save
plt.tight_layout()
output_dir = "fig3"
os.makedirs(output_dir, exist_ok=True)
plot_filename = os.path.join(output_dir, "llm_gnn_wall_with_modnet.pdf")
plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
plt.show()
