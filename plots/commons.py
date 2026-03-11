"""Common utilities, constants, and mappings shared across plotting scripts."""

from pathlib import Path

import numpy as np

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Representation Mappings ---
REPRESENTATION_MAPPING = {
    "slices": "SLICES",
    "slice_": "SLICES",
    "crystal_text_llm": "Crys.-text.",
    "crystal_llm_rep_": "Crys.-text.",
    "composition": "Composition",
    "composition_": "Composition",
    "cif_p1": "CIF P$_1$",
    "cif_p1_": "CIF P$_1$",
    "cif_symmetrized": "CIF Sym.",
    "cif_symmetrized_": "CIF Sym.",
    "atom_sequences": "Atom Seq.",
    "atoms_": "Atom Seq.",
    "atom_sequences_plusplus": "Atom Seq.++",
    "atoms_params_": "Atom Seq.++",
    "zmatrix": "Z-Matrix",
    "zmatrix_": "Z-Matrix",
    "local_env": "Local-Env",
    "local_env_": "Local-Env",
    "robocrys": "Robocrys",
}


# --- Property Display Mappings ---
PROPERTY_DISPLAY_MAPPING = {
    "matbench_log_gvrh": r"$\mu$",
    "matbench_log_kvrh": r"$K$",
    "matbench_perovskites": r"$E_{\mathrm{f}}$",
    "dielectric": "Dielectric",
    "phonons": "Phonons",
    "gvrh": "GVRH",
    "form_energy": "Form Energy",
    "bandgap": "Bandgap",
    "perovskites": "Perovskites",
}

PROPERTY_KEY_IN_MODEL_JSON = {
    "matbench_log_gvrh": "matbench_log_gvrh",
    "matbench_log_kvrh": "matbench_log_kvrh",
    "matbench_perovskites": "matbench_perovskites",
}

DATA_SIZE_MAPPING = {"30k": "30K", "100k": "100K", "300k": "300K", "2m": "2M"}


# --- Group Styles ---
GROUP_STYLES = {
    "compositional": {
        "color": "#e76f51",
        "members": [
            "composition",
            "atom_sequences_plusplus",
            "atom_sequences",
            "composition_",
            "atoms_params_",
            "atoms_",
        ],
    },
    "local": {
        "color": "#c1121f",
        "members": ["local_env", "slices", "robocrys", "local_env_", "slice_"],
    },
    "geometric": {
        "color": "#79155B",
        "members": [
            "crystal_text_llm",
            "cif_symmetrized",
            "cif_p1",
            "zmatrix",
            "crystal_llm_rep_",
            "cif_symmetrized_",
            "cif_p1_",
            "zmatrix_",
        ],
    },
}

# Per-representation color overrides
REP_COLOR_OVERRIDES = {
    "robocrys": "#6c757d",  # grey, distinct from local group red for readability
}


# --- Model Styles (for comparison plots) ---
MODEL_STYLES = {
    "MatText": {
        "color": "#c1121f",
        "linestyle": "-",
        "marker": "o",
        "label": "MatText (LLM)",
    },
    "CoGN": {
        "color": "#1a6faf",
        "linestyle": "--",
        "marker": "o",
        "label": "CoGN (GNN)",
    },
    "MODNet": {
        "color": "#6c757d",
        "linestyle": ":",
        "marker": "o",
        "label": "MODNet (ANN)",
    },
}

NON_MATTEXT_REPS = {"cogn_rep", "modnet"}


# --- Property Colors (for dumbbell plots) ---
PROPERTY_COLORS = {
    "form_energy": "#f54952",
    "bandgap": "#ae2d68",
    "dielectric": "#d00000",
    "gvrh": "#660f56",
    "phonons": "#731013",
    "perovskites": "#280659",
}
DEFAULT_COLOR = "#000000"


# --- HuggingFace Task Mappings ---
TASK_REVERSE_MAP = {
    "gvrh": "matbench_log_gvrh",
    "kvrh": "matbench_log_kvrh",
    "perovskites": "matbench_perovskites",
    "dielectric": "matbench_dielectric",
}


# --- Utility Functions ---
def range_frame(ax, x, y, pad=0.1):
    """Apply range frame styling to an axis.

    Args:
        ax: Matplotlib axis object
        x: Array-like of x values
        y: Array-like of y values
        pad: Padding fraction for axis limits
    """
    x = np.array(x)
    y = np.array(y)

    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)

    # Handle edge cases
    if x_min == x_max:
        if x_min > 0:  # For log scale
            x_min *= 0.5
            x_max *= 2.0
        else:
            x_min -= 1
            x_max += 1

    y_range = y_max - y_min
    x_range = x_max - x_min

    ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)

    # Handle x-axis limits for log scale
    if ax.get_xscale() == "log":
        if x_min <= 0:
            x_min = min(x[x > 0]) if np.any(x > 0) else 0.1
        log_pad = 1 + pad
        ax.set_xlim(x_min / log_pad, x_max * log_pad)
    else:
        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)

    # Set spine properties
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Set spine bounds to actual data range
    ax.spines["left"].set_bounds(np.min(y), np.max(y))
    ax.spines["bottom"].set_bounds(np.min(x), np.max(x))


def get_representation_group_and_color(representation):
    """Get the group and color for a given representation.

    Args:
        representation: Representation name

    Returns:
        Tuple of (group_name, color)
    """
    for group_name, group_info in GROUP_STYLES.items():
        if representation in group_info["members"]:
            return group_name, group_info["color"]
    return "unknown", "#808080"


def get_output_path(script_name: str, extension: str = "png") -> Path:
    """Get standardized output path for a plotting script.

    Args:
        script_name: Name of the plotting script (without .py extension)
        extension: File extension (png, pdf, etc.)

    Returns:
        Path object for output file
    """
    return OUTPUT_DIR / f"{script_name}.{extension}"


def save_figure(fig, script_name: str, dpi: int = 300):
    """Save figure in both PNG and PDF formats.

    Args:
        fig: Matplotlib figure object
        script_name: Name of the plotting script (without .py extension)
        dpi: DPI for saved figures
    """
    png_path = get_output_path(script_name, "png")
    pdf_path = get_output_path(script_name, "pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")

    return png_path, pdf_path
