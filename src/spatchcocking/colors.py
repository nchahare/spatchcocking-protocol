"""The project's color map, in one place so every figure agrees.

ColorBrewer Spectral, ten stops, as chosen by the owner. Defined from explicit
hex rather than pulled from matplotlib by name, so the figures do not shift if a
future matplotlib alters its built-in.

Ordering runs dark red (low) to purple (high). `spectral(reverse=True)` gives
the other orientation, which is the more common convention for heatmaps where
cool should read as low.
"""
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["SPECTRAL_HEX", "spectral"]

SPECTRAL_HEX = [
    "#9e0142",
    "#d53e4f",
    "#f46d43",
    "#fdae61",
    "#fee08b",
    "#e6f598",
    "#abdda4",
    "#66c2a5",
    "#3288bd",
    "#5e4fa2",
]


def spectral(reverse=False, name="spatchcocking_spectral"):
    """The project color map as a matplotlib colormap."""
    stops = SPECTRAL_HEX[::-1] if reverse else SPECTRAL_HEX
    return LinearSegmentedColormap.from_list(name, stops)


# Per-field presentation, as the published figures use it. It is here once so that
# the CLI, the notebooks and any future script agree, and so a reader comparing a
# figure to the paper is not comparing two color schemes.
#
# `symmetric` marks the signed quantities, whose range is forced to +/- max so that
# the neutral color always sits at zero. The unsigned ones start at zero instead.
# Getting that wrong is not cosmetic: a diverging map centered on zero applied to
# thickness, which is never negative, puts the whole dataset in one half of the
# color range and throws away most of the contrast.
FIELD_STYLE = {
    "Gauss_Curvature": ("Gaussian curvature [$\\mu m^{-2}$]", "PiYG", True),
    "Mean_Curvature": ("Mean curvature [$\\mu m^{-1}$]", "PiYG", True),
    "K1": ("Max principal $k_1$ [$\\mu m^{-1}$]", "Spectral_r", True),
    "K2": ("Min principal $k_2$ [$\\mu m^{-1}$]", "Spectral_r", True),
    "thickness": ("Thickness [$\\mu m$]", "GnBu", False),
    "thickness_um": ("Thickness [$\\mu m$]", "GnBu", False),
    "phh3": ("Local pHH3+ cell count", "viridis", False),
}

# The field to show when none is named. Gaussian curvature, by preference.
DEFAULT_FIELD = "Gauss_Curvature"

__all__ += ["FIELD_STYLE", "DEFAULT_FIELD", "field_style"]


def field_style(field):
    """(label, colormap, symmetric) for a field name.

    Unknown fields -- a sampled stain, say -- fall back to a sequential map
    starting at zero, which is right for an intensity and safe for anything else
    non-negative.
    """
    if field in FIELD_STYLE:
        return FIELD_STYLE[field]
    if "curvature" in field.lower():
        return (field, "PiYG", True)
    if field.endswith("_density"):
        # notebook 03 names a point file's density "<name>_density"
        return (f"Local count: {field[:-len('_density')].replace('_', ' ')}", "viridis", False)
    if field.endswith("_intensity"):
        return (f"{field[:-len('_intensity')].replace('_', ' ')} intensity", "viridis", False)
    return (field, "viridis", False)
