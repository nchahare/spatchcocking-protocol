"""Rendering the unwrapped map, the way the published code does it.

`visualize_flatmesh2` in the original package gets this right and says so in a
comment: interpolate in NORMALIZED (s, theta), then put the extent back in real
units. Interpolating in raw coordinates, where s spans 1 and theta spans 360,
gives a wildly anisotropic triangulation whose slivers show up as horizontal
banding across the map.

Kept here so every figure renders the map the same way instead of each script
inventing its own.
"""
import numpy as np
from scipy.interpolate import griddata

__all__ = ["grid_map", "paper_map"]


def grid_map(s, theta, values, resolution=500, method="linear"):
    """Interpolate scattered (s, theta, value) onto a regular grid.

    Returns (image, extent) ready for `imshow(..., origin="lower",
    extent=extent)`. Points outside the convex hull come back as NaN and render
    transparent, which is what gives the map its clean edge.
    """
    s = np.asarray(s, dtype=float)
    theta = np.asarray(theta, dtype=float)
    values = np.asarray(values, dtype=float)

    sn = (s - s.min()) / np.ptp(s)
    tn = (theta - theta.min()) / np.ptp(theta)

    gs, gt = np.meshgrid(np.linspace(0.0, 1.0, resolution),
                         np.linspace(0.0, 1.0, resolution))
    zi = griddata((sn, tn), values, (gs, gt), method=method)
    return zi, [s.min(), s.max(), theta.min(), theta.max()]


def paper_map(frame, field, ax, resolution=500):
    """Draw one field of an unwrapped map on `ax`, as the paper's figures do.

    - interpolated with `grid_map`, not scattered points
    - dorso-ventral across (V, D, V) and rostro-caudal up (C to R): `grid_map`
      returns s along the first axis, so the image is transposed and its extent
      reordered; without that the axis labels quietly describe the wrong axis
    - the field's own palette from `colors.field_style`, scaled to median +/- 3
      standard deviations as the published figures are. Signed fields are made
      symmetric so the neutral color is at zero; unsigned ones start at zero.

    `frame` is the DataFrame from `api.unwrap`, or a saved one. Returns the image,
    for a colorbar.
    """
    from .colors import field_style
    from .pipeline import getTightercmap

    values = frame[field].to_numpy()
    label, cmap_name, symmetric = field_style(field)
    image, extent = grid_map(frame["norm_height"].to_numpy(),
                             frame["angle_degrees"].to_numpy(), values,
                             resolution=resolution)
    vmin, vmax = getTightercmap(values, sigma=3)
    if symmetric:
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    else:
        vmin = 0.0
    im = ax.imshow(image.T, extent=[extent[2], extent[3], extent[0], extent[1]],
                   origin="lower", aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("dorso-ventral axis")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["V", "", "D", "", "V"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["C", "R"])
    return im
