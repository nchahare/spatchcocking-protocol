# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: spatchcocking
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Put the measurements on the surface
#
# Every measurement becomes a **field**: one number at each point of the
# reference surface. Once it is on the surface, a field is projected to 2D
# like any other. The fields come from three sources:
#
# | Source | Fields |
# |---|---|
# | the shape of the surface itself | curvature |
# | the image | the intensity of each channel, in a thin layer next to the surface |
# | the point files | the density of the points; and any value the points carry, e.g. thickness |
#
# At the end, the surface is saved with all its fields, for the projection.
#
# **Timing:** about 5 s.
#
# Equivalent one-liner: `spatchcock measure`.

# %%
import numpy as np
from vedo import Mesh, settings

from spatchcocking import api, io

from _paths import EXAMPLE_INPUTS, RESULTS

settings.default_backend = "vtk"

# %% [markdown]
# ## 1. The inputs
#
# The mesh made in notebook 02, and the same image and point files as in
# notebook 01.

# %%
MESH_PATH = RESULTS / "reference_surface.ply"  # made by notebook 02

IMAGE_PATH = EXAMPLE_INPUTS / "2025-10-22-12-30-image.tif"
IMAGE_VOXEL_SIZE = (4.5, 22.14, 22.14)  # z, y, x in µm
CHANNELS = ["pHH3", "DAPI"]

POINT_FILES = {
    "pHH3": EXAMPLE_INPUTS / "2025-10-22-12-30-points_phh3.csv",
    "thickness": EXAMPLE_INPUTS / "2025-10-22-12-30-points_thickness.csv",
}

FIELDS_PATH = RESULTS / "reference_surface_fields.vtk"  # where the result is saved

# %%
mesh = Mesh(str(MESH_PATH))
print(f"reference surface: {mesh.npoints} points")

# %% [markdown]
# ## 2. The settings
#
# The values the paper used. If you change them, report the values you used.
#
# - **Intensity** is the average of the image in a thin layer next to the
#   surface. `INTENSITY_DIRECTION` says which side of the surface the layer is on:
#
#   | `INTENSITY_DIRECTION` | Samples |
#   |---|---|
#   | `"both"` (default) | both sides of the surface, averaged together |
#   | `"in"` | only inside the surface, into the tissue |
#   | `"out"` | only outside it |
#
#   `"both"` doesn't depend on exactly which side of the tissue boundary the
#   segmentation drew the surface. `INTENSITY_DEPTH_UM` is how far the layer
#   reaches on each side, and `INTENSITY_SAMPLES` how many depths are averaged on
#   each side. Inside, the layer should stay within the tissue wall.
# - **Density** is the local density of the points near each place on the
#   surface: the points within `DENSITY_RADIUS_UM` are counted on a regular grid,
#   the grid is sampled at the surface, and the result is lightly smoothed, so the
#   values are not whole numbers. The radius should span the whole wall, so that a
#   cell anywhere through the wall counts for the surface above it.
# - **Values** carried by points need no setting: each place on the surface takes
#   its value from its 5 nearest points, as in the published analysis.

# %%
INTENSITY_DIRECTION = "both"  # "both", "in" or "out"
INTENSITY_DEPTH_UM = 15.0
INTENSITY_SAMPLES = 4
DENSITY_RADIUS_UM = 100.0

# %% [markdown]
# ## 3. From the shape: curvature
#
# Four fields, from how the surface bends at each point:
#
# | Field | What it is | Units |
# |---|---|---|
# | `K1`, `K2` | the most and the least the surface bends, in two perpendicular directions | 1/µm |
# | `Mean_Curvature` | their average, (K1 + K2) / 2. Positive where the surface bulges out, negative where it folds in | 1/µm |
# | `Gauss_Curvature` | their product, K1 × K2. Positive on a dome, negative on a saddle | 1/µm² |

# %%
mesh = api.measure(mesh, "curvature")

# %% [markdown]
# ## 4. From the image: intensity of each channel

# %%
for c, channel in enumerate(CHANNELS):
    volume, _ = io.load_stack(IMAGE_PATH, IMAGE_VOXEL_SIZE, channel=c)
    mesh = api.measure(mesh, "intensity", volume=volume, spacing=IMAGE_VOXEL_SIZE,
                       depth_um=INTENSITY_DEPTH_UM, n_samples=INTENSITY_SAMPLES,
                       direction=INTENSITY_DIRECTION, name=f"{channel}_intensity")

# %% [markdown]
# ## 5. From the point files
#
# A file with **positions only** gives the density of its points. A file whose
# points **carry values** gives one field per value column, named after the
# column.

# %%
for name, path in POINT_FILES.items():
    points, values = io.load_point_table(path)
    if not values:
        field = name.replace(" ", "_") + "_density"
        mesh = api.measure(mesh, "density", points=points, radius=DENSITY_RADIUS_UM)
        mesh.pointdata[field] = mesh.pointdata["phh3"]  # the published function's own name
        mesh.pointdata.remove("phh3")
    for column, v in values.items():
        mesh = api.measure(mesh, "values", points=points, values=v, name=column)

# %% [markdown]
# ## 6. What is on the surface now

# %%
print(f"{'field':<22}{'min':>11}{'max':>11}{'mean':>11}")
for field in mesh.pointdata.keys():
    v = np.asarray(mesh.pointdata[field])
    if v.ndim == 1:
        print(f"{field:<22}{v.min():>11.4g}{v.max():>11.4g}{v.mean():>11.4g}")

# %% [markdown]
# ## 7. Look at each field
#
# One picture per field, on the surface, seen from the side. Each color scale
# runs from the 1st to the 99th percentile of its field, so a few extreme points
# don't wash out the rest; the table above gives the true range.
#
# A window opens that you can rotate (all the views turn together); close it to
# carry on. To save a picture instead, set `SCREENSHOT` to a file, e.g.
# `RESULTS / "03_fields.png"`.

# %%
SCREENSHOT = None  # RESULTS / "03_fields.png"

from vedo import Plotter  # noqa: E402

names = [f for f in mesh.pointdata.keys() if np.asarray(mesh.pointdata[f]).ndim == 1]
columns = 3
rows = -(-len(names) // columns)
plotter = Plotter(shape=(rows, columns), sharecam=True, size=(1500, 450 * rows),
                  offscreen=bool(SCREENSHOT), title="fields on the reference surface")
for i, field in enumerate(names):
    low, high = np.percentile(np.asarray(mesh.pointdata[field]), [1, 99])
    shown = mesh.clone().cmap("viridis", field, vmin=low, vmax=high).add_scalarbar()
    plotter.at(i).show(shown, field, azimuth=-90 if i == 0 else 0)
plotter.interactive() if not SCREENSHOT else plotter.screenshot(str(SCREENSHOT))
plotter.close()

# %% [markdown]
# ## 8. Save

# %%
mesh.write(str(FIELDS_PATH))
print(f"saved {mesh.npoints} points with {len(names)} fields to {FIELDS_PATH}")

# %% [markdown]
# ## What you should see
#
# A table with the range and mean of every field, in a few seconds: the four
# curvature fields, one intensity field per channel, one density field per file
# of positions, and one field per value column.
#
# In the pictures: the shape of the tissue stands out in curvature; the
# intensity and density fields follow the signal and the cells in the image.
#
# Next: **04 — straighten and unwrap into 2D**.
