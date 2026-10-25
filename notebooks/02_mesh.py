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
# # 02 — From the mask to a surface mesh
#
# The reference mask becomes the **reference surface**: a mesh of the outer
# surface of the neural tube. Every measurement is put on this mesh, and it is
# the mesh that is projected to 2D.
#
# Then the mesh is drawn inside the image, to check that it sits on the tissue.
#
# **Timing:** about 1 minute.
#
# Equivalent one-liner: `spatchcock mesh`.

# %%
from vedo import settings

from spatchcocking import api, io, viewer

from _paths import EXAMPLE_INPUTS, RESULTS

settings.default_backend = "vtk"
RESULTS.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. The inputs
#
# The same files and voxel sizes as in notebook 01, which checked them.

# %%
REFERENCE_MASK_PATH = EXAMPLE_INPUTS / "2025-10-22-12-30-outer_mask.tif"
MASK_VOXEL_SIZE = (4.5, 5.535, 5.535)  # z, y, x in µm

IMAGE_PATH = EXAMPLE_INPUTS / "2025-10-22-12-30-image.tif"
IMAGE_VOXEL_SIZE = (4.5, 22.14, 22.14)  # z, y, x in µm
CHANNELS = ["pHH3", "DAPI"]

MESH_PATH = RESULTS / "reference_surface.ply"  # where the mesh is saved, for the next notebooks

# %% [markdown]
# ## 2. Make the mesh
#
# Four stages, all inside `api.condition`:
#
# 1. **Clean the mask.** Remove specks under 100 voxels, fill holes, smooth
#    lightly.
# 2. **Marching cubes.** Put a surface on the boundary between 0 and 1. The voxel
#    size is passed in, so the surface comes out in µm, with far more points
#    than needed.
# 3. **Smooth, then simplify.** Smooth 100 passes, keep 5% of the triangles,
#    then 10% of those, then smooth 50 more passes.
# 4. **Stop loudly if anything comes out empty**, naming the stage. The
#    published code could save an empty file and report success.
#
# ### The settings for stage 3
#
# Stage 3 decides how smooth and how fine the mesh is. The values below are the
# ones the paper used; **keep them to reproduce the published mesh exactly.**
#
# For your own data you may need others. The fractions are relative, so a mask
# at a different resolution starts with a different number of points and ends
# with a different number. Aim for a mesh fine enough to follow the tissue, not
# much finer.
#
# If the z step is much coarser than the pixel size, the surface can come out
# terraced, and more smoothing does not help: set `blur_um` (a few µm, about the
# pixel size) to blur the mask before the surface is placed. Off (`None`) gives
# the published mesh exactly.
#
# If you change them, **report the values you used**: every measurement
# afterwards depends on them.

# %%
MESH_SETTINGS = dict(
    smooth_iter=100,         # smoothing passes, before simplifying
    decimate_fraction=0.05,  # first simplification: keep this fraction of the triangles
    final_fraction=0.1,      # second simplification: keep this fraction of what is left
    final_smooth_iter=50,    # smoothing passes at the end (the open edge is held still)
    blur_um=None,            # blur the mask by this many µm first; None = published route
)

mesh = api.condition(REFERENCE_MASK_PATH, MASK_VOXEL_SIZE, out_path=str(MESH_PATH),
                     **MESH_SETTINGS)
print(f"\nreference surface: {mesh.npoints} points, {mesh.ncells} triangles")
print(f"area {mesh.area() / 1e6:.2f} mm², saved to {MESH_PATH}")

# A medial axis saved for an earlier surface no longer fits this one: delete it,
# so that notebook 04 computes a new one.
OLD_AXIS = RESULTS / "reference_surface-axis.npy"
if OLD_AXIS.exists():
    OLD_AXIS.unlink()
    print(f"deleted the medial axis of the previous surface: {OLD_AXIS}")

# %% [markdown]
# ## 3. Look at the mesh
#
# The mesh on its own, with its triangles drawn, so you can judge whether it is
# smooth enough and fine enough to follow the shape. If it looks blocky, keep
# more triangles; if it looks lumpy, smooth more (the settings above).
#
# The camera looks from the side, as in napari below, with the long axis of the
# tube across the picture. The caption gives the mesh's size in each direction.
#
# A window opens that you can rotate; close it to carry on. To save a picture
# instead, set `VEDO_SCREENSHOT` to a file, e.g. `RESULTS / "02_mesh_vedo.png"`.

# %%
VEDO_SCREENSHOT = None # RESULTS / "02_mesh_vedo.png"  # or None

from vedo import Plotter  # noqa: E402

shown = mesh.clone().c("lightblue").lw(0.5).lc("steelblue")
b = mesh.bounds()  # z0, z1, y0, y1, x0, x1; the mesh is (z, y, x)
size = f"{b[1] - b[0]:.0f} x {b[3] - b[2]:.0f} x {b[5] - b[4]:.0f} um (z, y, x)"  # vedo's font has no µ
caption = f"reference surface: {mesh.npoints} points, {mesh.ncells} triangles\n{size}"
plotter = Plotter(offscreen=bool(VEDO_SCREENSHOT), title="reference surface")
plotter.show(shown, caption, azimuth=-90, interactive=not VEDO_SCREENSHOT)
if VEDO_SCREENSHOT:
    plotter.screenshot(str(VEDO_SCREENSHOT))
plotter.close()

# %% [markdown]
# ## 4. Look: the mesh inside the image
#
# This answers what no number can: does the surface lie where the tissue is? The
# mesh is in µm, in (z, y, x) order, which is napari's order too, so it goes in
# as it is. The image gets its own voxel size. No shift, no flip.
#
# A napari window opens that you can rotate; close it to carry on. To save a
# picture instead, set `SCREENSHOT` to a file, e.g. `RESULTS / "02_mesh.png"`.

# %%
SCREENSHOT = None # RESULTS / "02_mesh.png"  # or None

image, _ = io.load_stack(IMAGE_PATH, IMAGE_VOXEL_SIZE)
viewer.show_in_context(
    image,
    IMAGE_VOXEL_SIZE,
    mesh,
    channel_names=CHANNELS,
    screenshot=str(SCREENSHOT) if SCREENSHOT else None,
)

# %% [markdown]
# ## What you should see
#
# - the number of points and triangles, and the area, in about a minute
# - the vedo picture: the whole tube from the side, the triangles evenly sized,
#   no jagged steps from the voxels
# - in napari: the gray surface sitting in the image signal and following the
#   tube wall
#
# If the surface floats outside the tissue, or is the wrong size by a constant
# factor, a voxel size is wrong. Notebook 01 should have caught that already.
#
# A small hole where the mask was thin is harmless: the later steps cope with it.
#
# Next: **03 — put the measurements on the surface**.
