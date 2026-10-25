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
# # 01 — Are the inputs right?
#
# A wrong input does not crash the pipeline. It gives a map that looks fine and
# is wrong. This notebook loads the inputs, checks them, and shows them together
# in napari, so that everything is known to line up before any work starts.
#
# It uses the example dataset in `data/example_inputs/`: one HH20 chick cranial
# neural tube.
#
# **Timing:** about 20 s, plus the viewer.

# %%
from spatchcocking import checks, io

from _paths import EXAMPLE_INPUTS, RESULTS

report = checks.Report()

# %% [markdown]
# ## 1. The inputs
#
# | Input | What it is |
# |---|---|
# | image | the raw image: 2 channels, pHH3 and DAPI |
# | reference mask | the outer surface of the neural tube, 0 and 1. It becomes the mesh, and is projected to 2D |
# | point files | CSV, one row per point. Columns `z, y, x` in µm; any other column is a value carried by the point |
#
# **Everything is in the image's frame of reference:** z, y, x order, in µm,
# starting at the image's first voxel. Points from other tools (Imaris, Fiji,
# CellProfiler) must be converted to that frame first.
#
# **Voxel size is typed here, never read from the file.** This microscope writes
# z = 3.0 µm; the true step is 4.5 µm after the 1.5× correction for DBE
# clearing. The image and the mask have different voxel sizes, because the
# image was shrunk 4× in x and y to fit in the repository.

# %%
IMAGE_PATH = EXAMPLE_INPUTS / "2025-10-22-12-30-image.tif"
IMAGE_VOXEL_SIZE = (4.5, 22.14, 22.14)  # z, y, x in µm
CHANNELS = ["pHH3", "DAPI"]  # one name per channel, in the file's order

REFERENCE_MASK_PATH = EXAMPLE_INPUTS / "2025-10-22-12-30-outer_mask.tif"
MASK_VOXEL_SIZE = (4.5, 5.535, 5.535)  # z, y, x in µm

POINT_FILES = {
    "pHH3": EXAMPLE_INPUTS / "2025-10-22-12-30-points_phh3.csv",
    "thickness": EXAMPLE_INPUTS / "2025-10-22-12-30-points_thickness.csv",
}

# %% [markdown]
# ## Your own data: what else is possible
#
# The example is one arrangement of inputs. With your own data, change the cell
# above to match; these are the variations it allows.
#
# ### Point files
#
# Any number of point files. Each is one kind of object: one cell type, or one
# set of measurements. Add a line to `POINT_FILES` for each, with a name of your
# choice. Every file is a CSV:
#
# | Columns | Meaning |
# |---|---|
# | `z`, `y`, `x` (required) | the position, in µm, in the image's frame |
# | anything else (optional) | a value carried by each point, named with its units: `area_um2`, `perimeter_um`, `volume_um3`, `mean_intensity`, `thickness_um` |
#
# What each gives on the 2D map:
#
# | A point file with | What goes on the map |
# |---|---|
# | positions only (like `pHH3`) | **how many points there are** near each place on the surface: their density |
# | values (like `thickness`) | **the value itself.** Each place on the surface takes its value from the nearest points (the 5 closest), then it is mapped like curvature |
#
# The protocol doesn't know what a value *means*. A column called `area` could be
# a 3D surface area or a 2D cross-section. Say in the column name what it is,
# and its units.
#
# ### Voxel sizes
#
# Give the image and the mask their own voxel sizes, as above. If both were
# saved at the same resolution, give the same numbers twice. Always type them:
# never trust the file's metadata.
#
# ### Channels
#
# Any number of channels. Name them in `CHANNELS`, in the order they are stored in
# the file. Any channel can be measured as intensity later.
#
# ### What is *not* an input: your own mesh
#
# The protocol always makes the mesh from the reference mask. A mesh from
# elsewhere carries no measurements, so only its curvature could be mapped, and
# it may be in a different coordinate system from the image.

# %% [markdown]
# ## 2. The image

# %%
image, _ = io.load_stack(IMAGE_PATH, IMAGE_VOXEL_SIZE)
image_um = checks.extent_um(image.shape, IMAGE_VOXEL_SIZE)
report.check(len(CHANNELS) == len(image), "one name per channel",
             f"{len(image)} channels: {CHANNELS}",
             problem=f"CHANNELS has {len(CHANNELS)} names")
channels = list(CHANNELS) + [f"channel {i}" for i in range(len(CHANNELS), len(image))]

print(f"{image.shape} {image.dtype}")
print(f"size (z, y, x): {image_um.round().tolist()} µm")

# %% [markdown]
# ## 3. The reference mask
#
# It must be one object. Tiny specks are fine, because meshing removes anything
# under 100 voxels. A second real piece is not: the surface would come out in
# several parts.

# %%
reference = checks.load_mask(REFERENCE_MASK_PATH)
n_pieces, n_specks = checks.pieces(reference)

report.check(n_pieces == 1, "reference mask is one object",
             f"{n_pieces} piece, plus {n_specks} specks meshing will remove")
report.check(not checks.touches_edge(reference), "reference mask is clear of the stack edge",
             problem="it touches the edge, so the surface will be open there",
             warn_only=True)

# %% [markdown]
# ## 4. Do the image and the mask cover the same space?
#
# Compare their size in µm. If this fails, a voxel size is wrong. This is the
# check that catches the uncorrected z = 3.0.

# %%
mask_um = checks.extent_um(reference.shape, MASK_VOXEL_SIZE)
report.check(checks.same_space(image.shape, IMAGE_VOXEL_SIZE, reference.shape, MASK_VOXEL_SIZE),
             "image and mask cover the same space",
             f"image {image_um.round().tolist()} µm, mask {mask_um.round().tolist()} µm",
             problem="a voxel size is wrong")

# %% [markdown]
# ## 5. The point files
#
# Each file is read as positions plus values. The points must land on the
# tissue, inside the reference mask. If they don't, they are not in the image's
# frame, and the check says which of the usual causes fits: axis order, an
# origin offset (e.g. stage coordinates), or units that are not µm.

# %%
points, values = {}, {}
for name, path in POINT_FILES.items():
    points[name], values[name] = io.load_point_table(path)
    carried = ", ".join(values[name]) or "none: positions only"
    print(f"{name}: {len(points[name]):,} points; values: {carried}")
    for column, v in values[name].items():
        print(f"    {column}: {v.min():.2f} to {v.max():.2f}, mean {v.mean():.2f}")
    on, why = checks.diagnose_points(points[name], reference, MASK_VOXEL_SIZE, image_um)
    report.check(on >= 0.95, f"{name}: points land on the tissue", f"{on:.2%}", problem=why)

# %% [markdown]
# ## 6. Look
#
# The checks say the inputs line up. Only looking shows that the mask
# **follows the tissue**. Each layer gets its own voxel size, and nothing else:
# no shift, no flip.
#
# `SCREENSHOT` saves one picture and closes. Set it to `None` to get a window
# you can rotate.

# %%
SCREENSHOT = RESULTS / "01_inputs.png"  # or None

import napari  # noqa: E402

viewer = napari.Viewer(ndisplay=3, title="inputs")
for vol, name, color in zip(image, channels, ["magenta", "cyan"]):
    viewer.add_image(vol, name=name, scale=IMAGE_VOXEL_SIZE, units="um", colormap=color,
                     blending="additive", rendering="attenuated_mip", attenuation=0.25)
viewer.add_labels(reference.astype("uint8"), name="reference mask", scale=MASK_VOXEL_SIZE,
                  units="um", opacity=0.35)
for (name, pts), color in zip(points.items(), ["white", "yellow"]):
    viewer.add_points(pts, name=name, size=12, face_color=color, units="um", visible=False)
camera = getattr(viewer, "scene", viewer).camera  # napari 0.9 moved it to viewer.scene
camera.angles = (0, 0, 90)
camera.zoom = 0.13
viewer.canvas.overlays.scale_bar.visible = True

if SCREENSHOT:
    RESULTS.mkdir(exist_ok=True)
    viewer.screenshot(str(SCREENSHOT), canvas_only=True)
    viewer.close()
else:
    napari.run()

# %% [markdown]
# ## 7. Verdict

# %%
report.verdict()

# %% [markdown]
# ## What you should see
#
# - the image, with one name per channel
# - reference mask: one piece, plus perhaps a few specks that meshing removes
# - image and mask cover the same space
# - each point file: its points land on the tissue. Points that sit *on* the
#   outer surface land a little less often than cells inside the wall, because
#   the smoothed surface runs slightly outside the mask in places.
# - every check ok and `inputs ready`, then the napari picture: the mask on the
#   tube, inside the image signal. The point sets are hidden; tick them in the
#   layer list.
#
# ## If a check fails
#
# | The message says | What is wrong | What to do |
# |---|---|---|
# | image and mask do not cover the same space | a voxel size is wrong, often the uncorrected z | fix it in section 1 |
# | reference mask is not one object | the mask has several real pieces | keep only the tube in the mask |
# | a point file "has no column(s) ['z', ...]" | the columns are named differently | name them `z`, `y`, `x` |
# | points: "probably (x, y, z)" | columns in the wrong order | reorder them to z, y, x |
# | points: "probably an origin offset" | coordinates start at the stage position | subtract the image's origin |
# | points: "probably not in µm" | wrong units | convert to µm |
#
# Next: **02 — mask to surface**.
