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
# # 04 — Spatchcocking: from the 3D surface to a 2D map
#
# The curved tube, with all its fields, becomes a flat map: position around the
# tube across, with **dorsal at the center**, and position along the tube up, from
# the tail to the head. The same place on the map is the same place in every
# embryo, so maps can be compared and averaged.
#
# **Timing:** about 15 s, plus picking the dorsal anchors once per embryo.

# %%
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
from vedo import Mesh

from spatchcocking import api, io
from spatchcocking.plotting import paper_map

from _paths import EXAMPLE_OUTPUTS, RESULTS, ROOT

# %% [markdown]
# ## 1. The inputs

# %%
SURFACE_PATH = RESULTS / "reference_surface.ply"  # made by notebook 02
FIELDS_PATH = RESULTS / "reference_surface_fields.vtk"  # made by notebook 03
ANCHORS_PATH = RESULTS / "reference_surface-anchors.json"  # made by the picker, below
MAP_PATH = RESULTS / "spatchcocked.csv"  # where the 2D map is saved

# %% [markdown]
# ## 2. Where is dorsal?
#
# Dorsal has to be shown by a person, once per embryo. Set `PICK_ANCHORS = True`
# to open the picker:
#
# 1. click the **tail** end of the tube;
# 2. click along the **dorsal midline** until the red line stops moving;
# 3. press **q**.
#
# It saves the anchors to `ANCHORS_PATH`. With `PICK_ANCHORS = False`, the
# anchors already saved there are used.
#
# For the example, `USE_EXAMPLE_ANCHORS = True` uses the dorsal points made for
# it, from `data/example_outputs/`, instead of clicking. They fit only the
# surface made with the default settings of notebook 02.
#
# The picker first draws the medial axis. Check it runs once through the middle
# of the tube, from end to end. If it loops at an end (most often the anterior
# one) or wanders off-center, raise `AXIS_ITERATIONS` and set
# `RECOMPUTE_AXIS = True`: without that, the axis saved last time is reused.
# A new axis needs new anchors.
#
# Between 10 and 20 is safe. With too few iterations the axis loops at an end;
# with too many it shortens, and stations fall outside the tube.

# %%
PICK_ANCHORS = False
USE_EXAMPLE_ANCHORS = True  # the example only: use its saved dorsal points instead of clicking
AXIS_ITERATIONS = 10    # smoothing rounds for the medial axis; more remove a loop at an end
RECOMPUTE_AXIS = False  # True: compute the axis afresh instead of reusing the saved one

if PICK_ANCHORS:
    command = [sys.executable, str(ROOT / "scripts" / "pick_anchors.py"),
               str(SURFACE_PATH), "--out", str(ANCHORS_PATH),
               "--iterations", str(AXIS_ITERATIONS)]
    if RECOMPUTE_AXIS:
        command.append("--recompute")
    subprocess.run(command, check=True)
elif USE_EXAMPLE_ANCHORS:
    for name in ("reference_surface-anchors.json", "reference_surface-axis.npy"):
        shutil.copy(EXAMPLE_OUTPUTS / name, RESULTS / name)
    print("using the example's dorsal points, from", EXAMPLE_OUTPUTS)

mesh = Mesh(str(FIELDS_PATH))
anchors = io.load_anchors(ANCHORS_PATH, mesh)  # stops if they were picked on another surface
print(f"{len(anchors['stations'])} anchors")

# %% [markdown]
# ## 3. Straighten, then unwrap
#
# At each station the tube is cut by a plane, and the outline of the tube there
# is moved onto a straight axis. Where the tube bends back, the same plane also
# cuts the other arm of the bend; `DISTS_THRESHOLD` (µm) decides which cut points
# belong to this station: closer ones are kept, the rest discarded. `None`
# derives it from the tube's own width (1.1 × the 95th percentile of its radius).
#
# - a region of the map stretched or smeared: part of the outline was discarded,
#   so raise it to just above the tube's widest radius;
# - tissue from elsewhere in a region: the other arm of a bend was kept, so lower
#   it below the distance to that arm.

# %%
DISTS_THRESHOLD = None  # µm; None = 1.1 × the 95th percentile of the tube's radius

straight, info = api.project(mesh, anchors["axis"], anchors["stations"], anchors["directions"],
                             dists_threshold=DISTS_THRESHOLD)
print(f"distance threshold: {info['dists_threshold']:.0f} µm")
flat = api.unwrap(straight)
print(f"2D map: {len(flat)} points")

# %% [markdown]
# ## 4. The 2D map: Gaussian curvature
#
# Around the tube across (**V**entral, **D**orsal, **V**entral), along it up
# (**C**, the tail, at the bottom; **R**, the head, at the top). Green is
# dome-like (positive), pink saddle-like (negative), white flat in one direction.

# %%
FIELD = "Gauss_Curvature"

fig, ax = plt.subplots(figsize=(4, 5))
fig.colorbar(paper_map(flat, FIELD, ax), ax=ax, shrink=0.75)
ax.set_ylabel("rostro-caudal axis")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Save

# %%
flat.to_csv(MAP_PATH, index=False)
print(f"saved to {MAP_PATH}")

# %% [markdown]
# ## What you should see
#
# The number of anchors, the number of points in the map, and a Gaussian
# curvature map.
#
# Next: **05 — every map**.
