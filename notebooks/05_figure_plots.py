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
# # 05 — Every map
#
# Notebook 04 drew one field. This draws all of them, from the map notebook 04
# saved, so each can be checked against the anatomy: one panel per field, in the
# same layout as the published figures.
#
# Equivalent one-liner, one field at a time: `spatchcock map results/spatchcocked.csv --field <name>`.
#
# **Timing:** a few seconds.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatchcocking.plotting import paper_map

from _paths import RESULTS

# %% [markdown]
# ## 1. The map

# %%
MAP_PATH = RESULTS / "spatchcocked.csv"  # made by notebook 04

flat = pd.read_csv(MAP_PATH)
COORDINATES = ["radius", "norm_height", "angle_degrees"]
fields = [c for c in flat.columns if c not in COORDINATES]
print(f"{len(flat)} points; fields: {fields}")

# %% [markdown]
# ## 2. One panel per field
#
# Across: around the tube, **V**entral, **D**orsal, **V**entral. Up: along it,
# **C** (the tail) to **R** (the head). Each field has its own color scale,
# from median − 3σ to median + 3σ; fields that change sign (the curvatures) are
# centered on zero.
#
# `SCREENSHOT` saves the figure as well as showing it.

# %%
SCREENSHOT = RESULTS / "05_maps.png"  # or None

columns = 4
rows = int(np.ceil(len(fields) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 4.2 * rows))
for ax in axes.flat[len(fields):]:
    ax.set_visible(False)
for ax, field in zip(axes.flat, fields):
    fig.colorbar(paper_map(flat, field, ax), ax=ax, shrink=0.75)
for ax in axes[:, 0]:
    ax.set_ylabel("rostro-caudal axis")
fig.tight_layout()
if SCREENSHOT:
    fig.savefig(SCREENSHOT, dpi=200)
plt.show()

# %% [markdown]
# ## 3. Is dorsal where it should be?
#
# In the example, the thinnest tissue (the roof plate) runs along the dorsal
# midline. Averaged along the tube, thickness should therefore dip at 0°. If the dip sits
# elsewhere, dorsal was marked off the midline: go back to the picker in
# notebook 04.

# %%
if "thickness_um" in flat.columns:
    edges = np.arange(-180, 190, 10)
    profile = flat.groupby(pd.cut(flat["angle_degrees"], edges), observed=False)["thickness_um"]
    mean, sd = profile.mean().to_numpy(), profile.std().to_numpy()
    centers = edges[:-1] + 5
    print(f"thinnest at {centers[np.nanargmin(mean)]:+d}°: {np.nanmin(mean):.1f} µm")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(centers, mean)
    ax.fill_between(centers, mean - sd, mean + sd, alpha=0.2)
    ax.set_xlabel("around the tube (degrees)")
    ax.set_ylabel("thickness (µm)")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["V", "", "D", "", "V"])
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Optional: pooling many embryos
#
# Once several embryos have been spatchcocked, their maps can be averaged.
# Every map is on the same coordinates: position along the tube from 0 at the
# tail to 1 at the head, and angle around it from the dorsal midline. The points
# themselves differ from embryo to embryo, so maps are pooled **by bin, not by
# point**: each map is cut into the same grid, 50 bins along the tube by 10°
# around it.
#
# Each embryo is averaged within each bin first, then the embryos are averaged,
# so every embryo counts once.
#
# The example: your map from notebook 04 stands in for its embryo
# (2025-10-22-12-30), pooled with the maps of the other five in
# `data/example_maps/`, giving two groups of three. Testing whether the groups
# differ is analysis, and outside this protocol.
#
# Equivalent one-liner: `spatchcock compare <map.csv> ... --labels ... --field ...`.

# %%
from spatchcocking import api  # noqa: E402
from spatchcocking.colors import field_style  # noqa: E402

from _paths import EXAMPLE_MAPS  # noqa: E402

FIELD = "Gauss_Curvature"  # also "Mean_Curvature", "thickness_um", "pHH3_density"
STAGES = {  # each example map, and its stage
    "2025-09-18-13-02": "HH17",
    "2025-09-18-14-26": "HH17",
    "2025-09-18-16-46": "HH17",
    "2025-09-23-15-48": "HH20",
    "2025-10-22-12-30": "HH20",
    "2025-10-23-13-06": "HH20",
}
EXAMPLE_EMBRYO = "2025-10-22-12-30"

maps, labels = [], []
for embryo, stage in STAGES.items():
    mine = embryo == EXAMPLE_EMBRYO
    maps.append(flat if mine else pd.read_csv(EXAMPLE_MAPS / embryo / "spatchcocked.csv"))
    labels.append(stage)
    print(f"{stage}  {embryo}  {'your map' if mine else 'example map'}")

_, grids = api.compare(maps, FIELD, labels=labels)

# %% [markdown]
# One panel per group on a shared color scale, and their difference, with the
# tail at the bottom, as in the maps above.

# %%
SCREENSHOT_POOLED = RESULTS / "05_pooled.png"  # or None

label, cmap_name, symmetric = field_style(FIELD)
groups = {g: grids[g].to_numpy() for g in sorted(grids)}
values = np.concatenate([g[np.isfinite(g)] for g in groups.values()])
limit = np.nanpercentile(np.abs(values), 99)
vmin, vmax = (-limit, limit) if symmetric else (0.0, np.nanpercentile(values, 99))
first, second = list(groups)
difference = groups[second] - groups[first]
dlim = np.nanpercentile(np.abs(difference), 99)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
panels = [(groups[first], first, cmap_name, vmin, vmax),
          (groups[second], second, cmap_name, vmin, vmax),
          (difference, f"{second} − {first}", "PuOr_r", -dlim, dlim)]
for ax, (image, title, cmap, lo, hi) in zip(axes, panels):
    im = ax.imshow(image, origin="lower", aspect="auto", extent=(-180, 180, 0, 1),
                   cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dorso-ventral axis")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["V", "", "D", "", "V"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["C", "R"])
    fig.colorbar(im, ax=ax, shrink=0.75)
axes[0].set_ylabel("rostro-caudal axis")
fig.suptitle(f"{label}, mean of 3 embryos per group", fontsize=10)
fig.tight_layout()
if SCREENSHOT_POOLED:
    fig.savefig(SCREENSHOT_POOLED, dpi=200)
plt.show()

# %% [markdown]
# ## What you should see
#
# - one map per field, each on its own color scale
# - patterns that follow the anatomy of the tissue, and that are roughly
#   symmetric about the dorsal midline where the tissue is
# - in the dorsal check, the thinnest tissue close to 0°, if the tissue is
#   thinnest dorsally
# - pooled: the mean map of each group, and their difference
