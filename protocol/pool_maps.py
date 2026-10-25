"""Protocol step: pool the maps of several samples on a common grid.

Run from the working directory: python protocol/pool_maps.py
Each sample is averaged within each bin first, and the samples are then
averaged, so that every sample counts once in every bin it reaches.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from spatchcocking import api
from spatchcocking.colors import field_style
from spatchcocking.pipeline import getTightercmap

# Pooling settings.
MAPS = {  # each sample's map, and the group it belongs to, e.g. a stage
    "data/example_maps/2025-09-23-15-48/spatchcocked.csv": "HH20",
    "data/example_maps/2025-10-22-12-30/spatchcocked.csv": "HH20",
    "data/example_maps/2025-10-23-13-06/spatchcocked.csv": "HH20",
}
FIELD = "Gauss_Curvature"  # the field to pool: a column that every map has
RC_BINS = 50               # number of bins along the tube
DV_BIN_DEG = 10            # width of the bins around the tube, in degrees

pooled, grids = api.compare(list(MAPS), FIELD, rc_bins=RC_BINS, dv_bin_width=DV_BIN_DEG,
                            labels=list(MAPS.values()))
# Colors: the field's palette, median +/- 3 standard deviations, symmetric about
# zero for fields that change sign.
_, cmap, symmetric = field_style(FIELD)
values = np.concatenate([g.to_numpy().ravel() for g in grids.values()])
vmin, vmax = getTightercmap(values[np.isfinite(values)], sigma=3)
if symmetric:
    vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))

os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, len(grids), figsize=(4 * len(grids), 5.5), squeeze=False)
for ax, (group, grid) in zip(axes[0], grids.items()):
    grid.to_csv(f"results/pooled_{FIELD}_{group}.csv")
    print(group, "mean map saved to", f"results/pooled_{FIELD}_{group}.csv")
    n = sum(1 for g in MAPS.values() if g == group)
    image = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto",
                      extent=(-180, 180, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{group}: {FIELD}\nmean of {n} samples", pad=14)
    ax.set_xlabel("dorso-ventral axis")
    ax.set_xticks([-180, -90, 0, 90, 180], ["V", "", "D", "", "V"])
    ax.set_yticks([0, 1], ["C", "R"])
    fig.colorbar(image, ax=ax)
plt.show()
