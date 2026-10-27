"""Protocol step: show the map, one panel per field.

Run from the working directory: python protocol/show_map.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatchcocking.colors import field_grid
from spatchcocking.plotting import paper_map
from sample import *

FIELD = "all"  # "all" for every field, or one column of spatchcocked.csv

COORDINATES = ["radius", "norm_height", "angle_degrees"]

flat = pd.read_csv(f"{RESULTS}/spatchcocked.csv")
fields = [c for c in flat.columns if c not in COORDINATES] if FIELD == "all" else [FIELD]

grid = field_grid(fields)  # curvature first, then intensity, then the point fields
rows, columns = len(grid), len(grid[0])
fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 4.2 * rows), squeeze=False)
for ax, field in zip(axes.flat, sum(grid, [])):
    if field is None:
        ax.set_visible(False)
    else:
        fig.colorbar(paper_map(flat, field, ax), ax=ax, shrink=0.75)
for ax in axes[:, 0]:
    ax.set_ylabel("rostro-caudal axis")
fig.tight_layout()
plt.show()
