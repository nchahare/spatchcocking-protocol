"""Protocol step: show the map, one panel per field.

Run from the working directory: python protocol/show_map.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatchcocking.plotting import paper_map
from sample import *

FIELD = "all"  # "all" for every field, or one column of spatchcocked.csv

COORDINATES = ["radius", "norm_height", "angle_degrees"]

flat = pd.read_csv(f"{RESULTS}/spatchcocked.csv")
fields = [c for c in flat.columns if c not in COORDINATES] if FIELD == "all" else [FIELD]

columns = min(4, len(fields))
rows = int(np.ceil(len(fields) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 4.2 * rows), squeeze=False)
for ax in axes.flat[len(fields):]:
    ax.set_visible(False)
for ax, field in zip(axes.flat, fields):
    fig.colorbar(paper_map(flat, field, ax), ax=ax, shrink=0.75)
for ax in axes[:, 0]:
    ax.set_ylabel("rostro-caudal axis")
fig.tight_layout()
plt.show()
