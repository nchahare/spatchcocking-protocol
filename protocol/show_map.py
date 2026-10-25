"""Protocol step: show one field of the map.

Run from the working directory: python protocol/show_map.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from spatchcocking.plotting import paper_map
from sample import *

FIELD = "Gauss_Curvature"  # the field to show: any column of spatchcocked.csv

flat = pd.read_csv(f"{RESULTS}/spatchcocked.csv")
fig, ax = plt.subplots(figsize=(4, 5))
fig.colorbar(paper_map(flat, FIELD, ax), ax=ax)
ax.set_title(FIELD)
plt.show()
