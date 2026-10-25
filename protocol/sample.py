"""The sample to analyze. Every protocol script reads this file.

Paths are relative to the working directory, the unpacked protocol folder,
from which the scripts are run, e.g. `python scripts/check_inputs.py`.
Edit this file for each new sample.
"""
IMAGE = "data/example_inputs/2025-10-22-12-30-image.tif"  # None if there is no image
IMAGE_VOXEL = (4.5, 22.14, 22.14)  # z, y, x, in micrometers
CHANNELS = ["pHH3", "DAPI"]  # one name per channel, in the file's order

MASK = "data/example_inputs/2025-10-22-12-30-outer_mask.tif"
MASK_VOXEL = (4.5, 5.535, 5.535)  # z, y, x, in micrometers

POINTS = {  # a name for each point file, and its path; {} if there are none
    "pHH3": "data/example_inputs/2025-10-22-12-30-points_phh3.csv",
    "thickness": "data/example_inputs/2025-10-22-12-30-points_thickness.csv",
}
# Point files are read as CSV: one row per point, columns z, y, x in micrometers,
# in the image's frame of reference, and one more column per value carried by the
# points, named with its unit, e.g. thickness_um.
#
# Points saved as a NumPy file (.npy) must be converted to such a CSV first. Run
# these lines once in Python, with the environment active, changing the file
# names, then list the CSV in POINTS above:
#
#   import numpy as np
#   import pandas as pd
#   points = np.load("data/my_points.npy")      # shape (N, 3): z, y, x in micrometers
#   table = pd.DataFrame(points, columns=["z", "y", "x"])
#   # table["thickness_um"] = np.load("data/my_values.npy")  # optional: one value per point
#   table.to_csv("data/my_points.csv", index=False)
#
# If the array is in x, y, z order, reverse it first: points = points[:, ::-1].
# If it is in pixels, multiply each column by the voxel size of the image.

RESULTS = "results/2025-10-22-12-30"  # one folder per sample
