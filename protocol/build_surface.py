"""Protocol step: build the reference surface from the mask.

Run from the working directory: python protocol/build_surface.py
"""
import os
from spatchcocking import api, io, viewer
from sample import *

# Mesh settings: change them until the mesh is smooth and keeps the shape.
# They are listed in the order in which they act.
BLUR_UM = None            # blur the mask by this many micrometers first; None = off
SMOOTH_ITER = 100         # smoothing passes before simplifying
DECIMATE_FRACTION = 0.05  # fraction of triangles kept in the first simplification
FINAL_FRACTION = 0.1      # fraction of those kept in the second simplification
FINAL_SMOOTH_ITER = 50    # smoothing passes at the end
SHOW_IN_IMAGE = False     # True: also show the mesh inside the image in napari

os.makedirs(RESULTS, exist_ok=True)
mesh = api.condition(MASK, MASK_VOXEL, out_path=f"{RESULTS}/reference_surface.ply",
                     smooth_iter=SMOOTH_ITER, decimate_fraction=DECIMATE_FRACTION,
                     final_fraction=FINAL_FRACTION, final_smooth_iter=FINAL_SMOOTH_ITER,
                     blur_um=BLUR_UM)
print(mesh.npoints, "vertices,", round(mesh.area() / 1e6, 2), "square millimeters")

# A medial axis saved for an earlier surface no longer fits this one: delete it,
# so that pick_dorsal_points.py computes a new one.
old_axis = f"{RESULTS}/reference_surface-axis.npy"
if os.path.exists(old_axis):
    os.remove(old_axis)
    print("deleted the medial axis of the previous surface:", old_axis)
mesh.show()

if SHOW_IN_IMAGE and IMAGE:
    image, _ = io.load_stack(IMAGE, IMAGE_VOXEL)
    viewer.show_in_context(image, IMAGE_VOXEL, mesh, channel_names=CHANNELS)
