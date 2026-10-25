"""Protocol step: straighten and unwrap the surface into a map.

Run from the working directory: python protocol/unwrap.py
"""
from vedo import Mesh
from spatchcocking import api, io
from sample import *

# Straightening setting.
DISTS_THRESHOLD = None  # cut points farther from the axis are discarded, micrometers;
                        # None = 1.1 x the 95th percentile of the tube's radius

mesh = Mesh(f"{RESULTS}/reference_surface_fields.vtk")
anchors = io.load_anchors(f"{RESULTS}/reference_surface-anchors.json", mesh)

straight, info = api.project(mesh, anchors["axis"], anchors["stations"],
                             anchors["directions"], dists_threshold=DISTS_THRESHOLD)
print(f"distance threshold: {info['dists_threshold']:.0f} micrometers")
flat = api.unwrap(straight)
flat.to_csv(f"{RESULTS}/spatchcocked.csv", index=False)
print(len(flat), "points on the map, saved to", f"{RESULTS}/spatchcocked.csv")
