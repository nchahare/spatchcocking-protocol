"""Protocol step: mark the dorsal points on the reference surface.

Run from the working directory: python protocol/pick_dorsal_points.py
Opens the picker on the sample's reference surface; the dorsal points are
saved beside it, in the results folder of the sample.
"""
import subprocess
import sys
from sample import *

# Medial axis settings.
AXIS_ITERATIONS = 10   # smoothing rounds for the medial axis; raise it, up to 20,
                       # if the axis loops back at one end of the tube
RECOMPUTE_AXIS = False # True: compute the axis afresh instead of reusing the saved one

command = [sys.executable, "scripts/pick_anchors.py", f"{RESULTS}/reference_surface.ply",
           "--iterations", str(AXIS_ITERATIONS)]
if RECOMPUTE_AXIS:
    command.append("--recompute")
subprocess.run(command)  # the picker reports for itself when nothing is saved
