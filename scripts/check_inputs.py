"""Protocol step: check that the image, the mask, and the points line up.

Run from the working directory: python scripts/check_inputs.py
Only the mask is required; the image and the point files are checked if
protocol/sample.py names them.
"""
import sys
sys.path.insert(0, "protocol")  # where sample.py lives
from spatchcocking import checks, io
from sample import *  # noqa: E402,F403

mask = checks.load_mask(MASK)
size = checks.extent_um(mask.shape, MASK_VOXEL)

report = checks.Report()
report.check(checks.pieces(mask)[0] == 1, "the mask is one object")
report.check(not checks.touches_edge(mask), "the mask is clear of the stack edge",
             warn_only=True)
if IMAGE:
    image, _ = io.load_stack(IMAGE, IMAGE_VOXEL)
    n_channels = image.shape[0] if image.ndim == 4 else 1
    report.check(n_channels == len(CHANNELS), "one name per channel",
                 f"{n_channels} channels, {len(CHANNELS)} names")
    report.check(checks.same_space(image.shape, IMAGE_VOXEL, mask.shape, MASK_VOXEL),
                 "the image and the mask cover the same space",
                 problem="a voxel size is wrong")
    size = checks.extent_um(image.shape, IMAGE_VOXEL)
for name, path in POINTS.items():
    points, values = io.load_point_table(path)
    on, why = checks.diagnose_points(points, mask, MASK_VOXEL, size)
    report.check(on >= 0.95, f"{name}: points land on the tissue", f"{on:.1%}", problem=why)
report.verdict()

import napari  # noqa: E402

viewer = napari.Viewer(ndisplay=3)
if IMAGE:
    viewer.add_image(image, channel_axis=0 if image.ndim == 4 else None,
                     scale=IMAGE_VOXEL, name=CHANNELS if image.ndim == 4 else CHANNELS[0])
viewer.add_labels(mask.astype("uint8"), scale=MASK_VOXEL, opacity=0.35, name="mask")
napari.run()
