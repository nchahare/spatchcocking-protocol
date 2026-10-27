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
point_tables = {}
for name, path in POINTS.items():
    points, values = io.load_point_table(path)
    point_tables[name] = points, values
    on, why = checks.diagnose_points(points, mask, MASK_VOXEL, size)
    report.check(on >= 0.95, f"{name}: points land on the tissue", f"{on:.1%}", problem=why)
report.verdict()

import napari  # noqa: E402

CHANNEL_STYLE = {  # colormap and opacity per channel; channels left out keep napari's
    "pHH3": dict(colormap="magenta", opacity=0.5),
    "DAPI": dict(colormap="gray", opacity=1.0),
}
POINTS_BACKDROP = "DAPI"    # channel also shown under the points; None for none
POINT_STYLE = {  # size in micrometers, opacity and symbol per point file
    "pHH3": dict(size=30, opacity=1.0, symbol="cross"),  # thin, so the points under show
}
DEFAULT_POINT_STYLE = dict(size=40, opacity=1.0, symbol="disc")


# One window, two panels side by side: the image channels with the mask, and the
# point files on one channel. The panels share one camera, so they turn together.
viewer = napari.Viewer(ndisplay=3, title="inputs")
channel_layers = []
if IMAGE:
    added = viewer.add_image(image, channel_axis=0 if image.ndim == 4 else None,
                             scale=IMAGE_VOXEL, units="um",
                             name=CHANNELS if image.ndim == 4 else CHANNELS[0])
    channel_layers = added if isinstance(added, list) else [added]
    for layer in channel_layers:
        for key, value in CHANNEL_STYLE.get(layer.name, {}).items():
            setattr(layer, key, value)
viewer.add_labels(mask.astype("uint8"), scale=MASK_VOXEL, units="um", opacity=0.5,
                  name="mask")

# Each panel holds the same number of consecutive layers, so the shorter group
# is padded with empty layers.
backdrop = [layer for layer in channel_layers if layer.name == POINTS_BACKDROP]
left, right = len(channel_layers) + 1, len(backdrop) + len(point_tables)
for _ in range(right - left):
    viewer.add_points(name="(empty)", ndim=3, units="um")
for layer in backdrop:
    viewer.add_image(layer.data, scale=IMAGE_VOXEL, units="um", name=f"{layer.name} (under points)",
                     colormap=layer.colormap, contrast_limits=layer.contrast_limits)
# Points are already in micrometers, napari's world units here, so need no scale.
# A file with value columns is colored by its first one.
for name, (points, values) in point_tables.items():
    style = {**DEFAULT_POINT_STYLE, **POINT_STYLE.get(name, {})}
    if values:
        column = next(iter(values))
        viewer.add_points(points, name=f"{name} points", units="um", border_width=0, **style,
                          features=values, face_color=column, face_colormap="viridis")
    else:
        viewer.add_points(points, name=f"{name} points", units="um", border_width=0, **style,
                          face_color="magenta")
for _ in range(left - right):
    viewer.add_points(name="(empty)", ndim=3, units="um")
viewer.grid.stride = max(left, right)
viewer.grid.enabled = bool(point_tables)
# napari 0.9 moved the scale bar into viewer.canvas.overlays; the layers' units
# make it read in micrometers.
try:
    viewer.canvas.overlays.scale_bar.visible = True
except AttributeError:
    viewer.scale_bar.visible = True
napari.run()
