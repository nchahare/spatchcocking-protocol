"""Protocol step: put the measurements on the surface.

Run from the working directory: python protocol/measure.py
Four types of measurement become fields on the surface, one number per vertex.
Curvature is always computed; the others need an image or point files
named in protocol/sample.py.
"""
from vedo import Mesh
from spatchcocking import api, io
from sample import *

# 1. Curvature, from the shape of the surface: at each vertex, a polynomial
#    surface is fitted to the neighboring vertices.
CURVATURE = dict(
    depth=5,   # rings of neighboring vertices used in each fit
    degree=2,  # degree of the fitted polynomial surface
)

# 2. Image intensity next to the surface, per channel: which side to sample
#    ("both", "in" or "out"), how deep in micrometers, and at how many depths on
#    each side. A channel left out is not measured; {} measures none. Ignored
#    if sample.py has no image.
INTENSITY = {
    "pHH3": dict(direction="both", depth_um=15, n_samples=4),
    "DAPI": dict(direction="both", depth_um=15, n_samples=4),
}

# 3. Point density, per point file with positions only: the local density of
#    the points, counted within a radius, in micrometers. The radius must span
#    the tissue wall. {} if there are no such files.
DENSITY = {
    "pHH3": dict(radius_um=100),
}

# 4. Point values, per point file with value columns: each column is carried
#    onto the surface from the five nearest points. No settings.

mesh = Mesh(f"{RESULTS}/reference_surface.ply")
mesh = api.measure(mesh, "curvature", **CURVATURE)

for channel, settings in (INTENSITY.items() if IMAGE else []):
    if channel not in CHANNELS:
        raise SystemExit(f"INTENSITY names {channel!r}, which is not in CHANNELS {CHANNELS}")
    volume, _ = io.load_stack(IMAGE, IMAGE_VOXEL, channel=CHANNELS.index(channel))
    mesh = api.measure(mesh, "intensity", volume=volume, spacing=IMAGE_VOXEL,
                       name=f"{channel}_intensity", **settings)

for name, path in POINTS.items():
    points, values = io.load_point_table(path)
    if not values:
        if name not in DENSITY:
            raise SystemExit(f"{name} has positions only: add it to DENSITY")
        mesh = api.measure(mesh, "density", points=points,
                           radius=DENSITY[name]["radius_um"], name=f"{name}_density")
    for column, v in values.items():
        mesh = api.measure(mesh, "values", points=points, values=v, name=column)

mesh.write(f"{RESULTS}/reference_surface_fields.vtk")
fields = [f for f in mesh.pointdata.keys() if f != "Normals"]
print("fields:", fields)

# Show every field computed, curvature first, then intensity, then the point
# fields, with the colormaps and ranges of show_map.py: median +/- 3 standard
# deviations, symmetric about zero for signed fields, from zero for the others.
import numpy as np  # noqa: E402
from vedo import Plotter  # noqa: E402
from spatchcocking.colors import field_grid, field_style  # noqa: E402
from spatchcocking.pipeline import getTightercmap  # noqa: E402

grid = field_grid(fields)
rows, columns = len(grid), len(grid[0])
plotter = Plotter(shape=(rows, columns), sharecam=True, size=(400 * columns, 450 * rows),
                  title="fields on the reference surface")
for i, field in enumerate(sum(grid, [])):
    if field is None:
        continue
    _, cmap_name, symmetric = field_style(field)
    low, high = getTightercmap(np.asarray(mesh.pointdata[field]), sigma=3)
    if symmetric:
        high = max(abs(low), abs(high))
        low = -high
    else:
        low = 0.0
    shown = mesh.clone().cmap(cmap_name, field, vmin=low, vmax=high).add_scalarbar()
    plotter.at(i).show(shown, field)
plotter.interactive().close()
