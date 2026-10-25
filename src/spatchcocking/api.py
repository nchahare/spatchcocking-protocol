"""The protocol's vocabulary: condition, measure, axis, project, unwrap, compare.

Each protocol step is one call, under the name the protocol uses. The published
function names -- `getPlanes`, `getDeformedmesh2`, `get_flatdata2`,
`normalize_values2` -- are internal history, and their `2` suffixes say nothing
to a reader. This module is the layer that gives each step its name.

It is a layer, not a rewrite. Every published function stays exactly where it is,
byte for byte. What changes is what a reader has to know to call it.

Two behaviors differ deliberately from the published path, both because the
published one cannot be written into a protocol:

* `medial_axis` does not open a window. `pipeline.getAxis` opens a fifteen-panel
  `Plotter` on every call whether or not `check` is set, and writes its result into
  the current working directory under a name derived from whatever `.tif` it finds
  there. Neither is something a protocol can ask a reader to do.
* `project` derives `dists_threshold` from the geometry instead of taking a
  hand-tuned constant. The published notebooks set it by hand for each specimen,
  with no stated rule, and a constant with no rule cannot go in a protocol.
"""

import numpy as np

from . import pipeline as sp
from .frame import dorsal_frame_from_anchors, tangents

__all__ = [
    "condition",
    "measure",
    "medial_axis",
    "axis_diagnostics",
    "project",
    "unwrap",
    "compare",
]

NUM_AXIS_POINTS = 25
MLS_ITERATIONS = 10

# dists_threshold culls intersection points farther than this from their axis
# station. Its job is to drop the second cluster a plane cuts on the far limb of a
# bend while keeping the true cross-section, so it belongs just above the real tube
# radius -- hence a high percentile of the measured station radius, with margin.
THRESHOLD_PERCENTILE = 95
THRESHOLD_MARGIN = 1.10

# Azimuth is measured from x_axis = cross(dorsal, tangent) while dorsal points
# along y_axis, so dorsal comes out at +90 deg. This brings it to zero. It is the
# same -90 the published notebooks used without saying why.
DORSAL_SHIFT_DEG = -90.0


def _published_output(verbose):
    """Hide what the published functions print, unless `verbose`.

    They print progress bars, "i am warpping now --", and whole mesh descriptions.
    The CLI has always hidden this behind --verbose; this does the same for the
    notebooks.
    """
    import contextlib
    import io as _io

    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(_io.StringIO())


def condition(mask_path, spacing, out_path=None, **kwargs):
    """Mask stack -> a clean surface mesh. One mask, one surface.

    Wraps `meshing.mask_to_mesh`, which marches the mask and then smooths and
    decimates it to a workable vertex count. It raises if any stage produces an
    empty mesh, naming the stage -- unlike the published `get_mesh`, which could
    write an empty STL while printing "Saved:" and exiting cleanly.

    `spacing` is (z, y, x) micrometers per voxel and is never read from the file.
    """
    import os

    from .meshing import mask_to_mesh

    # Check the file before marching it. The published `load_tiff_stack` catches
    # every exception and returns None, so a missing or unreadable file arrives at
    # marching cubes indistinguishable from a mask that is genuinely empty -- and
    # the reader is told "the mask is empty" when they have really mistyped a
    # path. That function is parity-locked, so the check belongs here.
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"no mask at {mask_path}")
    try:
        import tifffile

        with tifffile.TiffFile(mask_path) as handle:
            n_pages = len(handle.pages)
    except Exception as exc:
        raise ValueError(
            f"{os.path.basename(mask_path)} could not be read as a TIFF: {exc}"
        ) from exc
    if n_pages == 0:
        raise ValueError(f"{os.path.basename(mask_path)} contains no image data.")

    px2umz, px2umxy = float(spacing[0]), float(spacing[1])
    if not np.isclose(spacing[1], spacing[2]):
        raise ValueError(
            f"in-plane spacing is anisotropic ({spacing[1]} vs {spacing[2]} um); "
            "mask_to_mesh takes a single xy value."
        )
    return mask_to_mesh(mask_path, px2umz, px2umxy, out_path=out_path, **kwargs)


def measure(mesh, which, **kwargs):
    """Attach a named field to a surface. Returns the mesh.

    `which` is one of "curvature", "thickness", "density" or "intensity", and the
    remaining arguments go to the matching function in `fields`:

    ============  ===========================================================
    curvature     (no extra arguments needed)
    thickness     inner=<vedo.Mesh>
    density       points=<(N, 3) array>, radius=<um>, name=<str> (default "phh3")
    intensity     volume=<(z, y, x) array>, spacing=<(z, y, x) um>
    values        points=<(N, 3) array>, values=<(N,) array>, name=<str>
    ============  ===========================================================

    "values" carries a number measured at each point (a thickness, a cell area)
    onto the surface: each vertex takes it from its 5 nearest points, through the
    published `pipeline.transfer_points_to_mesh`.
    """
    from . import fields

    if which == "curvature":
        # Fitted on the mesh as given, so every field has one row per vertex and
        # the identities K = k1*k2 and H = (k1 + k2)/2 hold exactly. Pass
        # normalise=True for the published face-area normalization, which makes
        # the estimate comparable across meshes of different resolution but
        # changes the vertex count -- the fields then have to be interpolated
        # back, and interpolating each component independently breaks those
        # identities by around 1e-06 on this specimen.
        kwargs.setdefault("normalise", False)
        # The published fit draws a progress bar per vertex.
        with _published_output(kwargs.pop("verbose", False)):
            curv = fields.curvature(mesh, **kwargs)
        if curv.npoints != mesh.npoints:
            return mesh.clone().interpolate_data_from(curv, n=5)
        out = mesh.clone()
        for name in ("Gauss_Curvature", "Mean_Curvature", "K1", "K2"):
            out.pointdata[name] = np.asarray(curv.pointdata[name])
        return out

    if which == "thickness":
        inner = kwargs.pop("inner")
        return fields.attach(mesh.clone(), "thickness",
                             fields.thickness(mesh, inner, **kwargs))

    if which == "density":
        points = kwargs.pop("points")
        # "phh3" is the published name, kept as the default for parity.
        name = kwargs.pop("name", "phh3")
        return fields.attach(mesh.clone(), name,
                             fields.density_from_points(mesh, points, **kwargs))

    if which == "intensity":
        volume = kwargs.pop("volume")
        spacing = kwargs.pop("spacing")
        name = kwargs.pop("name", "intensity")
        return fields.attach(
            mesh.clone(), name,
            fields.intensity_on_surface(mesh, volume, spacing, **kwargs),
        )

    if which == "values":
        from .pipeline import transfer_points_to_mesh

        points = np.asarray(kwargs.pop("points"), dtype=np.float64)
        values = np.asarray(kwargs.pop("values"), dtype=np.float64)
        name = kwargs.pop("name")
        if points.ndim != 2 or points.shape[1] != 3 or len(values) != len(points):
            raise ValueError(
                f"points {points.shape} and values {values.shape} must be (N, 3) "
                "and (N,), one value per point."
            )
        out = transfer_points_to_mesh(mesh, np.column_stack([points, values]), name,
                                      show_result=False)
        return fields.attach(out, name, np.asarray(out.pointdata[name]))

    raise ValueError(
        f"unknown field {which!r}; expected one of "
        "'curvature', 'thickness', 'density', 'intensity', 'values'."
    )


def medial_axis(mesh, seed=None, num_points=NUM_AXIS_POINTS, n_iter=MLS_ITERATIONS):
    """The tube's longitudinal medial axis, computed headlessly.

    Iterated moving-least-squares smoothing collapses a subsample of the surface
    onto its own centerline; the result is ordered end to end, splined, and
    resampled to `num_points` evenly spaced stations.

    Parameters
    ----------
    mesh : vedo.Mesh
    seed : (3,) array, optional
        A point near one end, used only to decide which end the ordering starts
        from. Defaults to the extreme along the mesh's first principal component,
        which is an end by construction -- so this costs the reader no clicks.
        Which end is posterior is settled later, by the tail click in the picker.
    num_points, n_iter : int
        Number of axis stations, and MLS smoothing iterations.

    Returns
    -------
    (num_points, 3) array
    """
    from vedo import Points, Spline

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if seed is None:
        center = verts.mean(axis=0)
        _, _, vh = np.linalg.svd(verts - center, full_matrices=False)
        proj = (verts - center) @ vh[0]
        seed = verts[np.argmin(proj)]

    pcl = Points(mesh.vertices).subsample(0.02)
    for _ in range(n_iter):
        pcl = pcl.clone().smooth_mls_1d(f=1)
    pcl.subsample(0.1)

    idx = pcl.closest_point(np.asarray(seed), return_point_id=True)
    pts = pcl.vertices
    ordered = pts[sp.nearest_neighbor_order(pts, start=idx)]

    # Extend past both ends before splining, so the spline is not fitted short of
    # the tube it is meant to describe.
    delta = 2
    extended = np.vstack([
        ordered[0] - (ordered[1] - ordered[0]) * delta,
        ordered,
        ordered[-1] + (ordered[-1] - ordered[-2]) * delta,
    ])
    spline = Spline(extended, closed=False).vertices
    return spline[np.linspace(0, len(spline) - 1, num_points).astype(int)]


def station_radii(verts, axis):
    """Median distance from each axis station to the surface points nearest it.

    NaN for a station that no surface point is nearest to. The radius of a tube
    can change several-fold along its length, which is why a dorsal line drawn at
    one radius floats off the narrow end.
    """
    verts = np.asarray(verts, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    d = np.linalg.norm(verts[:, None, :] - axis[None, :, :], axis=2)
    owner = np.argmin(d, axis=1)
    radii = np.full(len(axis), np.nan)
    for i in range(len(axis)):
        mine = owner == i
        if mine.any():
            radii[i] = float(np.median(d[mine, i]))
    return radii


def inside_stations(mesh, axis):
    """Indices of the axis stations that lie inside the tube.

    The medial axis runs past both ends of the tissue, so the stations there lie
    outside the surface. The longest unbroken run of inside
    stations is returned, so one station poking out at a tight bend does not
    split the axis.
    """
    ids = set(int(i) for i in np.atleast_1d(mesh.inside_points(axis, return_ids=True)))
    best, run = [], []
    for i in range(len(axis)):
        if i in ids:
            run.append(i)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    return best


def dorsal_line(mesh, axis, stations, directions):
    """A line along the dorsal side of the surface, for checking the anchors.

    Returns the points, one per inside station, each just outside the surface in
    the dorsal direction the anchors give, at that station's own radius.
    """
    axis = np.asarray(axis, dtype=np.float64)
    valid = inside_stations(mesh, axis)
    radii = station_radii(mesh.vertices, axis)
    known = [i for i in valid if np.isfinite(radii[i])]
    r = np.interp(valid, known, radii[known])
    frame = dorsal_frame_from_anchors(axis, list(stations), np.asarray(directions))
    return axis[valid] + frame[valid] * (r * 1.05)[:, None]


def axis_diagnostics(axis, verts):
    """Whether the cross-sectional planes stay disjoint, station by station.

    Normal planes taken along a curved axis intersect each other where the tube
    radius exceeds the axis's local radius of curvature. That is a computable
    criterion, not a visual judgement, which is what makes it writable as a
    protocol step: the reader reads a table instead of eyeballing a render.

    Returns a dict with `s` (normalized arclength), `rho` (radius of curvature),
    `r_max` (station radius), `ratio` (r_max / rho) and `violating` (station
    indices where ratio >= 1).
    """
    axis = np.asarray(axis, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)

    d1 = np.gradient(axis, axis=0)
    d2 = np.gradient(d1, axis=0)
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / (
        np.linalg.norm(d1, axis=1) ** 3 + 1e-12
    )
    rho = 1.0 / (kappa + 1e-12)

    d = np.linalg.norm(verts[:, None, :] - axis[None, :, :], axis=2)
    owner = np.argmin(d, axis=1)
    dmin = d[np.arange(len(verts)), owner]
    r_max = np.full(len(axis), np.nan)
    for i in range(len(axis)):
        sel = dmin[owner == i]
        if sel.size:
            r_max[i] = sel.max()

    seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s / s[-1]

    ratio = r_max / rho
    return {
        "s": s,
        "rho": rho,
        "r_max": r_max,
        "ratio": ratio,
        "violating": np.flatnonzero(np.nan_to_num(ratio) >= 1.0),
    }


def project(mesh, axis, stations, directions, skip_index=None, dists_threshold=None,
            verbose=False):
    """Straighten the tube: cut normal planes and warp them onto a common line.

    The dorsal direction is carried between anchors by a rotation-minimizing
    frame, so it is defined at every station by construction. The published
    scheme took each station's direction from the nearest point on a traced
    spline, which leaves the stations near the ends with a numerically
    meaningless direction however carefully the trace is drawn.

    Parameters
    ----------
    mesh : vedo.Mesh
        With whatever fields are to be carried through already attached.
    axis : (N, 3) array
    stations : sequence of int
        Axis station index of each anchor.
    directions : (K, 3) array
        Dorsal direction at each anchor.
    skip_index : array of int, optional
        Stations to drop. Off by default: on a synthetic tube with a known
        dorsal direction, dropping overlapping stations made the map slightly
        worse. The warp absorbs a mild overlap better than removing stations
        does.
    dists_threshold : float, optional
        Defaults to 1.10 x the 95th percentile of station radius, derived from
        this specimen's own geometry.

    Returns
    -------
    (deformed_mesh, info) where info records the threshold and diagnostics.
    """
    axis = np.asarray(axis, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    stations = list(int(i) for i in stations)
    skip_index = np.array([] if skip_index is None else skip_index, dtype=int)

    diag = axis_diagnostics(axis, mesh.vertices)
    if dists_threshold is None:
        dists_threshold = float(
            np.nanpercentile(diag["r_max"], THRESHOLD_PERCENTILE) * THRESHOLD_MARGIN
        )

    axis_info = {
        "axis": axis,
        "axis normals": tangents(axis),
        "dorsal normals": dorsal_frame_from_anchors(axis, stations, directions),
    }
    with _published_output(verbose):
        deformed = sp.getDeformedmesh2(
            mesh, axis_info, namefile=None, skip_index=skip_index,
            dists_threshold=dists_threshold, check=None,
        )
    info = dict(diag)
    info["dists_threshold"] = dists_threshold
    info["n_anchors"] = len(stations)
    return deformed, info


def unwrap(deformed, shift_deg=DORSAL_SHIFT_DEG, verbose=False):
    """Cut the straightened cylinder open and lay it flat.

    Returns a DataFrame with `radius`, `norm_height` (rostro-caudal, 0 to 1),
    `angle_degrees` (dorso-ventral, -180 to 180) and every scalar field carried
    through the warp, one row per resampled point.

    `shift_deg` rotates the azimuthal origin, and the default of -90 is not a
    magic number. `transform_to_radial` builds its frame as
    `y_axis = dorsal`, `x_axis = cross(dorsal, tangent)`, then measures
    `angle = arctan2(proj . y_axis, proj . x_axis)`. Azimuth is therefore measured
    from x, while dorsal points along y -- so dorsal lands at +90 deg, and -90
    brings it to zero. Pinning the frame to dorsal with anchors does not change
    this: it fixes which direction is dorsal, not where arctan2 starts counting.

    Checked against the biology as well as the algebra. The roof plate is the
    thinnest part of the tube, so a correctly oriented map puts the thickness
    minimum close to theta = 0 with this default, and about 90 deg away
    without it.
    """
    import pandas as pd

    with _published_output(verbose):
        radius, angle, height, flat = sp.get_flatdata2(deformed, namefile=None, check=None)
    norm_height, angle_deg = sp.normalize_values2(height, angle, shift_deg=shift_deg)

    cols = {"radius": radius, "norm_height": norm_height, "angle_degrees": angle_deg}
    for name in flat.pointdata.keys():
        arr = np.asarray(flat.pointdata[name])
        if arr.ndim == 1 and len(arr) == len(radius):
            cols[name] = arr
    return pd.DataFrame(cols)


def compare(maps, field, rc_bins=50, dv_bin_width=10, labels=None):
    """Pool several unwrapped maps and average them on a common grid.

    Once every specimen is in (s, theta), a bin means the same anatomical place
    in all of them. The points themselves differ from specimen to specimen, so
    maps are pooled by bin, not by point.

    Each specimen is averaged within each bin first, and the specimens are then
    averaged, so every specimen counts once in every bin it reaches. Averaging
    all points together instead weights a specimen by how many points it happens
    to put in a bin.

    Binning: `rc_bins` equal-width bins along norm_height, and
    `dv_bin_width`-degree bins across the full -180 to 180 azimuth. Rows run from
    the tail (norm_height 0) to the head (1), as in the maps.

    Parameters
    ----------
    maps : sequence of str or DataFrame
        One per specimen: a CSV written by `unwrap`, or the table itself.
    field : str
        Column to average.
    labels : sequence of str, optional
        A label per specimen -- a stage, say -- to group by. Defaults to file
        stems, or "map 0", "map 1", ... for tables.

    Returns
    -------
    (pooled, grids) : the concatenated per-point table, and a dict mapping each
    label to its mean grid as a DataFrame indexed by RC bin, tail first, columns
    DV bin.
    """
    import os

    import pandas as pd

    maps = list(maps)
    names = [m if isinstance(m, (str, os.PathLike)) else f"map {i}"
             for i, m in enumerate(maps)]
    if labels is None:
        labels = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
    if len(labels) != len(maps):
        raise ValueError(
            f"{len(labels)} labels given for {len(maps)} maps."
        )

    frames = []
    for m, name, label in zip(maps, names, labels):
        df = pd.read_csv(m) if isinstance(m, (str, os.PathLike)) else m.copy()
        if field not in df.columns:
            raise ValueError(
                f"{name} has no column {field!r}. It carries: {list(df.columns)}"
            )
        df["label"] = label
        df["source"] = str(name)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    dv_edges = np.arange(-180, 180 + dv_bin_width, dv_bin_width)
    pooled["rc_bin"] = pd.cut(pooled["norm_height"], bins=rc_bins)
    pooled["dv_bin"] = pd.cut(pooled["angle_degrees"], bins=dv_edges)

    grids = {}
    for label, part in pooled.groupby("label", observed=True):
        per_specimen = [
            specimen.pivot_table(index="rc_bin", columns="dv_bin", values=field,
                                 aggfunc="mean", observed=False, dropna=False)
            for _, specimen in part.groupby("source", observed=True)
        ]
        grids[label] = pd.concat(per_specimen).groupby(level=0, observed=False).mean()
    return pooled, grids
