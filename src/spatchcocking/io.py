"""Loading and saving, with voxel spacing always supplied by the caller.

One rule governs this module: **spacing is an argument, never a file lookup.**

The reason is specific and was paid for. The acquisition behind the shipped
example reports z = 3.0 um in its metadata, but the correct step is 4.5 um -- a
factor of 1.5, the refractive-index correction for DBE clearing. Every mesh, spot
and density in this dataset is already in the corrected frame. A metadata-aware
loader silently returns 3.0, and the image then sits 1.5x compressed inside its
own segmentation masks. Nothing raises; the maps just come out wrong.

So no function here reads spacing from a file header. `load_stack` takes it, and
`data/example_inputs/2025-10-22-12-30-meta.json` records it beside the data where a human
reads it. If you are tempted to add metadata sniffing, read that sidecar first.

The scalar-field interchange format is the one the published notebooks use: an
(N, 4) array of X, Y, Z, value, saved as .npy. `load_field`/`save_field` are the
named version of that convention.
"""

import json
import os

import numpy as np

__all__ = [
    "load_stack",
    "find_meta",
    "load_meta",
    "load_points",
    "load_field",
    "save_field",
    "load_anchors",
]


def find_meta(image_path):
    """The `*-meta.json` sidecar describing `image_path`, or None.

    Matched on the sidecar's own `image` field rather than by filename pattern,
    so a renamed image does not silently pick up the wrong description.
    """
    folder = os.path.dirname(os.path.abspath(image_path))
    target = os.path.basename(image_path)
    for name in sorted(os.listdir(folder)):
        if not name.endswith("-meta.json"):
            continue
        try:
            payload = load_meta(os.path.join(folder, name))
        except (OSError, ValueError):
            continue
        if payload.get("image") == target:
            return payload
    return None


def load_stack(path, spacing, channel=None):
    """Read a 3D or 4D TIFF stack. Returns (array, spacing).

    Parameters
    ----------
    path : str
        A TIFF stack. 3D files are read as (z, y, x). 4D files are read as
        (c, z, y, x); ImageJ hyperstacks are stored ZCYX and are transposed
        here, which is how the example image is stored.
    spacing : tuple of float
        (z, y, x) micrometers per voxel. Required, and never inferred -- see the
        module docstring.
    channel : int or str, optional
        Return this channel only, as (z, y, x). Prefer the name: the shipped
        acquisition names its channels only "Ch2 GaAsP-T1" and "Ch1-T2", and
        which of them is pHH3 was settled by eye, not by metadata. Passing
        `channel="pHH3"` resolves the index through the `*-meta.json` sidecar,
        so the assignment lives in one place instead of in every caller's memory
        of which integer meant what.

    Raises
    ------
    ValueError
        If the file is not 3D or 4D, if `spacing` is not three positive numbers,
        or if `channel` cannot be resolved. Each message names what to change.
    """
    import tifffile

    spacing = tuple(float(s) for s in np.atleast_1d(spacing))
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(
            f"spacing must be three positive numbers (z, y, x) in um, got {spacing}. "
            "For the shipped example it is (4.5, 22.14, 22.14) -- see "
            "data/example_inputs/2025-10-22-12-30-meta.json."
        )

    arr = tifffile.imread(path)

    if isinstance(channel, str):
        meta = find_meta(path)
        names = (meta or {}).get("channels")
        if not names:
            raise ValueError(
                f"channel={channel!r} was given by name, but no *-meta.json beside "
                f"{os.path.basename(path)} lists its channels. Pass an integer "
                "index, or add a sidecar with a 'channels' list."
            )
        lowered = [str(n).lower() for n in names]
        if channel.lower() not in lowered:
            raise ValueError(
                f"channel {channel!r} not found; {os.path.basename(path)} carries "
                f"{names}."
            )
        channel = lowered.index(channel.lower())

    if arr.ndim == 4:
        # ImageJ hyperstacks are ZCYX; the smaller of the first two axes is the
        # channel axis in every acquisition this package has seen.
        if arr.shape[1] < arr.shape[0]:
            arr = arr.transpose(1, 0, 2, 3)
        n_c = arr.shape[0]
        if channel is None:
            return arr, spacing
        if not 0 <= channel < n_c:
            raise ValueError(
                f"channel {channel} out of range: {os.path.basename(path)} has "
                f"{n_c} channels (0-{n_c - 1})."
            )
        return np.ascontiguousarray(arr[channel]), spacing

    if arr.ndim == 3:
        if channel not in (None, 0):
            raise ValueError(
                f"channel={channel} requested but {os.path.basename(path)} is a "
                "single-channel (z, y, x) stack."
            )
        return arr, spacing

    raise ValueError(
        f"{os.path.basename(path)} has shape {arr.shape}; expected a 3D (z, y, x) "
        "or 4D (c, z, y, x) stack."
    )


def load_meta(path):
    """Read a dataset sidecar such as `2025-10-22-12-30-meta.json`.

    The sidecar is documentation with a machine-readable shape, not a
    configuration file: it records spacing, channel identities and the reason the
    z step is what it is. Read `spacing_um_zyx` from it and pass that to
    `load_stack` explicitly, so the value appears in the caller's own code.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_points(path):
    """Read an (N, 3) point cloud in micrometers, ordered (z, y, x).

    The protocol's point files are CSV tables; see `load_point_table`.
    """
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"{os.path.basename(path)} has shape {pts.shape}; expected (N, 3) "
            "point coordinates in um."
        )
    return np.asarray(pts, dtype=np.float64)


def load_point_table(path):
    """Read a point CSV. Returns (points, values).

    The format is one row per point: columns `z`, `y`, `x` in micrometers, in the
    image's frame of reference, and any number of other columns, each a value
    carried by that point (a thickness, a cell area, a mean intensity). Units
    belong in the column name, e.g. `thickness_um`, because nothing here can
    know them.

    Returns
    -------
    points : (N, 3) float array, (z, y, x) in um
    values : dict of column name -> (N,) float array; empty if there are none
    """
    import pandas as pd

    table = pd.read_csv(path)
    table.columns = [str(c).strip() for c in table.columns]
    missing = [c for c in ("z", "y", "x") if c not in table.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} has no column(s) {missing}; its columns are "
            f"{list(table.columns)}. A point file needs z, y, x in um, in the "
            "image's frame of reference."
        )
    points = table[["z", "y", "x"]].to_numpy(dtype=np.float64)
    values = {c: table[c].to_numpy(dtype=np.float64)
              for c in table.columns if c not in ("z", "y", "x")}
    return points, values


def load_field(path):
    """Read an (N, 3+k) scalar-field sidecar. Returns (coords, values).

    Columns are X, Y, Z followed by one or more scalars, the format of the
    field files of the published analysis.
    """
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(
            f"{os.path.basename(path)} has shape {arr.shape}; expected (N, 3+k) "
            "with columns X, Y, Z, then one or more scalars."
        )
    values = arr[:, 3] if arr.shape[1] == 4 else arr[:, 3:]
    return np.ascontiguousarray(arr[:, :3]), np.ascontiguousarray(values)


def save_field(path, coords, values):
    """Write a scalar field in the (N, 3+k) X, Y, Z, value convention."""
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if len(coords) != len(values):
        raise ValueError(
            f"coords has {len(coords)} rows and values has {len(values)}; they "
            "must describe the same vertices."
        )
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(path, np.column_stack([coords, values]))
    return path


def mesh_fingerprint(mesh):
    """A short hash of a mesh's vertices, recorded with the anchors picked on it."""
    import hashlib

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    return hashlib.sha256(vertices.tobytes()).hexdigest()[:16]


def load_anchors(path, mesh=None):
    """Read a `spatchcocking.dorsal_anchors/1` sidecar written by `pick_anchors`.

    Returns the parsed payload with `stations`, `directions` and, when present,
    `points` as arrays, plus `axis`: the medial axis the anchors were picked on,
    read from the `axis_file` beside the sidecar. The anchors are station numbers
    on that axis, so it is the one to use. Older sidecars carry no `point` key --
    a direction alone cannot be turned back into a position on the surface, so
    the geodesic frame route is unavailable for those and the caller is told so
    rather than left to discover it.

    With `mesh`, also check the anchors were picked on that surface. If the mesh
    has been remade since (notebook 02 rerun with other settings, say), the old
    anchors would still load and put dorsal in the wrong place, with nothing to
    say so.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    schema = payload.get("schema", "")
    if not schema.startswith("spatchcocking.dorsal_anchors/"):
        raise ValueError(
            f"{os.path.basename(path)} has schema {schema!r}; expected "
            "'spatchcocking.dorsal_anchors/1'. Sidecars written by the older "
            "dorsal picker use 'spatchcocking.dorsal_reference/1' and hold a "
            "click trace, not anchors."
        )

    anchors = payload.get("anchors", [])
    if not anchors:
        raise ValueError(f"{os.path.basename(path)} contains no anchors.")

    payload["stations"] = np.array([a["station"] for a in anchors], dtype=int)
    payload["directions"] = np.array([a["direction"] for a in anchors], dtype=float)
    payload["points"] = (
        np.array([a["point"] for a in anchors], dtype=float)
        if all("point" in a for a in anchors)
        else None
    )
    axis_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                             payload.get("axis_file", ""))
    if payload.get("axis_file") and not os.path.isfile(axis_path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} names its medial axis as "
            f"{payload['axis_file']!r}, but there is no such file at {axis_path}. "
            "Keep the anchors file and the -axis.npy file where the picker saved them."
        )
    payload["axis"] = np.load(axis_path) if os.path.isfile(axis_path) else None

    if mesh is not None and "mesh_fingerprint" in payload:
        if mesh_fingerprint(mesh) != payload["mesh_fingerprint"]:
            raise ValueError(
                f"{os.path.basename(path)} was picked on a different surface than "
                "this one. Mark the dorsal points again on this surface: "
                "python protocol/pick_dorsal_points.py, or "
                "python scripts/pick_anchors.py <surface>"
            )
    return payload
