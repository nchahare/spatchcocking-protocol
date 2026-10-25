"""The dorsal reference frame: carrying one direction along the medial axis.

The azimuthal coordinate theta needs a dorsal direction at EVERY axis station.
A human supplies it at a handful.

The published approach splines the clicked points and gives each station the
direction towards the nearest point on that spline. It degenerates wherever the
axis outruns the trace - and `getAxis` extrapolates the axis past the data, so
that always happens at both ends: the stations there are left with a numerically
meaningless direction, even when the trace is deliberately clicked to both
extremes.

Here the clicks are ANCHORS instead. A rotation-minimizing frame is defined at
every station by construction, so there is nothing to degenerate; the anchors
only pin its phase. About five anchors are enough to follow a full hand trace
of the midline.
"""
import numpy as np

__all__ = [
    "tangents",
    "rotation_minimizing_frame",
    "dorsal_frame_from_anchors",
    "dorsal_frame_from_surface_path",
    "surface_path_through",
    "signed_angle_about",
]


def tangents(axis):
    """Unit tangents along a discrete axis."""
    t = np.gradient(np.asarray(axis, dtype=float), axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def rotation_minimizing_frame(axis, tang, seed=None):
    """Torsion-free frame along the axis, by double reflection.

    Wang et al. (2008), "Computation of rotation minimizing frames". Preferred
    over a Frenet frame, which is undefined where curvature vanishes and flips
    through inflection points - both of which a flexed neural tube has.

    The seed only fixes the overall phase; anchors correct it afterwards.
    """
    axis = np.asarray(axis, dtype=float)
    r = np.zeros_like(axis)

    if seed is None:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(tang[0] @ seed)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
    v = seed - (seed @ tang[0]) * tang[0]
    r[0] = v / np.linalg.norm(v)

    for i in range(len(axis) - 1):
        v1 = axis[i + 1] - axis[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-12:
            r[i + 1] = r[i]
            continue
        rL = r[i] - (2.0 / c1) * (v1 @ r[i]) * v1
        tL = tang[i] - (2.0 / c1) * (v1 @ tang[i]) * v1
        v2 = tang[i + 1] - tL
        c2 = float(v2 @ v2)
        r[i + 1] = rL if c2 < 1e-12 else rL - (2.0 / c2) * (v2 @ rL) * v2
        r[i + 1] /= np.linalg.norm(r[i + 1])
    return r


def signed_angle_about(a, b, axis_vec):
    """Signed angle from a to b measured about axis_vec, in degrees."""
    cross = np.cross(a, b)
    return np.degrees(np.arctan2(np.sum(cross * axis_vec, axis=-1),
                                 np.sum(a * b, axis=-1)))


def dorsal_frame_from_anchors(axis, anchor_idx, anchor_dirs):
    """Dorsal direction at every station, from directions known at a few.

    Args:
        axis: (N, 3) medial axis stations.
        anchor_idx: station indices where the dorsal direction is known.
        anchor_dirs: (k, 3) directions at those stations; need not be exactly
            perpendicular to the tangent, the component along it is removed.

    Returns:
        (N, 3) unit dorsal directions, perpendicular to the tangent everywhere.
    """
    axis = np.asarray(axis, dtype=float)
    anchor_idx = np.asarray(anchor_idx, dtype=int)
    anchor_dirs = np.atleast_2d(np.asarray(anchor_dirs, dtype=float))
    if len(anchor_idx) == 0:
        raise ValueError("at least one anchor is required")

    tang = tangents(axis)
    rmf = rotation_minimizing_frame(axis, tang)

    # Project each supplied direction into its station's cross-sectional plane.
    d = anchor_dirs - (np.sum(anchor_dirs * tang[anchor_idx], axis=1, keepdims=True)
                       * tang[anchor_idx])
    n = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("an anchor direction is parallel to the axis tangent; "
                         "pick a point further from the axis")
    d = d / n

    # How far the frame has drifted from the anatomy, at each anchor.
    delta = signed_angle_about(rmf[anchor_idx], d, tang[anchor_idx])

    # Interpolate the correction along arc length. Unwrap first so that a drift
    # passing through +/-180 does not wrap and produce a spurious jump.
    seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / s[-1]

    if len(anchor_idx) == 1:
        corr = np.full(len(axis), delta[0])
    else:
        order = np.argsort(anchor_idx)
        ai, du = anchor_idx[order], np.degrees(np.unwrap(np.radians(delta[order])))
        corr = np.interp(s, s[ai], du)

    # Rotate the frame by the correction, about the tangent (Rodrigues).
    c = np.cos(np.radians(corr))[:, None]
    sn = np.sin(np.radians(corr))[:, None]
    out = (rmf * c
           + np.cross(tang, rmf) * sn
           + tang * np.sum(tang * rmf, axis=1, keepdims=True) * (1.0 - c))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def surface_path_through(mesh, points):
    """Ordered points along the mesh surface passing through `points`.

    Consecutive anchors are joined by a geodesic - the shortest path across the
    mesh itself - so the resulting curve is constrained to the surface. This is
    the difference from the published scheme, which splines the clicks in free
    3D space: that chord can cut through the tissue or float off it, and the
    dorsal direction is then taken towards a curve that is not on the surface.

    Args:
        mesh: a vedo Mesh.
        points: (k, 3) ordered anchor points lying on (or near) the surface.

    Returns:
        (M, 3) ordered points along the surface, anchors included.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()

    out = []
    for a, b in zip(points[:-1], points[1:]):
        seg = mesh.geodesic(list(a), list(b)).vertices
        if len(out) and len(seg):
            seg = seg[1:]           # drop the shared endpoint
        out.append(seg)
    path = np.vstack([s for s in out if len(s)])
    return path


def dorsal_frame_from_surface_path(axis, path, anchor_idx=None):
    """Dorsal direction at every station, from a curve lying on the surface.

    Within the span the path covers, the direction points from each axis station
    towards the nearest point on the path - the published idea, but with a curve
    that genuinely follows the dorsal surface.

    Outside that span the path has nothing to say, so the direction is carried
    by the rotation-minimizing frame instead, which is defined everywhere. That
    is what removes the stations the published scheme leaves degenerate at the
    ends.

    Args:
        axis: (N, 3) medial axis stations.
        path: (M, 3) ordered points on the surface.
        anchor_idx: optional station indices of the anchors; the covered span is
            taken from them when given, otherwise inferred from the path.

    Returns:
        (N, 3) unit dorsal directions perpendicular to the tangent.
    """
    axis = np.asarray(axis, dtype=float)
    path = np.asarray(path, dtype=float)
    tang = tangents(axis)

    d = np.linalg.norm(axis[:, None, :] - path[None, :, :], axis=2)
    nearest = path[np.argmin(d, axis=1)]
    v = nearest - axis
    perp = v - np.sum(v * tang, axis=1, keepdims=True) * tang
    n = np.linalg.norm(perp, axis=1)

    if anchor_idx is not None and len(anchor_idx):
        lo, hi = int(min(anchor_idx)), int(max(anchor_idx))
    else:
        idx = np.argmin(d, axis=1)
        covered = (idx > 0) & (idx < len(path) - 1)
        lo = int(np.argmax(covered))
        hi = int(len(covered) - 1 - np.argmax(covered[::-1]))

    # A near-zero perpendicular component means the station sits beyond the
    # path and the direction is meaningless; treat it as uncovered.
    covered = np.zeros(len(axis), dtype=bool)
    covered[lo:hi + 1] = True
    covered &= n / np.maximum(np.linalg.norm(v, axis=1), 1e-12) > 0.5

    if not covered.any():
        raise ValueError("the surface path covers no axis station")

    out = np.zeros_like(axis)
    out[covered] = perp[covered] / n[covered, None]

    # Extrapolate past both ends with the frame, anchored to the outermost
    # covered station so the join is continuous.
    ci = np.flatnonzero(covered)
    for end, step in ((ci[0], -1), (ci[-1], +1)):
        rmf = rotation_minimizing_frame(axis, tang)
        delta = signed_angle_about(rmf[end], out[end], tang[end])
        c, s_ = np.cos(np.radians(delta)), np.sin(np.radians(delta))
        rot = (rmf * c + np.cross(tang, rmf) * s_
               + tang * np.sum(tang * rmf, axis=1, keepdims=True) * (1.0 - c))
        rng = range(end - 1, -1, -1) if step < 0 else range(end + 1, len(axis))
        for i in rng:
            if not covered[i]:
                out[i] = rot[i] / np.linalg.norm(rot[i])

    return out / np.linalg.norm(out, axis=1, keepdims=True)
