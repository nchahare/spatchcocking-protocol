"""The anchor frame must behave, and must match the measured sweep."""
import numpy as np
import pytest

from spatchcocking.frame import (
    dorsal_frame_from_anchors,
    rotation_minimizing_frame,
    signed_angle_about,
    tangents,
)


def helix(n=60, turns=1.5, radius=30.0, pitch=200.0):
    t = np.linspace(0, turns * 2 * np.pi, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t),
                            pitch * t / (2 * np.pi)])


def test_frame_is_perpendicular_to_tangent():
    axis = helix()
    tang = tangents(axis)
    r = rotation_minimizing_frame(axis, tang)
    assert np.allclose(np.sum(r * tang, axis=1), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(r, axis=1), 1, atol=1e-9)


def test_frame_adds_no_twist_on_a_straight_axis():
    """With no bending there is nothing to transport; the frame must be constant."""
    axis = np.column_stack([np.zeros(40), np.zeros(40), np.linspace(0, 100, 40)])
    r = rotation_minimizing_frame(axis, tangents(axis))
    assert np.allclose(r, r[0], atol=1e-9)


def test_single_anchor_is_honoured_exactly():
    axis = helix()
    tang = tangents(axis)
    want = np.cross(tang[10], [0, 0, 1.0])
    want /= np.linalg.norm(want)
    out = dorsal_frame_from_anchors(axis, [10], [want])
    assert np.allclose(out[10], want, atol=1e-8)


def test_every_anchor_is_honoured_exactly():
    axis = helix()
    tang = tangents(axis)
    idx = [5, 20, 35, 50]
    dirs = []
    for i in idx:
        v = np.cross(tang[i], [0, 0, 1.0])
        dirs.append(v / np.linalg.norm(v))
    out = dorsal_frame_from_anchors(axis, idx, dirs)
    for i, d in zip(idx, dirs):
        assert np.allclose(out[i], d, atol=1e-8), f"anchor {i} not reproduced"


def test_anchor_direction_need_not_be_perpendicular():
    """A click lands on the surface, not in the cross-sectional plane."""
    axis = helix()
    tang = tangents(axis)
    v = np.cross(tang[10], [0, 0, 1.0])
    v /= np.linalg.norm(v)
    tilted = v + 0.4 * tang[10]          # deliberately out of plane
    out = dorsal_frame_from_anchors(axis, [10], [tilted])
    assert np.allclose(np.sum(out * tang, axis=1), 0, atol=1e-9)
    assert np.allclose(out[10], v, atol=1e-8)


def test_rejects_direction_parallel_to_axis():
    axis = helix()
    tang = tangents(axis)
    with pytest.raises(ValueError, match="parallel"):
        dorsal_frame_from_anchors(axis, [10], [tang[10]])


def test_matches_measured_sweep_on_real_data():
    """Five anchors should reproduce the example's hand trace to ~1.6 deg median.

    The trace is the twelve dorsal clicks of the example, on its own medial
    axis; if the implementation drifts, this test is what catches it.
    """
    from vedo import Spline

    axis = np.load("data/example_outputs/reference_surface-axis.npy")
    endpts = np.load("data/example_outputs/2025-10-22-12-30-dorsal_clicks.npy")

    tang = tangents(axis)

    # reference directions, as the published getPlanes derives them
    dorsal_pts = Spline(endpts, 100, closed=False).vertices
    d = np.linalg.norm(axis[:, None, :] - dorsal_pts[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    v = dorsal_pts[idx] - axis
    perp = v - np.sum(v * tang, axis=1, keepdims=True) * tang
    pn = np.linalg.norm(perp, axis=1)
    ref = perp / pn[:, None]
    good = ~((idx == 0) | (idx == len(dorsal_pts) - 1)
             | (pn / np.maximum(np.linalg.norm(v, axis=1), 1e-12) < 0.5))

    gi = np.flatnonzero(good)
    anchors = sorted(set(int(round(x)) for x in np.linspace(gi[0], gi[-1], 5)))
    out = dorsal_frame_from_anchors(axis, anchors, ref[anchors])

    resid = signed_angle_about(out[good], ref[good], tang[good])
    resid = (resid + 180) % 360 - 180
    median = float(np.median(np.abs(resid)))
    assert median < 3.0, f"median residual {median:.2f} deg, expected ~1.6"
