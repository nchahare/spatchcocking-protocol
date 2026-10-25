"""Curvature must not drift from the values in data/example_outputs/.

Unlike test_vendor_identical.py, this test needs no special vedo build: it
compares the vendored quadric fit against the curvature fields the protocol
wrote for the example. It is the check that should run in CI and on any
reader's machine.
"""
import numpy as np
import pytest
from vedo import Mesh

from spatchcocking._vendor import project_point_on_variety

MESH = "data/example_outputs/reference_surface.ply"
REFERENCE = "data/example_outputs/reference_surface_fields.vtk"
DEPTH, DEGREE = 5, 2  # as in protocol/measure.py


@pytest.fixture(scope="module")
def reference():
    fields = Mesh(REFERENCE)
    return {"vertices": fields.vertices,
            "gauss": np.asarray(fields.pointdata["Gauss_Curvature"]),
            "mean": np.asarray(fields.pointdata["Mean_Curvature"])}


@pytest.fixture(scope="module")
def computed():
    depth, degree = DEPTH, DEGREE
    msh = Mesh(MESH)
    msh.compute_normals()
    adlist = msh.compute_adjacency()
    pts = msh.vertices
    normals = msh.pointdata["Normals"]

    g, m = [], []
    for i in range(msh.npoints):
        ids = msh.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
        _, poly, _ = project_point_on_variety(pts[i], pts[ids], degree=degree,
                                              normal=normals[i])
        g.append(poly[3])
        m.append(poly[4] * -1)
    return np.asarray(g), np.asarray(m)


def test_mesh_unchanged(reference):
    """The reference is only meaningful if it describes this exact mesh."""
    pts = Mesh(MESH).vertices
    assert pts.shape == reference["vertices"].shape
    np.testing.assert_allclose(pts, reference["vertices"], rtol=0, atol=0)


# Tolerances are absolute, not relative: both curvatures cross zero, so a
# relative tolerance is meaningless near the sign change.
#
# Measured drift between vedo 2026.6.2.dev8 and 2026.6.1, with identical
# vertices, normals, adjacency and MKL: 4.0e-11 on K, 2.8e-8 on H. The cause is
# neighbor ordering changing the row order of the least-squares fit, so the
# rounding differs. Within a single environment the computation is bit-identical
# run to run, independent of thread count.
#
# Thresholds sit an order of magnitude above the measured drift, and are still
# ~1e-5 of each quantity's full range (K spans ~7e-5, H spans ~1.2e-2), i.e. far
# below anything that could change a figure or a reported statistic.
ATOL_GAUSS = 1e-9
ATOL_MEAN = 1e-6


def test_gaussian_curvature_matches_reference(computed, reference):
    np.testing.assert_allclose(computed[0], reference["gauss"],
                               rtol=0, atol=ATOL_GAUSS)


def test_mean_curvature_matches_reference(computed, reference):
    np.testing.assert_allclose(computed[1], reference["mean"],
                               rtol=0, atol=ATOL_MEAN)


def test_vendored_function_needs_no_vedo_master():
    """The vendored copy must accept `normal=`, whatever vedo is installed."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(40, 3))
    out = project_point_on_variety(pts[0], pts, degree=2, normal=[0, 0, 1])
    assert len(out) == 3
    assert len(out[1]) == 5
