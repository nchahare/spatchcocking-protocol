"""The input checks behind notebooks/01_inputs.py, on small synthetic masks."""

import numpy as np
import pytest

from spatchcocking import checks


def ball(shape, center, radius):
    z, y, x = np.indices(shape)
    return (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius ** 2


SHAPE = (40, 40, 40)
OUTER = ball(SHAPE, (20, 20, 20), 15)
INNER = ball(SHAPE, (20, 20, 20), 8)


def test_one_object_and_specks():
    mask = OUTER.copy()
    mask[0, 0, 0] = True  # a one-voxel speck
    assert checks.pieces(mask) == (1, 1)
    second = OUTER | ball(SHAPE, (5, 5, 5), 4)
    assert checks.pieces(second)[0] == 2


def test_edge_contact():
    assert not checks.touches_edge(OUTER)
    assert checks.touches_edge(ball(SHAPE, (20, 20, 20), 25))


def test_same_space_catches_the_uncorrected_z_step():
    # image binned 4x in xy, masks at full resolution: same box
    assert checks.same_space((462, 192, 256), (4.5, 22.14, 22.14),
                             (462, 768, 1024), (4.5, 5.535, 5.535))
    assert not checks.same_space((462, 192, 256), (4.5, 22.14, 22.14),
                                 (462, 768, 1024), (3.0, 5.535, 5.535))


def test_tissue_region_by_direction():
    assert (checks.tissue_region(OUTER, "in") == OUTER).all()
    assert (checks.tissue_region(INNER, "out") == ~INNER).all()
    assert checks.tissue_region(OUTER, "both") is None
    wall = checks.tissue_region(OUTER, "in", INNER)
    assert (wall == (OUTER & ~INNER)).all()
    # the same wall whichever surface is the reference
    assert (checks.tissue_region(INNER, "out", OUTER) == wall).all()


def test_points_on_tissue_and_the_axis_order_hint():
    spacing = (1.0, 1.0, 1.0)
    wall = OUTER & ~INNER
    zyx = np.argwhere(wall)[::7].astype(float)
    # break the symmetry so that reversing the columns really moves the points
    zyx = zyx[zyx[:, 0] < 12]
    assert checks.fraction_on_tissue(zyx, wall, spacing, tolerance_voxels=0) == 1.0
    shifted = zyx + np.array([0.0, 0.0, 30.0])
    assert checks.fraction_on_tissue(shifted, wall, spacing, tolerance_voxels=0) < 0.5


def test_report_verdict():
    report = checks.Report()
    report.check(True, "fine", problem="never shown")
    assert report.lines[0] == ("ok", "fine", "")
    report.check(False, "broken", "detail", problem="why")
    assert report.lines[1] == ("FAIL", "broken", "detail; why")
    with pytest.raises(SystemExit):
        report.verdict()
