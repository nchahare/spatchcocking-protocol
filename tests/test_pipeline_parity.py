"""The split modules must stay identical to the published one, and complete.

The point of splitting was to organize code, not to change or lose any. So the
test is mostly textual: every function across `pipeline`, `meshing` and `attic`
is compared BYTE-FOR-BYTE against its counterpart in `spatchcocking_utils.py`,
and the union is checked to account for ALL 31 published functions. If the
bodies match, the numerics cannot differ - and the comparison is instant, so it
can run in CI rather than being something someone means to check.

Two numerical tests back it up, because identical text can still behave
differently if a name resolves to something else. The published module does
`from vedo import *`; this one imports explicitly, and an explicit import that
named the wrong thing would pass the text check and fail here.

The published module is not a dependency of this package. Point
SPATCHCOCKING_PUBLISHED at a checkout to run these; otherwise they skip.
"""
import ast
import importlib.util
import os

import numpy as np
import pytest

from spatchcocking import attic, meshing, pipeline

MODULES = (pipeline, meshing, attic)

# Ours, not the published module's. Named individually so that adding a
# function to these modules is a deliberate act rather than a silent one.
LOCAL = {"mask_to_mesh"}

# No default path. An absolute path to one machine's checkout used to live here,
# which meant these tests silently did something different depending on whose
# machine ran them -- passing here, skipping everywhere else, for a reason the
# skip message did not give. The environment variable is the only way in.
PUBLISHED = os.environ.get("SPATCHCOCKING_PUBLISHED", "")
MESH = "data/example_outputs/reference_surface.ply"


def _skip_unless_published():
    """Skip, saying which of the two reasons applies."""
    if not PUBLISHED:
        pytest.skip("SPATCHCOCKING_PUBLISHED is not set; point it at a checkout "
                    "of the published spatchcocking_utils.py to run parity tests")
    if not os.path.exists(PUBLISHED):
        pytest.skip(f"SPATCHCOCKING_PUBLISHED points at {PUBLISHED}, "
                    "which does not exist")


def _load_published():
    _skip_unless_published()
    spec = importlib.util.spec_from_file_location("_published", PUBLISHED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources(path):
    """name -> exact source text, for every top-level function in a file."""
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            # Trailing newlines are stripped: the published file's last
            # function ends without one, which is a property of the file, not
            # of the code.
            body = "".join(lines[n.lineno - 1:n.end_lineno])
            out[n.name] = body.rstrip()
    return out


@pytest.fixture(scope="module")
def published():
    return _load_published()


def _ours():
    out = {}
    for mod in MODULES:
        for name, text in _sources(mod.__file__).items():
            if name in LOCAL:
                continue
            assert name not in out, f"{name} appears in two modules"
            out[name] = text
    return out


def test_local_additions_are_declared():
    """Anything in these modules that is not published must be in LOCAL."""
    _skip_unless_published()
    theirs = set(_sources(PUBLISHED))
    for mod in MODULES:
        for name in _sources(mod.__file__):
            assert name in theirs or name in LOCAL, (
                f"{name} in {mod.__name__} is neither published nor declared "
                f"in LOCAL. Add it to LOCAL if it is deliberately ours.")


def test_every_function_is_byte_identical():
    _skip_unless_published()
    ours, theirs = _ours(), _sources(PUBLISHED)
    assert ours, "no functions found"
    extra = sorted(set(ours) - set(theirs))
    assert not extra, f"functions the published module does not have: {extra}"
    for name, text in sorted(ours.items()):
        assert text == theirs[name], (
            f"{name} differs from the published implementation.\n"
            f"The split may move a function between files, never edit one.")


def test_nothing_was_lost_in_the_split():
    """The whole published module must be accounted for, function by function.

    This is the check that matters. Twenty published functions are not called
    by the protocol's scripts, but four of them - the mask-to-mesh entry point
    and the point-cloud route - are part of the method, so "unused by the code"
    is not the same as "not needed". Splitting keeps every one; this asserts it.
    """
    _skip_unless_published()
    ours, theirs = _ours(), _sources(PUBLISHED)
    lost = sorted(set(theirs) - set(ours))
    assert not lost, f"functions dropped entirely: {lost}"
    assert len(ours) == len(theirs) == 31


def test_functions_are_in_the_module_their_role_says():
    _skip_unless_published()
    in_ = {m.__name__.rsplit(".", 1)[-1]: set(_sources(m.__file__))
           for m in MODULES}
    # Named individually, so moving one between files is a deliberate edit here.
    assert {"getAxis", "getPlanes", "getDeformedmesh2", "get_flatdata2",
            "normalize_values2", "compute_and_save_curvatures",
            "transfer_points_to_mesh"} <= in_["pipeline"]
    assert {"load_tiff_stack", "preprocess_mask", "tiff_stack_to_mesh",
            "get_mesh", "save_mesh", "extract_mesh_marching_cubes",
            "extract_mesh_dual_contouring"} <= in_["meshing"]
    assert {"getDeformedmesh", "get_flatdata", "normalize_values",
            "visualize_flatmesh", "visualize_flatmesh2", "selectPointsonMesh",
            "query4skip_index", "getProperCurvature", "getnormal"} <= in_["attic"]


def test_radial_transforms_match_numerically(published):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * 200.0
    center = np.array([10.0, -5.0, 3.0])
    normal = np.array([0.3, 0.5, 0.81])
    normal = normal / np.linalg.norm(normal)
    dnormal = np.cross(normal, [0.0, 0.0, 1.0])
    dnormal = dnormal / np.linalg.norm(dnormal)

    a = pipeline.transform_to_radial(pts, center, normal, dnormal)
    b = published.transform_to_radial(pts, center, normal, dnormal)
    np.testing.assert_array_equal(a, b)

    a2 = pipeline.transform_to_cartesian(a, center, normal, dnormal)
    b2 = published.transform_to_cartesian(b, center, normal, dnormal)
    np.testing.assert_array_equal(a2, b2)


def test_normalize_values_matches_numerically(published):
    rng = np.random.default_rng(1)
    height = rng.normal(size=4000) * 1000.0
    angle = rng.uniform(-np.pi, np.pi, size=4000)
    for shift in (0.0, -90.0, 137.0):
        a = pipeline.normalize_values2(height, angle, shift_deg=shift)
        b = published.normalize_values2(height, angle, shift_deg=shift)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.skipif(not os.path.exists(MESH), reason="example mesh missing")
def test_get_flatdata2_matches_on_the_example_mesh(published):
    """The one that exercises vedo: decimate, smooth, subdivide, then polar."""
    from vedo import Mesh
    a = pipeline.get_flatdata2(Mesh(MESH), namefile=None, check=None)
    b = published.get_flatdata2(Mesh(MESH), namefile=None, check=None)
    for k, (x, y) in enumerate(zip(a[:3], b[:3])):
        np.testing.assert_array_equal(
            x, y, err_msg=f"get_flatdata2 output {k} differs")
    assert a[3].npoints == b[3].npoints
