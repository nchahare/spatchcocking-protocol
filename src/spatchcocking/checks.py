"""Checks on the inputs, before any work is done.

Nothing in the pipeline can tell a wrong input from a right one: a spacing off by
a constant factor, a mask with two objects, or points in (x, y, z) all still give
a plausible map at the end. These functions catch that at the start. They back
`notebooks/01_inputs.py`, which is why each returns something short enough to
print in one line.

Every analysis has an image and one surface mask, the reference: the surface
that becomes the mesh and is projected to 2D. Optionally, a second mask for the
other side of the tissue wall, and point files.
`direction` says where the tissue is relative to it: "in" (outer surface), "out"
(inner surface) or "both" (a middle surface).
"""

import numpy as np

DIRECTIONS = ("in", "out", "both")


class Report:
    """Collects `ok` / `WARN` / `FAIL` lines and gives a verdict at the end."""

    def __init__(self):
        self.lines = []

    def check(self, ok, what, detail="", problem="", warn_only=False):
        """`detail` is always shown; `problem` only when the check does not pass."""
        status = "ok" if ok else ("WARN" if warn_only else "FAIL")
        text = "; ".join(t for t in (detail, "" if ok else problem) if t)
        self.lines.append((status, what, text))
        print(f"[{status:4s}] {what}" + (f"  —  {text}" if text else ""))
        return ok

    @property
    def failed(self):
        return [line for line in self.lines if line[0] == "FAIL"]

    def verdict(self):
        n_warn = sum(1 for s, _, _ in self.lines if s == "WARN")
        print(f"\n{len(self.lines)} checks: {len(self.lines) - len(self.failed) - n_warn} ok, "
              f"{n_warn} warnings, {len(self.failed)} failed")
        for _, what, detail in self.failed:
            print(f"  FIX: {what}" + (f" — {detail}" if detail else ""))
        if self.failed:
            raise SystemExit("inputs NOT ready: fix the items above, then rerun")
        print("inputs ready")


def load_mask(path):
    """Read a mask as a boolean (z, y, x) array. Raises if it cannot be one."""
    import tifffile

    mask = tifffile.imread(path)
    if mask.ndim != 3:
        raise ValueError(f"{path} has shape {mask.shape}; a mask must be 3D (z, y, x).")
    values = np.unique(mask[::4, ::4, ::4])
    if len(values) > 2:
        raise ValueError(
            f"{path} has more than two values ({values[:6].tolist()} ...); "
            "a mask must be binary, 0 and one other value.")
    mask = mask > 0
    if not mask.any() or mask.all():
        raise ValueError(f"{path} is {'empty' if not mask.any() else 'completely full'}; "
                         "there is no boundary to put a surface on.")
    return mask


def pieces(mask, min_voxels=100):
    """(number of pieces of at least `min_voxels`, number of specks below it).

    Specks are harmless: the mesh step removes anything under 100 voxels.
    """
    from scipy import ndimage

    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return int((sizes >= min_voxels).sum()), int((sizes < min_voxels).sum())


def touches_edge(mask):
    """True if the mask reaches any face of the stack, leaving the surface open there."""
    faces = (mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1])
    return any(f.any() for f in faces)


def extent_um(shape, spacing):
    """Physical size of a (z, y, x) grid, in µm."""
    return np.asarray(shape[-3:], dtype=float) * np.asarray(spacing, dtype=float)


def same_space(shape_a, spacing_a, shape_b, spacing_b):
    """True if two grids cover the same physical box, to within one voxel of the coarser."""
    tol = np.maximum(np.asarray(spacing_a, float), np.asarray(spacing_b, float))
    return bool(np.all(np.abs(extent_um(shape_a, spacing_a) - extent_um(shape_b, spacing_b)) <= tol))


def fraction_inside(a, b):
    """Fraction of the voxels of mask `a` that are also in mask `b`."""
    return float((a & b).sum() / a.sum())


def tissue_region(reference, direction, wall=None):
    """Where the tissue is, as a mask, given the reference and the direction.

    With a wall mask, the tissue is exactly the space between the two surfaces.
    Without one, it is the side of the reference that `direction` points to, and
    for "both" there is no side to test, so None.
    """
    if wall is not None:
        return reference ^ wall
    if direction == "in":
        return reference
    if direction == "out":
        return ~reference
    return None


def diagnose_points(points, tissue, spacing, image_extent_um):
    """(fraction on the tissue, the likeliest reason if it is low).

    Points come from many tools, and the three usual ways they end up outside
    the image's frame are tested in turn: axis order, origin, units.
    """
    points = np.asarray(points, float)
    on = fraction_on_tissue(points, tissue, spacing)
    flipped = fraction_on_tissue(points[:, ::-1], tissue, spacing)
    if flipped > on + 0.2:
        return on, (f"in reverse order {flipped:.2%} would land, so they are probably "
                    "(x, y, z): reorder the columns to (z, y, x)")
    extent = np.asarray(image_extent_um, float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = hi - lo
    # Size first: a cloud in mm or nm is the wrong size wherever it sits.
    if np.any(span > extent * 1.2) or np.all(span < extent * 0.1):
        return on, (f"the cloud spans {span.round(3).tolist()} against the image's "
                    f"{extent.round().tolist()} µm, so they are probably not in µm")
    in_box = np.all((points >= 0) & (points <= extent), axis=1).mean()
    if in_box < 0.5:
        return on, (f"the cloud is the right size but only {in_box:.0%} of it lies inside "
                    f"the image (it spans {lo.round().tolist()} to {hi.round().tolist()} µm). "
                    "Probably an origin offset, e.g. stage coordinates: subtract the "
                    "image's origin")
    return on, "they lie in the image but not on the tissue; look at them in napari"


def fraction_on_tissue(points, tissue, spacing, tolerance_voxels=3):
    """Fraction of (z, y, x) µm points within `tolerance_voxels` of the tissue."""
    from scipy import ndimage

    near = ndimage.binary_dilation(tissue, iterations=tolerance_voxels)
    idx = np.round(np.asarray(points, float) / np.asarray(spacing, float)).astype(int)
    inside_grid = np.all((idx >= 0) & (idx < tissue.shape), axis=1)
    hit = np.zeros(len(idx), dtype=bool)
    hit[inside_grid] = near[tuple(idx[inside_grid].T)]
    return float(hit.mean())
