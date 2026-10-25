"""Where the data is, found rather than assumed.

The published notebooks used `../data/example` and one of them called
`os.chdir` into it, so every later cell depended on the kernel's working
directory and on being launched from `notebooks/`. Restarting a kernel
halfway through the set left paths pointing somewhere else.

These resolve the repository root by looking for `pyproject.toml`, which works
whether the notebook is run from `notebooks/`, from the repository root, or
converted to a script and run from anywhere inside the tree.
"""

from pathlib import Path


def repo_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        f"no pyproject.toml above {start}; run this from inside the repository."
    )


ROOT = repo_root()
DATA = ROOT / "data"
EXAMPLE_INPUTS = DATA / "example_inputs"    # what a reader brings
EXAMPLE_OUTPUTS = DATA / "example_outputs"  # what the protocol makes from them
EXAMPLE_MAPS = DATA / "example_maps"        # six maps, for pooling
RESULTS = ROOT / "results"

# The worked example, one embryo, carried through every notebook here.
STEM = "2025-10-22-12-30"

# Voxel size of the shipped image, (z, y, x) in micrometers. The z step is the
# refractive-index-corrected 4.5, not the 3.0 the acquisition metadata reports;
# xy is 5.535 binned 4x. Never read from the file -- see spatchcocking.io.
IMAGE_SPACING = (4.5, 22.14, 22.14)

# Voxel size of the masks, which are on the full unbinned lattice.
MASK_SPACING = (4.5, 5.535, 5.535)
