# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: spatchcocking
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 — Is the environment working?
#
# Run this first, once, after
#
# ```
# conda env create -f environment.yml
# conda activate spatchcocking
# ```
#
# It checks that everything the protocol needs is installed and usable, and
# ends with one line: **ready**, or a list of what to fix. It reads no data and
# writes nothing.
#
# **Timing:** a few seconds.

# %%
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _paths import ROOT

results = []  # (status, check, detail)


def record(ok, check, detail="", warn=False):
    status = "ok" if ok else ("WARN" if warn else "FAIL")
    results.append((status, check, detail))
    print(f"[{status:4s}] {check}" + (f"  —  {detail}" if detail else ""))


# %% [markdown]
# ## Python, and whether the environment is activated
#
# Activation matters on Windows. numpy's math library (MKL) lives in a folder
# that only `conda activate` puts on `PATH`. Without it, the first real matrix
# operation kills the process with exit code `0xC06D007F`, halfway through
# whatever was running, which looks like a pipeline bug.
#
# The matrix test below runs in a **separate process**, so if it crashes, this
# notebook survives to tell you why.

# %%
record(sys.version_info >= (3, 10), "Python 3.10 or newer", sys.version.split()[0])

prefix = os.environ.get("CONDA_PREFIX", "")
active = bool(prefix) and Path(prefix).resolve() == Path(sys.prefix).resolve()
record(active, "running inside the activated conda environment",
       f"{Path(sys.prefix).name}" if active else
       f"this Python is {sys.prefix}, but the active environment is "
       f"{prefix or 'none'}. Run `conda activate spatchcocking` and restart.")

probe = subprocess.run(
    [sys.executable, "-c",
     "import numpy as n; a = n.random.rand(400, 400); "
     "print(float(n.linalg.inv(a @ a.T + 400 * n.eye(400)).trace()))"],
    capture_output=True, text=True,
)
record(probe.returncode == 0, "numpy can do linear algebra (MKL loads)",
       "" if probe.returncode == 0 else
       f"exit code {probe.returncode & 0xFFFFFFFF:#x}. The environment is not "
       "activated: run `conda activate spatchcocking` and restart the kernel.")

# %% [markdown]
# ## The packages
#
# Two versions are held on purpose:
#
# - **vedo is pinned** in `environment.yml` to the release the tests were
#   verified against. A different version is flagged, not failed: vedo builds
#   differ very slightly in curvature, so results agree to far better than
#   anything biological, but not bit for bit.
# - **scikit-image must be 0.26 or later**, the first version with the argument
#   that the mask cleanup uses.

# %%
PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "seaborn", "skimage",
            "tifffile", "trimesh", "vtk", "vedo", "napari", "qtpy"]
versions = {}
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        if name == "vtk":
            versions[name] = mod.vtkVersion.GetVTKVersion()
        else:
            versions[name] = getattr(mod, "__version__", "?")
        record(True, f"import {name}", versions[name])
    except Exception as exc:
        record(False, f"import {name}", f"{type(exc).__name__}: {exc}")

if "vedo" in versions:
    record(versions["vedo"] == "2026.6.1", "vedo is the verified release",
           f"{versions['vedo']} (verified: 2026.6.1)", warn=True)
if "skimage" in versions:
    major, minor = (int(v) for v in versions["skimage"].split(".")[:2])
    record((major, minor) >= (0, 26), "scikit-image 0.26 or later", versions["skimage"])

# %% [markdown]
# ## napari has a working Qt backend
#
# napari is only used for looking: notebooks 01 and 02, and `spatchcock view`.
# Nothing is measured with it. This checks that it can find a
# Qt backend; it does not open a window.

# %%
try:
    from qtpy import API_NAME, QT_VERSION

    record(True, "Qt backend for napari", f"{API_NAME} {QT_VERSION}")
except Exception as exc:
    record(False, "Qt backend for napari", f"{type(exc).__name__}: {exc}")

# %% [markdown]
# ## This repository is the package that gets imported
#
# `environment.yml` installs the package *editable*: the environment links to
# `src/spatchcocking/` in this folder rather than copying it. If the path below
# points anywhere else, an older copy is installed and edits here will not take
# effect.

# %%
try:
    import spatchcocking

    where = Path(spatchcocking.__file__).resolve()
    here = (ROOT / "src" / "spatchcocking").resolve()
    record(here in where.parents, "spatchcocking imported from this repo",
           str(where.parent))
except Exception as exc:
    record(False, "import spatchcocking", f"{type(exc).__name__}: {exc}")

cli = shutil.which("spatchcock")
record(cli is not None, "`spatchcock` command on PATH", cli or
       "not found. Usually the environment is not activated; if it is, the "
       "package is not installed: rerun `conda env create -f environment.yml`.")

# %% [markdown]
# ## Verdict

# %%
failed = [r for r in results if r[0] == "FAIL"]
warned = [r for r in results if r[0] == "WARN"]
print(f"{len(results)} checks: {len(results) - len(failed) - len(warned)} ok, "
      f"{len(warned)} warnings, {len(failed)} failed\n")
if failed:
    for _, check, detail in failed:
        print(f"  FIX: {check} — {detail}")
    raise SystemExit("environment NOT ready — fix the items above, then rerun")
print("ready" + (" (with warnings above)" if warned else ""))

# %% [markdown]
# ## Expected outcome
#
# Every check **ok**, **0 failed**, and then **`ready`**, in a few seconds.
# Run without activating, it reports failures, including the `0xc06d007f`
# crash, and says to activate.
#
# For a deeper check, run the test suite from the repository root:
#
# ```
# pytest
# ```
#
# Expected: every test passes or is skipped, in a few minutes.
#
# Next: **01 — are the inputs right?**
