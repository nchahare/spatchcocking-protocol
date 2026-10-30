"""spatchcocking - standardized 3D-to-2D projection of tubular organ surfaces.

Supporting software for Chahare, Imamura & Nerurkar (2026),
Biophysical Reports 6:100280, doi:10.1016/j.bpr.2026.100280

Unlike the original release, this package does not re-export vedo and numpy into
the caller's namespace. `from spatchcocking import *` pulling in all of vedo made
notebooks short but left the reader unable to tell which names belong to this
package - and a protocol has to name things precisely.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
