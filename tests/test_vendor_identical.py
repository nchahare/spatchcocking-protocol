"""The vendored quadric fit must reproduce vedo's results exactly.

This is the test that justifies vendoring: if it passes, dropping the git-master
vedo dependency changes no published number.

Run against the example's reference surface, every vertex, not a synthetic case.
"""
import numpy as np
from vedo import Mesh, progressbar
from vedo import project_point_on_variety as vedo_fn

from spatchcocking._vendor import project_point_on_variety as vendored_fn

MESH = r"data/example_outputs/reference_surface.ply"
DEPTH = 5
DEGREE = 2


def curvatures(fn, msh, adlist, pts, normals):
    g, m = [], []
    for i in range(msh.npoints):
        ids = msh.find_adjacent_vertices(i, depth=DEPTH, adjacency_list=adlist)
        _, poly, _ = fn(pts[i], pts[ids], degree=DEGREE, normal=normals[i])
        g.append(poly[3])
        m.append(poly[4] * -1)
    return np.asarray(g), np.asarray(m)


def main():
    msh = Mesh(MESH)
    msh.compute_normals()
    adlist = msh.compute_adjacency()
    pts = msh.vertices
    normals = msh.pointdata["Normals"]
    print(f"mesh: {msh.npoints} vertices, depth={DEPTH}, degree={DEGREE}")

    print("computing with vedo...")
    g_vedo, m_vedo = curvatures(vedo_fn, msh, adlist, pts, normals)
    print("computing with vendored copy...")
    g_ours, m_ours = curvatures(vendored_fn, msh, adlist, pts, normals)

    for name, a, b in [("Gaussian K", g_vedo, g_ours), ("Mean H", m_vedo, m_ours)]:
        identical = np.array_equal(a, b)
        maxdiff = np.max(np.abs(a - b))
        print(f"{name:12s} bit-identical={identical}  max|diff|={maxdiff:.3e}  "
              f"range=[{a.min():.3e}, {a.max():.3e}]")
        assert identical, f"{name} differs (max {maxdiff:.3e})"

    print("\nPASS - vendored function is bit-identical to vedo on all "
          f"{msh.npoints} vertices.")


if __name__ == "__main__":
    main()
