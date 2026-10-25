"""Vendored from vedo — `project_point_on_variety`.

WHY THIS FILE EXISTS
--------------------
The curvature computation in this package fits a local quadric to each vertex
neighbourhood via `project_point_on_variety`, passing the vertex normal so the
sign of the mean curvature is consistent across the surface.

That `normal=` keyword exists only on vedo's development branch; it has never
appeared in a PyPI release. The published version of this pipeline therefore
required installing vedo from a specific git commit, which is the single largest
obstacle to a reader reproducing the analysis from a clean environment.

Vendoring the function removes that obstacle without altering the mathematics.
The code below is copied verbatim from vedo so that curvature values remain
bit-identical to those reported in Chahare et al. (2026); only the optional
`return_grid` branch has been changed, to import vedo lazily so this module has
no hard dependency on it.

tests/test_fields.py checks that this copy reproduces the curvature values
in data/example_outputs/.

SOURCE
------
vedo/pointcloud/fits.py, lines 256-411
vedo 2026.6.2.dev8 (git commit 88f8b2f9c5c6267b5dceda0e4ab0e15b3130910a)
https://github.com/marcomusy/vedo

LICENSE
-------
vedo is distributed under the MIT License, Copyright (c) 2017 Marco Musy.
The full text is in LICENSE.vedo alongside this file. MIT permits redistribution
provided the copyright notice and permission notice are retained, which they are.

If a future vedo release exposes `normal=` in `project_point_on_variety`, this
module can be deleted and the import in `pipeline.py` pointed back at vedo.
"""

import numpy as np

__all__ = ["project_point_on_variety"]


def project_point_on_variety(
    pt, points, degree=3, normal=None, return_grid=False
) -> tuple:
    """
    Project a point in 3D space onto a polynomial surface defined by a set of points
    around it. The polynomial degree can be adjusted.

    Args:
        pt (list or np.ndarray):
            The 3D point to project onto the fitted surface.
        points (np.ndarray):
            Neighbourhood points (Nx3) used to fit the polynomial surface.
        degree (int):
            Degree of the fitting polynomial.
        normal (list or np.ndarray, optional):
            Reference normal used to orient the local frame consistently.
            Pass the vertex normal of `pt` for correct mean curvature sign.
        return_grid (bool, optional):
            If True, also returns a `vedo.Grid` of the fitted surface patch.

    Returns:
        projected_pt (np.ndarray):
            The point projected onto the fitted polynomial surface.
        poly (tuple):
            Polynomial results `(coeffs, R, centroid, gauss_curv, mean_curv)`:
            - `coeffs`: polynomial coefficients (length = (degree+1)(degree+2)/2)
            - `R`: 3x3 rotation matrix mapping world -> local frame (rows are v1, v2, normal)
            - `centroid`: centroid of the neighbourhood points
            - `gauss_curv`: Gaussian curvature K at the projected point (0 for degree < 2)
            - `mean_curv`: mean curvature H at the projected point (0 for degree < 2)
        grid (vedo.Grid or None):
            Surface patch of the fitted polynomial, or `None` if `return_grid=False`.

    Examples:
        ```python
        import vedo
        from vedo.pointcloud import project_point_on_variety

        mesh = vedo.Mesh(vedo.dataurl+"bunny.obj").subdivide().scale(100)
        mesh.wireframe().alpha(0.1)

        pt = mesh.coordinates[30]
        points = mesh.closest_point(pt, n=200)

        pt_trans, poly, grid = project_point_on_variety(pt, points, degree=3, return_grid=True)
        vpoints = vedo.Points(points, r=6, c="yellow2")

        plotter = vedo.Plotter(size=(1200, 800))
        plotter += mesh, vedo.Point(pt), vpoints, grid, f"Residue: {pt - pt_trans}"
        plotter.show(axes=1).close()
        ```

    Check out also the `fit_plane()` function for a simpler case of plane fitting.
    """

    def _fit_polynomial_3d(points, degree):
        x, y, z = points.T
        xpow = x[:, None] ** np.arange(degree + 1)   # (N, degree+1)
        ypow = y[:, None] ** np.arange(degree + 1)
        V = np.column_stack([xpow[:, i] * ypow[:, j]
                             for i in range(degree + 1)
                             for j in range(degree + 1 - i)])
        return np.linalg.lstsq(V, z, rcond=None)[0]

    def _predict_polynomial_3d(x, y, coeffs, degree):
        xpow = [1.0] * (degree + 1)
        ypow = [1.0] * (degree + 1)
        for k in range(1, degree + 1):
            xpow[k] = xpow[k-1] * x
            ypow[k] = ypow[k-1] * y
        z, idx = 0.0, 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                z += coeffs[idx] * xpow[i] * ypow[j]
                idx += 1
        return z

    def _compute_curvature(coeffs, degree, x0, y0):
        if degree < 2:
            return 0, 0
        xpow = [1.0] * (degree + 1)
        ypow = [1.0] * (degree + 1)
        for k in range(1, degree + 1):
            xpow[k] = xpow[k-1] * x0
            ypow[k] = ypow[k-1] * y0
        f_x = f_y = f_xx = f_yy = f_xy = 0.0
        idx = 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                c = coeffs[idx]
                if i >= 1: f_x  += i       * c * xpow[i-1] * ypow[j]
                if j >= 1: f_y  += j       * c * xpow[i]   * ypow[j-1]
                if i >= 2: f_xx += i*(i-1) * c * xpow[i-2] * ypow[j]
                if j >= 2: f_yy += j*(j-1) * c * xpow[i]   * ypow[j-2]
                if i >= 1 and j >= 1:
                    f_xy += i * j  * c * xpow[i-1] * ypow[j-1]
                idx += 1
        denom2 = 1.0 + f_x**2 + f_y**2
        gaussian = (f_xx * f_yy - f_xy**2) / denom2**2
        # negated to match VTK convention: H > 0 for convex with outward normals
        mean = -((1 + f_y**2)*f_xx - 2*f_x*f_y*f_xy + (1 + f_x**2)*f_yy) / (2 * denom2**1.5)
        return float(gaussian), float(mean)

    # Fit the plane: compute centroid and normal
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # to find the normal vector
    # eigh on the 3x3 covariance matrix is much faster than SVD on the full N×3 matrix
    _, vv = np.linalg.eigh(centered.T @ centered)
    svd_normal = vv[:, 0]  # eigenvector for smallest eigenvalue = normal direction
    if normal is not None and np.dot(svd_normal, normal) < 0:
        svd_normal = -svd_normal
    normal = svd_normal

    # Build a right-handed orthogonal basis with normal as the z-axis.
    # Cross with the least-aligned cardinal axis for numerical stability.
    min_idx = np.argmin(np.abs(normal))
    e = np.zeros(3); e[min_idx] = 1.0
    v1 = np.cross(normal, e); v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1); v2 /= np.linalg.norm(v2)
    # v2 = cross(normal, v1) guarantees det([v1,v2,normal]) = +1, no check needed.
    R = np.array([v1, v2, normal])  # rows: basis vectors; local = centered @ R.T

    # Transform points to new coordinate system (plane aligns with XY)
    transformed = np.dot(centered, R.T)

    tpt = (pt - centroid) @ R.T  # Transform point to new coordinate system

    # Fit polynomial of given degree
    x_new, y_new = tpt[0], tpt[1]
    coeffs = _fit_polynomial_3d(transformed, degree)
    gauss_curv, mean_curv = _compute_curvature(coeffs, degree, x_new, y_new)
    z_pred = _predict_polynomial_3d(x_new, y_new, coeffs, degree)

    # Transform back to original
    transformed_pt = np.array([x_new, y_new, z_pred])
    back_transformed = np.dot(transformed_pt, R) + centroid
    grid = None
    if return_grid:
        # Create a surface mesh from the polynomial fit
        x_min, x_max = transformed[:, 0].min(), transformed[:, 0].max()
        y_min, y_max = transformed[:, 1].min(), transformed[:, 1].max()
        import vedo  # lazy: only needed for the optional preview grid
        grid = vedo.Grid([x_min, x_max, y_min, y_max], res=(20, 20))
        grid.flat().use_bounds(False)
        gpts = grid.points
        for g in gpts:
            zg = _predict_polynomial_3d(g[0], g[1], coeffs, degree)
            tzg = np.array([g[0], g[1], zg])
            g[:] = np.dot(tzg, R)
        grid.shift(centroid).compute_normals()
        grid.lw(0).c("lightblue").alpha(0.5).lighting("glossy")
    return back_transformed, (coeffs, R, centroid, gauss_curv, mean_curv), grid


