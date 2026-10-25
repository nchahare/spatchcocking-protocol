"""Rebuild an anchor sidecar from a legacy dorsal pick file.

The published pipeline stored the human's dorsal midline trace as a bare
(N, 3) float64 .npy - no schema, no mesh hash, no metadata, and under names
that do not always say what they are. The example's
`data/example_outputs/2025-10-22-12-30-dorsal_clicks.npy` is such a trace: the
clicks along the dorsal midline.

That file is the only record of the one irreducibly manual step, so being able
to read it back matters: it means a specimen processed before this repo existed
can be re-run without asking anyone to click again.

The conversion is small. Each pick is assigned to its nearest axis station, and
the direction from that station to the pick - with the component along the
tangent removed - is the dorsal direction there. That is exactly what the anchor
schema stores.

    python scripts/anchors_from_picks.py \\
        --mesh results/2025-10-22-12-30/reference_surface.ply \\
        --picks data/example_outputs/2025-10-22-12-30-dorsal_clicks.npy \\
        --axis results/2025-10-22-12-30/reference_surface-axis.npy

Writes <mesh stem>-anchors.json and copies the axis to <mesh stem>-axis.npy,
where protocol/unwrap.py finds them.
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np
from vedo import Mesh

from spatchcocking.frame import tangents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--picks", required=True,
                    help="(N, 3) .npy of dorsal midline clicks")
    ap.add_argument("--axis", required=True, help="(M, 3) .npy medial axis")
    ap.add_argument("--anchors", type=int, default=0,
                    help="keep only this many, evenly spaced; 0 keeps all")
    args = ap.parse_args()

    mesh = Mesh(args.mesh)
    picks = np.load(args.picks)
    axis = np.load(args.axis)
    if picks.ndim != 2 or picks.shape[1] != 3:
        raise SystemExit(f"{args.picks} is {picks.shape}, expected (N, 3)")
    tang = tangents(axis)

    verts = np.asarray(mesh.vertices)
    d_surf = np.linalg.norm(
        verts[np.argmin(np.linalg.norm(picks[:, None, :] - verts[None, :, :],
                                       axis=2), axis=1)] - picks, axis=1)

    d_axis = np.linalg.norm(picks[:, None, :] - axis[None, :, :], axis=2)
    station = np.argmin(d_axis, axis=1)

    # Two clicks can land on the same station. Keep the one furthest from the
    # axis: picks are ray-surface intersections, so the further one is the more
    # confidently on the dorsal surface rather than clipping an edge.
    keep = {}
    for k, st in enumerate(station):
        if st not in keep or d_axis[k, st] > d_axis[keep[st], st]:
            keep[st] = k
    order = [keep[st] for st in sorted(keep)]
    if len(order) < len(station):
        print(f"collapsed {len(station)} picks onto {len(order)} stations")

    if args.anchors and args.anchors < len(order):
        sel = np.linspace(0, len(order) - 1, args.anchors).round().astype(int)
        order = [order[i] for i in sorted(set(sel.tolist()))]

    idx = [int(station[k]) for k in order]
    d = picks[order] - axis[idx]
    d = d - np.sum(d * tang[idx], axis=1, keepdims=True) * tang[idx]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(n < 1e-9):
        raise SystemExit("a pick projects onto its station along the tangent; "
                         "it cannot define a dorsal direction")
    d = d / n

    fp = hashlib.sha256(
        np.ascontiguousarray(verts, dtype=np.float64).tobytes()).hexdigest()[:16]
    stem = os.path.splitext(args.mesh)[0]
    axis_file = os.path.basename(stem) + "-axis.npy"
    np.save(os.path.join(os.path.dirname(args.mesh), axis_file), axis)

    payload = {
        "schema": "spatchcocking.dorsal_anchors/1",
        "mesh": args.mesh.replace("\\", "/"),
        "mesh_npoints": int(mesh.npoints),
        "mesh_fingerprint": fp,
        "axis_file": axis_file,
        "axis_stations": int(len(axis)),
        "anchors": [{"station": int(station[k]),
                     "direction": list(map(float, d[j])),
                     "point": list(map(float, picks[k]))}
                    for j, k in enumerate(order)],
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": f"recovered from legacy pick file {os.path.basename(args.picks)}; "
                f"no new human input",
    }
    out = f"{stem}-anchors.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"mesh        : {args.mesh}  ({mesh.npoints} vertices, fp {fp})")
    print(f"picks       : {len(picks)} from {args.picks}")
    print(f"              {d_surf.min():.1f}-{d_surf.max():.1f} um off the surface")
    print(f"axis        : {len(axis)} stations -> {axis_file}")
    print(f"anchors     : {idx}")
    print(f"              span stations {min(idx)}-{max(idx)} of 0-{len(axis)-1}, "
          f"so {min(idx) + len(axis) - 1 - max(idx)} stations are extrapolated")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
