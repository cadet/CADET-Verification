# -*- coding: utf-8 -*-
"""
Extract the chromatogram curves of Figure 1 of

    Meyer et al., 2026, Computers and Chemical Engineering,
    "ChromOps.jl: High-order simulation and discrete forward sensitivity
     analysis for chromatography models"

Figure 1 is a vector graphic, so the curves are read directly from the PDF
content stream (PyMuPDF) instead of being digitized from a raster image. Each
curve is a polyline whose stroke colour identifies its component; the axes are
calibrated on the tick marks, which are themselves vector line segments.

The calibration is exact: the recovered salt program reproduces the plateau
values of Eqs. (39)/(40) to four digits (0.2400 and 1.0400 mol/L).

Usage:  python Meyer2026_fig1_extract.py [path/to/chromops.pdf] [page_index]

Output: Meyer2026_fig1_digitized/<A..F>.csv  columns time_s, OD_AU_per_cm
        Meyer2026_fig1_digitized/salt.csv    columns time_s, salt_mol_per_L
"""
import os
import sys

import numpy as np
import pymupdf

PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(r"~\Desktop\chromops.pdf")
PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # zero-based; Figure 1 is on p. 11
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meyer2026_fig1_digitized")

# Stroke colours of the seven curves (matplotlib "colorblind"/Okabe-Ito palette),
# in the legend order of Figure 1.
PALETTE = {
    "A":    (0.0000000, 0.4470590, 0.6980390),
    "B":    (0.9019610, 0.6235290, 0.0000000),
    "C":    (0.0000000, 0.6196080, 0.4509800),
    "D":    (0.8000000, 0.4745100, 0.6549020),
    "E":    (0.3372550, 0.7058820, 0.9137250),
    "F":    (0.8352940, 0.3686270, 0.0000000),
    "salt": (0.5019610, 0.5019610, 0.5019610),
}

page = pymupdf.open(PDF)[PAGE]
drawings = page.get_drawings()


def black_segments():
    """Axis frame, gridlines and tick marks: the only black stroked lines."""
    segs = []
    for g in drawings:
        if g.get("color") == (0.0, 0.0, 0.0) and g["type"] == "s":
            for it in g["items"]:
                if it[0] == "l":
                    segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
    return segs


segs = black_segments()
if not segs:
    raise RuntimeError(f"no vector axes found on page index {PAGE} of {PDF}")

# The plot frame is the bounding box of all black segments.
xs = [v for s in segs for v in (s[0], s[2])]
ys = [v for s in segs for v in (s[1], s[3])]
fx0, fx1, fy0, fy1 = min(xs), max(xs), min(ys), max(ys)


def cluster(vals, tol=1.0):
    vals = sorted(vals)
    out, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            out.append(float(np.mean(cur)))
            cur = [v]
    out.append(float(np.mean(cur)))
    return out


# Tick marks are the short segments (a few points long); the frame lines and the
# gridlines span the whole plot. Vertical short segments are x ticks; horizontal
# ones are y ticks, on the left (OD) or right (salt) axis.
short = [s for s in segs if abs(s[2] - s[0]) + abs(s[3] - s[1]) < 8.0]
xmid = 0.5 * (fx0 + fx1)
xticks = cluster([s[0] for s in short if abs(s[0] - s[2]) < 0.1])
ylticks = cluster([s[1] for s in short if abs(s[1] - s[3]) < 0.1 and s[0] < xmid])
yrticks = cluster([s[1] for s in short if abs(s[1] - s[3]) < 0.1 and s[0] > xmid])

assert len(xticks) == 3, xticks    # 0, 5000, 10000 s
assert len(ylticks) == 3, ylticks  # OD 2, 1, 0 (top to bottom)
assert len(yrticks) == 3, yrticks  # salt 1.0, 0.5, 0.0 (top to bottom)

t_of_x = np.poly1d(np.polyfit(xticks, [0.0, 5000.0, 10000.0], 1))
od_of_y = np.poly1d(np.polyfit(ylticks, [2.0, 1.0, 0.0], 1))
salt_of_y = np.poly1d(np.polyfit(yrticks, [1.0, 0.5, 0.0], 1))

# The legend sits in the upper-left corner and repeats every curve colour as a
# short sample line; those samples must not enter the data.
legend = None
for g in drawings:
    if g.get("fill") == (1.0, 1.0, 1.0) and g.get("color") == (0.0, 0.0, 0.0):
        legend = g["rect"]
if legend is None:
    raise RuntimeError("legend box not found")

os.makedirs(OUTDIR, exist_ok=True)
for name, color in PALETTE.items():
    pts = []
    for g in drawings:
        c = g.get("color")
        if c is None or max(abs(a - b) for a, b in zip(c, color)) > 1e-3:
            continue
        for it in g["items"]:
            if it[0] != "l":
                continue
            for p in it[1:]:
                if not legend.contains(p):
                    pts.append((p.x, p.y))
    if not pts:
        raise RuntimeError(f"no polyline found for curve {name}")
    a = np.array(pts)
    t = t_of_x(a[:, 0])
    y = (salt_of_y if name == "salt" else od_of_y)(a[:, 1])
    # Vertices are shared between consecutive segments; average duplicates.
    tu, inv = np.unique(np.round(t, 3), return_inverse=True)
    yu = np.bincount(inv, weights=y) / np.bincount(inv)
    np.savetxt(os.path.join(OUTDIR, f"{name}.csv"), np.column_stack([tu, yu]),
               delimiter=",", comments="",
               header="time_s,salt_mol_per_L" if name == "salt" else "time_s,OD_AU_per_cm")
    print(f"{name:>4}: n={tu.size:4d}  t=[{tu.min():.0f}, {tu.max():.0f}] s  "
          f"peak {yu.max():.4f} at t={tu[yu.argmax()]:.0f} s")

salt = np.genfromtxt(os.path.join(OUTDIR, "salt.csv"), delimiter=",", skip_header=1)
print(f"\ncalibration check: salt hold plateau = "
      f"{salt[(salt[:, 0] > 2300) & (salt[:, 0] < 2700), 1].mean():.4f} mol/L (Eq. 39/40: 0.24), "
      f"final = {salt[salt[:, 0] > 10250, 1].mean():.4f} mol/L (Eq. 39/40: 1.04)")
print(f"saved to {OUTDIR}/")
