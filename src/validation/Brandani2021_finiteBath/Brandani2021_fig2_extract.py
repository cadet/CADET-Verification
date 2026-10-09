# -*- coding: utf-8 -*-
"""
Regenerate Brandani2021_fig2_digitized.csv from the published PDF.

    S. Brandani, "Kinetics of liquid phase batch adsorption experiments",
    Adsorption 27 (2021) 353-368, https://doi.org/10.1007/s10450-020-00258-9

The article is open access (CC BY 4.0); the PDF is not kept in this
repository. Download it from the DOI above and pass its path:

    python Brandani2021_fig2_extract.py --pdf brandani2021.pdf

Fig. 2 is a vector graphic, so there is nothing to digitize in the usual
sense: page 9 draws the four curves as stroked polylines and carries no
image XObject. This reads the polyline vertices straight out of the page
content stream, which removes the pixel-quantisation error of
WebPlotDigitizer or CLAUDE/digitize_figure.py. What is left is Brandani's
own plotting resolution.

Curve identification on page 9:
  - two dashed black paths  -> Sh = 2   (combined film and pore diffusion)
  - two solid blue paths    -> Sh = inf (pore diffusion only)
and within each pair the taller one is p-xylene. This script extracts the
m-xylene, Sh = 2 curve, i.e. the lower of the two dashed paths.
"""
import argparse
import os

import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))

PAGE_INDEX = 8  # journal page 361, the page carrying Fig. 2 and Table 1

# Axis calibration, taken from the bounding boxes of the tick labels, which
# the PDF carries as text next to the plot. Both sets are equally spaced to
# within the extraction precision, so two anchors per axis suffice.
X_PT_T0, T_0 = 335.05, 0.0        # tick label '0'    on the time axis
X_PT_T1, T_1 = 536.45, 3000.0     # tick label '3000' on the time axis
Y_PT_Q0, Q_0 = 407.90, 0.0        # tick label '0'    on the Q axis
Y_PT_Q1, Q_1 = 285.30, 2000.0     # tick label '2000' on the Q axis


def curve_points(drawing):
    """Vertices of a drawing's stroked path, flattening Bezier segments to
    their end points (the curves are drawn as dense polylines, so the few
    Bezier segments present carry no extra shape)."""
    pts = []
    for item in drawing["items"]:
        if item[0] == "l":
            pts += [item[1], item[2]]
        elif item[0] == "c":
            pts += [item[1], item[4]]
    return pts


def extract(pdf_path):
    doc = pymupdf.open(pdf_path)
    page = doc[PAGE_INDEX]

    dashed = []
    for d in page.get_drawings():
        pts = curve_points(d)
        dashes = d.get("dashes")
        if len(pts) > 50 and dashes and dashes != "[] 0":
            dashed.append((d["rect"], pts))
    if len(dashed) != 2:
        raise RuntimeError(
            f"expected the two dashed Sh=2 curves on page {PAGE_INDEX + 1}, "
            f"found {len(dashed)}")

    # Larger y0 in PDF coordinates = less tall on the page = m-xylene, whose
    # uptake levels off near 850 mol/m^3 against p-xylene's 1800.
    dashed.sort(key=lambda rp: rp[0].y0)
    _, pts = dashed[-1]

    t = (np.array([p.x for p in pts]) - X_PT_T0) * (T_1 - T_0) / (X_PT_T1 - X_PT_T0)
    q = (np.array([p.y for p in pts]) - Y_PT_Q0) * (Q_1 - Q_0) / (Y_PT_Q1 - Y_PT_Q0)

    # The chart clips the series at the right edge of the plot box, but
    # get_drawings() reports the whole path, so cut the clipped-away tail.
    inside = (t >= -1.0) & (t <= T_1 + 0.5)
    t, q = t[inside], q[inside]

    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]

    # Every 'l' item contributes both of its end points, so each interior
    # vertex appears twice. Drop the repeats to keep the time axis strictly
    # increasing for interpolation.
    keep = np.concatenate(([True], np.diff(t) > 0.0))
    return t[keep], q[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, help="path to the Brandani (2021) PDF")
    ap.add_argument("--output", default=os.path.join(HERE, "Brandani2021_fig2_digitized.csv"))
    args = ap.parse_args()

    t, q = extract(args.pdf)

    if not np.all(np.diff(q) >= -1.0):
        raise RuntimeError("extracted uptake curve is not monotone; check the calibration")

    np.savetxt(args.output, np.column_stack([t, q]), delimiter=",",
               header="time_s,Q_mol_per_m3", comments="", fmt="%.6g")
    print(f"{len(t)} points, t = {t.min():.1f} .. {t.max():.1f} s, "
          f"Q = {q.min():.1f} .. {q.max():.1f} mol/m^3")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
