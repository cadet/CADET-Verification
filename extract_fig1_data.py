# -*- coding: utf-8 -*-
"""
Extract chromatogram curves (proteins A-F, left OD axis; salt, right mol/L axis)
from the paper's Figure 1 PNG by color matching.

Calibration:
  - x axis via vertical gridlines at t = 0, 5000, 10000 s
  - left y axis via horizontal gridlines at OD = 0, 1, 2 AU/cm
  - salt axis via known program values: initial plateau 0.04 mol/L, final 1.04 mol/L

Output: chromops_fig1_extracted/<name>.csv with columns time_s, value
"""
import numpy as np
from PIL import Image

IMG = r"c:\Users\jmbr\AppData\Roaming\Code\agentSessionData\61967a37-9cfd-4a2a-aa0b-b21440d057b9\attachments\650abe4a-e0bd-4c6e-804c-dc4cc48989ac\Pasted Image.png"

img = np.asarray(Image.open(IMG).convert("RGB")).astype(int)
H, W, _ = img.shape
print("image size:", W, "x", H)

# ---------- find plot frame (long dark lines) ----------
dark = (img.max(axis=2) < 120)
col_runs = dark.sum(axis=0)
row_runs = dark.sum(axis=1)
# frame columns/rows have runs comparable to plot height/width
cols = np.where(col_runs > 0.5 * H)[0]
rows = np.where(row_runs > 0.5 * W)[0]
x0, x1 = cols.min(), cols.max()
y0, y1 = rows.min(), rows.max()
print("frame:", x0, x1, y0, y1)

inner = img[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
iy0, ix0 = y0 + 2, x0 + 2

# ---------- gridlines (light gray) ----------
gray = (np.abs(inner[:, :, 0] - inner[:, :, 1]) < 8) & \
       (np.abs(inner[:, :, 1] - inner[:, :, 2]) < 8) & \
       (inner.max(axis=2) > 200) & (inner.max(axis=2) < 245)
gcol = gray.sum(axis=0)
grow = gray.sum(axis=1)
vgrid = np.where(gcol > 0.7 * inner.shape[0])[0]
hgrid = np.where(grow > 0.7 * inner.shape[1])[0]

def cluster(idx):
    groups, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 3:
            cur.append(v)
        else:
            groups.append(int(np.mean(cur))); cur = [v]
    groups.append(int(np.mean(cur)))
    return groups

# ---------- calibrate via tick marks outside the frame ----------
def tick_centers(mask_1d_idx):
    return cluster(mask_1d_idx)

# x ticks: dark pixels just below the bottom frame line
band = dark[y1 + 2:y1 + 8, :].sum(axis=0)
xt = cluster(np.where(band >= 3)[0])
print("x ticks (px):", xt)
# y ticks left: dark pixels just left of frame
bandl = dark[:, x0 - 8:x0 - 2].sum(axis=1)
ytl = cluster(np.where(bandl >= 3)[0])
print("left y ticks (px):", ytl)
# y ticks right: dark pixels just right of frame
bandr = dark[:, x1 + 2:x1 + 8].sum(axis=1)
ytr = cluster(np.where(bandr >= 3)[0])
print("right y ticks (px):", ytr)

assert len(xt) == 3, f"x ticks: {xt}"       # 0, 5000, 10000
assert len(ytl) == 3, f"left ticks: {ytl}"  # 2, 1, 0 (top to bottom)
assert len(ytr) == 3, f"right ticks: {ytr}" # 1.0, 0.5, 0.0 (top to bottom)

tx = np.polyfit(xt, [0.0, 5000.0, 10000.0], 1)
ty = np.polyfit(ytl, [2.0, 1.0, 0.0], 1)
tsalt = np.polyfit(ytr, [1.0, 0.5, 0.0], 1)
px2t = lambda px: np.polyval(tx, px)
px2od = lambda py: np.polyval(ty, py)
px2salt = lambda py: np.polyval(tsalt, py)

# ---------- curve colors ----------
# collect saturated pixels inside the plot area, excluding the legend box region
r, g, b = inner[:, :, 0], inner[:, :, 1], inner[:, :, 2]
mx, mn = inner.max(axis=2), inner.min(axis=2)
colorful = (mx - mn > 40) & (mx > 80)
ys, xs = np.where(colorful)
pix = inner[ys, xs]
key = (pix // 24)
uniq, counts = np.unique(key, axis=0, return_counts=True)
order = np.argsort(-counts)
clusters = []
for k in order[:20]:
    c = uniq[k] * 24 + 12
    # merge near-duplicates
    if any(np.abs(np.array(cc) - c).max() < 40 for cc, _ in clusters):
        continue
    clusters.append((tuple(c), counts[k]))
print("color clusters:", clusters[:8])

def extract(color, tol=25, exclude_legend=True):
    c = np.array(color)
    d = np.abs(img - c[None, None, :]).max(axis=2)
    mask = d < tol
    # restrict to plot area
    m = np.zeros_like(mask); m[y0 + 2:y1 - 1, x0 + 2:x1 - 1] = mask[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
    if exclude_legend:  # legend occupies upper-left corner
        m[:int(y0 + 0.45 * (y1 - y0)), :int(x0 + 0.22 * (x1 - x0))] = False
    t_list, y_list = [], []
    for px in range(x0 + 2, x1 - 1):
        rows_ = np.where(m[:, px])[0]
        if rows_.size == 0:
            continue
        t_list.append(px2t(px))
        y_list.append(px2od(rows_.mean()))
    return np.array(t_list), np.array(y_list)

# extract with explicit palette (core line colors detected above; legend order)
palette = {
    "A": (12, 108, 180),   # blue
    "B": (228, 156, 12),   # yellow
    "C": (12, 156, 108),   # green
    "D": (204, 132, 156),  # pink
    "E": (84, 180, 228),   # light blue
    "F": (204, 84, 12),    # dark orange
}
# tight tolerance: only pure core-line pixels, so anti-aliased halos of one
# curve (blends toward white) cannot be attributed to another curve's color
assigned = {}
for name, c in palette.items():
    t, y = extract(c, tol=12)
    j = np.argmax(y)
    assigned[name] = {"color": c, "t": t, "y": y}
    print(f"{name}: n={t.size}, peak t={t[j]:.0f}s, od={y[j]:.3f}")

# ---------- salt (gray dashed, low saturation, medium gray) ----------
gray_curve = (np.abs(img[:, :, 0] - img[:, :, 1]) < 12) & (np.abs(img[:, :, 1] - img[:, :, 2]) < 12) & \
             (img.max(axis=2) > 130) & (img.max(axis=2) < 200)
m = np.zeros_like(gray_curve); m[y0 + 2:y1 - 1, x0 + 2:x1 - 1] = gray_curve[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
m[:int(y0 + 0.45 * (y1 - y0)), :int(x0 + 0.22 * (x1 - x0))] = False  # legend
ts, ys_ = [], []
for px in range(x0 + 2, x1 - 1):
    rows_ = np.where(m[:, px])[0]
    if rows_.size == 0:
        continue
    ts.append(px2t(px)); ys_.append(rows_.mean())
ts, ys_ = np.array(ts), np.array(ys_)
salt = px2salt(ys_)
# sanity: known program plateaus
chk1 = salt[(ts > 1300) & (ts < 1900)].mean()
chk2 = salt[ts > 10150].mean()
print(f"salt sanity: plateau ~0.04 -> {chk1:.3f}, final ~1.04 -> {chk2:.3f}")

# ---------- save ----------
import os
outdir = "chromops_fig1_extracted"
os.makedirs(outdir, exist_ok=True)
for name in "ABCDEF":
    res = assigned[name]
    np.savetxt(os.path.join(outdir, f"{name}.csv"),
               np.column_stack([res["t"], res["y"]]), delimiter=",",
               header="time_s,OD_AU_per_cm", comments="")
np.savetxt(os.path.join(outdir, "salt.csv"),
           np.column_stack([ts, salt]), delimiter=",",
           header="time_s,salt_mol_per_L", comments="")
print("saved CSVs to", outdir)

# quick sanity: protein A area vs injected mass (497.16 mol*s/m^3 * w/1000 = 5801.8 AU*s/cm)
w = 11.67
tA, yA = assigned["A"]["t"], assigned["A"]["y"]
print("A: peak", yA.max(), "area", np.trapz(yA, tA), "(paper model: ~2.42 peak, 5802 area)")

# verification overlay
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(11, 6))
for name in "ABCDEF":
    res = assigned[name]
    ax.plot(res["t"], res["y"], label=name)
ax2 = ax.twinx()
ax2.plot(ts, salt, "k--", alpha=0.5, label="salt")
ax2.set_ylabel("salt (mol/L)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("OD (AU/cm)"); ax.legend()
fig.savefig(os.path.join(outdir, "extracted_overlay.png"), dpi=130)
print("saved overlay plot")
