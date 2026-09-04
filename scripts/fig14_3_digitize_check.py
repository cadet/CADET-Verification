"""
Sanity-check plot for the digitized data extracted from Gu (2015),
Fig. 14.3 "Simulation of binary frontal adsorption in inward flow RFC"
(T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
Springer, 2015, p. 199).

Overlays the digitized points (fig14_3_digitized.csv) on axes matching
the original figure (x: Dimensionless Time 0-6, y: Dimensionless
Concentration 0-1.4) so the extraction can be visually compared against
the scanned page image.
"""
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\jmbr\software\CADET-Verification\scripts\fig14_3_digitized.csv"
OUT_PATH = r"C:\Users\jmbr\software\CADET-Verification\scripts\fig14_3_digitize_check.png"

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(7, 5))

# Line showing the full digitized trace
ax.plot(df["time_dimensionless"], df["c1_dimensionless"], "-", color="tab:blue",
        linewidth=1.2, label="Curve 1 (digitized)")
ax.plot(df["time_dimensionless"], df["c2_dimensionless"], "-", color="tab:red",
        linewidth=1.2, label="Curve 2 (digitized)")

# Sparse markers on top so individual sample density/quality is visible
step = 6
ax.plot(df["time_dimensionless"][::step], df["c1_dimensionless"][::step], "o",
        color="tab:blue", markersize=2.5)
ax.plot(df["time_dimensionless"][::step], df["c2_dimensionless"][::step], "o",
        color="tab:red", markersize=2.5)

ax.set_xlim(0, 6)
ax.set_ylim(0, 1.4)
ax.set_xticks(range(0, 7))
ax.set_yticks([i / 10 for i in range(0, 15, 2)])
ax.set_xlabel("Dimensionless Time")
ax.set_ylabel("Dimensionless Concentration")
ax.set_title("Digitized Fig. 14.3 (Gu, 2015) - overlay check")
ax.legend(loc="upper right")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved check plot to {OUT_PATH}")
