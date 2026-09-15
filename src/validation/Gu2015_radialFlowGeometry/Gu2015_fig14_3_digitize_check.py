"""
Overlay plot for the data digitized from Gu (2015), Fig. 14.3, "Simulation
of binary frontal adsorption in inward flow RFC" (T. Gu, "Mathematical
Modeling and Scale-Up of Liquid Chromatography", Springer, 2015, p. 199).

The digitized points are drawn on the original figure's axes, x from 0 to 6
and y from 0 to 1.4, so that the extraction can be held next to the scanned
page.
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "Gu2015_fig14_3_digitized.csv")
OUT_PATH = os.path.join(HERE, "Gu2015_fig14_3_digitize_check.png")

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(7, 5))

# The full digitized trace
ax.plot(df["time_dimensionless"], df["c1_dimensionless"], "-", color="tab:blue",
        linewidth=1.2, label="Curve 1 (digitized)")
ax.plot(df["time_dimensionless"], df["c2_dimensionless"], "-", color="tab:red",
        linewidth=1.2, label="Curve 2 (digitized)")

# Sparse markers on top, so that the sampling density stays visible
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
