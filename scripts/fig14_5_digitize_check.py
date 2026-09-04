"""
Sanity-check plot for the digitized data extracted from Gu (2015),
Fig. 14.5 "Binary elution with an inert mobile phase in inward flow RFC"
(T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
Springer, 2015, p. 201) - a screenshot of the "Chromulator RateRFC Model
Simulator" GUI.

Reproduces the paper figure's appearance (red Component 1: narrow early
peak ~0.58 near tau~2.4; black Component 2: broader later peak ~0.15-0.16
near tau~5.4 with a long decaying tail out past tau=16) from the digitized
CSV (fig14_5_digitized.csv) so the extraction can be visually compared
against the scanned GUI screenshot.
"""
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\jmbr\software\CADET-Verification\scripts\fig14_5_digitized.csv"
OUT_PATH = r"C:\Users\jmbr\software\CADET-Verification\scripts\fig14_5_digitize_check.png"

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(df["tau"], df["c1_dimensionless"], "-", color="red",
        linewidth=1.4, label="Component 1 (digitized)")
ax.plot(df["tau"], df["c2_dimensionless"], "-", color="black",
        linewidth=1.4, label="Component 2 (digitized)")

ax.set_xlim(0, 16)
ax.set_ylim(0, 0.6)
ax.set_xticks(range(0, 17))
ax.set_yticks([i / 10 for i in range(0, 7)])
ax.set_xlabel("Dimensionless Time")
ax.set_ylabel("Dimensionless Concentration")
ax.set_title("Digitized Fig. 14.5 (Gu, 2015) - overlay check")
ax.legend(loc="upper right")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved check plot to {OUT_PATH}")

c1_peak_idx = df["c1_dimensionless"].idxmax()
c2_peak_idx = df["c2_dimensionless"].idxmax()
print(f"Component 1 peak: tau={df['tau'][c1_peak_idx]:.3f}, "
      f"C={df['c1_dimensionless'][c1_peak_idx]:.4f}")
print(f"Component 2 peak: tau={df['tau'][c2_peak_idx]:.3f}, "
      f"C={df['c2_dimensionless'][c2_peak_idx]:.4f}")
print(f"Component 2 value at tau=16: "
      f"{df['c2_dimensionless'].iloc[-1]:.5f}")
