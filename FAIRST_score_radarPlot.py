import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data
# -----------------------------
categories = [
    "Findability",
    "Accessibility",
    "Interoperability\nAnd\nReusability",
    "Scientific\nBasis",
    "Technical\nBasis",
]

series = {
    "CADET v4":   [1.5, 13.0 / 3.0, 8.0 / 3.0, 10.0 / 3.0, 10.0 / 5.0],
    "CADET v5":   [4.5, 5.0, 9.0 / 3.0, 11.0 / 3.0, 19.0 / 5.0],
    "CADET v6":   [4.5, 5.0, 12.0 / 3.0, 11.0 / 3.0, 23.0 / 5.0],
}

colors = {
    "CADET v4": "#e85b6c",
    "CADET v5": "#9fb0d9",
    "CADET v6": "#a8c64a",
}

# -----------------------------
# Radar plot setup
# -----------------------------
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig = plt.figure(figsize=(11, 7), dpi=160)
ax = plt.subplot(111, polar=True)

# Put first axis at the top and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=18)

# Radial scale
r_max = 5
ax.set_ylim(0, r_max)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels([])  # hide radial tick labels to match the style

# Grid / spine styling
ax.grid(color="#999999", alpha=0.35, linewidth=0.7)
ax.spines["polar"].set_visible(False)

# Optional background color
fig.patch.set_facecolor("#f2f2f2")
ax.set_facecolor("#f2f2f2")

# -----------------------------
# Plot each series
# -----------------------------
for name, values in series.items():
    values_closed = values + values[:1]
    ax.plot(
        angles,
        values_closed,
        color=colors[name],
        linewidth=3,
        solid_capstyle="round",
        label=name,
    )
    # Uncomment to add fill
    # ax.fill(angles, values_closed, color=colors[name], alpha=0.08)

# -----------------------------
# Legend
# -----------------------------
legend = ax.legend(
    loc="upper left",
    bbox_to_anchor=(-0.28, 1.22),
    frameon=True,
    fancybox=False,
    edgecolor="gray",
    facecolor="white",
    fontsize=18,
    handlelength=2.0,
    handletextpad=0.6,
)

# -----------------------------
# Layout / save / show
# -----------------------------
plt.tight_layout()
# plt.savefig("FAIRST.png", bbox_inches="tight", dpi=300)
# ax.set_title("FAIR ST", fontsize=22, pad=25)
plt.show()