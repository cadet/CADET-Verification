import json

import numpy as np
import matplotlib.pyplot as plt


def plot_metric(
    methods,
    x_key,
    y_key,
    *,
    section="outlet",
    ax=None,
    **kwargs,
):
    """
    Plot one convergence metric against another for one or more methods.

    Parameters
    ----------
    methods : dict
        Dictionary such as data["convergence"].

        Example:
            {
                "FVWENO2": {...},
                "DG_P3": {...},
                ...
            }

    x_key : str
        Quantity for x-axis (e.g. "Sim. time").

    y_key : str
        Quantity for y-axis (e.g. "Max. error").

    section : str, default="outlet"
        Subgroup to plot.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Other Parameters
    ----------------
    title : str
    xlabel : str
    ylabel : str
    figsize : tuple, default=(6,4)
    xscale : {"linear","log"}, default="linear"
    yscale : {"linear","log"}, default="linear"
    xlim : tuple
    ylim : tuple
    grid : bool, default=True
    grid_kwargs : dict
    font_scale : float, default=1.0
    linewidth : float, default=2.5
    marker : str, default="o"
    markersize : float, default=6
    linestyle : str, default="-"
    legend : bool, default=True

    Any remaining kwargs are forwarded to matplotlib.axes.Axes.plot().

    Returns
    -------
    fig, ax
    """

    figsize = kwargs.pop("figsize", (6, 4))

    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", x_key)
    ylabel = kwargs.pop("ylabel", y_key)

    xscale = kwargs.pop("xscale", "linear")
    yscale = kwargs.pop("yscale", "linear")

    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)

    grid = kwargs.pop("grid", True)
    grid_kwargs = kwargs.pop(
        "grid_kwargs",
        {"which": "both", "linestyle": "--", "alpha": 0.35},
    )

    font_scale = kwargs.pop("font_scale", 1.0)

    linewidth = kwargs.pop("linewidth", 2.5)
    marker = kwargs.pop("marker", "o")
    markersize = kwargs.pop("markersize", 6)
    linestyle = kwargs.pop("linestyle", "-")

    legend = kwargs.pop("legend", True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for method_name, method in methods.items():

        if section not in method:
            continue

        data = method[section]

        if x_key not in data or y_key not in data:
            continue

        x = np.asarray(data[x_key], dtype=float)
        y = np.asarray(data[y_key], dtype=float)

        # sort by x-value
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        ax.plot(
            x,
            y,
            label=method_name,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            linestyle=linestyle,
            **kwargs,
        )

    ax.set_xlabel(xlabel, fontsize=12 * font_scale)
    ax.set_ylabel(ylabel, fontsize=12 * font_scale)

    if title is not None:
        ax.set_title(title, fontsize=14 * font_scale)

    ax.tick_params(axis="both", labelsize=11 * font_scale)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(**grid_kwargs)

    if legend:
        ax.legend(fontsize=11 * font_scale)

    fig.tight_layout()

    return fig, ax

file_name = r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\transport\convergence_COL1D_frustumTransport_1comp_benchmark1.json"

with open(file_name) as f:
    data = json.load(f)

plot_metric(
    data["convergence"],
    "DoF",
    "Max. error",
    xscale="log",
    yscale="log",
    title=r"$L^\infty$ error vs simulation time",
)

plt.savefig(r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\transport\COL1D_frustumTransport_performance_dof_all.png", dpi=300)
plt.show()
