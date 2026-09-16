"""

Plots performance benchmarks from the convergence json files that the
verification scripts write, in the style of the benchmark figures of

    J.M. Breuer, S. Leweke, J. Schmoelder, G. Gassner, E. von Lieres,
    "Spatial discontinuous Galerkin spectral element method for a family of
    chromatography models in CADET", Computers and Chemical Engineering 177
    (2023) 108340, doi:10.1016/j.compchemeng.2023.108340,

that is the error of the column outlet over the compute time (Figs. 3 to 5 and
Fig. 7) and over the degrees of freedom (Figs. 3 to 5), and the largest negative
concentration value over the degrees of freedom (Fig. 8). One line per spatial
method, log-log on square axes, which is what makes the methods comparable at
equal accuracy.

Run this file. It plots whatever INPUT points at, a convergence json or a folder
holding several, and writes one figure per metric next to it. Paths given on the
command line override INPUT:

    python src/utility/plotting.py
    python src/utility/plotting.py output/chromatography
    python src/utility/plotting.py output/chromatography/convergence_radial_GRM_reqSMA_4comp_benchmark1.json

plot_metric is also importable on its own, for a metric pair the figures below
do not cover.

"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# %% What to plot -- edit these and run the file

_REPO_ROOT_ = Path(__file__).resolve().parent.parent.parent

# A convergence json, or a folder holding several of them.
INPUT = _REPO_ROOT_ / "output" / "paper_performance"

# Where the figures go; None writes them next to the json.
OUTPUT_PATH = None

# Solution part to plot, and whether to open the figures after writing them.
SECTION = "outlet"
SHOW = True

DPI = 300

# The publication names its figures in the caption and carries no title in the
# figure itself. Set to True to put the setting name above the axes, which helps
# when a run produces many of them.
TITLE = False


# %% Style of the published figures
#
# Square axes, dashed lines through filled markers, circles for the DG series
# and squares for FV, a light grey grid on the decades, and a boxed legend in
# the upper right.
FIGSIZE = (6.0, 6.0)
LINEWIDTH = 1.8
MARKERSIZE = 7
LINESTYLE = "--"
GRID_KWARGS = dict(which="major", color="0.8", linestyle="-", linewidth=0.8)
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# The DG series take the default colour cycle in ascending polynomial degree,
# which is what the publication does: its Fig. 5 shows P1 to P5 and so gives P3
# the third colour, while its Fig. 7 starts at P3 and gives it the first. FV is
# the dark magenta of both figures.
DG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2"]
FV_COLOR = "#8b008b"
DG_MARKER = "o"
FV_MARKER = "s"

# One entry per figure: file suffix, x quantity, y quantity, axis labels, and
# whether the y values are magnitudes of negative numbers.
FIGURES = (
    dict(suffix="performance_compute", x_key="Sim. time", y_key="Max. error",
         xlabel="Compute time in seconds",
         ylabel=r"$L^\infty$ error in mol $/m^3$", absolute=False),
    dict(suffix="performance_dof", x_key="DoF", y_key="Max. error",
         xlabel="Degrees of freedom",
         ylabel=r"$L^\infty$ error in mol $/m^3$", absolute=False),
    dict(suffix="negative_values_dof", x_key="DoF", y_key="Min. value",
         xlabel="Degrees of freedom",
         ylabel=r"Min. value in mol $/m^3$", absolute=True),
    )


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


# %% Benchmark figures


def setting_name(json_path):
    """Name of the setting a convergence json holds."""

    return Path(json_path).stem.replace("convergence_", "")


def method_label(name):
    """Readable name of one spatial method, as the published legends have it.

    The group names of the convergence json carry the discretization of both the
    bulk and the particles, such as DG_P3parP3 or FVWENO3, of which the legends
    of the publication keep the scheme and the polynomial degree.
    """

    if name.upper().startswith("FV"):
        return "FV"

    degree = re.search(r"_P(\d+)", name)

    return "DG P" + degree.group(1) if degree else name


def method_order(item):
    """Sort key putting the DG series first, by degree, and FV last."""

    name = item[0]

    if name.upper().startswith("FV"):
        return (1, 0)

    degree = re.search(r"_P(\d+)", name)

    return (0, int(degree.group(1)) if degree else 0)


def method_style(name, dg_index):
    """Colour and marker of one series, following the published figures."""

    if name.upper().startswith("FV"):
        return FV_COLOR, FV_MARKER

    return DG_COLORS[dg_index % len(DG_COLORS)], DG_MARKER


def absolute_values(method, y_key):
    """A copy of one method with the magnitudes of one quantity.

    The largest negative value is negative by definition and the published
    Fig. 8 draws it on a logarithmic axis, so its magnitude is what is plotted.
    """

    converted = {}

    for section, data in method.items():
        if not isinstance(data, dict):
            continue
        converted[section] = {
            key: (np.abs(np.asarray(values, dtype=float)).tolist()
                  if key == y_key else values)
            for key, values in data.items()
            }

    return converted


def plot_convergence_file(json_path, output_path=None, section=SECTION,
                          dpi=DPI, figures=FIGURES, title=TITLE):
    """Plot every benchmark figure of one convergence json.

    Figures whose quantities the json does not carry are skipped, so that a
    study without a Min. value column simply yields no Fig. 8.
    """

    json_path = Path(json_path)
    output_path = Path(output_path) if output_path else json_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    with open(json_path) as handle:
        methods = json.load(handle)["convergence"]

    methods = dict(sorted(
        ((name, method) for name, method in methods.items()
         if isinstance(method, dict) and section in method),
        key=method_order,
        ))

    name = setting_name(json_path)
    written = []

    for figure in figures:

        plotted = False
        fig, ax = plt.subplots(figsize=FIGSIZE)
        dg_index = 0

        for method_name, method in methods.items():

            data = method[section]
            if figure["x_key"] not in data or figure["y_key"] not in data:
                continue

            color, marker = method_style(method_name, dg_index)
            if not method_name.upper().startswith("FV"):
                dg_index += 1

            plot_metric(
                {method_label(method_name):
                    absolute_values(method, figure["y_key"]) if figure["absolute"]
                    else method},
                figure["x_key"], figure["y_key"], section=section, ax=ax,
                xscale="log", yscale="log",
                xlabel=figure["xlabel"], ylabel=figure["ylabel"],
                grid=False, legend=False,
                color=color, marker=marker, linestyle=LINESTYLE,
                linewidth=LINEWIDTH, markersize=MARKERSIZE,
                )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.grid(True, **GRID_KWARGS)
        ax.set_xlabel(figure["xlabel"], fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(figure["ylabel"], fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
        if title:
            ax.set_title(name, fontsize=LABEL_FONTSIZE)

        # Square axes, as the published figures have them. set_box_aspect fixes
        # the box itself, so the surrounding margins may still differ.
        ax.set_box_aspect(1)
        fig.tight_layout()

        target = output_path / (name + "_" + figure["suffix"] + ".png")
        fig.savefig(target, dpi=dpi)
        written.append(target)
        print("Figure written to " + str(target))

    if not written:
        print("Nothing to plot in " + str(json_path) + ": no method carries the "
              "section " + repr(section) + " with the expected quantities.")

    return written


def collect_convergence_files(target):
    """The convergence jsons of a file or a directory."""

    target = Path(target)

    if target.is_dir():
        return sorted(target.glob("convergence_*.json"))

    return [target]


def main(paths=None, output_path=OUTPUT_PATH, section=SECTION, dpi=DPI,
         title=TITLE, show=SHOW):
    """Plot every convergence json of the given files or folders."""

    paths = paths or [INPUT]

    written = []
    for path in paths:
        for json_path in collect_convergence_files(path):
            written += plot_convergence_file(
                json_path, output_path=output_path, section=section, dpi=dpi,
                title=title,
                )

    print(str(len(written)) + " figures written.")

    if show and written:
        plt.show()
    else:
        plt.close("all")

    return written


if __name__ == "__main__":
    main(sys.argv[1:] or None)
