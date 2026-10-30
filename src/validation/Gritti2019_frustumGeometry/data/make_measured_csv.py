# -*- coding: utf-8 -*-
"""Build the reference chromatograms of Gritti et al. (2019), Figs. 6-8.

Run this file to regenerate `Gritti2019_fig{6,7,8}_measured.csv` in this
directory from the raw exports that sit beside it. Nothing else imports it;
the validation scripts read the generated CSVs.

Source
------
`Data Peaks Fig 6-7-8.opju` is the authors' Origin project. Its worksheets
were exported to the `Cyl*.csv` and `Con*.csv` files here with Origin Viewer.
Each figure is built from one `Cyl*` worksheet, which holds the cylinder run
together with both cone runs at the time positions the published panel uses:

    Fig. 6  CylIsoAlkano.csv     isocratic, valerophenone
    Fig. 7  CylGrdAlkano.csv     gradient, valerophenone
    Fig. 8  CylGrdBombesin.csv   gradient, bombesin

Columns 1, 2 and 3 of each are the cylinder, the rho_s = 2 cone and the
rho_s = 0.5 cone. The two cone columns are the standalone runs in
`ConFWD*.csv` and `ConREV*.csv` rigidly moved in time, by -1.897/-2.889 s for
Fig. 6, -0.189/-1.027 s for Fig. 7 and +0.529/-0.126 s for Fig. 8, so that the
peaks coincide; the values are otherwise identical to the last digit, except
that Fig. 6's rho_s = 2 copy carries a 1.00046 amplitude factor. That is the
display adjustment Sec. 4.2.2 of the paper describes, so these columns are
what the published figures show and what a comparison against them must use.
`Gritti2019_fig6.md` explains what that costs: the retention-time error is
then carried by Delta mu_1 against Tables 1-3 rather than by the NRMSE.

Format of the raw exports
-------------------------
Origin wrote them with a German locale, so both the decimal separator and the
field separator are a comma: a row reads `125,000001862645,1,28830253355493E-6`
for t = 125.000001862645 s, y = 1.28830253355493e-6 AU. Fields are therefore
rejoined pairwise rather than split on commas. Missing cells are empty or `--`.
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: figure -> (source worksheet, time window of the published panel [s])
FIGURES = {
    'fig6': ('CylIsoAlkano.csv', (221.7, 247.7)),
    'fig7': ('CylGrdAlkano.csv', (269.9, 295.0)),
    'fig8': ('CylGrdBombesin.csv', (272.0, 306.0)),
}

#: worksheet columns 1, 2, 3 in order
CURVES = ('cylinder', 'cone_s2', 'cone_s05')

_INT = re.compile(r'^-?\d+$')
_FRAC = re.compile(r'^\d+(?:[Ee][+-]?\d+)?$')


def read_origin_csv(path):
    """Rows of an Origin export, as lists of float or None."""
    with open(path, encoding='utf-8-sig') as handle:
        lines = [line.rstrip('\r\n') for line in handle if line.strip()]
    rows = []
    for line in lines[3:]:            # 3 header lines: names, units, comments
        fields = line.split(',')
        values = []
        i = 0
        while i < len(fields):
            if (_INT.match(fields[i]) and i + 1 < len(fields)
                    and _FRAC.match(fields[i + 1])):
                values.append(float(fields[i] + '.' + fields[i + 1]))
                i += 2
            else:
                values.append(None)   # empty cell or '--'
                i += 1
        rows.append(values)
    return rows


def column(rows, index):
    """One (time, absorbance) series, dropping rows where it is empty."""
    pts = [(r[0], r[index]) for r in rows
           if len(r) > index and r[0] is not None and r[index] is not None]
    return (np.round(np.array([p[0] for p in pts]), 3),
            np.array([p[1] for p in pts]))


def build(figure, source, window):
    lo, hi = window
    rows = read_origin_csv(os.path.join(HERE, source))
    series = {}
    for index, name in enumerate(CURVES, start=1):
        t, y = column(rows, index)
        inside = (t >= lo) & (t <= hi)
        series[name] = (t[inside], y[inside])

    # One time column for all three, with blanks where a trace has no sample.
    grid = np.unique(np.concatenate([t for t, _ in series.values()]))
    values = {name: np.full(grid.shape, np.nan) for name in CURVES}
    for name, (t, y) in series.items():
        values[name][np.searchsorted(grid, t)] = y

    path = os.path.join(HERE, 'Gritti2019_%s_measured.csv' % figure)
    with open(path, 'w', newline='') as out:
        out.write('time_s,cylinder_AU,cone_s2_AU,cone_s05_AU\n')
        for i, t in enumerate(grid):
            cells = ['' if np.isnan(values[n][i]) else '%.10g' % values[n][i]
                     for n in CURVES]
            out.write('%.3f,%s,%s,%s\n' % (t, *cells))
    return path, grid, series


def main():
    for figure, (source, window) in FIGURES.items():
        path, grid, series = build(figure, source, window)
        peaks = '  '.join(
            '%s %.2f s' % (name, t[int(np.argmax(y))])
            for name, (t, y) in series.items())
        print('%-28s from %-22s n=%4d  t=[%.1f, %.1f] s  peaks: %s'
              % (os.path.basename(path), source, len(grid),
                 grid[0], grid[-1], peaks))


if __name__ == '__main__':
    main()
