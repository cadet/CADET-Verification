# -*- coding: utf-8 -*-
"""Assemble the LaTeX validation tables from the case studies' metric JSONs.

Each of the six case studies under ``src/validation`` writes a
``<case_id>_metrics.json`` next to its plots (see
``src/validation/validation_metrics.dump_metrics``). This script reads those
files back and emits the paper's validation tables, so the numbers never have
to be transcribed by hand.

The two reference sources get a table of their own, because the meaning of
their columns differs. Gritti et al. (2019) tabulate measured moments, so
Delta mu_1 and Delta mu_2 there are errors against the paper's own Tables 1-3
and the rows are the frustum flow modes. Gu (2015) tabulates nothing, so the
moments are compared against the digitized chromatogram and the rows are
individual components of a multi-component separation.

"""

import json
import os
import sys

# The cylindrical column (rho_s = 1) that every Gritti case study also
# simulates is deliberately NOT listed below. It is a reference and
# calibration case rather than one of the geometries under validation --
# Gritti2019_fig8.py calibrates its dispersion coefficient on the cylinder
# and reuses it unchanged for both cones -- so its numbers stay in the
# console output and in the metric JSONs but are not reported in the tables.
# A case-study name is given as the sequence of lines it is broken into. Each
# name already spans as many rows as the case has curves, so setting the line
# breaks by hand lets the name use that vertical space instead of forcing the
# first column as wide as the whole name on one line. Keep the number of lines
# at or below the number of curves in the case.
GRITTI_CASES = [
    ('Gritti2019_fig6', ('Isocratic', 'valerophenone'), [
        ('cone_s05', r'$\rho_s=0.5$'),
        ('cone_s2', r'$\rho_s=2$'),
    ]),
    ('Gritti2019_fig7', ('Gradient', 'valerophenone'), [
        ('cone_s05', r'$\rho_s=0.5$'),
        ('cone_s2', r'$\rho_s=2$'),
    ]),
    ('Gritti2019_fig8', ('Gradient', 'Bombesin'), [
        ('cone_s05', r'$\rho_s=0.5$'),
        ('cone_s2', r'$\rho_s=2$'),
    ]),
]

GU_CASES = [
    ('Gu2015_fig14_3', ('Binary frontal', 'adsorption'), [
        ('component_1', 'Component 1'),
        ('component_2', 'Component 2'),
    ]),
    ('Gu2015_fig14_5', ('Binary', 'elution'), [
        ('component_1', 'Component 1'),
        ('component_2', 'Component 2'),
    ]),
    ('Gu2015_fig14_6', ('Affinity', 'displacement'), [
        ('protein', 'Protein'),
        ('soluble_ligand', 'Soluble ligand'),
        ('complex', 'Complex'),
    ]),
]

#: One entry per emitted table: which case studies it holds, what the second
#: column is called, the caption and the label, and the file it is written to.
TABLES = [
    dict(
        name='gritti',
        cases=GRITTI_CASES,
        row_header='Flow mode',
        filename='validation_metrics_table_gritti.tex',
        label='tab:gritti2019_numerical_errors',
        caption=(
            r'Approximation errors of the simulated chromatograms of the conical '
            r'(frustum) column with respect to the experimental reference data of '
            r'Gritti et al. $\Delta \mu_1$ and $\Delta \mu_2$ are the relative '
            r'errors of the first and the second central moment against the '
            r"moments measured in the paper's own Tables 1--3; the NRMSE is taken "
            r'against the digitized chromatogram and normalised by its peak '
            r'height.'),
        # Appended to the caption only when the mass balance column is shown.
        caption_mass_balance=(
            r' The mass balance compares the simulated outlet integral with the '
            r'analytically known injected mass and verifies the solver rather '
            r'than validating against reference data.'),
    ),
    dict(
        name='gu',
        cases=GU_CASES,
        row_header='Component',
        filename='validation_metrics_table_gu.tex',
        label='tab:gu_numerical_errors',
        caption=(
            r'Approximation errors of the simulated chromatograms of the radial '
            r'flow column with respect to the reference data of Gu (2015). That '
            r'reference tabulates no moments, so $\Delta \mu_1$ and '
            r'$\Delta \mu_2$ -- the relative errors of the first and the second '
            r'central moment -- are taken against the digitized chromatogram, as '
            r'is the NRMSE, which is normalised by its peak height. For the '
            r'breakthrough curves, which approach a plateau instead of returning '
            r'to the baseline, the moments are those of the underlying residence '
            r'time distribution.'),
        # Appended to the caption only when the mass balance column is shown.
        caption_mass_balance=(
            r' The mass balance compares the simulated outlet integral with the '
            r'injected mass and verifies the solver rather than validating '
            r'against reference data.'),
    ),
]

#: Footnote marker attached to a Delta mu_2 entry that is not a mu_2 at all.
PEAK_HEIGHT_MARKER = r'\textsuperscript{a}'
#: Footnote marker for a mass balance that carries a physical residual on top
#: of the solver's own conservation error.
INEXACT_MASS_MARKER = r'\textsuperscript{b}'

#: Below this magnitude a column switches to scientific notation -- for all of
#: its entries, since siunitx aligns an S column on a single table-format and
#: mixing the two notations inside one column reads badly.
SCIENTIFIC_THRESHOLD = 1e-3

#: Declarations placed inside the float, ahead of the tabular. Empty because
#: breaking the case-study names across the rows they already span is enough
#: to fit both tables into the 345pt text width of a stock ``article`` at
#: full size. A narrower text block (a two-column layout, say) can be served
#: by putting back ``\small`` and a smaller ``\tabcolsep`` here; both are
#: scoped to the float.
TABLE_SIZE_DECLARATIONS = []


def load_cases(metrics_dir, case_spec):
    """Read the per-case JSON files; missing cases are reported, not fatal."""
    cases = []
    for case_id, case_label, curves in case_spec:
        path = os.path.join(metrics_dir, f'{case_id}_metrics.json')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found -- run the case study first, "
                  f"skipping '{plain_label(case_label)}'", file=sys.stderr)
            continue
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        by_name = {curve['name']: curve for curve in payload['curves']}
        cases.append((case_label, [(label, by_name[name])
                                   for name, label in curves if name in by_name]))
    return cases


def case_lines(case_label):
    """The lines a case-study name is broken into, as a tuple."""
    return (case_label,) if isinstance(case_label, str) else tuple(case_label)


def plain_label(case_label):
    """The name as a single line of plain text, for log messages."""
    return ' '.join(case_lines(case_label))


def stacked_label(case_label):
    r"""The name as a left-aligned ``\shortstack``, one line per row spanned."""
    return r'\shortstack[l]{' + r' \\ '.join(case_lines(case_label)) + '}'


def fmt_number(value, scientific, significant=3):
    """Format one entry with ``significant`` significant digits."""
    if value is None:
        return None
    if value == 0.0:
        return '0'
    if scientific:
        mantissa, _, exponent = f'{value:.{significant - 1}e}'.partition('e')
        return f'{mantissa}e{int(exponent)}'
    text = f'{value:#.{significant}g}'
    return text.rstrip('.') if '.' in text else text


def use_scientific(values):
    """Whether a column's entries should all be written in scientific form."""
    magnitudes = [abs(value) for value in values if value]
    return bool(magnitudes) and min(magnitudes) < SCIENTIFIC_THRESHOLD


def table_format(entries):
    """Derive an siunitx ``table-format`` covering every entry of a column."""
    int_digits = decimals = exponent_digits = 1
    has_exponent = False
    for entry in entries:
        if entry is None:
            continue
        mantissa, _, exponent = entry.partition('e')
        if exponent:
            has_exponent = True
            exponent_digits = max(exponent_digits, len(exponent.lstrip('-')))
        head, _, tail = mantissa.partition('.')
        int_digits = max(int_digits, len(head.lstrip('-')))
        decimals = max(decimals, len(tail))
    spec = f'{int_digits}.{decimals}'
    return spec + (f'e-{exponent_digits}' if has_exponent else '')


def build_table(cases, spec, include_mass_balance=True):
    """Render one LaTeX table: preamble, rows, footnotes and caption.

    ``include_mass_balance`` governs whether the solver verification column is
    part of the table at all. Dropping it also drops its caption sentence and
    its footnote, so the table stays self-consistent.
    """
    columns = ['delta_mu1_%', 'delta_mu2_%', 'nrmse_%']
    if include_mass_balance:
        columns.append('mass_balance_%')
    raw = {column: [] for column in columns}
    peak_height_rows = []
    inexact_mass_rows = []
    calibrated_rows = []
    for _, curves in cases:
        for _, curve in curves:
            for column in columns:
                raw[column].append(curve.get(column))
            peak_height_rows.append(curve.get('mu2_mode') == 'peak_height')
            calibrated_rows.append(curve.get('mu2_mode') == 'calibrated')
            inexact_mass_rows.append(curve.get('mass_balance_%') is not None
                                     and not curve.get('mass_balance_exact', True))

    cells = {column: [fmt_number(value, use_scientific(raw[column]))
                      for value in raw[column]]
             for column in columns}
    formats = {column: table_format(cells[column]) for column in columns}
    any_peak_height = any(peak_height_rows)
    any_inexact_mass = include_mass_balance and any(inexact_mass_rows)

    mu2_column = f"S[table-format={formats['delta_mu2_%']}"
    if any_peak_height:
        mu2_column += f', table-space-text-post={PEAK_HEIGHT_MARKER}'
    mu2_column += ']'

    column_specs = [
        f"S[table-format={formats['delta_mu1_%']}]",
        mu2_column,
        f"S[table-format={formats['nrmse_%']}]",
    ]
    headers = [r'{$\Delta \mu_1$}', r'{$\Delta \mu_2$}', r'{NRMSE}']
    if include_mass_balance:
        mass_column = f"S[table-format={formats['mass_balance_%']}"
        if any_inexact_mass:
            mass_column += f', table-space-text-post={INEXACT_MASS_MARKER}'
        mass_column += ']'
        column_specs.append(mass_column)
        headers.append(r'{Mass balance}')

    units = ' & '.join([r'{[\%]}'] * len(headers))
    lines = [
        r'\begin{table}[htbp]',
        r'  \centering',
        *TABLE_SIZE_DECLARATIONS,
        r'  \sisetup{table-number-alignment=center}',
        r'  \begin{tabular}{@{}l l ' + ' '.join(column_specs) + r'@{}}',
        r'    \toprule',
        r'    \multirow{2}{*}{Case study} & \multirow{2}{*}{' + spec['row_header'] + r'}',
        r'      & ' + ' & '.join(headers) + r' \\',
        r'      & & ' + units + r' \\',
        r'    \midrule',
    ]

    index = 0
    for case_number, (case_label, curves) in enumerate(cases):
        if case_number:
            lines.append(r'    \addlinespace')
        span = len(curves)
        for row, (row_label, _) in enumerate(curves):
            head = (rf'    \multirow{{{span}}}{{*}}{{{stacked_label(case_label)}}} '
                    if row == 0 else r'    ')
            values = []
            for column in columns:
                text = cells[column][index]
                values.append('{--}' if text is None else text)
            if peak_height_rows[index]:
                values[1] += PEAK_HEIGHT_MARKER
            if any_inexact_mass and inexact_mass_rows[index]:
                values[-1] += INEXACT_MASS_MARKER
            index += 1
            lines.append(head + f'& {row_label} & ' + ' & '.join(values) + r' \\')

    caption = spec['caption']
    if include_mass_balance:
        caption += spec['caption_mass_balance']
    if any(calibrated_rows):
        caption += (r' Empty $\Delta \mu_2$ entries mark cases whose dispersion '
                    r'coefficient was calibrated against exactly that $\mu_2$, so '
                    r'that the agreement would be fitted rather than predicted.')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
    ]

    footnotes = []
    if any_peak_height:
        footnotes.append(rf'{PEAK_HEIGHT_MARKER} peak-height error instead of '
                         r'$\Delta \mu_2$: the reference reports no second moment '
                         r'for this case.')
    if any_inexact_mass:
        footnotes.append(rf'{INEXACT_MASS_MARKER} not a pure conservation check: '
                         r'part of the loaded protein is still bound to the column '
                         r'at the end of the run, and that physical residual is '
                         r'contained in the deviation.')
    if footnotes:
        lines.append(r'  \smallskip')
        lines.append(r'  \footnotesize{' + ' '.join(footnotes) + '}')

    lines += [
        r'  \caption{' + caption + '}',
        r'  \label{' + spec['label'] + '}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def main(metrics_dir=None):
    """Write one LaTeX table per reference source and return them by name.

    The settings block below is the place to adapt what the tables contain;
    this module is driven from Python (scripts/verify_geometries.py calls it
    after the validation studies) rather than from a command line.
    """
    # -----------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------
    # Whether the tables carry the mass balance column at all. The mass
    # balance is a verification of the solver rather than a comparison
    # against reference data, so a paper may well want the tables to report
    # only the three validation metrics. Setting this to False also drops
    # the caption sentence and the footnote that belong to the column, and
    # it changes nothing about what the case studies themselves compute and
    # print -- only what ends up in the tables.
    include_mass_balance = False

    # Directory holding the case studies' <case_id>_metrics.json files.
    if metrics_dir is None:
        metrics_dir = os.path.join('output', 'validation')
    # -----------------------------------------------------------------

    tables = {}
    for spec in TABLES:
        cases = load_cases(metrics_dir, spec['cases'])
        if not cases:
            print(f"  WARNING: no metric JSON files for the '{spec['name']}' "
                  f"table, skipping it", file=sys.stderr)
            continue
        table = build_table(cases, spec, include_mass_balance=include_mass_balance)
        out_path = os.path.join(metrics_dir, spec['filename'])
        with open(out_path, 'w', encoding='utf-8') as handle:
            handle.write(table + '\n')
        tables[spec['name']] = table
        print(table)
        print()
        print(f"Wrote {out_path}", file=sys.stderr)
    if not tables:
        raise SystemExit(f"No metric JSON files found in {metrics_dir!r}; "
                         f"run the validation case studies first.")
    return tables


if __name__ == '__main__':
    main()
