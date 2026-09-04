from dataclasses import dataclass
import math


# =============================================================================
# Parameter definitions
# =============================================================================

@dataclass
class Parameter:
    label: str
    symbol: str
    unit: str
    key: str


PARAMETERS = {
    "FRUSTUM_COLUMN_MODEL_1D": [

        Parameter("Flow rate",                r"$Q$",                     r"$m^3/s$", "flow_rate"),

        Parameter("Bed length",             r"$L_{\mathrm{bed}}$",      r"$m$", "bed_length"),
        Parameter("Small-end cross section area", r"$A_{\mathrm{small}}$", r"$m^2$", "cross_section_area_small_end"),
        Parameter("Large-end cross section area", r"$A_{\mathrm{large}}$", r"$m^2$", "cross_section_area_large_end"),

        Parameter("Column porosity",          r"$\varepsilon_c$",         r"$-$", "col_porosity"),
        Parameter("Total porosity",           r"$\varepsilon_t$",         r"$-$", "total_porosity"),

        Parameter("Column dispersion",        r"$D_{\mathrm{ax}}$",       r"$m^2/s$", "col_dispersion"),

        Parameter("Initial bulk conc.",       r"$c^{\mathrm{b},\mathrm{init}}_i$",     r"$mol/L$", "init_c"),
        Parameter("Initial pore conc.",       r"$c^{\mathrm{p},\mathrm{init}}_i$",     r"$mol/L$", "init_cp"),
        Parameter("Initial solid conc.",       r"$c^{\mathrm{s},\mathrm{init}}_i$",     r"$mol/L$", "init_cs"),

        Parameter("Number of components",     r"$N^{\mathrm{c}}$",     r"$-$", "ncomp"),
        Parameter("Number of particle types", r"$N^{\mathrm{p}}$", r"$-$", "npartype"),
    ],
}


# =============================================================================
# Helpers
# =============================================================================

def _unit(model, unit_id):
    return getattr(model.input.model, f"unit_{unit_id:03d}")


def _latex_number(x):
    """Formats numbers nicely for LaTeX."""

    if isinstance(x, (list, tuple)):
        return ", ".join(_latex_number(v) for v in x)

    if isinstance(x, str):
        return x

    if isinstance(x, bool):
        return str(int(x))

    if isinstance(x, int):
        return str(x)

    if x == 0:
        return "0"

    if abs(x) < 1e-3 or abs(x) >= 1e3:
        mantissa, exponent = f"{x:.3e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        return rf"{mantissa:g} \cdot 10^{{{exponent}}}"

    return f"{x:g}"


# =============================================================================
# Connection handling
# =============================================================================

def _detect_connection_stride(connection_list):
    """
    Returns the row length (5 or 7).
    """

    possible = []

    for stride in (5, 7):
        if len(connection_list) % stride == 0:
            possible.append(stride)

    if len(possible) == 1:
        return possible[0]

    # Both fit -> determine whether flow rate is column 5 or 7.
    rows5 = [connection_list[i:i+5] for i in range(0, len(connection_list), 5)]
    rows7 = [connection_list[i:i+7] for i in range(0, len(connection_list), 7)]

    def score(values):
        """
        Flow rates are usually
        * positive
        * finite
        * not huge
        """
        return sum(
            (v > 0)
            and math.isfinite(v)
            and (v < 1.0)
            for v in values
        )

    score5 = score([r[4] for r in rows5])
    score7 = score([r[6] for r in rows7])

    return 5 if score5 >= score7 else 7


def find_flow_rate(model, unit_id):
    """
    Returns the flow rate associated with a unit.
    """

    conn = list(model.input.model.connections.switch_000.connections)

    stride = _detect_connection_stride(conn)

    rows = [
        conn[i:i+stride]
        for i in range(0, len(conn), stride)
    ]

    flow_col = 4 if stride == 5 else 6

    for row in rows:

        from_unit = int(row[0])
        to_unit = int(row[1])

        if from_unit == unit_id or to_unit == unit_id:
            return row[flow_col]

    return None


# =============================================================================
# Table generation
# =============================================================================

def generate_parameter_table(model, unit_id):
    """
    Generates a LaTeX parameter table for a unit.
    """

    unit = _unit(model, unit_id)

    if unit.unit_type not in PARAMETERS:
        raise ValueError(f"No parameter dictionary for unit type '{unit.unit_type}'")

    params = PARAMETERS[unit.unit_type]

    lines = [
        r"\begin{table}[!htb]",
        r"  \begin{center}",
        rf"  \caption{{Model parameters for unit {unit_id}.}}",
        rf"  \label{{tab:unit_{unit_id:03d}}}",
        r"    \begin{tabular}{lcc|c}",
        r"    \toprule",
        r"    Parameter & Symbol & Unit & Value \\",
        r"    \midrule",
    ]

    for p in params:

        if p.key == "flow_rate":
            value = find_flow_rate(model, unit_id)

        elif p.key in unit:
            value = unit[p.key]

        else:
            continue
        
        value = _latex_number(value)

        lines.append(
            f"    {p.label} & {p.symbol} & {p.unit} & ${value}$ \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"  \end{center}",
        r"\end{table}",
    ])

    return "\n".join(lines)

#########################################################################################
#########################################################################################
#########################################################################################

from src.benchmark_models.setting_COL1D_frustum_transport import get_model

latex = generate_parameter_table(get_model(), 1)

print(latex)
