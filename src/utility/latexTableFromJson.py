import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable


@dataclass(frozen=True)
class Format:
    """
    Formatting description for one column.

    kind:
        "int"   -> integer
        "float" -> fixed-point
        "sci"   -> scientific notation
        "g"     -> general format
        "str"   -> default string conversion
    """
    kind: str = "str"
    precision: int = 3
    formatter: Optional[Callable] = None


class LatexTableGenerator:
    def __init__(
        self,
        row_columns: List[str],
        include_columns: Optional[List[str]] = None,
        column_names: Optional[Dict[str, str]] = None,
        formats: Optional[Dict[str, Format]] = None,
    ):

        self.row_columns = row_columns
        self.include_columns = include_columns
        self.column_names = column_names or {}
        self.formats = formats or {}

    def _fmt(self, value, column):

        fmt = self.formats.get(column, Format())

        if fmt.formatter is not None:
            return fmt.formatter(value)

        try:

            if fmt.kind == "int":
                return f"{int(round(float(value)))}"

            elif fmt.kind == "float":
                return f"{float(value):.{fmt.precision}f}"

            elif fmt.kind == "sci":
                return f"{float(value):.{fmt.precision}e}"

            elif fmt.kind == "g":
                return f"{float(value):.{fmt.precision}g}"

        except (ValueError, TypeError):
            pass

        return str(value)

    def generate(
        self,
        json_file,
        scheme=None,
        group="outlet",
    ):

        with open(json_file, "r") as f:
            data = json.load(f)

        conv = data["convergence"]

        if scheme is None:
            scheme = next(iter(conv.keys()))

        values = conv[scheme][group]

        if self.include_columns is None:
            columns = list(values.keys())
        else:
            columns = self.include_columns

        data_columns = [c for c in columns if c not in self.row_columns]

        headers = (
            [self.column_names.get(c, c) for c in self.row_columns]
            + [self.column_names.get(c, c) for c in data_columns]
        )

        nrows = len(values[self.row_columns[0]])

        alignment = (
            "l" * len(self.row_columns)
            + "r" * len(data_columns)
        )

        lines = []

        lines.append(r"\begin{tabular}{" + alignment + "}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r"\\")
        lines.append(r"\hline")

        for i in range(nrows):

            row = []

            for c in self.row_columns:
                row.append(self._fmt(values[c][i], c))

            for c in data_columns:
                row.append(self._fmt(values[c][i], c))

            lines.append(" & ".join(row) + r"\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

        return "\n".join(lines)


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":

    # reusable formats

    INT = Format("int")
    EOC = Format("float", 2)
    ERROR = Format("sci", 3)
    TIME = Format("float", 4)
    GENERAL = Format("g", 4)

    generator = LatexTableGenerator(

        column_names={
            "Max. error": r"$L^\infty$ error",
            "Max. EOC": r"$L^\infty$ EOC",
            "Sim. time": "CPU [s]",
            "DoF": "DoFs",
        },

        formats={

            "$N_d$": INT,
            "$N_e^z$": INT,

            "DoF": INT,

            "Max. error": ERROR,
            "$L^1$ error": ERROR,
            "$L^2$ error": ERROR,

            "Max. EOC": EOC,
            "$L^1$ EOC": EOC,
            "$L^2$ EOC": EOC,

            "Sim. time": TIME,

            # "Min. value": ERROR,
        },

        # one, two or three row columns
        row_columns=[
            "$N_d$",
            "$N_e^z$",
        ],

        include_columns=[
            "$N_d$",
            "$N_e^z$",
            "Max. error",
            "Max. EOC",
            # "$L^1$ error",
            # "$L^1$ EOC",
            "$L^2$ error",
            "$L^2$ EOC",
            # "Sim. time",
            # "Min. value",
            # "DoF",
        ],
    )

    latex = generator.generate(
        r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\transport\convergence_COL1D_frustumTransport_1comp_benchmark1.json",
        scheme="DG_P3",
        group="outlet",
    )

    print(latex)

    # Path("table.tex").write_text(latex)