import json
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
    math_mode: bool = True


@dataclass(frozen=True)
class Geometry:
    """
    Description of one geometry / JSON input.
    """

    name: str
    json_file: str
    group: str = "outlet"
    include_columns: Optional[List[str]] = None


class LatexTableGenerator:

    def __init__(
        self,
        row_columns: List[str],
        column_names: Optional[Dict[str, str]] = None,
        formats: Optional[Dict[str, Format]] = None,
        default_float: Format = Format("sci", 3),
    ):

        self.row_columns = row_columns
        self.column_names = column_names or {}
        self.formats = formats or {}
        self.default_float = default_float

    # ==================================================================
    # Formatting
    # ==================================================================

    def _fmt(self, value, column):

        fmt = self.formats.get(column)

        if fmt is None:
            if isinstance(value, float):
                fmt = self.default_float
            else:
                fmt = Format()

        try:

            if fmt.kind == "int":
                result = f"{int(round(float(value)))}"

            elif fmt.kind == "float":
                result = f"{float(value):.{fmt.precision}f}"

            elif fmt.kind == "sci":
                result = f"{float(value):.{fmt.precision}e}"

            elif fmt.kind == "g":
                result = f"{float(value):.{fmt.precision}g}"

            else:
                result = str(value)

        except (ValueError, TypeError):
            result = str(value)

        if fmt.math_mode:
            return f"${result}$"

        return result

    # ==================================================================
    # Load one geometry for one scheme
    # ==================================================================

    def _load_geometry(
        self,
        geometry: Geometry,
        scheme: str,
    ):

        with open(geometry.json_file, "r") as f:
            data = json.load(f)

        conv = data["convergence"]

        if scheme not in conv:
            raise ValueError(
                f"Scheme '{scheme}' not found in "
                f"'{geometry.json_file}'.\n"
                f"Available schemes: {list(conv.keys())}"
            )

        if geometry.group not in conv[scheme]:
            raise ValueError(
                f"Group '{geometry.group}' not found for scheme "
                f"'{scheme}' in '{geometry.json_file}'."
            )

        values = conv[scheme][geometry.group]

        if geometry.include_columns is None:
            columns = list(values.keys())
        else:
            columns = geometry.include_columns

        # Check that requested columns exist
        for column in columns:

            if column not in values:
                raise ValueError(
                    f"Column '{column}' not found in scheme "
                    f"'{scheme}', group '{geometry.group}', "
                    f"file '{geometry.json_file}'."
                )

        return values, columns

    # ==================================================================
    # Determine Nd for a scheme
    # ==================================================================

    def _get_nd(self, values, nd_column):
        """
        Determine the polynomial degree Nd for one scheme.

        Normally all entries in the Nd column are identical, e.g.

            DG_P1 -> [1, 1, 1]
            DG_P2 -> [2, 2, 2]

        We verify this and return the unique value.
        """

        if nd_column not in values:
            raise ValueError(
                f"Column '{nd_column}' not found."
            )

        nd_values = values[nd_column]

        if len(nd_values) == 0:
            raise ValueError(
                f"Column '{nd_column}' is empty."
            )

        normalized = [
            int(round(float(value)))
            for value in nd_values
        ]

        unique_nd = set(normalized)

        if len(unique_nd) != 1:
            raise ValueError(
                f"Expected exactly one Nd value for a scheme, "
                f"but found {sorted(unique_nd)}."
            )

        return normalized[0]

    # ==================================================================
    # Generate table
    # ==================================================================

    def generate(
        self,
        geometries: List[Geometry],
        schemes: List[str],
    ):
        """
        Generate ONE combined LaTeX table.

        The first column is the shared Nd column.

        Each geometry then gets its own columns, e.g.

            Nd | Radial flow                  | Conical frustum
               | Ne^z | L2 error | L2 EOC    | Ne^z | L2 error | L2 EOC

        Different geometries are allowed to have different Ne^z
        refinement levels.

        Each scheme forms one block and blocks are separated by
        \\hline.
        """

        # --------------------------------------------------------------
        # Validate geometries
        # --------------------------------------------------------------

        if not 1 <= len(geometries) <= 3:
            raise ValueError(
                "The table must contain between one and three geometries."
            )

        # --------------------------------------------------------------
        # Validate schemes
        # --------------------------------------------------------------

        if not schemes:
            raise ValueError(
                "At least one scheme must be provided."
            )

        # Option 2:
        # only Nd is shared
        if len(self.row_columns) != 1:
            raise ValueError(
                "For this table format, row_columns must contain "
                "exactly one column: ['$N_d']."
            )

        nd_column = self.row_columns[0]

        # --------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------

        all_data = []

        for scheme in schemes:

            scheme_data = []

            nd_reference = None

            for geometry in geometries:

                values, columns = self._load_geometry(
                    geometry,
                    scheme,
                )

                nd = self._get_nd(
                    values,
                    nd_column,
                )

                # ------------------------------------------------------
                # Verify Nd is identical between geometries
                # ------------------------------------------------------

                if nd_reference is None:
                    nd_reference = nd

                elif nd != nd_reference:
                    raise ValueError(
                        f"Different Nd values for scheme '{scheme}': "
                        f"geometry '{geometry.name}' has Nd={nd}, "
                        f"but another geometry has Nd={nd_reference}."
                    )

                # ------------------------------------------------------
                # Geometry-specific columns
                # ------------------------------------------------------

                data_columns = [
                    column
                    for column in columns
                    if column != nd_column
                ]

                # Number of refinement levels
                nrows = len(
                    values[data_columns[0]]
                )

                # Verify all columns have same length
                for column in data_columns:

                    if len(values[column]) != nrows:
                        raise ValueError(
                            f"Column '{column}' has a different number "
                            f"of rows in geometry '{geometry.name}', "
                            f"scheme '{scheme}'."
                        )

                scheme_data.append(
                    {
                        "geometry": geometry,
                        "values": values,
                        "columns": data_columns,
                        "nrows": nrows,
                    }
                )

            all_data.append(
                {
                    "scheme": scheme,
                    "nd": nd_reference,
                    "geometries": scheme_data,
                }
            )

        # --------------------------------------------------------------
        # Column structure
        # --------------------------------------------------------------

        first_scheme = all_data[0]

        alignment = "c"

        for item in first_scheme["geometries"]:

            alignment += "|"
            alignment += "r" * len(item["columns"])

        # --------------------------------------------------------------
        # Begin table
        # --------------------------------------------------------------

        lines = []

        lines.append(
            rf"\begin{{tabular}}{{{alignment}}}"
        )

        lines.append(r"\hline")

        # --------------------------------------------------------------
        # Major headers
        # --------------------------------------------------------------

        major_headers = [
            r"\multicolumn{1}{c}{}"
        ]

        for item in first_scheme["geometries"]:

            ncols = len(item["columns"])

            major_headers.append(
                rf"\multicolumn{{{ncols}}}{{c}}"
                rf"{{\textbf{{{item['geometry'].name}}}}}"
            )

        lines.append(
            " & ".join(major_headers) + r"\\"
        )

        # --------------------------------------------------------------
        # Subheaders
        # --------------------------------------------------------------

        headers = [
            self.column_names.get(
                nd_column,
                nd_column,
            )
        ]

        for item in first_scheme["geometries"]:

            headers.extend(
                self.column_names.get(
                    column,
                    column,
                )
                for column in item["columns"]
            )

        lines.append(
            " & ".join(headers) + r"\\"
        )

        lines.append(r"\hline")

        # --------------------------------------------------------------
        # Data blocks
        # --------------------------------------------------------------

        for scheme_index, scheme_data in enumerate(all_data):

            nd = scheme_data["nd"]
            geometry_data = scheme_data["geometries"]

            # ----------------------------------------------------------
            # Maximum number of refinement levels
            # ----------------------------------------------------------

            max_rows = max(
                item["nrows"]
                for item in geometry_data
            )

            # ----------------------------------------------------------
            # Rows
            # ----------------------------------------------------------

            for i in range(max_rows):

                row = []

                # ------------------------------------------------------
                # Nd only appears once
                # ------------------------------------------------------

                if i == 0:
                    row.append(
                        self._fmt(
                            nd,
                            nd_column,
                        )
                    )
                else:
                    row.append("")

                # ------------------------------------------------------
                # Geometry-specific columns
                # ------------------------------------------------------

                for item in geometry_data:

                    values = item["values"]

                    for column in item["columns"]:

                        column_values = values[column]

                        if i < len(column_values):

                            row.append(
                                self._fmt(
                                    column_values[i],
                                    column,
                                )
                            )

                        else:

                            # Geometry has fewer refinement levels
                            row.append("")

                lines.append(
                    " & ".join(row) + r"\\"
                )

            # ----------------------------------------------------------
            # Separate polynomial degrees
            # ----------------------------------------------------------

            if scheme_index < len(all_data) - 1:
                lines.append(r"\hline")

        # --------------------------------------------------------------
        # End table
        # --------------------------------------------------------------

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

        return "\n".join(lines)


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Reusable formats
    # -------------------------------------------------------------------------

    INT = Format("int")
    EOC = Format("float", 2)
    ERROR = Format("sci", 3)
    TIME = Format("float", 4)
    GENERAL = Format("g", 4)

    # -------------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------------

    generator = LatexTableGenerator(

        # Only Nd is shared between geometries
        row_columns=[
            "$N_d$",
        ],

        column_names={

            "$N_d$": r"$N_d$",
            "$N_e^z$": r"$N_e^z$",

            "Max. error": r"$L^\infty$ error",
            "Max. EOC": r"$L^\infty$ EOC",

            "$L^1$ error": r"$L^1$ error",
            "$L^1$ EOC": r"$L^1$ EOC",

            "$L^2$ error": r"$L^2$ error",
            "$L^2$ EOC": r"$L^2$ EOC",

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
        },
    )

    # -------------------------------------------------------------------------
    # Geometries
    # -------------------------------------------------------------------------

    geometries = [

        Geometry(
            name="Radial flow",
            json_file=(
                r"C:\Users\jmbr\software\CADET-Verification"
                r"\output\test_cadet-core\chromatography"
                r"\convergence_radialAdvDPFR_1comp_benchmark1.json"
            ),
            group="bulk_component_000",
            include_columns=[
                "$N_d$",
                "$N_e^z$",
                "$L^2$ error",
                "$L^2$ EOC",
            ],
        ),

        Geometry(
            name="Conical frustum",
            json_file=(
                r"C:\Users\jmbr\software\CADET-Verification"
                r"\output\test_cadet-core\chromatography"
                r"\convergence_frustumAdvDPFR_1comp_benchmark1.json"
            ),
            group="bulk_component_000",
            include_columns=[
                "$N_d$",
                "$N_e^z$",
                "$L^2$ error",
                "$L^2$ EOC",
            ],
        ),
    ]

    # -------------------------------------------------------------------------
    # Schemes
    # -------------------------------------------------------------------------

    schemes = [
        "FVWENO3",
        "DG_P1",
        "DG_P2",
        "DG_P3",
        "DG_P4",
    ]

    # -------------------------------------------------------------------------
    # Generate table
    # -------------------------------------------------------------------------

    latex = generator.generate(
        geometries=geometries,
        schemes=schemes,
    )

    print(latex)