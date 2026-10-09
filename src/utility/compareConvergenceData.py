"""
Compare convergence-test JSON data between two directory trees and generate
both Markdown and JSON reports.

Example usage:
    path1 = r"C:/Users/user1/software/CADET-Verification/test/data/verify_cadet_core_dummyData"
    path2 = r"C:/Users/user1/software/CADET-Verification/test/data/verify_cadet_core_v600alpha3"
    output_dir = r"C:/Users/user1/software/CADET-Verification/test/data/comparison_reports"
    
    exit_code = compare_convergence_data(
        [
            path1,
            path2,
            "--output-dir",
            output_dir,
            "--abs-tol",
            "1e-8",
            "--rel-tol",
            "1e-1",
            "--minor-multiplier",
            "10.0",
            "--verbose",
        ]
    )
    
    print(f"compare_convergence_data finished with exit code {exit_code}")

"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TOOL_NAME = "compare_convergence"
TOOL_VERSION = "1.0"

CONVERGENCE_KEY = "convergence"

EXACT_KEYS: Tuple[str, ...] = (
    "$N_e^z$",
    "$N_e^p$",
    "$N_e^r$",
    "$N_e^x$",
)

NUMERIC_KEYS: Tuple[str, ...] = (
    "Max. error",
    "$L^1$ error",
    "$L^2$ error",
    "Max. EOC",
    "$L^1$ EOC",
    "$L^2$ EOC",
)

SEVERITY_ORDER: Dict[str, int] = {
    "OK": 0,
    "MINOR": 1,
    "MAJOR": 2,
    "MISSING": 3,
    "ERROR": 4,
}

SEVERITY_DESCRIPTIONS: Dict[str, str] = {
    "OK": "Values match or are within tolerance",
    "MINOR": "Small deviation (within abstol, reltol * minor multiplier)",
    "MAJOR": "Significant deviation",
    "MISSING": "Missing file, group, or key",
    "ERROR": "Parse, schema, or type error",
}

SIM_TIME_KEY = "Sim. time"

# Error and EOC keys of the same norm, used by the convergence screen.
ERROR_EOC_KEYS: Tuple[Tuple[str, str], ...] = (
    ("Max. error", "Max. EOC"),
    ("$L^1$ error", "$L^1$ EOC"),
    ("$L^2$ error", "$L^2$ EOC"),
)

# Compute times below this many seconds are dominated by start-up and say
# nothing about performance, so they are left out of the timing analysis.
SIM_TIME_FLOOR = 0.05

# A file whose median compute time ratio deviates from the median ratio of the
# whole run by more than this factor is reported as a timing outlier. Runs on
# different machines differ by a factor that applies to every simulation, which
# is what normalizing against the run median removes.
SIM_TIME_OUTLIER_FACTOR = 2.0

# A refinement level whose error did not fall by at least this factor compared
# to the previous level has reached the accuracy of the reference solution. Its
# EOC describes the reference, not the method, so the screen ignores it. Even a
# series converging at order 0.2 falls to 0.87 per level, so the threshold
# separates a saturated series from a slowly converging one.
EOC_SATURATION_FACTOR = 0.9

# An EOC below this is not convergence.
EOC_NONCONVERGENT = 0.5

# A drop of at least this much, ending below EOC_SUSPICIOUS and accompanied by
# an error grown by at least ERROR_GROWTH_FACTOR, is reported.
EOC_DROP = 0.5
EOC_SUSPICIOUS = 2.0
ERROR_GROWTH_FACTOR = 1.5

class Severity(str, Enum):
    """Allowed severity levels for findings."""

    OK = "OK"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    MISSING = "MISSING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Tolerances:
    """Tolerance settings for floating-point comparison."""

    abs_tol: float
    rel_tol: float
    minor_multiplier: float


@dataclass
class ComparisonEntry:
    """Represents a single key comparison within a solution group."""

    key: str
    category: str  # "exact" or "numeric"
    present_in_a: bool
    present_in_b: bool
    value_a: Any
    value_b: Any
    abs_diff: Optional[Any]
    rel_diff: Optional[Any]
    severity: Severity
    status: str


@dataclass
class StructuralDifference:
    """Represents a structural discrepancy in convergence/method/solution groups."""

    level: str
    parent: str
    missing_in_a: List[str] = field(default_factory=list)
    missing_in_b: List[str] = field(default_factory=list)
    severity: Severity = Severity.MISSING
    comment: str = ""


@dataclass
class SolutionResult:
    """Comparison results for a specific solution group under a method."""

    solution: str
    comparisons: List[ComparisonEntry] = field(default_factory=list)
    highest_severity: Severity = Severity.OK


@dataclass
class MethodSimTimeSummary:
    """Aggregate Sim. time deviation summary for one method."""

    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    worst_solution_abs: Optional[str] = None
    worst_solution_rel: Optional[str] = None
    worst_index_abs: Optional[int] = None
    worst_index_rel: Optional[int] = None


@dataclass
class EocFinding:
    """One payload whose order of convergence got worse from A to B.

    Advisory: these findings are reported but do not influence the severity
    counts or the exit code, since an EOC can legitimately change with the
    refinement range or the reference solution.
    """

    relative_path: str
    method: str
    solution: str
    key: str
    kind: str  # "new_nonconvergent" or "worse"
    eoc_a: float
    eoc_b: float
    error_a: float
    error_b: float
    level_a: int
    level_b: int


@dataclass
class FileSimTime:
    """Compute time ratios B/A of one file."""

    relative_path: str
    n_values: int
    median_ratio: float
    normalized_ratio: Optional[float] = None  # median_ratio / run median


@dataclass
class SimTimeAnalysis:
    """Compute time ratios B/A over the whole run.

    A run on another machine scales every simulation by a similar factor, so the
    median over all compared simulations is the machine difference and the files
    that deviate from it are the ones worth looking at.
    """

    n_values: int = 0
    median_ratio: Optional[float] = None
    p10_ratio: Optional[float] = None
    p90_ratio: Optional[float] = None
    min_ratio: Optional[float] = None
    max_ratio: Optional[float] = None
    files: List[FileSimTime] = field(default_factory=list)
    outliers: List[FileSimTime] = field(default_factory=list)


@dataclass
class MethodInventory:
    """How often each method name occurs in either tree.

    A method renamed between two runs shows up here as one name losing and
    another gaining files, which the per-file comparison can only report as
    missing on both sides.
    """

    counts_a: Dict[str, int] = field(default_factory=dict)
    counts_b: Dict[str, int] = field(default_factory=dict)
    only_in_a: List[str] = field(default_factory=list)
    only_in_b: List[str] = field(default_factory=list)


@dataclass
class MethodResult:
    """Comparison results for a specific numerical method."""

    method: str
    structural_differences: List[StructuralDifference] = field(default_factory=list)
    solutions: List[SolutionResult] = field(default_factory=list)
    sim_time_summary: MethodSimTimeSummary = field(default_factory=MethodSimTimeSummary)
    highest_severity: Severity = Severity.OK


@dataclass
class FileError:
    """Represents a parse or schema error associated with one file."""

    side: str  # "A", "B", or "both"
    error_type: str
    message: str
    severity: Severity = Severity.ERROR


@dataclass
class FileResult:
    """Comparison results for a single matched file path."""

    relative_path: str
    status: str  # e.g. "compared", "error"
    errors: List[FileError] = field(default_factory=list)
    structural_differences: List[StructuralDifference] = field(default_factory=list)
    methods: List[MethodResult] = field(default_factory=list)
    highest_severity: Severity = Severity.OK
    methods_a: List[str] = field(default_factory=list)
    methods_b: List[str] = field(default_factory=list)
    sim_time_ratios: List[float] = field(default_factory=list)
    eoc_findings: List[EocFinding] = field(default_factory=list)


@dataclass
class MissingFiles:
    """Lists of files present only on one side."""

    only_in_a: List[str] = field(default_factory=list)
    only_in_b: List[str] = field(default_factory=list)


@dataclass
class Summary:
    """Global summary of comparison results."""

    shared_files: int
    files_only_in_a: int
    files_only_in_b: int
    counts_by_severity: Dict[str, int]
    has_non_ok: bool


@dataclass
class Report:
    """Complete machine-readable report."""

    metadata: Dict[str, Any]
    inputs: Dict[str, str]
    tolerances: Dict[str, float]
    summary: Summary
    missing_files: MissingFiles
    files: List[FileResult]
    method_inventory: MethodInventory = field(default_factory=MethodInventory)
    sim_time_analysis: SimTimeAnalysis = field(default_factory=SimTimeAnalysis)
    eoc_findings: List[EocFinding] = field(default_factory=list)


def severity_max(*severities: Severity) -> Severity:
    """Return the highest severity according to SEVERITY_ORDER."""
    return max(severities, key=lambda sev: SEVERITY_ORDER[sev.value])


def update_highest(current: Severity, candidate: Severity) -> Severity:
    """Return the more severe of two severities."""
    return severity_max(current, candidate)


def utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 form with trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_number(value: Any) -> bool:
    """Return True for int/float values excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_number_sequence(value: Any) -> bool:
    """Return True if value is a list/tuple containing only numeric entries."""
    return isinstance(value, (list, tuple)) and all(is_number(v) for v in value)


def is_scalar_or_number_sequence(value: Any) -> bool:
    """Return True for a scalar number or a sequence of scalar numbers."""
    return is_number(value) or is_number_sequence(value)


def normalize_numeric_value(value: Any) -> List[float]:
    """
    Normalize a numeric scalar or numeric sequence to a list of floats.
    Scalars become a single-element list.
    """
    if is_number(value):
        return [float(value)]
    if is_number_sequence(value):
        return [float(v) for v in value]
    raise TypeError(f"Value is not numeric or numeric sequence: {value!r}")


def highest_numeric_severity(abs_diffs: List[float], rel_diffs: List[float], tolerances: Tolerances) -> Severity:
    """Return the highest severity across all elementwise numeric comparisons."""
    severity = Severity.OK
    for abs_diff, rel_diff in zip(abs_diffs, rel_diffs):
        severity = update_highest(severity, classify_numeric_difference(abs_diff, rel_diff, tolerances))
    return severity


def format_value(value: Any) -> str:
    """Format a Python value for Markdown output."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.16g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def format_float(value: Optional[Any]) -> str:
    """Format an optional float or list of floats for Markdown output."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_float(v) for v in value) + "]"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6e}"


def update_sim_time_summary(
    summary: MethodSimTimeSummary,
    label: str,
    a_payload: Mapping[str, Any],
    b_payload: Mapping[str, Any],
) -> None:
    """Update method-level Sim. time summary from one compared payload.

    Stores signed differences:
        diff = B - A
        rel_diff = (B - A) / max(abs(A), abs(B), tiny)

    The reported entry is the one with the largest magnitude.
    """
    a_has = SIM_TIME_KEY in a_payload
    b_has = SIM_TIME_KEY in b_payload

    if not a_has and not b_has:
        return
    if not a_has or not b_has:
        return

    a_value = a_payload[SIM_TIME_KEY]
    b_value = b_payload[SIM_TIME_KEY]

    if not is_scalar_or_number_sequence(a_value) or not is_scalar_or_number_sequence(b_value):
        return

    a_values = normalize_numeric_value(a_value)
    b_values = normalize_numeric_value(b_value)

    if len(a_values) != len(b_values):
        return

    for idx, (a_num, b_num) in enumerate(zip(a_values, b_values)):
        diff = b_num - a_num

        if a_num == 0.0 and b_num == 0.0:
            rel_diff = 0.0
        else:
            denom = max(abs(a_num), abs(b_num), sys.float_info.min)
            rel_diff = diff / denom

        if summary.max_abs_diff is None or abs(diff) > abs(summary.max_abs_diff):
            summary.max_abs_diff = diff
            summary.worst_solution_abs = label
            summary.worst_index_abs = idx

        if summary.max_rel_diff is None or abs(rel_diff) > abs(summary.max_rel_diff):
            summary.max_rel_diff = rel_diff
            summary.worst_solution_rel = label
            summary.worst_index_rel = idx
            
        
def json_safe(value: Any) -> Any:
    """
    Convert values to JSON-safe equivalents while preserving useful detail.

    This is mostly needed for Enum values and any incidental Path objects.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def dataclass_to_jsonable(obj: Any) -> Any:
    """Convert nested dataclasses and enums into deterministic JSON-safe objects."""
    return json_safe(asdict(obj))


def relative_difference(a: float, b: float) -> float:
    """
    Compute a symmetric relative difference safely.

    Definition:
        abs(a - b) / max(abs(a), abs(b), tiny)
    Special case:
        if a == b == 0, returns 0.0
    """
    if a == 0.0 and b == 0.0:
        return 0.0
    denom = max(abs(a), abs(b), sys.float_info.min)
    return abs(a - b) / denom


def classify_numeric_difference(
    abs_diff: float,
    rel_diff: float,
    tolerances: Tolerances,
) -> Severity:
    """
    Classify floating-point difference using configured tolerances.

    Rules:
    - OK if abs_diff <= abs_tol OR rel_diff <= rel_tol
    - MINOR if within (minor_multiplier * tolerance) on either scale
    - MAJOR otherwise
    """
    if abs_diff <= tolerances.abs_tol or rel_diff <= tolerances.rel_tol:
        return Severity.OK

    abs_minor = tolerances.abs_tol * tolerances.minor_multiplier
    rel_minor = tolerances.rel_tol * tolerances.minor_multiplier

    if abs_diff <= abs_minor or rel_diff <= rel_minor:
        return Severity.MINOR

    return Severity.MAJOR


def collect_json_files(root: Path) -> Dict[str, Path]:
    """
    Recursively collect all JSON files beneath a root directory.

    The returned mapping keys are normalized POSIX-style relative paths,
    which are used as the matching keys between the two trees.
    """
    files: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files[rel] = path
    return files


def load_json_file(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load a JSON file.

    Returns:
        (data, None) on success
        (None, error_message) on failure
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except OSError as exc:
        return None, f"I/O error while reading file: {exc}"


def validate_convergence_root(data: Any) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """
    Validate the expected top-level structure and return the convergence mapping.

    Expected structure:
        {
          "convergence": { ... }
        }
    """
    if not isinstance(data, Mapping):
        return None, "Top-level JSON value is not an object"
    if CONVERGENCE_KEY not in data:
        return None, f'Missing top-level "{CONVERGENCE_KEY}" group'
    convergence = data[CONVERGENCE_KEY]
    if not isinstance(convergence, Mapping):
        return None, f'Top-level "{CONVERGENCE_KEY}" group is not an object'
    return convergence, None


def make_file_error(side: str, error_type: str, message: str) -> FileError:
    """Construct a standardized file error."""
    return FileError(side=side, error_type=error_type, message=message, severity=Severity.ERROR)


def compare_exact_key(key: str, a_has: bool, b_has: bool, a_value: Any, b_value: Any) -> ComparisonEntry:
    """Compare an exact-match key."""
    if not a_has and not b_has:
        raise ValueError("compare_exact_key should not be called when both sides are absent")

    if not a_has:
        return ComparisonEntry(
            key=key,
            category="exact",
            present_in_a=False,
            present_in_b=True,
            value_a=None,
            value_b=b_value,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.MISSING,
            status="Missing in A",
        )
    if not b_has:
        return ComparisonEntry(
            key=key,
            category="exact",
            present_in_a=True,
            present_in_b=False,
            value_a=a_value,
            value_b=None,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.MISSING,
            status="Missing in B",
        )
    if a_value == b_value:
        return ComparisonEntry(
            key=key,
            category="exact",
            present_in_a=True,
            present_in_b=True,
            value_a=a_value,
            value_b=b_value,
            abs_diff=0.0 if is_number(a_value) and is_number(b_value) else None,
            rel_diff=0.0 if is_number(a_value) and is_number(b_value) else None,
            severity=Severity.OK,
            status="Exact match",
        )
    return ComparisonEntry(
        key=key,
        category="exact",
        present_in_a=True,
        present_in_b=True,
        value_a=a_value,
        value_b=b_value,
        abs_diff=(abs(float(a_value) - float(b_value)) if is_number(a_value) and is_number(b_value) else None),
        rel_diff=(relative_difference(float(a_value), float(b_value)) if is_number(a_value) and is_number(b_value) else None),
        severity=Severity.MAJOR,
        status="Exact-key mismatch",
    )


def compare_numeric_key(
    key: str,
    a_has: bool,
    b_has: bool,
    a_value: Any,
    b_value: Any,
    tolerances: Tolerances,
) -> ComparisonEntry:
    """Compare a numeric key using absolute and relative differences.

    Supports both scalar numeric values and lists/tuples of numeric values.
    Lists are compared elementwise.
    """
    if not a_has and not b_has:
        raise ValueError("compare_numeric_key should not be called when both sides are absent")

    if not a_has:
        return ComparisonEntry(
            key=key,
            category="numeric",
            present_in_a=False,
            present_in_b=True,
            value_a=None,
            value_b=b_value,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.MISSING,
            status="Missing in A",
        )
    if not b_has:
        return ComparisonEntry(
            key=key,
            category="numeric",
            present_in_a=True,
            present_in_b=False,
            value_a=a_value,
            value_b=None,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.MISSING,
            status="Missing in B",
        )

    if not is_scalar_or_number_sequence(a_value) or not is_scalar_or_number_sequence(b_value):
        return ComparisonEntry(
            key=key,
            category="numeric",
            present_in_a=True,
            present_in_b=True,
            value_a=a_value,
            value_b=b_value,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.ERROR,
            status="Expected numeric scalar or numeric list on both sides",
        )

    a_values = normalize_numeric_value(a_value)
    b_values = normalize_numeric_value(b_value)

    if len(a_values) != len(b_values):
        return ComparisonEntry(
            key=key,
            category="numeric",
            present_in_a=True,
            present_in_b=True,
            value_a=a_value,
            value_b=b_value,
            abs_diff=None,
            rel_diff=None,
            severity=Severity.ERROR,
            status=f"Length mismatch for numeric list: len(A)={len(a_values)}, len(B)={len(b_values)}",
        )

    abs_diffs = [abs(a - b) for a, b in zip(a_values, b_values)]
    rel_diffs = [relative_difference(a, b) for a, b in zip(a_values, b_values)]
    severity = highest_numeric_severity(abs_diffs, rel_diffs, tolerances)

    if severity is Severity.OK:
        status = "Within tolerance"
    elif severity is Severity.MINOR:
        status = "Numerical difference exceeds tolerance slightly"
    else:
        status = "Numerical difference exceeds tolerance strongly"

    # Keep scalar-style output for true scalar inputs
    if is_number(a_value) and is_number(b_value):
        abs_diff_out: Any = abs_diffs[0]
        rel_diff_out: Any = rel_diffs[0]
    else:
        abs_diff_out = abs_diffs
        rel_diff_out = rel_diffs

    return ComparisonEntry(
        key=key,
        category="numeric",
        present_in_a=True,
        present_in_b=True,
        value_a=a_value,
        value_b=b_value,
        abs_diff=abs_diff_out,
        rel_diff=rel_diff_out,
        severity=severity,
        status=status,
    )

def compare_solution_payload(
    payload_a: Mapping[str, Any],
    payload_b: Mapping[str, Any],
    tolerances: Tolerances,
) -> List[ComparisonEntry]:
    """
    Compare a level-4 solution payload.

    Only keys in EXACT_KEYS and NUMERIC_KEYS are considered, and only when present
    in at least one side.
    """
    entries: List[ComparisonEntry] = []

    for key in sorted(EXACT_KEYS):
        a_has = key in payload_a
        b_has = key in payload_b
        if a_has or b_has:
            entries.append(compare_exact_key(key, a_has, b_has, payload_a.get(key), payload_b.get(key)))

    for key in sorted(NUMERIC_KEYS):
        a_has = key in payload_a
        b_has = key in payload_b
        if a_has or b_has:
            entries.append(compare_numeric_key(key, a_has, b_has, payload_a.get(key), payload_b.get(key), tolerances))

    return entries


def median(values: Sequence[float]) -> Optional[float]:
    """Median of a sequence, None if it is empty."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank quantile of a sequence, None if it is empty."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


def collect_sim_time_ratios(
    payload_a: Mapping[str, Any],
    payload_b: Mapping[str, Any],
) -> List[float]:
    """Compute time ratios B/A of one solution payload, short runs excluded."""
    a_value = payload_a.get(SIM_TIME_KEY)
    b_value = payload_b.get(SIM_TIME_KEY)

    if not is_scalar_or_number_sequence(a_value) or not is_scalar_or_number_sequence(b_value):
        return []

    a_values = normalize_numeric_value(a_value)
    b_values = normalize_numeric_value(b_value)

    return [
        b / a for a, b in zip(a_values, b_values)
        if a > SIM_TIME_FLOOR and b > 0.0
    ]


def last_converging_level(
    errors: Sequence[float],
    eocs: Sequence[float],
) -> Optional[Tuple[float, float, int]]:
    """EOC, error and index of the finest level that still resolves the solution.

    Walks the series from the finest level down to the first level whose error
    fell compared to its predecessor. Levels beyond that have reached the
    accuracy of the reference solution, where the EOC says nothing about the
    method, which is what would otherwise look like a sudden loss of order.
    """
    n_levels = min(len(errors), len(eocs))

    for index in range(n_levels - 1, 0, -1):
        if errors[index] < errors[index - 1] * EOC_SATURATION_FACTOR:
            return eocs[index], errors[index], index

    return None


def screen_eoc(
    relative_path: str,
    method: str,
    solution: str,
    payload_a: Mapping[str, Any],
    payload_b: Mapping[str, Any],
) -> List[EocFinding]:
    """Report the norms whose order of convergence got worse from A to B."""
    findings: List[EocFinding] = []

    for error_key, eoc_key in ERROR_EOC_KEYS:

        series = [payload.get(key)
                  for payload in (payload_a, payload_b)
                  for key in (error_key, eoc_key)]

        if not all(is_number_sequence(entry) for entry in series):
            continue

        level_a = last_converging_level(payload_a[error_key], payload_a[eoc_key])
        level_b = last_converging_level(payload_b[error_key], payload_b[eoc_key])

        if level_a is None or level_b is None:
            continue

        eoc_a, error_a, index_a = level_a
        eoc_b, error_b, index_b = level_b

        # A method that is more accurate than before has lost nothing, whatever
        # its EOC does: a series approaching the accuracy of its reference flattens
        # out, which is a statement about the reference and not about the method.
        if error_b <= error_a * ERROR_GROWTH_FACTOR:
            continue

        if eoc_b < EOC_NONCONVERGENT <= eoc_a:
            kind = "new_nonconvergent"
        elif eoc_b < eoc_a - EOC_DROP and eoc_b < EOC_SUSPICIOUS:
            kind = "worse"
        else:
            continue

        findings.append(EocFinding(
            relative_path=relative_path,
            method=method,
            solution=solution,
            key=error_key,
            kind=kind,
            eoc_a=eoc_a,
            eoc_b=eoc_b,
            error_a=error_a,
            error_b=error_b,
            level_a=index_a,
            level_b=index_b,
        ))

    return findings


def collect_json_methods(path: Path) -> List[str]:
    """Method names of one convergence file, empty if it cannot be read."""
    data, error = load_json_file(path)
    if error is not None:
        return []

    convergence, schema_error = validate_convergence_root(data)
    if schema_error is not None or convergence is None:
        return []

    return sorted(str(key) for key in convergence.keys())


def build_method_inventory(
    file_results: Sequence[FileResult],
    methods_only_in_a: Mapping[str, Sequence[str]],
    methods_only_in_b: Mapping[str, Sequence[str]],
) -> MethodInventory:
    """Count how many files each method name occurs in, per tree."""
    counts_a: Counter[str] = Counter()
    counts_b: Counter[str] = Counter()

    for file_result in file_results:
        counts_a.update(file_result.methods_a)
        counts_b.update(file_result.methods_b)

    for methods in methods_only_in_a.values():
        counts_a.update(methods)
    for methods in methods_only_in_b.values():
        counts_b.update(methods)

    return MethodInventory(
        counts_a=dict(sorted(counts_a.items())),
        counts_b=dict(sorted(counts_b.items())),
        only_in_a=sorted(set(counts_a) - set(counts_b)),
        only_in_b=sorted(set(counts_b) - set(counts_a)),
    )


def build_sim_time_analysis(file_results: Sequence[FileResult]) -> SimTimeAnalysis:
    """Aggregate the compute time ratios of all files, normalized by the run."""
    all_ratios: List[float] = []
    files: List[FileSimTime] = []

    for file_result in file_results:
        ratios = file_result.sim_time_ratios
        if not ratios:
            continue
        all_ratios.extend(ratios)
        file_median = median(ratios)
        assert file_median is not None
        files.append(FileSimTime(
            relative_path=file_result.relative_path,
            n_values=len(ratios),
            median_ratio=file_median,
        ))

    analysis = SimTimeAnalysis(n_values=len(all_ratios), files=files)

    if not all_ratios:
        return analysis

    run_median = median(all_ratios)
    assert run_median is not None

    analysis.median_ratio = run_median
    analysis.p10_ratio = quantile(all_ratios, 0.1)
    analysis.p90_ratio = quantile(all_ratios, 0.9)
    analysis.min_ratio = min(all_ratios)
    analysis.max_ratio = max(all_ratios)

    if run_median > 0.0:
        for entry in files:
            entry.normalized_ratio = entry.median_ratio / run_median

        analysis.outliers = sorted(
            (entry for entry in files
             if entry.normalized_ratio is not None
             and (entry.normalized_ratio > SIM_TIME_OUTLIER_FACTOR
                  or entry.normalized_ratio < 1.0 / SIM_TIME_OUTLIER_FACTOR)),
            key=lambda entry: entry.normalized_ratio or 0.0,
            reverse=True,
        )

    analysis.files = sorted(files, key=lambda entry: entry.relative_path)
    return analysis


def compare_shared_file(
    relative_path: str,
    path_a: Path,
    path_b: Path,
    tolerances: Tolerances,
) -> FileResult:
    """
    Compare one JSON file that exists in both trees.

    This function is deliberately robust: all parse and schema issues are reported
    in the FileResult rather than raising exceptions.
    """
    result = FileResult(relative_path=relative_path, status="compared")

    data_a, err_a = load_json_file(path_a)
    data_b, err_b = load_json_file(path_b)

    if err_a:
        result.errors.append(make_file_error("A", "parse_error", err_a))
    if err_b:
        result.errors.append(make_file_error("B", "parse_error", err_b))

    if result.errors:
        result.status = "error"
        result.highest_severity = Severity.ERROR
        return result

    conv_a, schema_err_a = validate_convergence_root(data_a)
    conv_b, schema_err_b = validate_convergence_root(data_b)

    if schema_err_a:
        result.errors.append(make_file_error("A", "schema_error", schema_err_a))
    if schema_err_b:
        result.errors.append(make_file_error("B", "schema_error", schema_err_b))

    if result.errors:
        result.status = "error"
        result.highest_severity = Severity.ERROR
        return result

    assert conv_a is not None
    assert conv_b is not None

    methods_a = sorted(str(k) for k in conv_a.keys())
    methods_b = sorted(str(k) for k in conv_b.keys())

    result.methods_a = methods_a
    result.methods_b = methods_b

    missing_methods_in_a = sorted(set(methods_b) - set(methods_a))
    missing_methods_in_b = sorted(set(methods_a) - set(methods_b))

    if missing_methods_in_a or missing_methods_in_b:
        comment_parts: List[str] = []
        if missing_methods_in_a:
            comment_parts.append("Method(s) missing in A")
        if missing_methods_in_b:
            comment_parts.append("Method(s) missing in B")
        diff = StructuralDifference(
            level="method",
            parent=CONVERGENCE_KEY,
            missing_in_a=missing_methods_in_a,
            missing_in_b=missing_methods_in_b,
            severity=Severity.MISSING,
            comment="; ".join(comment_parts),
        )
        result.structural_differences.append(diff)
        result.highest_severity = update_highest(result.highest_severity, diff.severity)

    shared_methods = sorted(set(methods_a) & set(methods_b))

    for method in shared_methods:
        method_result = MethodResult(method=method)
        method_payload_a = conv_a.get(method)
        method_payload_b = conv_b.get(method)

        if not isinstance(method_payload_a, Mapping):
            method_result.structural_differences.append(
                StructuralDifference(
                    level="method_payload",
                    parent=method,
                    missing_in_a=[],
                    missing_in_b=[],
                    severity=Severity.ERROR,
                    comment='Method payload in A is not an object under "convergence"',
                )
            )
            method_result.highest_severity = Severity.ERROR
            result.methods.append(method_result)
            result.highest_severity = update_highest(result.highest_severity, method_result.highest_severity)
            continue

        if not isinstance(method_payload_b, Mapping):
            method_result.structural_differences.append(
                StructuralDifference(
                    level="method_payload",
                    parent=method,
                    missing_in_a=[],
                    missing_in_b=[],
                    severity=Severity.ERROR,
                    comment='Method payload in B is not an object under "convergence"',
                )
            )
            method_result.highest_severity = Severity.ERROR
            result.methods.append(method_result)
            result.highest_severity = update_highest(result.highest_severity, method_result.highest_severity)
            continue

        solutions_a = sorted(str(k) for k in method_payload_a.keys())
        solutions_b = sorted(str(k) for k in method_payload_b.keys())

        missing_solutions_in_a = sorted(set(solutions_b) - set(solutions_a))
        missing_solutions_in_b = sorted(set(solutions_a) - set(solutions_b))

        if missing_solutions_in_a or missing_solutions_in_b:
            comment_parts = []
            if missing_solutions_in_a:
                comment_parts.append("Solution group(s) missing in A")
            if missing_solutions_in_b:
                comment_parts.append("Solution group(s) missing in B")
            diff = StructuralDifference(
                level="solution",
                parent=method,
                missing_in_a=missing_solutions_in_a,
                missing_in_b=missing_solutions_in_b,
                severity=Severity.MISSING,
                comment="; ".join(comment_parts),
            )
            method_result.structural_differences.append(diff)
            method_result.highest_severity = update_highest(method_result.highest_severity, diff.severity)

        shared_solutions = sorted(set(solutions_a) & set(solutions_b))

        for solution in shared_solutions:
            solution_payload_a = method_payload_a.get(solution)
            solution_payload_b = method_payload_b.get(solution)

            solution_result = SolutionResult(solution=solution)

            if not isinstance(solution_payload_a, Mapping):
                solution_result.comparisons.append(
                    ComparisonEntry(
                        key="<solution_payload>",
                        category="schema",
                        present_in_a=True,
                        present_in_b=True,
                        value_a=solution_payload_a,
                        value_b=solution_payload_b,
                        abs_diff=None,
                        rel_diff=None,
                        severity=Severity.ERROR,
                        status="Solution payload in A is not an object",
                    )
                )
                solution_result.highest_severity = Severity.ERROR
                method_result.solutions.append(solution_result)
                method_result.highest_severity = update_highest(method_result.highest_severity, solution_result.highest_severity)
                continue

            if not isinstance(solution_payload_b, Mapping):
                solution_result.comparisons.append(
                    ComparisonEntry(
                        key="<solution_payload>",
                        category="schema",
                        present_in_a=True,
                        present_in_b=True,
                        value_a=solution_payload_a,
                        value_b=solution_payload_b,
                        abs_diff=None,
                        rel_diff=None,
                        severity=Severity.ERROR,
                        status="Solution payload in B is not an object",
                    )
                )
                solution_result.highest_severity = Severity.ERROR
                method_result.solutions.append(solution_result)
                method_result.highest_severity = update_highest(method_result.highest_severity, solution_result.highest_severity)
                continue

            update_sim_time_summary(
                method_result.sim_time_summary,
                solution,
                solution_payload_a,
                solution_payload_b,
            )

            result.sim_time_ratios.extend(
                collect_sim_time_ratios(solution_payload_a, solution_payload_b)
            )

            result.eoc_findings.extend(screen_eoc(
                relative_path, method, solution,
                solution_payload_a, solution_payload_b,
            ))

            solution_result.comparisons = compare_solution_payload(solution_payload_a, solution_payload_b, tolerances)
            for entry in solution_result.comparisons:
                solution_result.highest_severity = update_highest(solution_result.highest_severity, entry.severity)

            method_result.solutions.append(solution_result)
            method_result.highest_severity = update_highest(method_result.highest_severity, solution_result.highest_severity)

        result.methods.append(method_result)
        result.highest_severity = update_highest(result.highest_severity, method_result.highest_severity)

    return result


def count_severities(report_files: Sequence[FileResult], missing_files: MissingFiles) -> Dict[str, int]:
    """Count all findings by severity across the whole report."""
    counts: Counter[str] = Counter()

    counts[Severity.MISSING.value] += len(missing_files.only_in_a)
    counts[Severity.MISSING.value] += len(missing_files.only_in_b)

    for file_result in report_files:
        for error in file_result.errors:
            counts[error.severity.value] += 1

        for diff in file_result.structural_differences:
            counts[diff.severity.value] += 1

        for method in file_result.methods:
            for diff in method.structural_differences:
                counts[diff.severity.value] += 1
            for solution in method.solutions:
                for entry in solution.comparisons:
                    counts[entry.severity.value] += 1

    for severity in Severity:
        counts.setdefault(severity.value, 0)

    return dict(sorted(counts.items(), key=lambda kv: SEVERITY_ORDER[kv[0]]))


def has_failure_condition(counts_by_severity: Mapping[str, int]) -> bool:
    """Return True if any MAJOR, MISSING, or ERROR findings are present."""
    return (
        counts_by_severity.get(Severity.MAJOR.value, 0) > 0
        or counts_by_severity.get(Severity.MISSING.value, 0) > 0
        or counts_by_severity.get(Severity.ERROR.value, 0) > 0
    )


def make_report(
    path_a: Path,
    path_b: Path,
    tolerances: Tolerances,
    missing_files: MissingFiles,
    file_results: List[FileResult],
    methods_only_in_a: Optional[Mapping[str, Sequence[str]]] = None,
    methods_only_in_b: Optional[Mapping[str, Sequence[str]]] = None,
) -> Report:
    """Assemble the full report object."""
    counts = count_severities(file_results, missing_files)
    summary = Summary(
        shared_files=len(file_results),
        files_only_in_a=len(missing_files.only_in_a),
        files_only_in_b=len(missing_files.only_in_b),
        counts_by_severity=counts,
        has_non_ok=has_failure_condition(counts) or counts.get(Severity.MINOR.value, 0) > 0,
    )

    exit_code = 1 if has_failure_condition(counts) else 0

    metadata = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "timestamp_utc": utc_timestamp(),
        "python_version": platform.python_version(),
        "exit_code": exit_code,
    }

    return Report(
        metadata=metadata,
        inputs={
            "path_a": str(path_a.resolve()),
            "path_b": str(path_b.resolve()),
        },
        tolerances={
            "abs_tol": tolerances.abs_tol,
            "rel_tol": tolerances.rel_tol,
            "minor_multiplier": tolerances.minor_multiplier,
        },
        summary=summary,
        missing_files=missing_files,
        files=file_results,
        method_inventory=build_method_inventory(
            file_results, methods_only_in_a or {}, methods_only_in_b or {}
        ),
        sim_time_analysis=build_sim_time_analysis(file_results),
        eoc_findings=[finding
                      for file_result in file_results
                      for finding in file_result.eoc_findings],
    )


def markdown_escape(text: str) -> str:
    """Escape Markdown table separators minimally."""
    return text.replace("|", "\\|")


def render_summary_table(counts_by_severity: Mapping[str, int]) -> str:
    """Render counts-by-severity as a Markdown table with descriptions."""
    lines = [
        "| Severity | Count | Description |",
        "|---|---:|---|",
    ]
    for severity in sorted(counts_by_severity.keys(), key=lambda s: SEVERITY_ORDER[s]):
        description = SEVERITY_DESCRIPTIONS.get(severity, "")
        lines.append(
            f"| {severity} | {counts_by_severity[severity]} | {markdown_escape(description)} |"
        )
    return "\n".join(lines)


def render_sim_time_summary_table(summary: MethodSimTimeSummary) -> str:
    """Render method-level Sim. time deviation summary."""
    lines = [
        "| Max diff | Worst solution | Index | Max rel diff | Worst solution | Index |",
        "|---:|---|---:|---:|---|---:|",
        "| {abs_diff} | {abs_sol} | {abs_idx} | {rel_diff} | {rel_sol} | {rel_idx} |".format(
            abs_diff=format_float(summary.max_abs_diff),
            abs_sol=markdown_escape(summary.worst_solution_abs or "-"),
            abs_idx=summary.worst_index_abs if summary.worst_index_abs is not None else "-",
            rel_diff=format_float(summary.max_rel_diff),
            rel_sol=markdown_escape(summary.worst_solution_rel or "-"),
            rel_idx=summary.worst_index_rel if summary.worst_index_rel is not None else "-",
        ),
    ]
    return "\n".join(lines)


def render_structural_differences_table(differences: Sequence[StructuralDifference]) -> str:
    """Render structural differences as a Markdown table."""
    lines = [
        "| Level | Parent | Missing in A | Missing in B | Severity | Comment |",
        "|---|---|---|---|---|---|",
    ]
    for diff in differences:
        lines.append(
            "| {level} | {parent} | {mia} | {mib} | {severity} | {comment} |".format(
                level=markdown_escape(diff.level),
                parent=markdown_escape(diff.parent),
                mia=markdown_escape(", ".join(diff.missing_in_a) if diff.missing_in_a else "-"),
                mib=markdown_escape(", ".join(diff.missing_in_b) if diff.missing_in_b else "-"),
                severity=diff.severity.value,
                comment=markdown_escape(diff.comment or "-"),
            )
        )
    return "\n".join(lines)


def render_comparison_table(entries: Sequence[ComparisonEntry]) -> str:
    """Render solution-level comparisons as a Markdown table."""
    lines = [
        "| Key | Value A | Value B | Abs diff | Rel diff | Severity | Status |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| {key} | {value_a} | {value_b} | {abs_diff} | {rel_diff} | {severity} | {status} |".format(
                key=markdown_escape(entry.key),
                value_a=markdown_escape(format_value(entry.value_a)),
                value_b=markdown_escape(format_value(entry.value_b)),
                abs_diff=format_float(entry.abs_diff),
                rel_diff=format_float(entry.rel_diff),
                severity=entry.severity.value,
                status=markdown_escape(entry.status),
            )
        )
    return "\n".join(lines)


def has_non_ok_comparisons(entries: Sequence[ComparisonEntry]) -> bool:
    """Return True if any comparison entry is not OK."""
    return any(entry.severity is not Severity.OK for entry in entries)


def render_method_inventory_table(inventory: MethodInventory) -> str:
    """Render the method inventory of both trees as a Markdown table."""
    names = sorted(set(inventory.counts_a) | set(inventory.counts_b))

    if not names:
        return "_No methods found._"

    rows = ["| Method | Files in A | Files in B | |", "|---|---:|---:|---|"]
    for name in names:
        count_a = inventory.counts_a.get(name, 0)
        count_b = inventory.counts_b.get(name, 0)
        if name in inventory.only_in_a:
            note = "only in A"
        elif name in inventory.only_in_b:
            note = "only in B"
        else:
            note = ""
        rows.append(f"| {markdown_escape(name)} | {count_a} | {count_b} | {note} |")

    return "\n".join(rows)


def render_eoc_findings_table(findings: Sequence[EocFinding]) -> str:
    """Render the convergence screen findings as a Markdown table."""
    if not findings:
        return "_No norm lost order of convergence._"

    rows = [
        "| File | Method | Solution | Norm | Finding | EOC A | EOC B | Error A | Error B |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for finding in findings:
        rows.append(
            f"| `{finding.relative_path}` | {markdown_escape(finding.method)} "
            f"| {markdown_escape(finding.solution)} | {markdown_escape(finding.key)} "
            f"| {finding.kind} | {finding.eoc_a:.3f} | {finding.eoc_b:.3f} "
            f"| {format_float(finding.error_a)} | {format_float(finding.error_b)} |"
        )

    return "\n".join(rows)


def render_sim_time_analysis(analysis: SimTimeAnalysis) -> str:
    """Render the compute time ratios and their outliers as Markdown."""
    if analysis.median_ratio is None:
        return "_No comparable compute times._"

    lines = [
        f"- Simulations compared: **{analysis.n_values}**",
        f"- Median ratio B/A: **{analysis.median_ratio:.3f}**",
        f"- p10 / p90: `{analysis.p10_ratio:.3f}` / `{analysis.p90_ratio:.3f}`",
        f"- Minimum / maximum: `{analysis.min_ratio:.3f}` / `{analysis.max_ratio:.3f}`",
        "",
        "### Outliers, normalized by the median of the run",
        "",
    ]

    if not analysis.outliers:
        lines.append("_No file deviates from the median of the run._")
        return "\n".join(lines)

    lines.append("| File | Simulations | Median B/A | Normalized |")
    lines.append("|---|---:|---:|---:|")
    for entry in analysis.outliers:
        normalized = entry.normalized_ratio if entry.normalized_ratio is not None else float("nan")
        lines.append(
            f"| `{entry.relative_path}` | {entry.n_values} "
            f"| {entry.median_ratio:.3f} | {normalized:.2f}x |"
        )

    return "\n".join(lines)


def render_markdown_report(report: Report) -> str:
    """Create the human-readable Markdown report."""
    lines: List[str] = []

    lines.append("# Convergence Comparison Report")
    lines.append("")
    lines.append(f"- Compared directory A: `{report.inputs['path_a']}`")
    lines.append(f"- Compared directory B: `{report.inputs['path_b']}`")
    lines.append(f"- Timestamp (UTC): `{report.metadata['timestamp_utc']}`")
    lines.append(f"- Absolute tolerance: `{report.tolerances['abs_tol']}`")
    lines.append(f"- Relative tolerance: `{report.tolerances['rel_tol']}`")
    lines.append(f"- Minor multiplier: `{report.tolerances['minor_multiplier']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Shared files compared: **{report.summary.shared_files}**")
    lines.append(f"- Files only in A: **{report.summary.files_only_in_a}**")
    lines.append(f"- Files only in B: **{report.summary.files_only_in_b}**")
    lines.append(f"- Exit code: **{report.metadata['exit_code']}**")
    lines.append("")
    lines.append("### Counts by severity")
    lines.append("")
    lines.append(render_summary_table(report.summary.counts_by_severity))
    lines.append("")

    lines.append("## Method inventory")
    lines.append("")
    lines.append(
        "Number of files each method name occurs in. A method that lost files "
        "in B while another gained them is a rename, which the per-file "
        "comparison can only report as missing on both sides."
    )
    lines.append("")
    lines.append(render_method_inventory_table(report.method_inventory))
    lines.append("")

    lines.append("## Convergence screen")
    lines.append("")
    lines.append(
        "Norms whose order of convergence got worse from A to B, judged on the "
        "finest refinement level whose error still fell, so that levels "
        "saturated at the accuracy of the reference are left out. Advisory: "
        "these findings do not enter the severity counts or the exit code."
    )
    lines.append("")
    lines.append(render_eoc_findings_table(report.eoc_findings))
    lines.append("")

    lines.append("## Compute times")
    lines.append("")
    lines.append(
        "Ratios B/A of simulations longer than "
        f"{SIM_TIME_FLOOR} s. Runs on different machines differ by a factor "
        "that applies to the whole run, which the median gives; the files "
        "listed as outliers are the ones that deviate from it by more than a "
        f"factor of {SIM_TIME_OUTLIER_FACTOR}."
    )
    lines.append("")
    lines.append(render_sim_time_analysis(report.sim_time_analysis))
    lines.append("")

    lines.append("## Files only in A")
    lines.append("")
    if report.missing_files.only_in_a:
        for rel in report.missing_files.only_in_a:
            lines.append(f"- `{rel}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Files only in B")
    lines.append("")
    if report.missing_files.only_in_b:
        for rel in report.missing_files.only_in_b:
            lines.append(f"- `{rel}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Per-file results")
    lines.append("")

    for file_result in sorted(report.files, key=lambda f: f.relative_path):
        lines.append(f"## File: `{file_result.relative_path}`")
        lines.append("")
        lines.append(f"- Status: **{file_result.status}**")
        lines.append(f"- Highest severity: **{file_result.highest_severity.value}**")
        lines.append("")

        if file_result.errors:
            lines.append("### Errors")
            lines.append("")
            lines.append("| Side | Type | Severity | Message |")
            lines.append("|---|---|---|---|")
            for err in file_result.errors:
                lines.append(
                    f"| {markdown_escape(err.side)} | {markdown_escape(err.error_type)} | "
                    f"{err.severity.value} | {markdown_escape(err.message)} |"
                )
            lines.append("")

        if file_result.structural_differences:
            lines.append("### Structural differences")
            lines.append("")
            lines.append(render_structural_differences_table(file_result.structural_differences))
            lines.append("")

        if not file_result.methods and not file_result.errors:
            lines.append("_No shared methods available for detailed comparison._")
            lines.append("")

        for method in sorted(file_result.methods, key=lambda m: m.method):
            lines.append(f"### Method: `{method.method}`")
            lines.append("")
            lines.append(f"- Highest severity: **{method.highest_severity.value}**")
            lines.append("")

            lines.append("#### Sim. time summary")
            lines.append("")
            lines.append(render_sim_time_summary_table(method.sim_time_summary))
            lines.append("")

            if method.structural_differences:
                lines.append("#### Method-level structural differences")
                lines.append("")
                lines.append(render_structural_differences_table(method.structural_differences))
                lines.append("")

            if not method.solutions:
                lines.append("_No shared solution groups available for this method._")
                lines.append("")

            for solution in sorted(method.solutions, key=lambda s: s.solution):
                # Skip perfectly matching solutions entirely
                if solution.highest_severity is Severity.OK:
                    continue
            
                lines.append(f"#### Solution: `{solution.solution}`")
                lines.append("")
                lines.append(f"- Highest severity: **{solution.highest_severity.value}**")
                lines.append("")
            
                relevant_entries = [entry for entry in solution.comparisons if entry.severity is not Severity.OK]
            
                if relevant_entries:
                    lines.append(render_comparison_table(relevant_entries))
                else:
                    lines.append("_No non-OK compared keys to report._")
            
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_json_report(report: Report, output_path: Path) -> None:
    """Write deterministic machine-readable JSON report."""
    jsonable = dataclass_to_jsonable(report)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text_file(text: str, output_path: Path) -> None:
    """Write text to a file using UTF-8 encoding."""
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def print_terminal_summary(report: Report, markdown_path: Path, json_path: Path) -> None:
    """Print a concise terminal summary after execution."""
    print("Comparison complete.")
    print(f"Directory A: {report.inputs['path_a']}")
    print(f"Directory B: {report.inputs['path_b']}")
    print(f"Shared files compared: {report.summary.shared_files}")
    print(f"Files only in A: {report.summary.files_only_in_a}")
    print(f"Files only in B: {report.summary.files_only_in_b}")
    print("Counts by severity:")
    for severity in sorted(report.summary.counts_by_severity.keys(), key=lambda s: SEVERITY_ORDER[s]):
        print(f"  {severity}: {report.summary.counts_by_severity[severity]}")

    inventory = report.method_inventory
    if inventory.only_in_a or inventory.only_in_b:
        print(f"Method names only in A: {', '.join(inventory.only_in_a) or 'none'}")
        print(f"Method names only in B: {', '.join(inventory.only_in_b) or 'none'}")

    print(f"Convergence screen findings (advisory): {len(report.eoc_findings)}")

    analysis = report.sim_time_analysis
    if analysis.median_ratio is not None:
        print(
            f"Compute time ratio B/A: median {analysis.median_ratio:.3f} "
            f"over {analysis.n_values} simulations, "
            f"{len(analysis.outliers)} file(s) off the median of the run"
        )

    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare convergence-test JSON data between two directory trees."
    )
    parser.add_argument("path_a", help="First input directory tree")
    parser.add_argument("path_b", help="Second input directory tree")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the Markdown and JSON reports will be written (default: current directory)",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for floating-point comparisons (default: 1e-12)",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-9,
        help="Relative tolerance for floating-point comparisons (default: 1e-9)",
    )
    parser.add_argument(
        "--minor-multiplier",
        type=float,
        default=10.0,
        help="Multiplier used to distinguish MINOR from MAJOR (default: 10.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress information",
    )
    return parser


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Path, Tolerances]:
    """Validate command-line arguments and convert them to structured values."""
    path_a = Path(args.path_a)
    path_b = Path(args.path_b)
    output_dir = Path(args.output_dir)

    if not path_a.exists():
        raise ValueError(f"path_a does not exist: {path_a}")
    if not path_a.is_dir():
        raise ValueError(f"path_a is not a directory: {path_a}")

    if not path_b.exists():
        raise ValueError(f"path_b does not exist: {path_b}")
    if not path_b.is_dir():
        raise ValueError(f"path_b is not a directory: {path_b}")

    if args.abs_tol < 0:
        raise ValueError("--abs-tol must be non-negative")
    if args.rel_tol < 0:
        raise ValueError("--rel-tol must be non-negative")
    if args.minor_multiplier < 1.0:
        raise ValueError("--minor-multiplier must be >= 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)

    tolerances = Tolerances(
        abs_tol=float(args.abs_tol),
        rel_tol=float(args.rel_tol),
        minor_multiplier=float(args.minor_multiplier),
    )
    return path_a, path_b, output_dir, tolerances


def compare_directory_trees(
    path_a: Path,
    path_b: Path,
    tolerances: Tolerances,
    verbose: bool = False,
) -> Tuple[MissingFiles, List[FileResult], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compare all JSON files in two directory trees.

    Matching is based on the normalized relative path under each root.

    Returns the missing files, the per-file results, and the method names of the
    files that exist on one side only, keyed by relative path.
    """
    files_a = collect_json_files(path_a)
    files_b = collect_json_files(path_b)

    rels_a = set(files_a.keys())
    rels_b = set(files_b.keys())

    only_in_a = sorted(rels_a - rels_b)
    only_in_b = sorted(rels_b - rels_a)
    shared = sorted(rels_a & rels_b)

    missing_files = MissingFiles(only_in_a=only_in_a, only_in_b=only_in_b)

    file_results: List[FileResult] = []
    for rel in shared:
        if verbose:
            print(f"Comparing: {rel}")
        file_results.append(compare_shared_file(rel, files_a[rel], files_b[rel], tolerances))

    file_results.sort(key=lambda fr: fr.relative_path)

    # The method inventory covers the one-sided files too, since a method that
    # only occurs in a file added to B belongs in the overview.
    methods_only_in_a = {rel: collect_json_methods(files_a[rel]) for rel in only_in_a}
    methods_only_in_b = {rel: collect_json_methods(files_b[rel]) for rel in only_in_b}

    return missing_files, file_results, methods_only_in_a, methods_only_in_b


def compare_convergence_data(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point.

    Returns:
        Process exit code:
        - 0 if no MAJOR, MISSING, or ERROR findings are present
        - 1 otherwise
        - 2 for invalid invocation or fatal execution errors
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        path_a, path_b, output_dir, tolerances = validate_args(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    try:
        (missing_files, file_results,
         methods_only_in_a, methods_only_in_b) = compare_directory_trees(
            path_a=path_a,
            path_b=path_b,
            tolerances=tolerances,
            verbose=args.verbose,
        )

        report = make_report(
            path_a=path_a,
            path_b=path_b,
            tolerances=tolerances,
            missing_files=missing_files,
            file_results=file_results,
            methods_only_in_a=methods_only_in_a,
            methods_only_in_b=methods_only_in_b,
        )

        markdown_name = "convergence_comparison_report.md"
        json_name = "convergence_comparison_report.json"

        markdown_path = output_dir / markdown_name
        json_path = output_dir / json_name

        markdown_text = render_markdown_report(report)
        write_text_file(markdown_text, markdown_path)
        write_json_report(report, json_path)

        print_terminal_summary(report, markdown_path, json_path)
        return int(report.metadata["exit_code"])

    except Exception as exc:
        # Fatal unexpected error: do not hide it, but report clearly.
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 2


#%% Usage Example

# path1 = r"C:/Users/jmbr/software/CADET-Verification/test/data/verify_cadet_core_dummyData"
# path2 = r"C:/Users/jmbr/software/CADET-Verification/test/data/verify_cadet_core_v600alpha3"
# output_dir = r"C:/Users/jmbr/software/CADET-Verification/output/test_cadet-core/comparison_report"

# exit_code = compare_convergence_data(
#     [
#         path1,
#         path2,
#         "--output-dir",
#         output_dir,
#         "--abs-tol",
#         "1e-8",
#         "--rel-tol",
#         "1e-1",
#         "--minor-multiplier",
#         "10.0",
#         "--verbose",
#     ]
# )

# print(f"compare_convergence_data finished with exit code {exit_code}")
