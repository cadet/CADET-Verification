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
) -> Tuple[MissingFiles, List[FileResult]]:
    """
    Compare all JSON files in two directory trees.

    Matching is based on the normalized relative path under each root.
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
    return missing_files, file_results


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
        missing_files, file_results = compare_directory_trees(
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
