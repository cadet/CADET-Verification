import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def str2list(v):
    """A comma separated command line value as a list of its entries."""
    if isinstance(v, (list, tuple)):
        return list(v)
    entries = [entry.strip() for entry in v.split(",") if entry.strip()]
    if not entries:
        raise argparse.ArgumentTypeError("Expected at least one comma separated entry.")
    return entries


def pytest_addoption(parser):
    parser.addoption("--small-test", type=str2bool, default=True)
    parser.addoption("--n-jobs", type=int, default=-1)
    parser.addoption("--delete-h5-files", type=str2bool, default=True)

    # Extra repeats per simulation of a performance benchmark, keeping the
    # fastest compute time; a single measurement carries whatever else the
    # machine was doing.
    parser.addoption("--n-reruns", type=int, default=0)

    # Column geometries the performance benchmarks are run on, comma separated.
    # One at a time keeps the expensive sweeps tractable.
    parser.addoption(
        "--column-geometries", type=str2list,
        default=['RADIAL_FLOW_CYLINDER_SHELL', 'AXIAL_FLOW_FRUSTUM',
                 'AXIAL_FLOW_CYLINDER']
        )

    parser.addoption("--run-performance-tests", type=str2bool, default=False)
    # The column geometry performance benchmarks are selected per physical case,
    # so that the two can be run separately, see scripts/verify_geometries.py
    parser.addoption("--run-performance-sma-tests", type=str2bool, default=False)
    parser.addoption("--run-performance-langmuir-tests", type=str2bool, default=False)
    # Particle treatments the SMA benchmark is run with, comma separated. A
    # general rate particle is resolved in space, which is the case of the
    # publication; a homogeneous one carries no particle grid and so leaves the
    # bulk discretizations as the only difference between the methods.
    parser.addoption(
        "--sma-particle-resolutions", type=str2list,
        default=['HOMOGENEOUS_PARTICLE', 'GENERAL_RATE_PARTICLE']
        )
    # Spatial methods of the performance benchmarks, comma separated, as the
    # polynomial degree of the DG discretization or zero for the WENO finite
    # volume scheme. Defaults to all of them; a subset reruns part of a sweep.
    parser.addoption("--spatial-methods", type=str2list, default=None)
    parser.addoption("--run-validation-tests", type=str2bool, default=False)
    parser.addoption("--run-eoc-tests", type=str2bool, default=False)

    # The reference solution of a column geometry performance benchmark, see
    # scripts/verify_geometries.py. One setting and one geometry per run: a
    # reference is expensive enough that its run is the whole job. Naming
    # neither of the two computes no reference, which is the default.
    # The discretization and the tolerance a reference is computed at are
    # bench_configs.GEOMETRY_REFERENCES and are not overridden per run, so that
    # a reference always carries the resolution the repository records for it.
    parser.addoption("--reference-setting", type=str, default=None)
    parser.addoption("--reference-geometry", type=str, default=None)

    parser.addoption("--commit-message", type=str, default="CADET model test run")
    parser.addoption("--rdm-debug-mode", type=str2bool, default=True)
    parser.addoption("--rdm-push", type=str2bool, default=False)
    parser.addoption("--branch-name", type=str, default="main")
