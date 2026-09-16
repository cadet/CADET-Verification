def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def pytest_addoption(parser):
    parser.addoption("--small-test", type=str2bool, default=True)
    parser.addoption("--n-jobs", type=int, default=-1)
    parser.addoption("--delete-h5-files", type=str2bool, default=True)

    # Extra repeats per simulation of a performance benchmark, keeping the
    # fastest compute time; a single measurement carries whatever else the
    # machine was doing.
    parser.addoption("--n-reruns", type=int, default=0)

    parser.addoption("--run-performance-tests", type=str2bool, default=True)
    # The column geometry performance benchmarks are selected per physical case,
    # so that the two can be run separately, see scripts/verify_geometries.py
    parser.addoption("--run-performance-sma-tests", type=str2bool, default=True)
    parser.addoption("--run-performance-langmuir-tests", type=str2bool, default=True)
    parser.addoption("--run-validation-tests", type=str2bool, default=True)
    parser.addoption("--run-eoc-tests", type=str2bool, default=True)

    parser.addoption("--commit-message", type=str, default="CADET model test run")
    parser.addoption("--rdm-debug-mode", type=str2bool, default=True)
    parser.addoption("--rdm-push", type=str2bool, default=False)
    parser.addoption("--branch-name", type=str, default="main")
