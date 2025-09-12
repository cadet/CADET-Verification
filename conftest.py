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

    parser.addoption("--run-binding-tests", type=str2bool, default=True)
    parser.addoption("--run-chromatography-tests", type=str2bool, default=True)
    parser.addoption("--run-chromatography-sensitivity-tests", type=str2bool, default=True)
    parser.addoption("--run-chromatography-system-tests", type=str2bool, default=True)
    parser.addoption("--run-crystallization-tests", type=str2bool, default=True)
    parser.addoption("--run-mct-tests", type=str2bool, default=True)
    parser.addoption("--run-2dmodels-tests", type=str2bool, default=True)

    parser.addoption("--commit-message", type=str, default="CADET model test run")
    parser.addoption("--rdm-debug-mode", type=str2bool, default=True)
    parser.addoption("--rdm-push", type=str2bool, default=False)
    parser.addoption("--branch-name", type=str, default="main")
