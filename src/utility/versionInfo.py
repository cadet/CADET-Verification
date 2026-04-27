# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path


def get_git_info(path=None):
    try:
        repo_path = Path(path or ".").resolve()

        # Get full commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Get branch name (may fail in detached HEAD)
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            # In detached HEAD, git returns "HEAD"
            if branch == "HEAD":
                branch = "detached"
        except Exception:
            branch = "unknown"

        return branch, commit

    except Exception:
        return "unknown", "unknown"
    
    
def print_cadet_versions(cadet_core_path=None):
    
    from cadet import Cadet
    from importlib.metadata import version
    jo = Cadet()
        
    if cadet_core_path is not None:
        print(f"::notice::CADET-Core environment version: {jo.version}")
        jo.install_path=cadet_core_path
        print(f"::notice::CADET-Core provided cli (not environment) version: {jo.version}")
    else:
        print(f"::notice::CADET-Core version: {jo.version}")

    print(f"::notice::CADET-Python version: {version('CADET-Python')}")
    print(f"::notice::CADET-RDM version: {version('CADET-RDM')}")
    # print(f"::notice::CADET-Process version: {version('CADET-Process')}")
    
    branch, commit = get_git_info()
    print(f"::notice::CADET-Verification branch: {branch}")
    print(f"::notice::CADET-Verification commit: {commit}")
