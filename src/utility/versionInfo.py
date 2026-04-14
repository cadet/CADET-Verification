# -*- coding: utf-8 -*-


def print_cadet_versions():
    
    from cadet import Cadet
    from importlib.metadata import version
    jo = Cadet()
    print(f"::notice::CADET-Core version: {jo.version}")
    print(f"::notice::CADET-Python version: {version('CADET-Python')}")
    print(f"::notice::CADET-RDM version: {version('CADET-RDM')}")
    # print(f"::notice::CADET-Process version: {version('CADET-Process')}")
