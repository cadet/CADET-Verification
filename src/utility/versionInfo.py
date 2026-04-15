# -*- coding: utf-8 -*-


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
