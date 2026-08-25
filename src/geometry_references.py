# -*- coding: utf-8 -*-
"""

Reference solutions for the column geometry convergence studies, see
src/column_geometries.py.

Every physical setting of the study has at most one reference, which all spatial
methods of that setting are compared against. It is the best one available:

Analytical reference
    Used wherever an analytical solution of the setting exists, i.e. for the
    pure advection and the pure dispersion transport settings in all
    geometries, see src/analytical.py. The references are stored in
    data/CADET-Verification_reference and are found by the setting name.

CADET reference
    Used for the remaining settings, i.e. combined advection and dispersion and
    the settings with particles, for which no analytical solution is available.
    Such a reference is a stored CADET solution at a resolution finer than every
    point of the sweep. It is named explicitly by the setting, i.e. hardcoded to
    a file of data/CADET-Core_reference/transport, which the geometry studies
    share with the other transport EOC studies, see src/transport_convDisp.py.
    Being no point of the sweep, it costs no refinement level.

Neither kind is required. A setting that names no CADET reference, or names one
that is not in the reference data directory, is reported as None, upon which
bench_func.run_convergence_analysis falls back to its built-in self-convergence,
i.e. it computes the finest discretization of each method itself and excludes it
from that method's sweep.

References are inputs, not results: running a study never computes or rewrites
one. A new CADET reference is produced by simulating the intended resolution
once, storing it in the reference data directory and naming it in the setting.

"""

import os

from src import analytical
from src.utility import convergence


# Subdirectories of the data directory holding the two kinds of reference. The CADET
# references share the transport directory with the other transport EOC studies, see
# src/transport_convDisp.py, from which the geometry studies are run.
ANALYTICAL_SUBDIR = 'CADET-Verification_reference'
CADET_SUBDIR = os.path.join('CADET-Core_reference', 'transport')


def analytical_reference(setting_name, data_dir):
    """Return the analytical reference of a setting, or None.

    Parameters
    ----------
    setting_name : string
        Name of the setting.
    data_dir : string
        Path of the data directory, i.e. the parent of the reference
        subdirectories. None disables all references.

    Returns
    -------
    Cadet object or None
    """
    if data_dir is None:
        return None
    return analytical.load_reference(
        setting_name, os.path.join(data_dir, ANALYTICAL_SUBDIR)
    )


def cadet_reference_path(file_name, data_dir):
    """Return the full path of a stored CADET reference solution.

    Parameters
    ----------
    file_name : string
        File name of the reference, as named by the setting.
    data_dir : string
        Path of the data directory.

    Returns
    -------
    string
        Full path of the reference file.
    """
    return os.path.join(data_dir, CADET_SUBDIR, file_name)


def resolve(setting_name, ax_methods, data_dir, cadet_reference=None, verbose=True):
    """Return the reference of a setting, once per method.

    The analytical reference of the setting is used if it exists. Otherwise the
    CADET reference named by the setting is used, if it is present in the reference
    data directory. If neither is available the reference is None, which makes
    bench_func.run_convergence_analysis fall back to self-convergence.

    Parameters
    ----------
    setting_name : string
        Name of the setting.
    ax_methods : list of int
        Polynomial degree (DG) or 0 (FV) per method. Only its length is used, since
        all methods share the reference.
    data_dir : string
        Path of the data directory. None disables all references, i.e. falls
        back to self-convergence.
    cadet_reference : string
        File name of the CADET reference of this setting in the reference data
        directory. None for a setting that has no such reference.
    verbose : bool
        Report which reference is used.

    Returns
    -------
    list
        The reference, repeated once per method, entries are Cadet objects or None.
    """
    reference = analytical_reference(setting_name, data_dir)
    if reference is not None:
        if verbose:
            print(f"{setting_name}: analytical reference for all methods")
        return [reference] * len(ax_methods)

    if data_dir is not None and cadet_reference is not None:
        path = cadet_reference_path(cadet_reference, data_dir)
        if os.path.exists(path):
            if verbose:
                print(f"{setting_name}: CADET reference {cadet_reference} "
                      "for all methods")
            return [convergence.get_simulation(path)] * len(ax_methods)

        if verbose:
            print(f"WARNING {setting_name}: the reference {cadet_reference} named by "
                  "this setting is not in the reference data directory, falling back "
                  "to self-convergence")
        return [None] * len(ax_methods)

    if verbose:
        print(f"{setting_name}: no reference, using self-convergence")

    return [None] * len(ax_methods)
