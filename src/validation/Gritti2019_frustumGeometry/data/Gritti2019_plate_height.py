# -*- coding: utf-8 -*-
"""Measured plate heights for the Gritti et al. (2019) conical-column study.

The table below is the measurement behind Fig. 5 of the paper, provided by
Gritti et al. per personal correspondence: the plate height H_1/2 obtained
from the half-height peak width of six n-alkanophenones on the 3.0 mm i.d.
cylindrical reference column, at seven flow rates.

The two z/L columns are the paper's own assignment of each flow rate to the
axial position at which a cone's local velocity matches that of the reference
column, one for each flow direction of the 2.1/4.2 mm tube. They are what
Fig. 5 is plotted against and are kept for that purpose; the fit below does
not use them, since the velocity follows from the flow rate directly.

Figs. 6, 7 and 8 all run on the same particle batch, so they share one H(v),
fitted here once and imported by the three scripts.
"""
import numpy as np

MICRON = 1e-6
ML_MIN = 1e-6 / 60.0   # 1 mL/min -> m^3/s
MIN = 60.0             # 1 min -> s

#: Columns of PLATE_HEIGHT_TABLE below.
PLATE_HEIGHT_COLUMNS = (
    'Fv_mL_min',             # flow rate on the 3.0 mm i.d. reference column
    'zL_cone_2p1_to_4p2',    # matching axial position, 2.1 -> 4.2 mm cone
    'zL_cone_4p2_to_2p1',    # matching axial position, 4.2 -> 2.1 mm cone
    'uracil_H50_um',
    'acetophenone_H50_um',
    'propiophenone_H50_um',
    'butyrophenone_H50_um',
    'valerophenone_H50_um',
    'hexanophenone_H50_um',
)

#: H_1/2 in micron, from the half-height peak width. Gritti et al., personal
#: correspondence; the paper itself shows only the drawn curve of Fig. 5.
PLATE_HEIGHT_TABLE = (
    (0.204, 1.000, 0.000,  7.78,  8.62,  8.81, 10.59, 10.59, 10.79),
    (0.306, 0.931, 0.367,  7.74,  7.94,  8.13,  9.49,  9.60,  9.84),
    (0.408, 0.845, 0.586,  8.23,  7.98,  8.14,  9.34,  9.47,  9.71),
    (0.510, 0.735, 0.735,  9.28,  8.39,  8.55,  9.60,  9.82, 10.04),
    (0.612, 0.586, 0.845, 10.58,  9.06,  9.16, 10.08, 10.32, 10.62),
    (0.714, 0.367, 0.931, 11.35,  9.76,  9.81, 10.70, 10.89, 11.21),
    (0.816, 0.000, 1.000, 12.62, 10.58, 10.60, 11.39, 11.54, 11.88),
)

# The reference column the plate heights were measured on, and the quantities
# its total porosity is derived from (Table 1, p. 44; k on p. 43).
R_CYL = 1.50e-3                    # m, 3.0 mm i.d.
L_BED = 0.15                       # m
FV_CYL = 0.35 * ML_MIN             # m^3/s
MU1_CYL = 3.869 * MIN              # s, first moment of valerophenone
K_RET = 1.08                       # valerophenone retention factor

V_BED_CYL = np.pi * R_CYL ** 2 * L_BED
ET_REF = (MU1_CYL / (1.0 + K_RET)) * FV_CYL / V_BED_CYL   # total porosity


def load_table():
    """The measured plate-height table as a structured array."""
    return np.rec.fromrecords(PLATE_HEIGHT_TABLE, names=','.join(PLATE_HEIGHT_COLUMNS))


def interstitial_velocity(Fv_mL_min, et=ET_REF):
    """Interstitial velocity in the cylindrical reference column [m/s]."""
    return np.asarray(Fv_mL_min) * ML_MIN / (np.pi * R_CYL ** 2) / et


def fit_van_deemter(et=ET_REF, compound='valerophenone'):
    """Least-squares fit of H(v) = A + B/v + C*v to the measured points.

    Returns (A, B, C) in m, m^2/s and s, the coefficients CADET's
    COL_DISPERSION_DEP='VAN_DEEMTER' expects. The van Deemter form is an
    interpolant between the seven measured flow rates; the paper draws a
    smooth curve through the same points and states no functional form.
    """
    table = load_table()
    v = interstitial_velocity(table['Fv_mL_min'], et)
    H = table[compound + '_H50_um'] * MICRON
    basis = np.column_stack([np.ones_like(v), 1.0 / v, v])
    coeffs, *_ = np.linalg.lstsq(basis, H, rcond=None)
    return tuple(coeffs)


def fit_residuals(et=ET_REF, compound='valerophenone'):
    """(RMSE, max|residual|) of the fit above, in m."""
    table = load_table()
    v = interstitial_velocity(table['Fv_mL_min'], et)
    H = table[compound + '_H50_um'] * MICRON
    A, B, C = fit_van_deemter(et, compound)
    resid = (A + B / v + C * v) - H
    return float(np.sqrt(np.mean(resid ** 2))), float(np.max(np.abs(resid)))


# Valerophenone is the analyte of Figs. 6 and 7; Fig. 8 reuses the same curve
# because the paper reports no plate height for bombesin.
VD_A, VD_B, VD_C = fit_van_deemter()
