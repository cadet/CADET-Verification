"""CADET H5 configuration helpers for chromatography surrogate workflows.

This module extracts and consolidates the repeated CADET model-building logic
from the notebooks into reusable Python functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Literal, Optional, Sequence

import h5py
import numpy as np

from cadet import Cadet

AdsorptionMode = Literal["MCL", "GPR", "ANN", "SPLINE"]


@dataclass(frozen=True)
class ColumnConfig:
    """Physical and operating settings for one chromatography condition."""

    epsilon_b: float
    epsilon_p: float
    dax: float
    de: float
    do: float
    column_length: float
    column_diameter: float
    residence_time: float
    volumetric_flow_rate_m3_per_sec: float
    loading_concentration: float
    packed_bed_volume_mL: float
    particle_diameter: float
    density_water: float
    water_dynamic_viscosity: float
    kf: float
    time_in_minutes: float
    factor: float
    qmax: float

COLUMN_CONFIGS: Dict[str, ColumnConfig] = {
    "favorable_lysozyme": ColumnConfig(
        epsilon_b=0.256288,
        epsilon_p=0.75,
        dax=7.550718e-08,
        de=2.179105e-11,
        do=1.1e-10,
        column_length=5 / 100,
        column_diameter=0.5046 / 100,
        residence_time=180,
        volumetric_flow_rate_m3_per_sec=5.56e-9,
        loading_concentration=4.55,
        packed_bed_volume_mL=2.5,
        particle_diameter=90e-6,
        density_water=1040.77,
        water_dynamic_viscosity=8.90e-4,
        kf=1e-5,
        time_in_minutes=100,
        factor=0.025280433,
        qmax=148.17,
    ),
    "less_favorable_igg": ColumnConfig(
        epsilon_b=0.215663,
        epsilon_p=0.75,
        dax=1.708813e-13,
        de=2.344211e-12,
        do=1.1e-10,
        column_length=5 / 100,
        column_diameter=0.5046 / 100,
        residence_time=480,
        volumetric_flow_rate_m3_per_sec=2.08333e-09,
        loading_concentration=5.15,
        packed_bed_volume_mL=2.5,
        particle_diameter=89.5e-6,
        density_water=1040.77,
        water_dynamic_viscosity=8.90e-4,
        kf=2e-6,
        time_in_minutes=601,
        factor=0.010730294,
        qmax=121.2622,
    ),
}


def get_cadet_template(n_units: int = 3) -> Cadet:
    """Create a CADET template with shared return flags and solver defaults."""

    cadet_template = Cadet()
    cadet_template.root.input.model.nunits = n_units

    rtn = cadet_template.root.input["return"]
    rtn.split_components_data = 0
    rtn.split_ports_data = 0
    rtn.unit_000.write_solution_inlet = 0
    rtn.unit_000.write_solution_outlet = 1
    rtn.unit_000.write_solution_bulk = 1
    rtn.unit_000.write_solution_particle = 1
    rtn.unit_000.write_solution_solid = 1
    rtn.unit_000.write_solution_flux = 0
    rtn.unit_000.write_solution_volume = 0
    rtn.unit_000.write_coordinates = 1
    rtn.unit_000.write_sens_outlet = 0

    for unit in range(n_units):
        rtn[f"unit_{unit:03d}"] = rtn.unit_000

    cadet_template.root.input.solver.time_integrator.abstol = 1e-6
    cadet_template.root.input.solver.time_integrator.algtol = 1e-6
    cadet_template.root.input.solver.time_integrator.reltol = 1e-6
    cadet_template.root.input.solver.time_integrator.init_step_size = 1e-6
    cadet_template.root.input.solver.time_integrator.max_steps = 1_000_000

    cadet_template.root.input.model.solver.gs_type = 1
    cadet_template.root.input.model.solver.max_krylov = 0
    cadet_template.root.input.model.solver.max_restarts = 10
    cadet_template.root.input.model.solver.schur_safety = 1e-6
    cadet_template.root.input.solver.nthreads = 1

    return cadet_template


def set_discretization(
    model: Cadet,
    n_bound: Optional[Sequence[int]] = None,
    n_col: int = 100,
    n_par: int = 30,
) -> None:
    """Apply GRM/LRM discretization settings from the notebook implementation."""

    columns = {
        "COLUMN_MODEL_1D",
        "GENERAL_RATE_MODEL",
        "LUMPED_RATE_MODEL_WITH_PORES",
        "LUMPED_RATE_MODEL_WITHOUT_PORES",
    }

    for unit_name, unit in model.root.input.model.items():
        if "unit_" not in unit_name or unit.unit_type not in columns:
            continue

        unit.particle_type_000.nbound = list(n_bound) if n_bound is not None else unit.ncomp * [0]
        unit.particle_type_000.discretization.spatial_method = "FV"
        unit.particle_type_000.discretization.ncells = n_par
        unit.particle_type_000.discretization.par_disc_type = "EQUIDISTANT_PAR"
        unit.discretization.spatial_method = "FV"
        unit.discretization.ncol = n_col
        unit.discretization.use_analytic_jacobian = 1
        unit.discretization.reconstruction = "WENO"
        unit.discretization.gs_type = 1
        unit.discretization.max_krylov = 0
        unit.discretization.max_restarts = 10
        unit.discretization.schur_safety = 1e-6
        unit.discretization.weno.boundary_model = 0
        unit.discretization.weno.weno_eps = 1e-8
        unit.discretization.weno.weno_order = 3


def _apply_column_hydrodynamics(model: Cadet, column: ColumnConfig) -> Dict[str, float]:
    """Populate geometry/transport values and return derived flow quantities."""

    cross_section_area = np.pi * (column.column_diameter / 2) ** 2
    volumetric_flow = column.volumetric_flow_rate_m3_per_sec

    model.root.input.model.unit_001.col_length = column.column_length
    model.root.input.model.unit_001.cross_section_area = cross_section_area
    model.root.input.model.unit_001.col_porosity = column.epsilon_b
    model.root.input.model.unit_001.col_dispersion = column.dax
    model.root.input.model.unit_001.npartype = 1
    model.root.input.model.unit_001.particle_type_000.par_porosity = column.epsilon_p
    model.root.input.model.unit_001.particle_type_000.par_radius = column.particle_diameter / 2
    model.root.input.model.unit_001.particle_type_000.has_film_diffusion = column.kf > 0.0
    model.root.input.model.unit_001.particle_type_000.has_pore_diffusion = column.de > 0.0
    model.root.input.model.unit_001.particle_type_000.has_surface_diffusion = False
    model.root.input.model.unit_001.particle_type_000.film_diffusion = [column.kf]
    model.root.input.model.unit_001.particle_type_000.pore_diffusion = [column.de]

    return {
        "Q": volumetric_flow,
        "time_in_seconds": column.time_in_minutes * 60,
        "cross_section_area": cross_section_area,
    }


def _apply_adsorption(
    model: Cadet,
    mode: AdsorptionMode,
    *,
    cp: np.ndarray,
    cs: np.ndarray,
    gpr_params: Optional[np.ndarray],
    gpr_kernel: str,
    ann_weights: Optional[Dict],
    keq: float,
    qm: float,
    epsilon_p: float,
    ann_norm_factor: Optional[float],
    ann_poros_factor: Optional[float],
    ann_impl: Optional[str],
    ann_layers: Optional[int],
) -> None:
    """Configure one of the four adsorption settings used in the notebooks."""

    adsorption = model.root.input.model.unit_001.particle_type_000.adsorption

    if mode == "MCL":
        model.root.input.model.unit_001.particle_type_000.adsorption_model = "MULTI_COMPONENT_LANGMUIR"
        adsorption.is_kinetic = False
        adsorption.mcl_ka = keq
        adsorption.mcl_kd = [1]
        adsorption.mcl_qmax = qm / (1 - epsilon_p)
        return

    if mode == "GPR":
        if gpr_params is None:
            raise ValueError("gpr_params must be provided for mode='GPR'.")
        model.root.input.model.unit_001.particle_type_000.adsorption_model = "GAUSSIAN_PROCESS_REGRESSION"
        adsorption.is_kinetic = True
        adsorption.GPR_KKIN = 1.0
        adsorption.CS_VALS = np.asarray(cs)
        adsorption.CP_VALS = np.asarray(cp)
        adsorption.TRAINED_PARAMS = np.asarray(gpr_params)
        adsorption.KERNAL = gpr_kernel
        adsorption.NDIM = 1
        return

    if mode == "ANN":
        if ann_weights is None:
            raise ValueError("ann_weights must be provided for mode='ANN'.")
        model.root.input.model.unit_001.particle_type_000.adsorption_model = "MACHINE_LEARNING"
        adsorption.ML_KKIN = 1
        adsorption.LAYERS = ann_layers if ann_layers is not None else 2
        adsorption.NORM_FACTOR = keq if ann_norm_factor is None else ann_norm_factor
        adsorption.bound_state_000.POROS_FACTOR = (
            1 / (1 - epsilon_p) if ann_poros_factor is None else ann_poros_factor
        )
        adsorption.bound_state_000.update(ann_weights)
        if ann_impl is not None:
            adsorption.IMPL = ann_impl
        return

    if mode == "SPLINE":
        model.root.input.model.unit_001.particle_type_000.adsorption_model = "SPLINE_INTERPOLATION"
        adsorption.is_kinetic = True
        adsorption.ML_KKIN = 1.0
        adsorption.CS_VALS = np.asarray(cs)
        adsorption.CP_VALS = np.asarray(cp)
        return

    raise ValueError(f"Unsupported adsorption mode: {mode}")


def get_model(
    *,
    file_name: str,
    mode: AdsorptionMode,
    column_key: str,
    cp: Optional[np.ndarray] = None,
    cs: Optional[np.ndarray] = None,
    loading: np.ndarray,
    keq: Optional[float] = None,
    qm: Optional[float] = None,
    gpr_params: Optional[np.ndarray] = None,
    gpr_kernel: str = "MLP",
    ann_weights: Optional[Dict] = None,
    ann_norm_factor: Optional[float] = None,
    ann_poros_factor: Optional[float] = None,
    ann_impl: Optional[str] = None,
    ann_layers: Optional[int] = None,
    **kwargs
) -> Any:
    """Build and return a CADET model object ready to save or run.

    Distinct settings covered by this function:
    - Column setting: favorable_lysozyme, less_favorable_igg
    - Adsorption setting: MCL, GPR, ANN, SPLINE
    """

    if column_key not in COLUMN_CONFIGS:
        raise KeyError(f"Unknown column_key '{column_key}'. Available: {list(COLUMN_CONFIGS)}")

    column = COLUMN_CONFIGS[column_key]
    model = get_cadet_template(n_units=3)
    model.filename = f"{file_name}.h5"

    model.root.input.model.unit_000.unit_type = "INLET"
    model.root.input.model.unit_000.ncomp = 1
    model.root.input.model.unit_000.inlet_type = "PIECEWISE_CUBIC_POLY"

    model.root.input.model.unit_001.unit_type = "COLUMN_MODEL_1D"
    model.root.input.model.unit_001.ncomp = 1

    derived = _apply_column_hydrodynamics(model, column)

    _apply_adsorption(
        model,
        mode,
        cp=np.asarray(cp),
        cs=np.asarray(cs),
        gpr_params=gpr_params,
        gpr_kernel=gpr_kernel,
        ann_weights=ann_weights,
        keq=keq,
        qm=qm,
        epsilon_p=column.epsilon_p,
        ann_norm_factor=ann_norm_factor,
        ann_poros_factor=ann_poros_factor,
        ann_impl=ann_impl,
        ann_layers=ann_layers,
    )

    model.root.input.model.unit_001.init_c = [0.0]
    model.root.input.model.unit_001.particle_type_000.init_cp = [0.0]
    model.root.input.model.unit_001.particle_type_000.init_cs = [0.0]

    model.root.input.model.unit_002.ncomp = 1
    model.root.input.model.unit_002.unit_type = "OUTLET"

    time_simulation = np.asarray(loading) / column.factor
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = [0.0, 5.0, float(time_simulation[-1])]
    model.root.input.solver.sections.section_continuity = [0, 0]

    for sec in ("sec_000", "sec_001"):
        getattr(model.root.input.model.unit_000, sec).const_coeff = [column.loading_concentration]
        getattr(model.root.input.model.unit_000, sec).lin_coeff = [0.0]
        getattr(model.root.input.model.unit_000, sec).quad_coeff = [0.0]
        getattr(model.root.input.model.unit_000, sec).cube_coeff = [0.0]

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0,
        1,
        -1,
        -1,
        derived["Q"],
        1,
        2,
        -1,
        -1,
        derived["Q"],
    ]

    model.root.input.solver.user_solution_times = time_simulation
    set_discretization(model, n_bound=[1], n_col=100, n_par=15)
    return model.root
