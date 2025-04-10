import os
from copy import deepcopy
from pathlib import Path
import multiprocessing

from CADETProcess import settings
from CADETProcess.comparison import Comparator
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.optimization import OptimizationProblem, U_NSGA3
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores, FlowSheet, ComponentSystem, Process, \
    LRMPDiscretizationFV
from CADETProcess.simulator import Cadet

# %% Global variables

USE_DLL = True
N_CYCLES = 100
# pre-lim testing on the IBT073 server showed very high variance in runtimes if > 50% of cores were used, so set to ~30%
N_CORES = int(multiprocessing.cpu_count() / 3)
POP_SIZE = 240

CACHE_DIRECTORY: Path | None = None  # e.g.: Path('/dev/shm/jaepel/CADET-Process/cache/')

# %% Pre-processing

if CACHE_DIRECTORY is not None:
    settings.temp_dir = CACHE_DIRECTORY.parent / "tmp"

    if not CACHE_DIRECTORY.exists():
        os.makedirs(CACHE_DIRECTORY, exist_ok=True)

    if not settings.temp_dir.exists():
        os.makedirs(settings.temp_dir, exist_ok=True)

# %% Setup Process

components = ComponentSystem(components=1)

inlet = Inlet(components, "inlet")
inlet.flow_rate = 6.7e-8

outlet = Outlet(components, "outlet")

column = LumpedRateModelWithPores(components, "column")

column.length = 0.014
column.diameter = 0.02
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.75
column.axial_dispersion = 5.75e-8
column.film_diffusion = column.n_comp * [6.9e-6]

flow_sheet = FlowSheet(components, "flow_sheet")
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(outlet)
flow_sheet.add_unit(column)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

process = Process(flow_sheet, "process")
process.cycle_time = 120
process.add_event("load", "flow_sheet.inlet.c", 1, time=0)
process.add_event("wash", "flow_sheet.inlet.c", 0, time=10)


# %% Generate ground truth and comparator

def generate_ground_truth(process):
    simulator = Cadet(use_dll=USE_DLL)
    simulator.n_cycles = 1
    simulator.time_integrator_parameters.abstol = 1e-10
    simulator.time_integrator_parameters.reltol = 1e-8
    local_process = deepcopy(process)
    local_process.flow_sheet.column.discretization.ncol = 500
    ground_truth = simulator.simulate(local_process)
    return ground_truth


ground_truth = generate_ground_truth(process)

comparator = Comparator("ground_truth")
comparator.add_reference(ground_truth.solution_cycles.outlet.outlet[-1])
comparator.add_difference_metric("NRMSE", ground_truth.solution_cycles.outlet.outlet[-1], "outlet.outlet")

# %% Setup optimization Problem

problem = OptimizationProblem(
    f"runtime_abstol_dll{USE_DLL}_ncycles{N_CYCLES}_popsize{POP_SIZE}_ncores{N_CORES}",
    cache_directory=CACHE_DIRECTORY
)

problem.add_variable(name='abstol', lb=1e-5, ub=1e-1, transform='log')
problem.add_variable(name='reltol', lb=1e-3, ub=1e0, transform='log')
problem.add_variable(name='axial_discretization', lb=25, ub=50, transform='auto')


# Because the simulator parameters are modified and the simulator is not part of the process: I don't know how to use
#  problem.add_variable(parameter_path=...) correctly, so it's all wrapped in an objective_function with positional x
def run_simulation(x, n_cycles=N_CYCLES):
    simulator = Cadet(use_dll=USE_DLL)
    simulator.n_cycles = n_cycles
    simulator.time_integrator_parameters.abstol = x[0]
    simulator.time_integrator_parameters.reltol = x[1]
    local_process = deepcopy(process)
    local_process.flow_sheet.column.discretization.ncol = int(x[2])
    sim_res = simulator.simulate(local_process)
    return sim_res


def time_elapsed(x):
    sim_results = run_simulation(x)
    return sim_results.time_elapsed


problem.add_objective(time_elapsed, n_objectives=1, )


def nrmse_comparator(x):
    sim_res = run_simulation(x, n_cycles=1)
    score = comparator.evaluate(sim_res)
    return score


problem.add_nonlinear_constraint(
    nrmse_comparator,
    name="NRMSE",
    n_nonlinear_constraints=comparator.n_metrics,
    bounds=0.01,
    comparison_operator="le",
)

# %% Setup optimizer and run optimization

optimizer = U_NSGA3()
optimizer.n_cores = N_CORES
optimizer.pop_size = POP_SIZE
optimizer.n_max_gen = 30

optimization_results = optimizer.optimize(problem)
