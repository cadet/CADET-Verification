'''
Created January 2025

This script implements the EOC tests to verify the aggregation and fragmentation,
part of the crystallization models implemented in CADET-Core. Additionally, all
combinations of aggregation, fragmentation and population balance model (PBM) are tested.
Further, the incorporation into a DPFR transport model is tested.

Similar verification studies were published in Zhang et al.
    'Solving crystallization/precipitation population balance models in CADET,
    Part II: Size-based Smoluchowski coagulation and fragmentation equations
    in batch and continuous modes' (2025)

@author: Wendi Zhang and Jan M. Breuer
'''

from mpmath import *  # used to compute a high precision reference solution
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import re
import json

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
from benchmark_models import settings_crystallization


# %% Helper functions

def calculate_normalized_error(ref, sim, x_ct, x_grid):
    area = np.trapezoid(ref, x_ct)

    L1_error = 0.0
    for i in range(0, len(ref)):
        L1_error += np.absolute(ref[i] - sim[i]) * (x_grid[i+1] - x_grid[i])

    return L1_error / area


def get_slope(error):
    return [-np.log2(error[i] / error[i-1]) for i in range(1, len(error))]


# %% Pure aggregation
def crystallization_aggregation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Pure aggregation tests against analytical solutions for the Golovin (sum)
    kernel, EOC tests. Assumes no solute (c) and solubility component (cs).
    The Golovin kernel should cover all functions implemented in the core simulator. 
    '''

    Cadet.cadet_path = cadet_path
    
    # define params
    x_c, x_max = 5e-3, 0.5e3  # m # crystal phase discretization
    cycle_time = 3.0
    time_res = 30
    t = np.linspace(0, cycle_time, time_res)
    v_0 = 1  # initial volume of particles
    N_0 = 1  # initial number of particles
    beta_0 = 1.0  # aggregation rate

    # analytical solution
    '''
    The analytical solution requires a high-precision floating point package to accurately calculate extremely large or small values from special functions. 
    mpmath is used here. 
    '''
    mp.dps = 50

    # analytical solution, see Zhang et al. (2025)
    def get_analytical_agg(n_x):
        T_t1 = 1.0 - exp(-N_0*beta_0*cycle_time*v_0)  # dimensionless time

        x_grid_mp = []
        for i in range(0, n_x+1): 
            x_grid_mp.append(
                power(10, linspace(log10(x_c), log10(x_max), n_x+1)[i]))

        x_ct = [(x_grid_mp[p+1] + x_grid_mp[p]) / 2 for p in range(0, n_x)]

        analytical_t1 = [3.0 * N_0 * (1.0-T_t1) * exp(-x_ct[k]**3 * (1.0+T_t1)/v_0) * besseli(
            1, 2.0*x_ct[k]**3 * sqrt(T_t1)/v_0) / x_ct[k] / sqrt(T_t1) for k in range(n_x)]

        return analytical_t1

    '''
    EOC tests, pure aggregation
    '''
    # run sims
    normalized_l1 = []

    Nx_grid = np.asarray([24, 48, 96, 192, 384]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, 384, 768])

    sim_times = []

    for n_x in Nx_grid:
        
        model, x_grid, x_ct = settings_crystallization.PureAgg_Golovin(
            n_x, x_c, x_max, v_0, N_0, beta_0, t, output_path=output_path
        )
        
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()
        
        sim_times.append(model.root.meta.time_sim)
        sim = model.root.output.solution.unit_001.solution_outlet[-1, :]
        analytical1 = get_analytical_agg(n_x)
        normalized_l1.append(calculate_normalized_error(
            analytical1, sim, x_ct, x_grid))
        
        if n_x == Nx_grid[-1]:
            plt.xscale("log")
            plt.xlabel(r'size/$\mu m$')
            plt.ylabel('particle number/1')
            plt.plot(x_ct, analytical1, label='Analytical')
            plt.plot(x_ct, sim, label='Numerical', linestyle='dashed')
            plt.legend(frameon=0)
            plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_CSTR_aggregation"), dpi=100, bbox_inches='tight')

    # print the slopes
    # The last value in this array should be around 1.2, see Zhang et al. (2025) for details
    EOC = []
    for i in range(0, len(normalized_l1)-1):
        EOC.append(log(normalized_l1[i] / normalized_l1[i+1], mp.e) / log(2.0))
    print(EOC)

    data = {
        'convergence': {
            "Nx": Nx_grid.tolist(),
            "L^1 error": [float(x) for x in normalized_l1],
            "L^1 EOC": [float(x) for x in EOC],
            "time_sim": sim_times
        }
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_aggregation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Pure fragmentation
def crystallization_fragmentation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Pure fragmentation tests against analytical solutions for the linear
    selection function with uniform particle binary fragmentation, EOC tests. 
    Assumes no solute (c) and solubility component (cs).
    '''

    Cadet.cadet_path = cadet_path

    def get_analytical_frag(n_x, x_ct, cycle_time):
        return np.asarray([3.0 * x_ct[j]**2 * (1.0+cycle_time)**2 * np.exp(-x_ct[j]**3 * (1.0+cycle_time)) for j in range(n_x)])

    # define params
    x_c, x_max = 1e-2, 1e2  # m # crystal phase discretization
    cycle_time = 4          # s
    time_res = 30
    t = np.linspace(0, cycle_time, time_res)
    S_0 = 1.0 # fragmentation rate

    '''
    EOC tests, pure fragmentation
    '''

    normalized_l1 = []
    sim_times = []

    Nx_grid = np.asarray([12, 24, 48, 96, 192, 384])

    for n_x in Nx_grid:
        
        model, x_grid, x_ct = settings_crystallization.PureFrag_LinBi(
            n_x, x_c, x_max, S_0, t, output_path)
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()
        
        sim = model.root.output.solution.unit_001.solution_outlet[-1, :]
        
        analytical = get_analytical_frag(n_x, x_ct, cycle_time)
        normalized_l1.append(calculate_normalized_error(
            analytical, sim, x_ct, x_grid))

        sim_times.append(model.root.meta.time_sim)

        if n_x == Nx_grid[-1]:
            # plot the result
            plt.xscale("log")
            plt.plot(x_ct, analytical, linewidth=2.5, label="Analytical")
            plt.plot(x_ct, sim, label="Numerical", linestyle='dashed')
            plt.xlabel(r'$Size/\mu m$')
            plt.ylabel('Particle count/1')
            plt.legend(frameon=0)
            plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_CSTR_fragmentation"), dpi=100, bbox_inches='tight')

    # print the slopes
    # The last value in this array should be around 2, see Zhang et al. (2025) for details
    EOC = []
    for i in range(0, len(normalized_l1)-1):
        EOC.append(np.log(normalized_l1[i] / normalized_l1[i+1]) / np.log(2.0))
    print(EOC)

    data = {
        'convergence': {
            "Nx": Nx_grid.tolist(),
            "L^1 error": [float(x) for x in normalized_l1],
            "L^1 EOC": [float(x) for x in EOC],
            "time_sim": sim_times
        }
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_fragmentation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Simultaneous aggregation and fragmentation
def crystallization_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Simultaneous aggregation and fragmentation tests against analytical
    solutions and EOC tests. Linear selection function with uniform particle binary
    fragmentation combined with a constant aggregation kernel. 
    Assumes no solute (c) and solubility component (cs).
    '''

    # combined aggregation and fragmentation
    Cadet.cadet_path = cadet_path

    def get_analytical_agg_frag(n_x, x_ct, t):
        x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

        # dimensionless time
        T_s = 1.0*beta_0*t
        d = T_s**2 + (10.0-2.0*np.exp(-T_s))*T_s + 25.0 - \
            26.0*np.exp(-T_s) + np.exp(-2.0*T_s)
        p_1 = 0.25*(np.exp(-T_s) - T_s - 9.0) + 0.25*np.sqrt(d)
        p_2 = 0.25*(np.exp(-T_s) - T_s - 9.0) - 0.25*np.sqrt(d)
        L_1 = 7.0+T_s+np.exp(-T_s)
        L_2 = 9.0+T_s-np.exp(-T_s)
        K_1 = L_1
        K_2 = 2.0-2.0*np.exp(-T_s)

        analytical = []
        for i in range(0, n_x):
            f = (K_1+p_1*K_2)/(L_2+4.0*p_1)*np.exp(p_1 *
                                                   x_ct[i]**3) + (K_1+p_2*K_2)/(L_2+4.0*p_2)*np.exp(p_2*x_ct[i]**3)
            f *= 3*x_ct[i]**2
            analytical.append(f)

        return analytical

    # define params
    x_c, x_max = 1e-2, 1e2  # m
    n_x = 100
    cycle_time = 5          # s
    time_res = 30
    t = np.linspace(0, cycle_time, time_res)
    S_0 = 0.1
    beta_0 = 0.2

    '''
    EOC tests, simultaneous aggregation and fragmentation
    '''

    # run sims
    normalized_l1 = []
    sim_times = []

    Nx_grid = np.asarray([12, 24, 48, 96, 192, 384, ]) if small_test else np.asarray(
        [25, 50, 100, 200, 400, 768, ])

    for n_x in Nx_grid:
        model, x_grid, x_ct = settings_crystallization.Agg_frag(
            n_x, x_c, x_max, beta_0, S_0, t, output_path)
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()
        sim = model.root.output.solution.unit_001.solution_outlet[-1, :]
        
        analytical = get_analytical_agg_frag(n_x, x_ct, cycle_time)
        
        normalized_l1.append(calculate_normalized_error(
            analytical, sim, x_ct, x_grid))

        sim_times.append(model.root.meta.time_sim)

        if n_x == Nx_grid[-1]:
            
            # plot the result
            plt.xscale("log")
            plt.plot(x_ct, analytical, linewidth=2.5, label="Analytical")
            plt.plot(x_ct, sim, label="Numerical")
            plt.xlabel(r'$Size/\mu m$')
            plt.ylabel('Particle count/1')
            plt.legend(frameon=0)
            plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_CSTR_aggregation_fragmentation"), dpi=100, bbox_inches='tight')

    # print the slopes
    # The last value in this array should be around 3, see Zhang et al. (2025) for details
    EOC = []
    for i in range(0, len(normalized_l1)-1):
        EOC.append(np.log(normalized_l1[i] / normalized_l1[i+1]) / np.log(2.0))
    print(EOC)

    data = {
        'convergence': {
            "Nx": Nx_grid.tolist(),
            "L^1 error": [float(x) for x in normalized_l1],
            "L^1 EOC": [float(x) for x in EOC],
            "time_sim": sim_times
        }
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_aggregation_fragmentation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% PBM, aggregation and fragmentation in CSTR
def crystallization_PBM_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Combined test with PBM: nucleation, growth, growth rate dispersion, aggregation and fragmentation in CSTR. 
    '''

    # define params
    x_c, x_max = 1e-6, 1000e-6  # m
    n_x = 100
    cycle_time = 300            # s
    time_res = 100
    t = np.linspace(0, cycle_time, time_res)

    # numerical ref solution
    model, x_grid, x_ct = settings_crystallization.CSTR_PBM_aggregation_fragmentation(
        800, x_c, x_max, 1, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
        raise Exception(f"simulation failed")
    model.load_from_file()
    c_x_ref = model.root.output.solution.unit_001.solution_outlet[-1, 1:-1]
    spl = UnivariateSpline(x_ct, c_x_ref)

    ## run sims
    normalized_l1 = []
    sim_times = []

    Nx_grid = np.asarray([12, 25, 50, 100, 200, 400])

    for n_x in Nx_grid:
        model, x_grid, x_ct = settings_crystallization.CSTR_PBM_aggregation_fragmentation(
        n_x, x_c, x_max, 1, t, output_path)
        model.save()
        return_data = model.run_simulation()
        model.load_from_file()
        sim = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
        normalized_l1.append(calculate_normalized_error(spl(x_ct), sim, x_ct, x_grid))

        sim_times.append(model.root.meta.time_sim)

        
    # print the slopes
    EOC = []
    for i in range(0, len(normalized_l1)-1):
        EOC.append(np.log(normalized_l1[i] / normalized_l1[i+1]) / np.log(2.0))
    print(EOC)

    data = {
        'convergence': {
            "Nx": Nx_grid.tolist(),
            "L^1 error": [float(x) for x in normalized_l1],
            "L^1 EOC": [float(x) for x in EOC],
            "time_sim": sim_times
        }
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_aggregation_fragmentation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Constant aggregation in a DPFR


def crystallization_DPFR_constAggregation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Constant aggregation kernel in a DPFR tests and EOC tests using a
    reference solution. 
    Assumes no solute (c) and solubility component (cs).
    '''

    Cadet.cadet_path = cadet_path

    # boundary condition
    # A: area, y0: offset, w:std, xc: center (A,w >0)

    # define params
    n_x = 100
    n_col = 100
    x_c, x_max = 1e-6, 1000e-6            # m
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300                      # s
    t = np.linspace(0, cycle_time, 200)

    '''
    @note: There is no analytical solution in this case, thus we use a numerical reference
    '''

    model, x_grid, x_ct = settings_crystallization.Agg_DPFR(
        n_x, n_col, x_c, x_max, 1, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()
    c_x = model.root.output.solution.unit_002.solution_outlet[-1, :]

    plt.xscale("log")
    plt.plot(x_ct, c_x, label="Numerical reference")
    plt.xlabel(r'$Size/\mu m$')
    plt.ylabel('Particle count')
    plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_DPFR_aggregation"), dpi=100, bbox_inches='tight')

    '''
    EOC tests in a DPFR, Constant aggregation kernel
    @note: the EOC is obtained along the Nx and Ncol coordinate, separately
    '''

    # get ref solution
    N_x_ref = 96 if small_test else 384
    N_col_ref = 48 if small_test else 192

    model, ref_x_grid, x_ct = settings_crystallization.Agg_DPFR(
        N_x_ref, N_col_ref, x_c, x_max, 3, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1, :]

    # interpolate the reference solution at the reactor outlet

    x_grid, x_ct = settings_crystallization.get_log_space(N_x_ref, x_c, x_max)

    spl = UnivariateSpline(x_ct, c_x_reference)

    # EOC for refinement in internal coordinate
    N_x_test = np.asarray([12, 24, 48]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, ])

    n_xs = []
    sim_times_int = []
    
    for Nx in N_x_test:
        model, x_grid, x_ct = settings_crystallization.Agg_DPFR(
            Nx, N_col_ref, x_c, x_max, 2, t, output_path)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])
        
        sim_times_int.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Nx = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Nx)

    data = {
        'convergence': {
            'internal_coordinate'
            "Nx": N_x_test.tolist(),
            "L^1 error": [float(x) for x in relative_L1_norms],
            "L^1 EOC": [float(x) for x in slopes_Nx],
            "time_sim": sim_times_int
        }
    }

    # EOC for refinement in the axial coordinate
    N_col_test = np.asarray(
        [12, 24, 48, ]) if small_test else np.asarray([12, 24, 48, 96, ])

    n_xs = []  # store the result nx here
    sim_times_ax = []
    
    for Ncol in N_col_test:
        model, x_grid, x_ct = settings_crystallization.Agg_DPFR(
            N_x_ref, Ncol, x_c, x_max, 2, t, output_path)  # test on WENO23
        model.save()
        return_return_data = model.run_simulation()
        if not return_return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])

        sim_times_ax.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Ncol = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Ncol)

    data['convergence']['axial_coordinate'] = {
        "Ncol": N_col_test.tolist(),
        "L^1 error": [float(x) for x in relative_L1_norms],
        "L^1 EOC": [float(x) for x in slopes_Ncol],
        "time_sim": sim_times_ax
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_aggregation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Constant fragmentation in a DPFR


def crystallization_DPFR_constFragmentation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Constant fragmentation kernel in a DPFR tests and EOC tests using a
    reference solution. 
    Assumes no solute (c) and solubility component (cs).
    '''

    Cadet.cadet_path = cadet_path

    # system setup
    n_x = 100
    n_col = 100

    x_c, x_max = 1e-6, 1000e-6            # m
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300                      # s
    t = np.linspace(0, cycle_time, 200)

    '''
    @note: There is no analytical solution in this case. Hence, we use a numerical reference
    '''

    model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
        n_x, n_col, x_c, x_max, 1, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()
    c_x = model.root.output.solution.unit_002.solution_outlet[-1, :]

    plt.xscale("log")
    plt.plot(x_ct, c_x, label="Numerical reference")
    plt.xlabel(r'$Size/\mu m$')
    plt.ylabel('Particle count')
    plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_DPFR_fragmentation"), dpi=100, bbox_inches='tight')

    '''
    EOC tests in a DPFR, Fragmentation
    @note: the EOC is obtained along the Nx and Ncol coordinate, separately
    '''

    # get ref solution
    N_x_ref = 96 if small_test else 384 * 2
    N_col_ref = 96 if small_test else 192 * 2

    model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
        N_x_ref, N_col_ref, x_c, x_max, 3, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1, :]

    # interpolate the reference solution at the reactor outlet

    x_grid, x_ct = settings_crystallization.get_log_space(N_x_ref, x_c, x_max)

    spl = UnivariateSpline(x_ct, c_x_reference)

    # EOC for refinement in internal coordinate
    N_x_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, 384, ])

    n_xs = []
    sim_times_int = []
    
    for Nx in N_x_test:
        model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
            Nx, 250, x_c, x_max, 2, t, output_path)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])

        sim_times_int.append(model.root.meta_time_sim)        

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Nx = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Nx)

    data = {
        'convergence': {
            'internal_coordinate'
            "Nx": N_x_test.tolist(),
            "L^1 error": [float(x) for x in relative_L1_norms],
            "L^1 EOC": [float(x) for x in slopes_Nx],
            "time_sim": sim_times_int
        }
    }

    # EOC for refinement in axial coordinate
    N_col_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, ])

    n_xs = []  # store the result nx here
    sim_times_ax = []
    
    for Ncol in N_col_test:
        model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
            450, Ncol, x_c, x_max, 2, t, output_path)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])

        sim_times_ax.append(model.root.meta_time_sim) 

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Ncol = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Ncol)

    data['convergence']['axial_coordinate'] = {
        "Ncol": N_col_test.tolist(),
        "L^1 error": [float(x) for x in relative_L1_norms],
        "L^1 EOC": [float(x) for x in slopes_Ncol],
        "time_sim": sim_times_ax
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_fragmention.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Nucleation, growth, growth rate dispersion and aggregation in a DPFR
def crystallization_DPFR_NGGR_aggregation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Nucleation, growth, growth rate dispersion and aggregation in a DPFR
    tests and EOC tests using a reference solution. 
    There are solute (c) and solubility components (cs).
    '''

    Cadet.cadet_path = cadet_path

    # set up
    n_x = 100
    n_col = 100
    x_c, x_max = 1e-6, 1000e-6       # m
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    # simulation time
    cycle_time = 200                 # s
    t = np.linspace(0, cycle_time, 200+1)

    model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
        n_x, n_col, x_c, x_max, 1, 1, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()

    t = model.root.input.solver.user_solution_times
    c_x = model.root.output.solution.unit_001.solution_outlet[-1, 1:-1]

    plt.xscale("log")
    plt.plot(x_ct, c_x, label="Numerical reference")
    plt.xlabel(r'$Size/\mu m$')
    plt.ylabel(r'$n/(1/m / m)$')
    plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_DPFR_PBM_aggregation"), dpi=100, bbox_inches='tight')

    '''
    EOC tests in a DPFR, Nucleation, growth, growth rate dispersion and aggregation
    @note: the EOC is obtained along the Nx and Ncol coordinate, separately
    '''

    # get ref solution
    N_x_ref = 96 if small_test else 384 * 2
    N_col_ref = 96 if small_test else 384 * 2

    model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
        N_x_ref, N_col_ref, x_c, x_max, 3, 3, t, output_path)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1, 1:-1]

    # interpolate the reference solution at the reactor outlet

    x_grid, x_ct = settings_crystallization.get_log_space(N_x_ref, x_c, x_max)

    spl = UnivariateSpline(x_ct, c_x_reference)

    # EOC for refinement in internal coordinate
    N_x_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, 384, ])

    n_xs = []
    sim_times_int = []
    
    for Nx in N_x_test:
        model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
            Nx, 400, x_c, x_max, 3, 2, t, output_path)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(
            model.root.output.solution.unit_001.solution_outlet[-1, 1:-1])
        
        sim_times_int.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Nx = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Nx)

    data = {
        'convergence': {
            'internal_coordinate'
            "Nx": N_x_test.tolist(),
            "L^1 error": [float(x) for x in relative_L1_norms],
            "L^1 EOC": [float(x) for x in slopes_Nx],
            "time_sim": sim_times_int
        }
    }

    # EOC for refinement in axial coordinate
    N_col_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, 384, ])

    n_xs = []  # store the result nx here
    sim_times_ax = []
    
    for Ncol in N_col_test:
        model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
            400, Ncol, x_c, x_max, 2, 3, t, output_path)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(
            model.root.output.solution.unit_001.solution_outlet[-1, 1:-1])
        
        sim_times_ax.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Ncol = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Ncol)

    data['convergence']['axial_coordinate'] = {
        "Ncol": N_col_test.tolist(),
        "L^1 error": [float(x) for x in relative_L1_norms],
        "L^1 EOC": [float(x) for x in slopes_Ncol],
        "time_sim": sim_times_ax
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_PBM_aggregation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Simultaneous aggregation and fragmentation in a DPFR
def crystallization_DPFR_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path):
    '''
    @detail: Simultaneous aggregation and fragmentation in a DPFR tests and EOC tests using a reference solution. 
    There are no solute (c) and solubility components (cs).
    '''

    Cadet.cadet_path = cadet_path

    # system setup
    n_x = 100
    n_col = 100

    x_c, x_max = 1e-6, 1000e-6            # m
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300                      # s
    t = np.linspace(0, cycle_time, 200)

    '''
    @note: There is no analytical solution in this case. We are using a result as the reference solution.
    '''

    model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
        n_x, n_col, x_c, x_max, 1)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()
    c_x = model.root.output.solution.unit_002.solution_outlet[-1, :]

    plt.xscale("log")
    plt.plot(x_ct, c_x, label="Numerical reference")
    plt.xlabel(r'$Size/\mu m$')
    plt.ylabel('Particle count/1')
    plt.savefig(re.sub(".h5", ".png", str(output_path) + "/fig_DPFR_aggregation_fragmentation"), dpi=100, bbox_inches='tight')

    '''
    EOC tests in a DPFR, Aggregation and Fragmentation
    @note: the EOC is obtained along the Nx and Ncol coordinate, separately
    '''

    # get ref solution
    N_x_ref = 96 if small_test else 384 * 2
    N_col_ref = 96 if small_test else 192 * 2

    model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
        N_x_ref, N_col_ref, x_c, x_max, 3)
    model.save()
    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        print(return_data.error_message)
    model.load_from_file()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1, :]

    # interpolate the reference solution at the reactor outlet

    x_grid, x_ct = settings_crystallization.get_log_space(N_x_ref, x_c, x_max)

    spl = UnivariateSpline(x_ct, c_x_reference)

    # EOC for refinement in internal coordinate
    N_x_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, 384, ])

    n_xs = []
    sim_times_int = []
    
    for Nx in N_x_test:
        model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
            Nx, 250, x_c, x_max, 2)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])

        sim_times_int.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Nx = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Nx)

    data = {
        'convergence': {
            'internal_coordinate'
            "Nx": N_x_test.tolist(),
            "L^1 error": [float(x) for x in relative_L1_norms],
            "L^1 EOC": [float(x) for x in slopes_Nx],
            "time_sim": sim_times_int
        }
    }

    # EOC for refinement in axial coordinate
    N_col_test = np.asarray([12, 24, 48, ]) if small_test else np.asarray(
        [12, 24, 48, 96, 192, ])

    n_xs = []  # store the result nx here
    sim_times_ax = []
    
    for Ncol in N_col_test:
        model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
            450, Ncol, x_c, x_max, 2)  # test on WENO23
        model.save()
        return_data = model.run_simulation()
        if not return_data.return_code == 0:
            print(return_data.error_message)
        model.load_from_file()

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1, :])

        sim_times_ax.append(model.root.meta.time_sim)

    relative_L1_norms = []  # store the relative L1 norms here
    for nx in n_xs:
        # interpolate the ref solution on the test case grid

        x_grid, x_ct = settings_crystallization.get_log_space(
            len(nx), x_c, x_max)

        relative_L1_norms.append(
            calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

    slopes_Ncol = get_slope(relative_L1_norms)  # calculate slopes
    print(slopes_Ncol)

    data['convergence']['axial_coordinate'] = {
        "Ncol": N_col_test.tolist(),
        "L^1 error": [float(x) for x in relative_L1_norms],
        "L^1 EOC": [float(x) for x in slopes_Ncol],
        "time_sim": sim_times_ax
    }

    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_aggregation_fragmentation.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# %% Main function calling all the tests
def crystallization_tests(n_jobs, database_path, small_test,
                          output_path, cadet_path,
                          run_CSTR_aggregation_test = 1,
                          run_CSTR_fragmentation_test = 1,
                          run_CSTR_aggregation_fragmentation_test = 0, # not included in test pipeline per default, due to redundancy
                          run_CSTR_PBM_aggregation_fragmentation_test = 1,
                          run_DPFR_constAgg_test = 0, # not included in test pipeline per default, due to redundancy
                          run_DPFR_constFrag_test = 1,
                          run_DPFR_NGGR_aggregation_test = 1,
                          run_DPFR_aggregation_fragmentation_test = 0 # not included in test pipeline per default, due to redundancy
                          ):
    
    os.makedirs(output_path, exist_ok=True)
    
    if run_CSTR_aggregation_test:
        crystallization_aggregation_EOC_test(cadet_path, small_test, output_path)
    
    if run_CSTR_fragmentation_test:
        crystallization_fragmentation_EOC_test(cadet_path, small_test, output_path)
    
    if run_CSTR_aggregation_fragmentation_test:
        crystallization_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path)
    
    if run_CSTR_PBM_aggregation_fragmentation_test:
        crystallization_PBM_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path)
        
    if run_DPFR_constAgg_test:
        crystallization_DPFR_constAggregation_EOC_test(cadet_path, small_test, output_path)
    
    if run_DPFR_constFrag_test:
        crystallization_DPFR_constFragmentation_EOC_test(cadet_path, small_test, output_path)
    
    if run_DPFR_NGGR_aggregation_test:
        crystallization_DPFR_NGGR_aggregation_EOC_test(cadet_path, small_test, output_path)
    
    if run_DPFR_aggregation_fragmentation_test:
        crystallization_DPFR_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path)

        
# # %% run in file

# cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master4_crysPartII_d0888cb\aRELEASE\bin\cadet-cli.exe"

# small_test = 1
# n_jobs = -1 # todo
# database_path = None

# sys.path.append(str(Path(".")))
# project_repo = ProjectRepo()
# output_path = str(project_repo.output_path /
#                   "test_cadet-core") + "/crystallization"

# os.makedirs(output_path, exist_ok=True)
        
# crystallization_tests(n_jobs, database_path, small_test,
#                       output_path, cadet_path,
#                       run_CSTR_aggregation_test = 1,
#                       run_CSTR_fragmentation_test = 0,
#                       run_CSTR_aggregation_fragmentation_test = 0,
#                       run_CSTR_PBM_aggregation_fragmentation_test = 0,
#                       run_DPFR_constAgg_test = 0,
#                       run_DPFR_constFrag_test = 0,
#                       run_DPFR_NGGR_aggregation_test = 0,
#                       run_DPFR_aggregation_fragmentation_test = 0
#                       )
        
        
        