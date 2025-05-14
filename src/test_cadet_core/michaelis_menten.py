import settings_michaelis_menten as mm
from scipy.integrate import solve_ivp
import numpy as np
from cadet import Cadet

import matplotlib.pyplot as plt


def michaelis_menten_complex_inhibition_test(output_path, cadet_path):

    print("Running Michaelis-Menten test...")

    parameter = {
        'ncomp': 5,
        'init_c': [10.0, 2, 5, 1e-5, 1e-5],
        'sim_time': 300.0,
        'km_a': 0.5,
        'km_b': 1.0,
        'km_c_1': 0.8,
        'km_c_2': 0.5,
        'ki_c_uc': 4.0,
        'ki_a_nc': 2.0,
        'ki_e_nc': 2.5,
        'ki_b_c': 5.0,
        'ki_d_c': 6.0,
        'vmax_1': 1.0,
        'vmax_2': 0.8,
        'vmax_3': 1.2
    }

    cadet_model = Cadet()
    cadet_model.install_path = cadet_path
    # simulation CADET
    cadet_model = mm.complex_inhibition_system_cadet_settings(cadet_model, parameter, output_path, cadet_path)

    cadet_model.filename = cadet_model.filename = str(output_path) + "/ref_CSTR_mm_complex_inhibition.h5"
    cadet_model.save()
    return_data = cadet_model.run()
    cadet_model.load()

    if not return_data.return_code == 0:
        raise RuntimeError(f"Simulation failed with return code {return_data.return_code}")
    else:
        print("Michaelis Menten Test: Simulation completed successfully.")

    c_cadet = cadet_model.root.output.solution.unit_001.solution_outlet
    times_cadet = cadet_model.root.output.solution.solution_times

    # simulation ODE
    ode_model = mm.complex_inhibition_system_ode(parameter)
        # Initial conditions

    y0 = parameter['init_c']
    t_span = (0, parameter['sim_time'])
    t_eval = np.linspace(0, parameter['sim_time'], 1000)

    sol = solve_ivp(ode_model, t_span, y0, t_eval=t_eval)

    # compare results
    # Extract ODE solution
    a_ode = sol.y[0]
    b_ode = sol.y[1]
    c_ode = sol.y[2]
    d_ode = sol.y[3]
    e_ode = sol.y[4]

    # Compare CADET and ODE solutions
    cadet_a = c_cadet[:, 0]
    cadet_b = c_cadet[:, 1]
    cadet_c = c_cadet[:, 2]
    cadet_d = c_cadet[:, 3]
    cadet_e = c_cadet[:, 4]

    # Strady state
    steady_state_a_cadet = cadet_a[-1]
    steady_state_b_cadet = cadet_b[-1]
    steady_state_c_cadet = cadet_c[-1]
    steady_state_d_cadet = cadet_d[-1]
    steady_state_e_cadet = cadet_e[-1]

    steady_state_a_ode = a_ode[-1]
    steady_state_b_ode = b_ode[-1]
    steady_state_c_ode = c_ode[-1]
    steady_state_d_ode = d_ode[-1]
    steady_state_e_ode = e_ode[-1]

    # Initial slope
    initial_slope_a_cadet = (cadet_a[1] - cadet_a[0]) / (times_cadet[1] - times_cadet[0])
    initial_slope_b_cadet = (cadet_b[1] - cadet_b[0]) / (times_cadet[1] - times_cadet[0])
    initial_slope_c_cadet = (cadet_c[1] - cadet_c[0]) / (times_cadet[1] - times_cadet[0])
    initial_slope_d_cadet = (cadet_d[1] - cadet_d[0]) / (times_cadet[1] - times_cadet[0])
    initial_slope_e_cadet = (cadet_e[1] - cadet_e[0]) / (times_cadet[1] - times_cadet[0])

    initial_slope_a_ode = (a_ode[1] - a_ode[0]) / (sol.t[1] - sol.t[0])
    initial_slope_b_ode = (b_ode[1] - b_ode[0]) / (sol.t[1] - sol.t[0])
    initial_slope_c_ode = (c_ode[1] - c_ode[0]) / (sol.t[1] - sol.t[0])
    initial_slope_d_ode = (d_ode[1] - d_ode[0]) / (sol.t[1] - sol.t[0])
    initial_slope_e_ode = (e_ode[1] - e_ode[0]) / (sol.t[1] - sol.t[0])

    max_acceptable_error  = 1e-3 # Set the maximum acceptable error for the test

    steadystat_err_a = np.abs(steady_state_a_cadet - steady_state_a_ode)
    steadystat_err_b =  np.abs(steady_state_b_cadet - steady_state_b_ode)
    steadystat_err_c = np.abs(steady_state_c_cadet - steady_state_c_ode)
    steadystat_err_d = np.abs(steady_state_d_cadet - steady_state_d_ode)
    steadystat_err_e = np.abs(steady_state_e_cadet - steady_state_e_ode)

    steadystate_err = np.max([steadystat_err_a, steadystat_err_b, steadystat_err_c, steadystat_err_d, steadystat_err_e])

    if steadystate_err > max_acceptable_error:
        raise ValueError(f"Steady state error exceeds set limit: {steadystate_err} > {max_acceptable_error}")
    else:
        print( f"Michaelis Menten Test: Steady state error is within set limits of {max_acceptable_error}.")

    slope_err_a = np.abs(initial_slope_a_cadet - initial_slope_a_ode)
    slope_err_b = np.abs(initial_slope_b_cadet - initial_slope_b_ode)
    slope_err_c = np.abs(initial_slope_c_cadet - initial_slope_c_ode)
    slope_err_d = np.abs(initial_slope_d_cadet - initial_slope_d_ode)
    slope_err_e = np.abs(initial_slope_e_cadet - initial_slope_e_ode)

    slope_err  = np.max([slope_err_a, slope_err_b, slope_err_c, slope_err_d, slope_err_e])

    if slope_err > max_acceptable_error:
        raise ValueError(f"Initial slope error exceeds set limit: {slope_err} > {max_acceptable_error}")
    else:
        print( f"Michaelis Menten Test: Initial slope error is within set limits of {max_acceptable_error}.")

    # Max absolute error
    abs_error_a = np.max(np.abs((cadet_a - a_ode)))
    abs_error_b = np.max(np.abs((cadet_b - b_ode)))
    abs_error_c = np.max(np.abs((cadet_c - c_ode)))
    abs_error_d = np.max(np.abs((cadet_d - d_ode)))
    abs_error_e = np.max(np.abs((cadet_e - e_ode)))

    abs_error = np.max([abs_error_a, abs_error_b, abs_error_c, abs_error_d, abs_error_e])

    if abs_error > max_acceptable_error:
        raise ValueError(f"Absolute error exceeds set limit: {abs_error} > {max_acceptable_error}")
    else:
        print(f"Michaelis Menten Test: Absolute error is within set limits of {max_acceptable_error}.")

    # Max relative error
    # note that the relative error for a and c is not defined, since they converge to 0
    # and the relative error would be infinite. Therefore, we only consider b, d, and e.
    rel_error_b = np.max(np.abs((cadet_b - b_ode) / b_ode))
    rel_error_d = np.max(np.abs((cadet_d - d_ode) / d_ode))
    rel_error_e = np.max(np.abs((cadet_e - e_ode) / e_ode))

    rel_error = np.max([ rel_error_b,  rel_error_d, rel_error_e])
    if rel_error > max_acceptable_error:
        raise ValueError(f"Relative error exceeds acceptable limit: {rel_error} > {max_acceptable_error}")
    else:
        print(f"Michaelis Menten Test: Relative error is within set limits of {max_acceptable_error}.")

    plt.figure(figsize=(10, 6))
    plt.plot(times_cadet, cadet_a, label='CADE A', color='blue')
    plt.plot(times_cadet, cadet_b, label='CADE B', color='orange')
    plt.plot(times_cadet, cadet_c, label='CADE C', color='green')
    plt.plot(times_cadet, cadet_d, label='CADE D', color='red')
    plt.plot(times_cadet, cadet_e, label='CADE E', color='purple')
    plt.plot(sol.t, a_ode, label='ODE A', linestyle='--', color='blue')
    plt.plot(sol.t, b_ode, label='ODE B', linestyle='--', color='orange')
    plt.plot(sol.t, c_ode, label='ODE C', linestyle='--', color='green')
    plt.plot(sol.t, d_ode, label='ODE D', linestyle='--', color='red')
    plt.plot(sol.t, e_ode, label='ODE E', linestyle='--', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mol/m^3)')
    plt.title('Michaelis-Menten Kinetics: CADE vs ODE')
    plt.legend()
    plt.grid()
    plt.savefig(str(output_path) + "/michaelis_menten_complex_inhibition_test_simulation.png")
    plt.show()
