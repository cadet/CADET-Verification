import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

'''
@note: Since how we modularize the code is not finalized, the following setup is subject to changes. 
@detail: Pure aggregation tests against analytical solutions for the Golovin (sum) kernel, EOC tests. Assumes no solute (c) and solubility component (cs).
The Golovin kernel should cover all functions implemented in the Core. 
'''

Cadet.cadet_path = r"C:\Users\zwend\CADET\cadet79\bin\cadet-cli.exe"                ## for aggtegation

def get_log_space(n_x, x_c, x_max):
    x_grid = np.logspace(np.log10(x_c), np.log10(x_max), n_x+1)                      ## log space
    x_ct = np.asarray([0.5 * x_grid[p+1] + 0.5 * x_grid[p] for p in range (0, n_x)])
    return x_grid, x_ct

def PureAgg_Golovin(n_x : 'int, number of bins', x_c, x_max):
    model = Cadet()
    
    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions    
    initial_c = np.asarray([3.0*x_ct[k]**2 * np.exp(-x_ct[k]**3/v_0)*N_0/v_0 for k in range(0, n_x)])   ## see our paper for the equation

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_index = 3                # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_rate_constant = beta_0

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = 'practice1.h5'                      ## change as needed
    
    return model

# number of bins
n_x = 150

# crystal phase discretization
x_c, x_max = 5e-3, 0.5e3 # m

# time
cycle_time = 3.0
time_res = 30
t = np.linspace(0, cycle_time, time_res)

# system setup
v_0 = 1              ## initial volume of particles
N_0 = 1              ## initial number of particles
beta_0 = 1.0         ## aggregation rate

x_grid, x_ct = get_log_space(n_x, x_c, x_max)
    
## run the model and get results
model = PureAgg_Golovin(n_x, x_c, x_max)
model.save()
data = model.run()
model.load() 
c_x = model.root.output.solution.unit_001.solution_outlet[-1,:]
#c_x[-1] = 0.0


## analytical solution 
'''
The analytical solution requires a high-precision floating point package to accurately calculate extremely large or small values from special functions. 
mpmath is used here. 
'''
from mpmath import * 
mp.dps = 50

# analytical solution, check our paper for details 
def get_analytical_agg(n_x):
    T_t1 = 1.0 - exp(-N_0*beta_0*cycle_time*v_0)  # dimensionless time

    x_grid_mp = []
    for i in range (0, n_x+1):
        x_grid_mp.append(power(10, linspace(log10(x_c), log10(x_max), n_x+1)[i]))
        
    x_ct = [(x_grid_mp[p+1] + x_grid_mp[p]) / 2 for p in range (0, n_x)]
    
    analytical_t1 = [3.0* N_0 * (1.0-T_t1) * exp(-x_ct[k]**3 * (1.0+T_t1)/v_0) * besseli(1,2.0*x_ct[k]**3 * sqrt(T_t1)/v_0) / x_ct[k] / sqrt(T_t1) for k in range (n_x)]

    return analytical_t1

analytical_1 = get_analytical_agg(n_x)

plt.plot(x_ct,analytical_1, label='Analytical')
plt.scatter(x_ct, c_x, label='Numerical')
plt.xscale("log")
plt.xlabel(r'size/$\mu m$')
plt.ylabel('particle number/1')
plt.legend(frameon=0)
plt.show()

'''
EOC tests, pure aggregation
'''

def calculate_normalized_error(ref, sim, x_ct, x_grid):    
    area = np.trapz(ref, x_ct)
    
    L1_error = 0.0
    for i in range (0, n_x):
        L1_error += np.absolute(ref[i] - sim[i]) * (x_grid[i+1] - x_grid[i])
    
    return L1_error / area

## run sims
normalized_l1 = []

Nx_grid = np.asarray([50,100,200,400,800,1600])

for n_x in Nx_grid:
    model = PureAgg_Golovin(n_x, x_c, x_max)
    model.save()
    data = model.run()
    model.load()
    sim = model.root.output.solution.unit_001.solution_outlet[-1,:]
    normalized_l1.append(calculate_normalized_error(get_analytical_agg(n_x), sim, x_ct, x_grid))

    
## print the slopes
## The last value in this array should be around 1.2, see our paper for details
EOC = []
for i in range (0, len(normalized_l1)-1):
    EOC.append(log(normalized_l1[i] / normalized_l1[i+1], mp.e) / log(2.0))
print(EOC)

'''
@note: The following setup is subject to changes depending on the core code configuration. 
@detail: Pure fragmentation tests against analytical solutions for the linear selection function with uniform particle binary fragmentation, EOC tests. 
Assumes no solute (c) and solubility component (cs).
'''

Cadet.cadet_path = r'C:\Users\zwend\CADET\cadet72\bin\cadet-cli.exe'   ## for breakage

def get_analytical_frag(n_x, x_ct, cycle_time):
    return np.asarray([3.0 * x_ct[j]**2 * (1.0+cycle_time)**2 * np.exp(-x_ct[j]**3 * (1.0+cycle_time)) for j in range (n_x)])

def PureFrag_LinBi(n_x : 'int, number of bins', x_c, x_max):
    model = Cadet()
    
    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions    
    initial_c = np.asarray([3.0*x_ct[k]**2 * np.exp(-x_ct[k]**3) for k in range(0, n_x)])   ## see our paper for the equation

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_kernel_gamma = 2.0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_selection_function_alpha = 1.0

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = 'practice1.h5'                      ## change as needed
    
    return model

## update params
# crystal phase discretization
x_c, x_max = 1e-2, 1e2  # m
x_grid, x_ct = get_log_space(n_x, x_c, x_max)

# time
cycle_time = 4          # s
time_res = 30
t = np.linspace(0, cycle_time, time_res)

## fragmentation rate
S_0 = 1.0

n_x = 100
model = PureFrag_LinBi(n_x, x_c, x_max)
model.save()
data = model.run()
model.load()
    
c_x1 = model.root.output.solution.unit_001.solution_outlet[-1,:]

analytical1 = get_analytical_frag(n_x, x_ct, cycle_time)


## plot the result
plt.xscale("log")
plt.scatter(x_ct,c_x1, label="Numerical")
plt.plot(x_ct,analytical1, linewidth=2.5, label="Analytical")
plt.xlabel(r'$Size/\mu m$')
plt.ylabel('Particle count/1')
plt.legend(frameon=0)
plt.show()

'''
EOC tests, pure fragmentation
'''

## run sims
normalized_l1 = []

Nx_grid = np.asarray([25,50,100,200])

for n_x in Nx_grid:
    model = PureFrag_LinBi(n_x, x_c, x_max)
    model.save()
    data = model.run()
    model.load()
    sim = model.root.output.solution.unit_001.solution_outlet[-1,:]
    normalized_l1.append(calculate_normalized_error(get_analytical_frag(n_x, x_ct, cycle_time), sim, x_ct, x_grid))

    
## print the slopes
## The last value in this array should be around 2, see our paper for details
EOC = []
for i in range (0, len(normalized_l1)-1):
    EOC.append(np.log(normalized_l1[i] / normalized_l1[i+1]) / np.log(2.0))
print(EOC)


'''
@note: The following setup is subject to changes depending on the core code configuration. 
@detail: Simultaneous aggregation and fragmentation tests against analytical solutions and EOC tests.
Linear selection function with uniform particle binary fragmentation combined with a constant aggregation kernel. 
Assumes no solute (c) and solubility component (cs).
'''

Cadet.cadet_path = r'C:\Users\zwend\CADET\cadet73\bin\cadet-cli.exe'   ## combined aggregation and fragmentation

def get_analytical_agg_frag(n_x, x_ct, t):  
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)
        
    T_s = 1.0*beta_0*t                                                                     # dimensionless time
    d = T_s**2 + (10.0-2.0*np.exp(-T_s))*T_s + 25.0 -26.0*np.exp(-T_s) + np.exp(-2.0*T_s)
    p_1 = 0.25*(np.exp(-T_s) - T_s - 9.0) + 0.25*np.sqrt(d)
    p_2 = 0.25*(np.exp(-T_s) - T_s - 9.0) - 0.25*np.sqrt(d)
    L_1 = 7.0+T_s+np.exp(-T_s)
    L_2 = 9.0+T_s-np.exp(-T_s)
    K_1 = L_1
    K_2 = 2.0-2.0*np.exp(-T_s)

    analytical = []
    for i in range (0,n_x):
        f = (K_1+p_1*K_2)/(L_2+4.0*p_1)*np.exp(p_1*x_ct[i]**3) + (K_1+p_2*K_2)/(L_2+4.0*p_2)*np.exp(p_2*x_ct[i]**3)
        f *= 3*x_ct[i]**2
        analytical.append(f)
        
    return analytical

def Agg_frag(n_x : 'int, number of bins', x_c, x_max):
    model = Cadet()
    
    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions    
    initial_c = np.asarray([3.0*x_ct[k]**2 * 4.0*x_ct[k]**3 * np.exp(-2.0*x_ct[k]**3) for k in range(0, n_x)])   ## see our paper for the equation

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_index = 0 # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_rate_constant = beta_0

    model.root.input.model.unit_001.reaction_bulk.cry_breakage_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_kernel_gamma = 2
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_selection_function_alpha = 1 

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = 'practice1.h5'                      ## change as needed
    
    return model

## update params
# crystal phase discretization
x_c, x_max = 1e-2, 1e2  # m
x_grid, x_ct = get_log_space(n_x, x_c, x_max)

# time
cycle_time = 5          # s
time_res = 30
t = np.linspace(0, cycle_time, time_res)

## rate constant
S_0 = 0.1
beta_0 = 0.2

n_x = 100
model = Agg_frag(n_x, x_c, x_max)
model.save()
data = model.run()
model.load()
    
c_x1 = model.root.output.solution.unit_001.solution_outlet[-1,:]

analytical = get_analytical_agg_frag(n_x, x_ct, cycle_time)


## plot the result
plt.xscale("log")
plt.scatter(x_ct,c_x1, label="Numerical")
plt.plot(x_ct,analytical, linewidth=2.5, label="Analytical")
plt.xlabel(r'$Size/\mu m$')
plt.ylabel('Particle count/1')
plt.legend(frameon=0)
plt.show()

'''
EOC tests, simultaneous aggregation and fragmentation
'''

## run sims
normalized_l1 = []

Nx_grid = np.asarray([25, 50, 100, 200, 400, 800, ])

for n_x in Nx_grid:
    model = Agg_frag(n_x, x_c, x_max)
    model.save()
    data = model.run()
    model.load()
    sim = model.root.output.solution.unit_001.solution_outlet[-1,:]
    normalized_l1.append(calculate_normalized_error(get_analytical_agg_frag(n_x, x_ct, cycle_time), sim, x_ct, x_grid))

    
## print the slopes
## The last value in this array should be around 3, see our paper for details
EOC = []
for i in range (0, len(normalized_l1)-1):
    EOC.append(np.log(normalized_l1[i] / normalized_l1[i+1]) / np.log(2.0))
print(EOC)

'''
@note: The following setup is subject to changes depending on the core code configuration. 
@detail: Constant aggregation kernel in a DPFR tests and EOC tests using a reference solution. 
Assumes no solute (c) and solubility component (cs).
'''

Cadet.cadet_path = r"C:\Users\zwend\CADET\cadet71\bin\cadet-cli.exe"

# boundary condition
# A: area, y0: offset, w:std, xc: center (A,w >0)
def log_normal(x, y0, A, w, xc):
    return y0 + A/(np.sqrt(2.0*np.pi) * w*x)* np.exp(-np.log(x/xc)**2 / 2.0/w**2)

def Agg_DPFR(n_x : 'int, number of x bins', n_col : 'int, number of z bins', x_c, x_max, axial_order):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = log_normal(x_ct*1e6, 0, 1e16, 0.4, 20)

    # Initial conditions
    initial_c = n_x*[0.0, ]

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # Tubular reactor
    model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    model.root.input.model.unit_001.col_length = 0.47
    model.root.input.model.unit_001.cross_section_area = 1.46e-4*0.21  # m^2
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05           # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8
    
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0              # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 3e-11

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6 / 60         # volumetric flow rate, m^3/s

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_coordinates = 0

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    
    model.filename = 'practice1.h5'                    ## change as needed

    return model

# system setup
x_c, x_max = 1e-6, 1000e-6            # m
x_grid, x_ct = get_log_space(n_x, x_c, x_max)

cycle_time = 300                      # s
t = np.linspace(0, cycle_time, 200)

n_x = 100
n_col = 100

'''
@note: There is no analytical solution in this case. We are using this result as the reference solution?
'''

model = Agg_DPFR(n_x, n_col, x_c, x_max, 1)
model.save()
data = model.run()
model.load() 
c_x = model.root.output.solution.unit_002.solution_outlet[-1,:]

plt.xscale("log")
plt.plot(x_ct, c_x)
plt.xlabel(r'$Size/\mu m$')
plt.ylabel('Particle count/1')
plt.show()

'''
EOC tests in a DPFR, Constant aggregation kernel
@note: the EOC is obtained along the Nx and Ncol coordinate, separately
'''
from scipy.interpolate import UnivariateSpline

def get_slope(error):
    return [np.log2(error[i] / error[i-1]) for i in range (1, len(error))]

## get ref solution
N_x_ref =   450
N_col_ref = 250
    
model = Agg_DPFR(N_x_ref, N_col_ref, x_c, x_max, 3)
model.save()
data = model.run()
model.load() 

c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,:]

## interpolate the reference solution at the reactor outlet

x_grid, x_ct = get_log_space(N_x_ref, x_c, x_max)

spl = UnivariateSpline(x_ct, c_x_reference)

## EOC, Nx

N_x_test = np.asarray([25, 50, 100, 200, 400, ])                          ## grid for EOC

n_xs = []
for Nx in N_x_test:
    model = Agg_DPFR(Nx, 250, x_c, x_max, 2)                ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,:])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))
    

slopes_Nx = get_slope(relative_L1_norms)      ## calculate slopes
print(slopes_Nx)

## EOC, Ncol

N_col_test = np.asarray([13, 25, 50, 100, 200, ])   ## grid for EOC

n_xs = []   ## store the result nx here
for Ncol in N_col_test:
    model = Agg_DPFR(450, Ncol, x_c, x_max, 2)        ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,:])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

slopes_Ncol = get_slope(relative_L1_norms) ## calculate slopes
print(slopes_Ncol)

'''
@note: The following setup is subject to changes depending on the core code configuration. 
@detail: Constant aggregation kernel in a DPFR tests and EOC tests using a reference solution. 
Assumes no solute (c) and solubility component (cs).
'''

Cadet.cadet_path = r'C:\Users\zwend\CADET\cadet72\bin\cadet-cli.exe' 

def Frag_DPFR(n_x : 'int, number of x bins', n_col : 'int, number of z bins', x_c, x_max, axial_order):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = log_normal(x_ct*1e6,0,1e16,0.4,150)  ## moved to larger sizes

    # Initial conditions
    initial_c = n_x*[0.0, ]

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # Tubular reactor
    model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    model.root.input.model.unit_001.col_length = 0.47
    model.root.input.model.unit_001.cross_section_area = 1.46e-4*0.21  # m^2
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05           # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8
    
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    model.root.input.model.unit_001.reaction.cry_breakage_rate_constant = 0.5e12
    model.root.input.model.unit_001.reaction.cry_breakage_kernel_gamma = 2 
    model.root.input.model.unit_001.reaction.cry_breakage_selection_function_alpha = 1

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6/60          ## m^3/s

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_coordinates = 0

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = 'practice1.h5'                    ## change as needed
    
    return model

# system setup
x_c, x_max = 1e-6, 1000e-6            # m
x_grid, x_ct = get_log_space(n_x, x_c, x_max)

cycle_time = 300                      # s
t = np.linspace(0, cycle_time, 200)

n_x = 100
n_col = 100

'''
@note: There is no analytical solution in this case. We are using this result as the reference solution?
'''

model = Frag_DPFR(n_x, n_col, x_c, x_max, 1)
model.save()
data = model.run()
model.load() 
c_x = model.root.output.solution.unit_002.solution_outlet[-1,:]

plt.xscale("log")
plt.plot(x_ct, c_x)
plt.xlabel(r'$Size/\mu m$')
plt.ylabel('Particle count/1')
plt.show()

'''
EOC tests in a DPFR, Fragmentation
@note: the EOC is obtained along the Nx and Ncol coordinate, separately
'''

## get ref solution
N_x_ref =   450
N_col_ref = 250
    
model = Frag_DPFR(N_x_ref, N_col_ref, x_c, x_max, 3)
model.save()
data = model.run()
model.load() 

c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,:]

## interpolate the reference solution at the reactor outlet

x_grid, x_ct = get_log_space(N_x_ref, x_c, x_max)

spl = UnivariateSpline(x_ct, c_x_reference)

## EOC, Nx

N_x_test = np.asarray([25, 50, 100, 200, 400, ])                          ## grid for EOC

n_xs = []
for Nx in N_x_test:
    model = Frag_DPFR(Nx, 250, x_c, x_max, 2)                ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,:])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))
    

slopes_Nx = get_slope(relative_L1_norms)      ## calculate slopes
print(slopes_Nx)

## EOC, Ncol

N_col_test = np.asarray([13, 25, 50, 100, 200, ])   ## grid for EOC

n_xs = []   ## store the result nx here
for Ncol in N_col_test:
    model = Frag_DPFR(450, Ncol, x_c, x_max, 2)        ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,:])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

slopes_Ncol = get_slope(relative_L1_norms) ## calculate slopes
print(slopes_Ncol)


'''
@note: The following setup is subject to changes depending on the core code configuration. 
@detail: Nucleation, growth, growth rate dispersion and aggregation in a DPFR tests and EOC tests using a reference solution. 
There are solute (c) and solubility components (cs).
'''

Cadet.cadet_path = r"C:\Users\zwend\CADET\cadet77\bin\cadet-cli.exe"

def NGRA(n_x : 'int, number of x bins + 2', n_col : 'int, number of z bins', x_c, x_max, axial_order : 'for weno schemes', growth_order):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x - 2, x_c, x_max)
    
    # c_feed
    c_feed = 9.0
    c_eq = 0.4

    # Boundary conditions
    boundary_c = []
    for i in range (0, n_x):
        if i == 0:
            boundary_c.append(c_feed)
        elif i == n_x - 1:
            boundary_c.append(c_eq)
        else:
            boundary_c.append(0.0)

    # Initial conditions
    initial_c = []
    for k in range(n_x):
        if k == 0:
            initial_c.append(0)
        elif k == n_x-1:
            initial_c.append(c_eq)
        else:
            initial_c.append(0)

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # Tubular reactor
    model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    model.root.input.model.unit_001.col_length = 0.47
    model.root.input.model.unit_001.cross_section_area = 3.066e-05
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8
    
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0             # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 5e-13
    
    model.root.input.model.unit_001.reaction.cry_nuclei_mass_density = 1.2e3
    model.root.input.model.unit_001.reaction.cry_vol_shape_factor = 0.524
    model.root.input.model.unit_001.reaction.cry_primary_nucleation_rate = 5.0
    model.root.input.model.unit_001.reaction.cry_secondary_nucleation_rate = 4e8

    model.root.input.model.unit_001.reaction.cry_growth_rate_constant = 5e-6
    model.root.input.model.unit_001.reaction.cry_g = 1.0

    model.root.input.model.unit_001.reaction.cry_a = 1.0
    model.root.input.model.unit_001.reaction.cry_growth_constant = 0.0
    model.root.input.model.unit_001.reaction.cry_p = 0.0

    model.root.input.model.unit_001.reaction.cry_k = 1.0
    model.root.input.model.unit_001.reaction.cry_u = 10.0
    model.root.input.model.unit_001.reaction.cry_b = 2.0

    model.root.input.model.unit_001.reaction.cry_growth_dispersion_rate = 2.5e-15
    model.root.input.model.unit_001.reaction.cry_growth_scheme_order = growth_order

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6/60     # Q, volumetric flow rate 

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-8
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_coordinates = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = 'practice1.h5'                    ## change as needed
    
    return model

# set up
n_x = 100 + 2
n_col = 100
x_c, x_max = 1e-6, 1000e-6       # m
x_grid, x_ct = get_log_space(n_x - 2, x_c, x_max)

# simulation time
cycle_time = 200                 # s
t = np.linspace(0, cycle_time, 200+1)

model = NGRA(n_x, n_col, x_c, x_max, 1, 1)
model.save()
data = model.run()
model.load()

t = model.root.input.solver.user_solution_times
c_x = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]


plt.xscale("log")
plt.plot(x_ct, c_x)
plt.xlabel(r'$Size/\mu m$')
plt.ylabel(r'$n/(1/m / m)$')
plt.show()


'''
EOC tests in a DPFR, Nucleation, growth, growth rate dispersion and aggregation
@note: the EOC is obtained along the Nx and Ncol coordinate, separately
'''

## get ref solution
N_x_ref =   500
N_col_ref = 500
    
model = NGRA(N_x_ref, N_col_ref, x_c, x_max, 3, 3)
model.save()
data = model.run()
model.load() 

c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]

## interpolate the reference solution at the reactor outlet

x_grid, x_ct = get_log_space(N_x_ref, x_c, x_max)

spl = UnivariateSpline(x_ct, c_x_reference)

## EOC, Nx

N_x_test = np.asarray([25, 50, 100, 200, 400, ])                                ## grid for EOC

n_xs = []
for Nx in N_x_test:
    model = NGRA(Nx, 400, x_c, x_max, 3, 2)                                     ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))
    

slopes_Nx = get_slope(relative_L1_norms)             ## calculate slopes
print(slopes_Nx)

## EOC, Ncol

N_col_test = np.asarray([25, 50, 100, 200, 400, ])   ## grid for EOC

n_xs = []   ## store the result nx here
for Ncol in N_col_test:
    model = NGRA(400, Ncol, x_c, x_max, 2, 3)        ## test on WENO23
    model.save()
    data = model.run()
    model.load() 

    n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])

relative_L1_norms = []  ## store the relative L1 norms here
for nx in n_xs:
    ## interpolate the ref solution on the test case grid

    x_grid, x_ct = get_log_space(len(nx), x_c, x_max)

    relative_L1_norms.append(calculate_normalized_error(spl(x_ct), nx, x_ct, x_grid))

slopes_Ncol = get_slope(relative_L1_norms) ## calculate slopes
print(slopes_Ncol)