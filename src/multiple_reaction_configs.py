import numpy as np

def cstr_setup(model):

    ncomp = 3
    init_c = [1.0, 2.0, 3.0]
    sim_time = 10.0

    model.root.input.model.nunits = 1

    # CSTR
    model.root.input.model.unit_000.unit_type = 'CSTR'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.init_liquid_volume = 1.0
    model.root.input.model.unit_000.init_c = init_c
    model.root.input.model.unit_000.const_solid_volume = 1.0
    model.root.input.model.unit_000.use_analytic_jacobian = 0 # jacobian has a bug

    #Configure solver settings
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [ ]

    return model

def lrmp_setup(model):
    
    ncomp = 3
    init_c = [1.0, 2.0, 3.0]
    init_q = init_c.copy()
    sim_time = 10.0

    model.root.input.model.nunits = 1

    model.root.input.model.unit_000.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.nbound = ncomp * [1.0]
    model.root.input.model.unit_000.init_liquid_volume = 1.0
    model.root.input.model.unit_000.init_c = init_c 
    model.root.input.model.unit_000.init_q = init_q
    model.root.input.model.unit_000.col_dispersion = 1e-5
    model.root.input.model.unit_000.col_length  = 0.6
    model.root.input.model.unit_000.col_porosity  = 0.37
    model.root.input.model.unit_000.film_diffusion = [1e-5] * ncomp

    model.root.input.model.unit_000.par_porosity = 0.33
    model.root.input.model.unit_000.par_radius = 1e-6
    model.root.input.model.unit_000.cross_section_area = 1.0386890710931253E-4

    model.root.input.model.unit_000.discretization.ncol                  = 1
    model.root.input.model.unit_000.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_000.discretization.weno.boundary_model   = 0 
    model.root.input.model.unit_000.discretization.weno.weno_eps         = 1e-10
    model.root.input.model.unit_000.discretization.weno.weno_order       = 1
    model.root.input.model.unit_000.discretization.max_restarts          = 10
    model.root.input.model.unit_000.discretization.gs_type              = 1
    model.root.input.model.unit_000.discretization.max_krylov           = 0
    model.root.input.model.unit_000.discretization.schur_safety          = 1e-8
    model.root.input.model.unit_000.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'

    # model.root.input.model.unit_000.reaction_model = 'MASS_ACTION_LAW'
    # model.root.input.model.unit_000.reaction_bulk.mal_kfwd_bulk = [0.5]
    # model.root.input.model.unit_000.reaction_bulk.mal_kbwd_bulk = [0.5]
    # model.root.input.model.unit_000.reaction_bulk.mal_stoichiometry_bulk = [[-1],[1],[0]]

    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = 0

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = []

    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_000.write_sens_outlet = 0

    # Adsorption model -> have to be set
    model.root.input.model.unit_000.adsorption_model = 'LINEAR'
    model.root.input.model.unit_000.adsorption.is_kinetic  = True
    model.root.input.model.unit_000.adsorption.lin_ka = [0,0,0]
    model.root.input.model.unit_000.adsorption.lin_kd = [0,0,0]

    return model

def grm_setup(model):
    
    ncomp = 3
    init_c = [1.0, 2.0, 3.0]
    init_q = init_c.copy()

    sim_time = 10.0

    model.root.input.model.nunits = 1

    # GRM
    model.root.input.model.unit_000.unit_type = 'GENERAL_RATE_MODEL'
    model.root.input.model.unit_000.ncomp = 3
    model.root.input.model.unit_000.nbound = [1.0] * ncomp
    model.root.input.model.unit_000.init_c = init_c 
    model.root.input.model.unit_000.init_q = init_q

    ## Geometry
    model.root.input.model.unit_000.col_length = 0.1    #h5            # m
    model.root.input.model.unit_000.col_porosity = 0.37 #h5            # -
    model.root.input.model.unit_000.par_porosity = 0.33 #5            # -
    model.root.input.model.unit_000.par_radius = 1e-6   #5            # m
                                                                    
    ## Transport
    model.root.input.model.unit_000.col_dispersion = 1e-8   #h5
    model.root.input.model.unit_000.col_dispersion_radial = 1e-6   #h5       # m^2 / s (interstitial volume)
    model.root.input.model.unit_000.film_diffusion = [1e-5] *ncomp   #5     # m / s
    model.root.input.model.unit_000.par_diffusion = [1e-10] *ncomp#h5       # m^2 / s (mobile phase)  
    model.root.input.model.unit_000.par_surfdiffusion = [0.0]*ncomp  #h5    # m^2 / s (solid phase)
    model.root.input.model.unit_000.par_coreradius = 0 #h5      # m^2 / s (solid phase)
    model.root.input.model.unit_000.total_porosity = 0.84 #h5      # m^2 / s (solid phase)
    model.root.input.model.unit_000.velocity = 5.75e-4 #h5      # m^2 / s (solid phase)

    ## Discretization
    ### Grid cells
    model.root.input.model.unit_000.discretization.ncol                  = 1
    model.root.input.model.unit_000.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_000.discretization.weno.boundary_model   = 0 
    model.root.input.model.unit_000.discretization.weno.weno_eps         = 1e-10
    model.root.input.model.unit_000.discretization.weno.weno_order       = 1
    model.root.input.model.unit_000.discretization.max_restarts          = 10
    model.root.input.model.unit_000.discretization.gs_type              = 1
    model.root.input.model.unit_000.discretization.max_krylov           = 0
    model.root.input.model.unit_000.discretization.schur_safety          = 1e-8
    model.root.input.model.unit_000.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'
    model.root.input.model.unit_000.discretization.npar = 1


    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_sens_outlet = 1


    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1


    """Connect the units together"""
    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = []

     # Adsorption model -> have to be set
    model.root.input.model.unit_000.adsorption_model = 'LINEAR'
    model.root.input.model.unit_000.adsorption.is_kinetic  = True
    model.root.input.model.unit_000.adsorption.lin_ka = [0,0,0]
    model.root.input.model.unit_000.adsorption.lin_kd = [0,0,0]

    return model

def mal_setup_particle_old(model, reaction_dict):
    
    #Configure the reaction system
    model.root.input.model.unit_000.reaction_model_particles = reaction_dict[f"reaction_type_000"]
    model.root.input.model.unit_000.reaction_particle_000.mal_kfwd_liquid = [reaction_dict[f'kfwd_liquid_000']]
    model.root.input.model.unit_000.reaction_particle_000.mal_kbwd_liquid = [reaction_dict[f'kbwd_liquid_000']]

        # Stoichiometry matrix 2D array [components][reaction]
    model.root.input.model.unit_000.reaction_particle_000.mal_stoichiometry_liquid = reaction_dict[f'stoichiometric_matrix_liquid_000']

    model.root.input.model.unit_000.reaction_particle_000.mal_kfwd_solid = [reaction_dict[f'kfwd_solid_000']]
    model.root.input.model.unit_000.reaction_particle_000.mal_kbwd_solid = [reaction_dict[f'kbwd_solid_000']]

        # Stoichiometry matrix 2D array [components][reaction]
    model.root.input.model.unit_000.reaction_particle_000.mal_stoichiometry_solid = reaction_dict[f'stoichiometric_matrix_solid_000']

    return model

def mal_setup_bulk_old(model, reaction_dict):
    
    #Configure the reaction system
    model.root.input.model.unit_000.reaction_model = reaction_dict[f"reaction_type_000"]
    model.root.input.model.unit_000.reaction_bulk.mal_kfwd_bulk = [reaction_dict[f'kfwd_000']]
    model.root.input.model.unit_000.reaction_bulk.mal_kbwd_bulk = [reaction_dict[f'kbwd_000']]

        # Stoichiometry matrix 2D array [components][reaction]
    model.root.input.model.unit_000.reaction_bulk.mal_stoichiometry_bulk = reaction_dict[f'stoichiometric_matrix_000']

    return model

def mal_setup_bulk(model, reaction_dict):
    
    #Configure the reaction system
    model.root.input.model.unit_000.reaction_bulk.nreac = reaction_dict['num_reactions']
    for i in range(reaction_dict['num_reactions']):
        getattr(model.root.input.model.unit_000.reaction_bulk, f"reaction_model_{i:03d}").reaction_type = reaction_dict[f"reaction_type_{i:03d}"]
        getattr(model.root.input.model.unit_000.reaction_bulk, f"reaction_model_{i:03d}").mal_kfwd_bulk = [reaction_dict[f'kfwd_{i:03d}']]
        getattr(model.root.input.model.unit_000.reaction_bulk, f"reaction_model_{i:03d}").mal_kbwd_bulk = [reaction_dict[f'kbwd_{i:03d}']]

        # Stoichiometry matrix 2D array [components][reaction]
        getattr(model.root.input.model.unit_000.reaction_bulk, f"reaction_model_{i:03d}").mal_stoichiometry_bulk = reaction_dict[f'stoichiometric_matrix_{i:03d}']

    return model

def mal_setup_cross_phase(model, reaction_dict):
    
    #Configure the reaction system

    model.root.input.model.unit_000.reaction_model_particles = reaction_dict[f"reaction_type_000"]
    model.root.input.model.unit_000.reaction_particle_000.mal_kfwd_liquid = [reaction_dict[f'kfwd_liquid_000']]
    model.root.input.model.unit_000.reaction_particle_000.mal_kbwd_liquid = [reaction_dict[f'kbwd_liquid_000']]

        # Stoichiometry matrix 2D array [components][reaction]
    model.root.input.model.unit_000.reaction_particle_000.mal_stoichiometry_liquid = reaction_dict[f'stoichiometric_matrix_liquid_000']

    model.root.input.model.unit_000.reaction_particle_000.mal_kfwd_solid = [reaction_dict[f'kfwd_solid_000']]
    model.root.input.model.unit_000.reaction_particle_000.mal_kbwd_solid = [reaction_dict[f'kbwd_solid_000']]

        # Stoichiometry matrix 2D array [components][reaction]
    model.root.input.model.unit_000.reaction_particle_000.mal_stoichiometry_solid = reaction_dict[f'stoichiometric_matrix_solid_000']

    return model

def mal_setup_paricle(model, reaction_dict):
    
    #Configure the reaction system
    model.root.input.model.unit_000.reaction_particle.nreac = reaction_dict['num_reactions']
    for i in range(reaction_dict['num_reactions']):
        getattr(model.root.input.model.unit_000.reaction_particle, f"reaction_model_{i:03d}").reaction_type = reaction_dict[f"reaction_type_{i:03d}"]
        getattr(model.root.input.model.unit_000.reaction_particle, f"reaction_model_{i:03d}").mal_kfwd = [reaction_dict[f'kfwd_{i:03d}']]
        getattr(model.root.input.model.unit_000.reaction_particle, f"reaction_model_{i:03d}").mal_kbwd = [reaction_dict[f'kbwd_{i:03d}']]

        # Stoichiometry matrix 2D array [components][reaction]
        getattr(model.root.input.model.unit_000.reaction_particle, f"reaction_model_{i:03d}").mal_stoichiometry = reaction_dict[f'stoichiometric_matrix_{i:03d}']

    return model

def get_dict_one_reaction_mal():
    
    return {
    "num_reactions": 1, 
    "reaction_type_000": "MASS_ACTION_LAW",
    "stoichiometric_matrix_000": [[-1, -1, 0],
                                [1, 1, -1],
                                [1, 0, 1]],
    "kfwd_000": [1.0, 0.5, 1.0],
    "kbwd_000": [1.0, 0.3, 1.0]
}

def get_dict_two_reaction_mal():
    
    return  {
    "num_reactions": 2, 
    "reaction_type_000": "MASS_ACTION_LAW",
    "stoichiometric_matrix_000": [[-1, -1],
                                    [1, 1],
                                    [1, 0]],
    "kfwd_000": [1.0, 0.5],
    "kbwd_000": [1.0, 0.3], 
    "reaction_type_001": "MASS_ACTION_LAW",
    "stoichiometric_matrix_001": [[0],
                                [-1],
                                [1]],
    "kfwd_001": [1.0],
    "kbwd_001": [1.0]

    }

def get_dict_three_reaction_mal():
    return {
    "num_reactions": 3,
    "reaction_type_000": "MASS_ACTION_LAW",
    "stoichiometric_matrix_000": [[-1], [1], [1]],
    "kfwd_000": [1.0],
    "kbwd_000": [1.0],
    "reaction_type_001": "MASS_ACTION_LAW",
    "stoichiometric_matrix_001": [[-1], [1], [0]],
    "kfwd_001": [0.5],
    "kbwd_001": [0.3],
    "reaction_type_002": "MASS_ACTION_LAW",
    "stoichiometric_matrix_002": [[0], [-1], [1]],
    "kfwd_002": [1.0],
    "kbwd_002": [1.0]
    }

def get_dict_one_reaction_cross_phase():
    return {
        "num_reactions": 1,
        "reaction_type_000": "MASS_ACTION_LAW",
        "kfwd_liquid_000": 1.0,
        "kbwd_liquid_000": 1.0,
        "stoichiometric_matrix_liquid_000": [[-1], [1], [0]],
        "kfwd_solid_000": 0.5,
        "kbwd_solid_000": 0.3,
        "stoichiometric_matrix_solid_000": [[0], [0], [1]]
    }

def get_dict_two_reactions_cross_phase():
    return {
        "num_reactions": 2,
        "reaction_type_000": "MASS_ACTION_LAW",
        "kfwd_liquid_000": 1.0,
        "kbwd_liquid_000": 1.0,
        "stoichiometric_matrix_liquid_000": [[-1], [1], [0]],
        "kfwd_solid_000": 0.5,
        "kbwd_solid_000": 0.3,
        "stoichiometric_matrix_solid_000": [[0], [0], [1]],
        "reaction_type_001": "MASS_ACTION_LAW",
        "kfwd_liquid_001": 0.8,
        "kbwd_liquid_001": 0.4,
        "stoichiometric_matrix_liquid_001": [[-1], [0], [1]],
        "kfwd_solid_001": 0.6,
        "kbwd_solid_001": 0.2,
        "stoichiometric_matrix_solid_001": [[0], [1], [-1]]
    }

def get_dict_three_reactions_cross_phase():
    return {
        "num_reactions": 3,
        "reaction_type_000": "MASS_ACTION_LAW",
        "kfwd_liquid_000": 1.0,
        "kbwd_liquid_000": 1.0,
        "stoichiometric_matrix_liquid_000": [[-1], [1], [0]],
        "kfwd_solid_000": 0.5,
        "kbwd_solid_000": 0.3,
        "stoichiometric_matrix_solid_000": [[0], [0], [1]],
        "reaction_type_001": "MASS_ACTION_LAW",
        "kfwd_liquid_001": 0.8,
        "kbwd_liquid_001": 0.4,
        "stoichiometric_matrix_liquid_001": [[-1], [0], [1]],
        "kfwd_solid_001": 0.6,
        "kbwd_solid_001": 0.2,
        "stoichiometric_matrix_solid_001": [[0], [1], [-1]],
        "reaction_type_002": "MASS_ACTION_LAW",
        "kfwd_liquid_002": 0.7,
        "kbwd_liquid_002": 0.5,
        "stoichiometric_matrix_liquid_002": [[0], [-1], [1]],
        "kfwd_solid_002": 0.9,
        "kbwd_solid_002": 0.1,
        "stoichiometric_matrix_solid_002": [[1], [0], [-1]]
    }

def get_dict_one_reaction_particle():
    return {
        "num_reactions": 1,
        "reaction_type_000": "MASS_ACTION_LAW",
        "stoichiometric_matrix_000": [[-1], [1], [0]],
        "kfwd_000": [1.0],
        "kbwd_000": [1.0]
    }

def get_dict_two_reactions_particle():
    return {
        "num_reactions": 2,
        "reaction_type_000": "MASS_ACTION_LAW",
        "stoichiometric_matrix_000": [[-1], [1], [0]],
        "kfwd_000": [1.0],
        "kbwd_000": [1.0],
        "reaction_type_001": "MASS_ACTION_LAW",
        "stoichiometric_matrix_001": [[0], [-1], [1]],
        "kfwd_001": [0.5],
        "kbwd_001": [0.3]
    }

def get_dict_three_reactions_particle():
    return {
        "num_reactions": 3,
        "reaction_type_000": "MASS_ACTION_LAW",
        "stoichiometric_matrix_000": [[-1], [1], [0]],
        "kfwd_000": [1.0],
        "kbwd_000": [1.0],
        "reaction_type_001": "MASS_ACTION_LAW",
        "stoichiometric_matrix_001": [[0], [-1], [1]],
        "kfwd_001": [0.5],
        "kbwd_001": [0.3],
        "reaction_type_002": "MASS_ACTION_LAW",
        "stoichiometric_matrix_002": [[-1], [0], [1]],
        "kfwd_002": [0.8],
        "kbwd_002": [0.4]
    }
