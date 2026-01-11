from addict import Dict
import numpy as np


def get_model(
        spatial_method_bulk,
        refinement=1,
        **kwargs):
    
    radNElem = 8 * kwargs.get('radRefinement', refinement)
    
    model = Dict()
    
    model.input.model.nunits = 3
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, 6.e-05,
        1.e+00, 2.e+00,-1.e+00,-1.e+00, 6.e-05
        ]
    model.input.model.connections.switch_000.section = 0
    
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08
    
    
    model.input.model.unit_000.unit_type = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [1.0]
    model.input.model.unit_000.sec_000.cube_coeff = [0.0]
    model.input.model.unit_000.sec_000.lin_coeff = [0.0]
    model.input.model.unit_000.sec_000.quad_coeff = [0.0]
    model.input.model.unit_000.sec_001.const_coeff = [0.0]
    model.input.model.unit_000.sec_001.cube_coeff = [0.0]
    model.input.model.unit_000.sec_001.lin_coeff = [0.0]
    model.input.model.unit_000.sec_001.quad_coeff = [0.0]
    
    column = Dict()
    column.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    column.ncomp = 1
    column.col_radius_inner = 0.01
    column.col_radius_outer = 0.2
    column.npartype = 1
    column.total_porosity = 0.8
    column.par_type_volfrac = 1
    column.velocity_coeff = 0.000575
    column.col_dispersion = 5.75e-08
    column.init_c = [0.0]
    
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    if spatial_method_bulk != 0:
        raise Exception("Radial flow model is only supported by an FV bulk discretization, no DG available (yet)")
    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = radNElem
    column.discretization.GS_TYPE = 0
    column.discretization.MAX_KRYLOV = 10
    column.discretization.MAX_RESTARTS = 100
    column.discretization.SCHUR_SAFETY = 0.1
    
    column.particle_type_000.has_film_diffusion = 0
    column.particle_type_000.has_pore_diffusion = 0
    column.particle_type_000.has_surface_diffusion = 0
    column.particle_type_000.nbound = [1]
    column.particle_type_000.init_cp = [0.0]
    column.particle_type_000.init_cs = [0.0]
    column.particle_type_000.adsorption_model = 'LINEAR'
    column.particle_type_000.adsorption.is_kinetic = 1
    column.particle_type_000.adsorption.lin_ka = [12.3]
    column.particle_type_000.adsorption.lin_kd = [45.]
    
    model.input.model.unit_001 = column
    
    model.input.model.unit_002.ncomp = 1
    model.input.model.unit_002.unit_type = 'OUTLET'
    
    
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = 0
    model.input['return'].unit_001.write_sens_bulk = 0
    model.input['return'].unit_001.write_sens_last = 0
    model.input['return'].unit_001.write_sens_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 0
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_bulk = 0
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    
    model.input.solver.CONSISTENT_INIT_MODE = 1
    model.input.solver.CONSISTENT_INIT_MODE_SENS = 1
    model.input.solver.NTHREADS = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.sections.section_times = [  0., 60.,130.]
    model.input.solver.time_integrator.ABSTOL = 1e-10
    model.input.solver.time_integrator.ALGTOL = 1e-08
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-08
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.time_integrator.RELTOL = 1e-08
    model.input.solver.user_solution_times = np.linspace(0.0, 130.0, 131)

    return model


def add_sensitivity_radLRM_dynLin_1comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(0)},
        'TOTAL_POROSITY': { },
        'LIN_KA': {'sens_comp': np.int64(0), 'sens_boundphase': np.int64(0)}
    }    

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': np.int64(1)}
        model['input']['sensitivity']['sens_method'] = np.bytes_(b'ad1')

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)
    
    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_partype'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_reaction'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_section'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_boundphase'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_comp'] = np.int64(-1)
    
    if sensName in sensDepIdx:
        param = model['input']['sensitivity'][f'param_{sensIdx}']
        for key, value in {**sensDepIdx[sensName]}.items():
            model['input']['sensitivity'][f'param_{sensIdx}'][key] = value

    return model


def get_sensbenchmark1(spatial_method_bulk):
    
    model = get_model(spatial_method_bulk)
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_radLRM_dynLin_1comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_radLRM_dynLin_1comp_benchmark1(model, 'TOTAL_POROSITY')
    model = add_sensitivity_radLRM_dynLin_1comp_benchmark1(model, 'LIN_KA')
    
    return model


# from cadet import Cadet

# Cadet.cadet_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master3_generalUnit_4cc363a\aRELEASE\bin\cadet-cli.exe"

# model = Cadet()

# model.root.input = get_model(0)

# model.filename = "test0.h5"

# model.save()

# return_data = model.run_simulation()

# if not return_data.return_code == 0:
#     print(return_data.error_message)
#     raise Exception(f"simulation failed")
# model.load_from_file()





