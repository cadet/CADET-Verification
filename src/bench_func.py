"""
Created April 2024

This script implement helper functions to create benchmarks.

@author: jmbr
"""

import urllib.request
import json
import re
import numpy as np
from cadet import Cadet
import utility.convergence as convergence
import os
from joblib import Parallel, delayed
import copy

from benchmark_models import settings_2Dchromatography


# %% Import packages and define helper functions


def write_result_json(path_and_name, group, table_, sub_group=None, is_dict=True):
    try:
        with open(path_and_name, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}

    if not is_dict:
        table_dict = {header: column for header,
                      column in zip(table_[0], zip(*table_[1]))}
    else:
        table_dict = table_

    # Check if 'convergence' exists, if not, create it
    if 'convergence' not in existing_data:
        existing_data['convergence'] = {}

    existing_data['convergence'].setdefault(group, {})

    if sub_group is not None:
        existing_data['convergence'][group].setdefault(sub_group, {})
        # Update the subgroup with new table entries
        for header, column in table_dict.items():
            existing_data['convergence'][group][sub_group][header] = column
    else:
        # Update the group with new table entries
        for header, column in table_dict.items():
            existing_data['convergence'][group][header] = column

    # Write the updated dictionary to the JSON file
    with open(path_and_name, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


def write_meta_json(path_and_name, meta):
    if os.path.exists(path_and_name):
        with open(path_and_name, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data['meta'] = meta

    with open(path_and_name, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


def run_simulation_in_verification(model, cadet_path):
    if cadet_path is not None:
        model[0].install_path = cadet_path
    model[0].save()
    data = model[0].run_simulation()
    if not data.return_code == 0:
        #     (f"simulation completed successfully")
        #     model.load_from_file()
        # else:
        print(data.error_message)
        raise Exception(f"simulation failed")


def create_object_from_database(
        database_url, cadet_config_json_name,
        unit_id=None,
        ax_method=None, ax_cells=None, par_method=None, par_cells=None,
        rad_method=None, rad_cells=None,
        output_path=None,
        idas_abstol=None, include_sens=True, **kwargs):
    """ Reads a CADET-configuration from the Database, adjusts it and returns 
        the Cadet-Object. Optionally saves the corresponding h5 config file.

    Parameters
    ----------
    database_url : String 
        URL to database files.
    cadet_config_json_name : List of Strings
        Name of model configuration files, including .json ending.
    unit_id : String
        Three character string for the unit ID to be discretized. Optional,
        unit discretization will not be changed.
    ax_methods : Int
        Specifies the axial discretization method.
    ax_cells : Int
        Specifies the number of axial cells/elements
    par_methods : Int
        Specifies the particle discretization method.
    par_cells : Int
        Specifies the number of particle cells/elements
    rad_methods : Int
        Specifies the particle discretization method.
    rad_cells : Int
        Specifies the number of radial cells/elements
    output_path : String 
        Path to output folder. Optional, h5 file will not be created/saved.
    include_sens : Bool 
        Specifies whether or not sensitivities should be configered if existent
        in configuration source file.
    Returns
    -------
    Cadet-Object
        Cadet object.
    """

    # Read configuration
    with urllib.request.urlopen(
            database_url + cadet_config_json_name) as url:

        config_data = json.loads(url.read().decode())

    setting_name = re.search(r'configuration_(.*?)(?:\.json|_FV|_DG)',
                             cadet_config_json_name).group(1)

    return create_object_from_config(
        config_data=config_data,
        setting_name=setting_name,
        unit_id=unit_id,
        ax_method=ax_method, ax_cells=ax_cells, par_method=par_method, par_cells=par_cells,
        output_path=output_path,
        idas_abstol=idas_abstol, include_sens=include_sens, **kwargs
    )


def create_object_from_config(
        config_data, setting_name,
        unit_id=None,
        ax_method=None, ax_cells=None, par_method=None, par_cells=None,
        rad_method=None, rad_cells=None,
        output_path=None,
        idas_abstol=None, include_sens=True, **kwargs):
    """ Takes a CADET-configuration as dictionary, adjusts it and returns the 
        Cadet-Object. Optionally saves the corresponding h5 config file.

    Parameters
    ----------
    config_data : Dictionary
        Dictionary with CADET configuration data
    setting_name : String
        Name of setting, will be sued to create the CADET filename
    unit_id : String
        Three character string for the unit ID to be discretized. Optional,
        unit discretization will not be changed.
    ax_methods : Int
        Specifies the axial discretization method.
    ax_cells : Int
        Specifies the number of axial cells/elements
    par_methods : Int
        Specifies the particle discretization method.
    par_cells : Int
        Specifies the number of particle cells/elements
    rad_methods : Int
        Specifies the particle discretization method.
    rad_cells : Int
        Specifies the number of radial cells/elements
    output_path : String 
        Path to output folder. Optional, h5 file will not be created/saved.
    include_sens : Bool 
        Specifies whether or not sensitivities should be configered if existent
        in configuration source file.
    Returns
    -------
    Cadet-Object
        Cadet object.
    """

    convergence_sim_names = []

    # Adjust configuration to desired numerical refinement
    if idas_abstol is not None:
        config_data['input']['solver']['time_integrator']['ABSTOL'] = idas_abstol
        config_data['input']['solver']['time_integrator']['RELTOL'] = idas_abstol*100
        config_data['input']['solver']['time_integrator']['ALGTOL'] = idas_abstol*100
    if 'USE_MODIFIED_NEWTON' in kwargs:
        config_data['input']['solver']['time_integrator']['USE_MODIFIED_NEWTON'] = kwargs['USE_MODIFIED_NEWTON']

    if kwargs.get('system_refinement_IDs', None) is not None:
        refinement_IDs = kwargs['system_refinement_IDs']
    else:
        refinement_IDs = [unit_id]
        
    for tmpID in refinement_IDs:
        if tmpID is not None:
            if ax_method == 0:
                config_data['input']['model']['unit_' +
                                              tmpID]['discretization']['SPATIAL_METHOD'] = "FV"
                config_data['input']['model']['unit_' +
                                              tmpID]['discretization']['NCOL'] = ax_cells
                if 'WENO_ORDER' in kwargs.keys():
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['weno']['WENO_ORDER'] = kwargs['WENO_ORDER']
            else:
                config_data['input']['model']['unit_' +
                                              tmpID]['discretization']['SPATIAL_METHOD'] = "DG"
                if rad_method is None:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['POLYDEG'] = ax_method
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['NELEM'] = ax_cells
                else:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['AX_POLYDEG'] = ax_method
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['AX_NELEM'] = ax_cells

            if par_method is not None:
                if par_method == 0:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['NPAR'] = par_cells
                else:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['PAR_POLYDEG'] = par_method
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['PAR_NELEM'] = par_cells
            if rad_method is not None:
            
                config_data['input']['model']['unit_'+tmpID].PORTS = (rad_method + 1 ) * rad_cells
            
                if rad_method == 0:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['NRAD'] = rad_cells
                else:
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['RAD_POLYDEG'] = rad_method
                    config_data['input']['model']['unit_' +
                                                  tmpID]['discretization']['RAD_NELEM'] = rad_cells

                if kwargs.get('rad_inlet_profile', None) is None:
                    # if we have more than 1 inlet, there are radial zones defined
                    n_units = config_data['input']['model']['nunits']
                    nInlets = n_units - 2 if kwargs.get('analytical_reference', 0) else n_units - 1
                    add_inlet_per_port = nInlets if nInlets > 2 else False
                else:
                    add_inlet_per_port = kwargs.get('rad_inlet_profile')
                    nOutlets = 1 if kwargs.get('analytical_reference', 0) else 0
                    n_units = (rad_method + 1 ) * rad_cells + 1 + nOutlets
                config_data['input']['model'].nunits = n_units
                
                connections, rad_coords = settings_2Dchromatography.generate_connections_matrix(
                    rad_method=rad_method, rad_cells=rad_cells,
                    velocity=config_data['input']['model']['unit_' +
                                                           tmpID].VELOCITY,
                    porosity=config_data['input']['model']['unit_' +
                                                           tmpID].COL_POROSITY,
                    col_radius=config_data['input']['model']['unit_' +
                                                             tmpID].COL_RADIUS,
                    add_inlet_per_port=add_inlet_per_port, add_outlet=int(kwargs.get('analytical_reference', 0))
                )

                if add_inlet_per_port is True:
                    for rad in range(rad_cells * (rad_method + 1)):
                
                        config_data['input']['model']['unit_' +
                                                      str(rad + 1).zfill(3)] = copy.deepcopy(config_data['input']['model']['unit_001'])
    
                        if kwargs.get('rad_inlet_profile', None) is not None:
                            config_data['input']['model']['unit_001'].sec_000.CONST_COEFF = kwargs['rad_inlet_profile'](
                                rad_coords[rad], config_data['input']['model']['unit_000'].COL_RADIUS)

                config_data['input'].model.connections.switch_000.connections = connections
            
                if 'radial_init_conc' in kwargs:
                    config_data['input']['model']['unit_'+tmpID].init_c = kwargs['radial_init_conc'](np.array(rad_coords))

            if 'LINEAR_SOLVER' in kwargs:
                config_data['input']['model']['unit_' +
                                          tmpID]['discretization']['LINEAR_SOLVER'] = kwargs['LINEAR_SOLVER']

    # Create configuration name
    if rad_method is not None:
        if par_method is not None:
            config_name = convergence.generate_2DGRM_name(
                setting_name, ax_method, ax_cells, rad_method, rad_cells, par_method, par_cells
            )
        elif ax_method is not None:
            config_name = convergence.generate_2D_name(
                setting_name, ax_method, ax_cells, rad_method, rad_cells
            )
    elif par_method is not None:
        config_name = convergence.generate_GRM_name(
            setting_name, ax_method, ax_cells, par_method, par_cells
        )
    elif ax_method is not None:
        config_name = convergence.generate_1D_name(
            setting_name, ax_method, ax_cells
        )
    else:
        config_name = re.sub('.json', '', setting_name) + '.h5'

    # Optionally exclude sensitivities (which influence the approximation)
    if 'sensitivity' in config_data['input']:
        if include_sens:
            sensitivity = config_data['input']['sensitivity']
            # config_data['input']['solver']['time_integrator']['SENS_ABSTOL'] = idas_abstol * 1000
        else:
            config_data['input'].pop('sensitivity')
            sensitivity = {'NSENS': 0}
            config_name = re.sub('sens|sensitivity', '', config_name)
    else:
        sensitivity = {'NSENS': 0}

    model = Cadet()
    model.root.input = copy.deepcopy(config_data['input'])
    if output_path is not None:
        if 'filename_prefix' not in kwargs.keys():
            model.filename = str(output_path) + '/' + config_name
        else:
            model.filename = str(output_path) + '/' + \
                kwargs['filename_prefix'] + config_name
        model.save()
    return model

# Note: if ref_file is not given but rerun ref is true, then we take the last disc as reference and exclude it from EOC


def generate_convergence_data(
        database_url, cadet_config_json,
        ax_method, ax_disc, rad_method=None, rad_disc=None,
        par_method=None, par_disc=None,
        output_path=None,
        write_result=1, ref_file=None,
        which='outlet', unitID='001',
        sens_included=True,
        write_sens=True,
        commit_hash=None,
        **kwargs
):
    """ Creates convergence data.

    Parameters
    ----------
    database_url : String 
        URL to datebase files.
    cadet_config_json : String
        Name of model configuration file, including .json ending.
    ax_methods : Int
        Specifies the axial discretization method.
    ax_cells : List of Ints
        Specifies the number of cells/elements.
    rad_methods : Int
        Specifies the radial discretization method.
    rad_cells : List of Ints
        Specifies the number of cells/elements.
    par_methods : Int
        Specifies the particle discretization method.
    par_cells : List of Ints
        Specifies the number of cells/elements.
    output_path : String 
        Path to output folder. Optional, output file will not be saved.
    which : String
        Solution part of interest. Optional, defaults to 'outlet'
    unitID : String
        Three character string for the unit ID whose discretization was varied.
        Defaults to '001'
    sens_included : Bool 
        Specifies whether or not sensitivities were included in the simulations
        if they exist in the source file.
    write_sens : Bool 
        Specifies whether or not sensitivity convergence data should be written
        to the output file.
    commit_hash : String
        Commit hash of CADET commit with which the results were computed
    Returns
    -------
    Bool
        Success.
    """
    GRM_setting = 0 if par_method is None else 1

    setting_name = re.search(r'configuration_(.*?)(?:\.json|_FV|_DG)',
                             cadet_config_json)
    if setting_name is None:
        setting_name = cadet_config_json
    else:
        setting_name = setting_name.group(1)

    convergence_sim_names = []

    # read setup
    if database_url is not None:
        with urllib.request.urlopen(
                database_url + cadet_config_json) as url:

            config_data = json.loads(url.read().decode())

        if 'sensitivity' in config_data['input']:
            if sens_included:
                sensitivity = config_data['input']['sensitivity']
            else:
                config_data['input'].pop('sensitivity')
                sensitivity = {'NSENS': 0}
                setting_name = re.sub('sens|sensitivity', '', setting_name)
        else:
            sensitivity = {'NSENS': 0}
    else:
        if sens_included:
            
            file_name = convergence.generate_simulation_names(
                prefix=setting_name,
                ax_methods=[ax_method], ax_cells=[ax_disc[0]],
                par_methods=None if par_method is None else [par_method],
                par_cells=None if par_method is None else [par_disc[0]],
                rad_methods=None if rad_method is None else [rad_method],
                rad_cells=None if rad_method is None else [rad_disc[0]]
                )[0]
            
            sensitivity = convergence.sim_go_to(
                convergence.get_simulation(output_path + '/' + file_name).root,
                ['input', 'sensitivity']
                )
            config_data = None
        else:
            sensitivity = {'NSENS': 0}
            config_data = None

    table = convergence.recalculate_results(
        file_path=str(output_path) + '/', models=[setting_name],
        ax_methods=[ax_method], ax_cells=ax_disc,
        par_methods=[par_method], par_cells=par_disc,
        rad_methods=[rad_method], rad_cells=rad_disc,
        exact_names=[ref_file],
        unit=unitID, which=which,
        incl_min_val=True,
        transport_model=kwargs.pop('transport_model', None),
        ncomp=kwargs.pop('ncomp', None), nbound=kwargs.pop('nbound', None),
        save_path_=None,
        simulation_names=None,
        save_names=None,
        export_results=False,
        **kwargs
    )

    table = table.to_dict(orient='list')
    print("Outlet convergence")
    print(table)

    if write_result:
        # Adjust meta data
        if config_data is not None:
            config_data['input']['meta'] = config_data.pop('meta')
            config_data['input']['meta']['source'] = 'CADET-Reference convergence data https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-reference_data'
            config_data['input']['meta']['info'] = 'CADET-Reference convergence data for the CADET-Database model configuration: ' + \
                setting_name + ' whose source is ' + \
                config_data['input']['meta']['source']
            config_data['input']['meta']['name'] = 'convergence_' + \
                setting_name
            if commit_hash is not None:
                config_data['input']['meta']['cadet_commit_last_run'] = commit_hash
        # get method name
        if ax_method == 0:
            method_name = 'FV'
        else:
            method_name = 'DG_P' + str(ax_method)
            if rad_method is not None:
                method_name += 'radP' + str(rad_method)
            if par_method is not None:
                method_name += 'parP' + str(par_method)

        if config_data is not None and 'meta' in config_data['input'].keys():
            write_meta_json(
                str(output_path) + '/convergence_' + setting_name + '.json',
                config_data['input']['meta']
            )
        write_result_json(
            str(output_path) + '/convergence_' + setting_name + '.json',
            method_name, table, sub_group='outlet', is_dict=True
        )

        result_name = 'convergence_' + setting_name + '.json'

    if not write_sens:
        return True
    
    for sensIdx in range(sensitivity.get('nsens') or sensitivity.get('NSENS')):

        sens_name = str(sensitivity['param_' + '{:03d}'.format(sensIdx)].get('sens_name') or sensitivity['param_' + '{:03d}'.format(sensIdx)].get('SENS_NAME'))

        table = convergence.recalculate_results(
            file_path=str(output_path) + '/', models=[setting_name],
            ax_methods=[ax_method], ax_cells=ax_disc,
            rad_methods=[rad_method], rad_cells=rad_disc,
            exact_names=[ref_file],
            unit=unitID, which='sens_' + which,
            par_methods=[par_method], par_cells=par_disc,
            incl_min_val=True,
            transport_model=None, ncomp=None, nbound=None,
            save_path_=None,
            simulation_names=None,
            save_names=None,
            export_results=False,
            **{'sensIdx': sensIdx}
        )

        table = table.to_dict(orient='list')
        if commit_hash is not None:
            table['cadet_commit'] = commit_hash
        print(sens_name + " sensitivity convergence")
        print(table)

        if write_result:
            write_result_json(
                str(output_path) + '/convergence_' +
                setting_name + '.json', method_name, table,
                sub_group='sens_' + sens_name, is_dict=True
            )

    return result_name


def disc_array(start, n):
    powers_of_two = np.zeros(n, dtype=int)
    powers_of_two[0] = start
    for i in range(1, n):
        powers_of_two[i] = powers_of_two[i - 1] * 2
    return powers_of_two


def disc_list(start, n):
    powers_of_two = [start]
    for i in range(1, n):
        powers_of_two.append(powers_of_two[i - 1] * 2)
    return powers_of_two


def run_convergence_analysis_from_configs(
        output_path=None,
        cadet_path=None,
        cadet_configs=None,
        cadet_config_names=[
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
        ],
        include_sens=[True, True],
        ref_files=[[None, None], [None, None]],
        unit_IDs=['001', '001'],
        which=['outlet', 'outlet'],
        ax_methods=[
        [0, 3],
        [0, 3]
        ],
        ax_discs=[
            [disc_list(8, 3), disc_list(1, 3)],
            [disc_list(8, 3), disc_list(4, 3)]
        ],
        par_methods=[
            [None, None],
            [0, 3]
        ],
        par_discs=[
            [None, None],
            [disc_list(1, 3), disc_list(1, 3)]
        ],
        rad_methods=None,
        rad_discs=None,
        idas_abstol=[[None, None], [None, None]],
        n_jobs=-1,
        rerun_sims=True,
        **kwargs
):
    """ Runs convergence analyses for the specified models and methods

    Parameters
    ----------

    idas_tol If set to None, the abstol from the configuration file will be used
    re_files: size model x methods
    unit_IDs size: model
    include_sens size: model TODO: * methods

    cadet_path : String
        Specifies the path to the cadet-cli executable, defaults to None.
        The default uses the Cadet executable installed for the current conda
        environment.
    cadet_configs : List of Dicts of size n_models
        CADET-configurations
    cadet_config_names : List of Strings of size n_models
        Names of configurations
    include_sens : List of Strings of size n_models
        Specifies the name of a reference h5 file that exists under
    ref_files : Lists of Strings of size n_models per method
        Specifies the name of a reference h5 file that exists under
        the output_path, defaults to Nones, which will use the last respective
        discretization as reference.
    ax_methods : list of Ints per model
        Specifies the axial discretization methods.
    ax_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all axial discretization methods.
    par_methods : list of Ints per model
        Specifies the particle discretization methods.
    par_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all particle discretization methods.
    rad_methods : list of Ints per model
        Specifies the radial discretization methods.
    rad_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all radial discretization methods.
    which : list of Strings per model
        Specifies the considered solution type, defaults to 'outlet'
    unitID : list of Strings per model
        Specifies the considered unit operation's ID, defaults to '001'
    idas_abstol: list of Floats per model and method
        Absolute time integration tolerance. Default None will make use of the
        tolerance from the configuration source file
    n_jobs: Int
        Specifies the number of workers. Defaults to one.
    rerun_sims: Bool
        Specifies whether or not the simulations should be rerun, as opposed to
        only comuting the tables from existing simulation files.
        Defaults to True


    Returns
    -------
    bool
        Success.
    """

    commit_hash = None

    if 'refinement_IDs' not in kwargs.keys():
        refinement_IDs = unit_IDs
    elif kwargs['refinement_IDs'] == []:
        refinement_IDs = unit_IDs
    else:
        refinement_IDs = kwargs['refinement_IDs']

    if rerun_sims:
        # Create simulation objects
        sims = []  # To be filled with all Cadet objects

        for modelIdx in range(0, len(cadet_config_names)): 

            for methodIdx in range(0, len(ax_methods[modelIdx])):

                for discIdx in range(0, len(ax_discs[modelIdx][methodIdx])):

                    par_method = par_methods[modelIdx][methodIdx]
                    par_cells = None
                    if par_method is not None:
                        par_cells = par_discs[modelIdx][methodIdx][discIdx]
                    if rad_methods is not None:
                        rad_method = rad_methods[modelIdx][methodIdx]
                        rad_cells = rad_discs[modelIdx][methodIdx][discIdx]
                    else:
                        rad_method = None
                        rad_cells = None

                    sims.append(
                        create_object_from_config(
                            config_data=cadet_configs[modelIdx],
                            setting_name=cadet_config_names[modelIdx],
                            unit_id=refinement_IDs[modelIdx],
                            ax_method=ax_methods[modelIdx][methodIdx],
                            ax_cells=ax_discs[modelIdx][methodIdx][discIdx],
                            par_method=par_method, par_cells=par_cells,
                            rad_method=rad_method, rad_cells=rad_cells,
                            output_path=output_path,
                            idas_abstol=idas_abstol[modelIdx][methodIdx],
                            include_sens=include_sens[modelIdx],
                            **kwargs
                        )
                    )

        # Run simulations in one global parallelization
        backend = Parallel(n_jobs=n_jobs, verbose=0)
        backend(delayed(run_simulation_in_verification)(sim, cadet_path)
                for sim in zip(sims))

        commit_hash = convergence.get_commit_hash(
            convergence.get_simulation(sims[0].filename)
            )

    return run_convergence_analysis_core(
        commit_hash=commit_hash,
        output_path=output_path,
        cadet_config_jsons=cadet_config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods,
        ax_discs=ax_discs,
        par_methods=par_methods,
        par_discs=par_discs,
        rad_methods=rad_methods,
        rad_discs=rad_discs,
        **kwargs
    )


def run_convergence_analysis_from_database(
        database_path, output_path=None,
        cadet_path=None,
        cadet_config_jsons=[
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
        ],
        include_sens=[True, True],
        ref_files=[[None, None], [None, None]],
        unit_IDs=['001', '001'],
        which=['outlet', 'outlet'],
        ax_methods=[
        [0, 3],
        [0, 3]
        ],
        ax_discs=[
            [disc_list(8, 3), disc_list(1, 3)],
            [disc_list(8, 3), disc_list(4, 3)]
        ],
        par_methods=[
            [None, None],
            [0, 3]
        ],
        par_discs=[
            [None, None],
            [disc_list(1, 3), disc_list(1, 3)]
        ],
        rad_methods=None,
        rad_discs=None,
        idas_abstol=[[None, None], [None, None]],
        n_jobs=-1,
        rerun_sims=True,
        **kwargs
):
    """ Runs convergence analyses for the specified models and methods

    Parameters
    ----------

    idas_tol If set to None, the abstol from the configuration file will be used
    re_files: size model x methods
    unit_IDs: size model
    include_sens: size model # TODO: * methods
    database_path : String or None
        URL to datebase files.
    cadet_path : String
        Specifies the path to the cadet-cli executable, defaults to None.
        The default uses the Cadet executable installed for the current conda
        environment.
    cadet_config_jsons : List of Strings of size n_models
        Names of database model configuration files, including .json ending
    include_sens : List of Strings of size n_models
        Specifies the name of a reference h5 file that exists under
    ref_files : Lists of Strings of size n_models per method
        Specifies the name of a reference h5 file that exists under
        the output_path, defaults to Nones, which will use the last respective
        discretization as reference.
    ax_methods : list of Ints per model
        Specifies the axial discretization methods.
    ax_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all axial discretization methods.
    par_methods : list of Ints per model
        Specifies the particle discretization methods.
    par_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all particle discretization methods.
    rad_methods : list of Ints per model
        Specifies the radial discretization methods.
    rad_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all radial discretization methods.
    which : list of Strings per model
        Specifies the considered solution type, defaults to 'outlet'
    unitID : list of Strings per model
        Specifies the considered unit operation's ID, defaults to '001'
    idas_abstol: list of Floats per model and method
        Absolute time integration tolerance. Default None will make use of the
        tolerance from the configuration source file
    n_jobs: Int
        Specifies the number of workers. Defaults to one.
    rerun_sims: Bool
        Specifies whether or not the simulations should be rerun, as opposed to
        only comuting the tables from existing simulation files.
        Defaults to True


    Returns
    -------
    bool
        Success.
    """

    commit_hash = None

    if rerun_sims:
        # Create simulation objects
        sims = []  # To be filled with all Cadet objects

        for modelIdx in range(0, len(cadet_config_jsons)):         

            for methodIdx in range(0, len(ax_methods[modelIdx])):

                for discIdx in range(0, len(ax_discs[modelIdx][methodIdx])):

                    par_method = par_methods[modelIdx][methodIdx]
                    par_cells = None
                    if par_method is not None:
                        par_cells = par_discs[modelIdx][methodIdx][discIdx]

                    sims.append(
                        create_object_from_database(
                            database_url=database_path,
                            cadet_config_json_name=cadet_config_jsons[modelIdx],
                            unit_id=unit_IDs[modelIdx],
                            ax_method=ax_methods[modelIdx][methodIdx],
                            ax_cells=ax_discs[modelIdx][methodIdx][discIdx],
                            par_method=par_method, par_cells=par_cells,
                            output_path=output_path,
                            idas_abstol=idas_abstol[modelIdx][methodIdx],
                            include_sens=include_sens[modelIdx],
                            **kwargs
                        )
                    )

        # Run simulations in one global parallelization
        backend = Parallel(n_jobs=n_jobs, verbose=0)
        backend(delayed(run_simulation_in_verification)(sim, cadet_path)
                for sim in zip(sims))

        commit_hash = convergence.get_commit_hash(
            convergence.get_simulation(sims[0].filename)
            )

    return run_convergence_analysis_core(
        database_path=database_path,
        commit_hash=commit_hash,
        output_path=output_path,
        cadet_config_jsons=cadet_config_jsons,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods,
        ax_discs=ax_discs,
        par_methods=par_methods,
        par_discs=par_discs,
        rad_methods=rad_methods,
        rad_discs=rad_discs,
        **kwargs
    )


def run_convergence_analysis_core(
        database_path=None,
        commit_hash=None,
        output_path=None,
        cadet_config_jsons=[
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
        ],
        include_sens=[True, True],
        ref_files=[[None, None], [None, None]],
        unit_IDs=['001', '001'],
        which=['outlet', 'outlet'],
        ax_methods=[
        [0, 3],
        [0, 3]
        ],
        ax_discs=[
            [disc_list(8, 3), disc_list(1, 3)],
            [disc_list(8, 3), disc_list(4, 3)]
        ],
        par_methods=[
            [None, None],
            [0, 3]
        ],
        par_discs=[
            [None, None],
            [disc_list(1, 3), disc_list(1, 3)]
        ],
        rad_methods=None,
        rad_discs=None,
        **kwargs
):
    """ Runs convergence analyses for the specified models and methods

    Parameters
    ----------

    idas_tol If set to None, the abstol from the configuration file will be used
    re_files: size model x methods
    unit_IDs size: model
    include_sens size: model TODO: * methods

    database_path : String or None
        URL to database files.
    cadet_config_jsons : List of Strings of size n_models
        Names of database model configuration files, including .json ending
    include_sens : List of Strings of size n_models
        Specifies the name of a reference h5 file that exists under
    ref_files : Lists of Strings of size n_models per method
        Specifies the name of a reference h5 file that exists under
        the output_path, defaults to Nones, which will use the last respective
        discretization as reference.
    ax_methods : list of Ints per model
        Specifies the axial discretization methods.
    ax_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all axial discretization methods.
    par_methods : list of Ints per model
        Specifies the particle discretization methods.
    par_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all particle discretization methods.
    rad_methods : list of Ints per model
        Specifies the radial discretization methods.
    rad_discs : list of Ints per model and method in ascending order
        Specifies the number of cells/elements for every considered refinement
        level for all radial discretization methods.
    which : list of Strings per model
        Specifies the considered solution type, defaults to 'outlet'
    unitID : list of Strings per model
        Specifies the considered unit operation's ID, defaults to '001'
    n_jobs: Int
        Specifies the number of workers. Defaults to one.
    rerun_sims: Bool
        Specifies whether or not the simulations should be rerun, as opposed to
        only comuting the tables from existing simulation files.
        Defaults to True


    Returns
    -------
    bool
        Success.
    """

    # Get setting names and set reference file names if required
    setting_names = []

    for modelIdx in range(0, len(cadet_config_jsons)):

        setting_name = re.search(r'configuration_(.*?)(?:\.json|_FV|_DG)',
                                 cadet_config_jsons[modelIdx])
        if setting_name is None:
            setting_name = cadet_config_jsons[modelIdx]
        else:
            setting_name = setting_name.group(1)

        if not include_sens[modelIdx]:
            setting_name = re.sub('sens|sensitivity', '', setting_name)

        setting_names.append(setting_name)

        for methodIdx in range(0, len(ax_methods[modelIdx])):

            if ref_files[modelIdx][methodIdx] is None:

                if rad_methods is None:
                    
                    if par_methods[modelIdx][methodIdx] is None:
    
                        ref_file = convergence.generate_1D_name(
                            setting_name,
                            ax_methods[modelIdx][methodIdx],
                            ax_discs[modelIdx][methodIdx][-1]
                        )
                    else:
    
                        ref_file = convergence.generate_GRM_name(
                            setting_name,
                            ax_methods[modelIdx][methodIdx],
                            ax_discs[modelIdx][methodIdx][-1],
                            par_methods[modelIdx][methodIdx],
                            par_discs[modelIdx][methodIdx][-1]
                        )
                        
                        par_discs[modelIdx][methodIdx] = par_discs[modelIdx][methodIdx][:-1]
                else:
                    
                    if par_methods[modelIdx][methodIdx] is None:
    
                        ref_file = convergence.generate_2D_name(
                            setting_name,
                            ax_methods[modelIdx][methodIdx],
                            ax_discs[modelIdx][methodIdx][-1],
                            rad_methods[modelIdx][methodIdx],
                            rad_discs[modelIdx][methodIdx][-1]
                        )
                    else:
    
                        ref_file = convergence.generate_2DGRM_name(
                            setting_name,
                            ax_methods[modelIdx][methodIdx],
                            ax_discs[modelIdx][methodIdx][-1],
                            rad_methods[modelIdx][methodIdx],
                            rad_discs[modelIdx][methodIdx][-1],
                            par_methods[modelIdx][methodIdx],
                            par_discs[modelIdx][methodIdx][-1]
                        )

                        par_discs[modelIdx][methodIdx] = par_discs[modelIdx][methodIdx][:-1]
                
                    rad_discs[modelIdx][methodIdx] = rad_discs[modelIdx][methodIdx][:-1]

                ax_discs[modelIdx][methodIdx] = ax_discs[modelIdx][methodIdx][:-1]

                ref_files[modelIdx][methodIdx] = ref_file

    result_names = []

    # Calculate and write convergence tables
    for modelIdx in range(0, len(cadet_config_jsons)):

        for methodIdx in range(0, len(ax_methods[modelIdx])):

            par_method = par_methods[modelIdx][methodIdx]
            if par_method is None:
                par_disc = None
            else:
                par_method = par_methods[modelIdx][methodIdx]
                par_disc = par_discs[modelIdx][methodIdx]
            if rad_methods is None:
                rad_method = None
                rad_disc = None
            else:
                rad_method = rad_methods[modelIdx][methodIdx]
                rad_disc = rad_discs[modelIdx][methodIdx]

            print('\n', cadet_config_jsons[modelIdx])
            print('\n Method: ', ax_methods[modelIdx][methodIdx], '\n')
            # print('Reference: ', ref_files[modelIdx][methodIdx], '\n')

            result_name = generate_convergence_data(
                database_path, cadet_config_jsons[modelIdx],
                ax_method=ax_methods[modelIdx][methodIdx], ax_disc=ax_discs[modelIdx][methodIdx],
                rad_method=rad_method, rad_disc=rad_disc,
                par_method=par_method, par_disc=par_disc,
                output_path=output_path,
                write_result=1, ref_file=ref_files[modelIdx][methodIdx],
                which=which[modelIdx], unitID=unit_IDs[modelIdx],
                sens_included=include_sens[modelIdx],
                write_sens=True,
                commit_hash=commit_hash,
                **kwargs
            )

            if methodIdx == 0:
                result_names.append(result_name)

    return result_names


def run_convergence_analysis(**kwargs):

    if 'database_path' in kwargs:
        if kwargs['database_path'] is not None:
            return run_convergence_analysis_from_database(**kwargs)

    return run_convergence_analysis_from_configs(**kwargs)
