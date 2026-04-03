# -*- coding: utf-8 -*-
"""

This script defines EOC tests of numerical methods for sole
convection diffusion equations.

"""

# %% Include packages
import os
import copy
import numpy as np
from functools import partial
import re

import src.benchmark_models.setting_COL1D_axial_transport as setting_axial_transport
import src.benchmark_models.setting_COL1D_radial_transport as setting_radial_transport
import src.benchmark_models.setting_COL1D_frustum_transport as setting_frustum_transport
import src.benchmark_models.setting_MCT_transport_2channel as setting_MCT_transport_2channel
import src.benchmark_models.setting_COL2D_axialTransport_2rad as setting_COL2D_axTransport
import src.bench_func as bench_func
import src.bench_configs as bench_configs

import src.utility.convergence as convergence

from cadet import Cadet

# %% 

def transport_tests(n_jobs, small_test,
                    output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)

    # grid fucntions
    
    def grid_equidistant(x0, x1, n):
        return  np.linspace(x0, x1, n + 1)

    def grid_square(x0, x1, n):
        
        x = np.zeros(n+1)
        
        for i in range(0,n+1):
            
            x[i] = (i/(n+1))**2 * (x1-x0)
            
        return x

    def grid_sinusoidal_perturbation(x0, x1, n, alpha=0.3):
        """
        Generate grid faces on [x0, x1] with a sinusoidal perturbation
        applied to interior nodes.

        Parameters
        ----------
        x0, x1 : float
            Domain boundaries.
        n : int
            Number of cells (n+1 faces returned).
        alpha : float, optional
            Perturbation strength relative to uniform spacing.

        Returns
        -------
        faces : ndarray
            Array of length n+1 containing face locations.
        """
        faces = np.linspace(x0, x1, n + 1)
        h = (x1 - x0) / n

        i = np.arange(1, n)
        faces[i] += alpha * h * np.sin(2.0 * np.pi * i / n)

        return faces


    def grid_tanh_mapping(x0, x1, n, alpha=3.0):
        """
        Generate smoothly stretched grid faces using a tanh mapping.

        Clusters nodes symmetrically near the boundaries.

        Parameters
        ----------
        x0, x1 : float
            Domain boundaries.
        n : int
            Number of cells (n+1 faces returned).
        alpha : float, optional
            Stretching strength (alpha=0 -> uniform grid).

        Returns
        -------
        faces : ndarray
            Array of length n+1 containing face locations.
        """
        xi = np.linspace(0.0, 1.0, n + 1)

        stretched = (
            np.tanh(alpha * (xi - 0.5)) + np.tanh(alpha / 2.0)
        ) / (2.0 * np.tanh(alpha / 2.0))

        faces = x0 + (x1 - x0) * stretched
        return faces


    def grid_left_cos_cluster(x0, x1, N, p=1.0):
        """
        Generate grid faces clustered toward the left boundary using
        a cosine-based stretching with tunable clustering.

        Parameters
        ----------
        x0, x1 : float
            Domain boundaries.
        N : int
            Number of cells (N+1 faces returned).
        p : float, optional
            Clustering exponent (>1 for stronger left clustering).

        Returns
        -------
        faces : ndarray
            Array of length N+1 containing face locations.
        """
        xi = np.linspace(0.0, 1.0, N + 1)
        stretched = 1.0 - np.cos(0.5 * np.pi * xi**p)
        faces = x0 + (x1 - x0) * stretched
        return faces
    
    def grid_radial_equivolume(r0, r1, n):
        """
        Returns radial faces of a cylindrical shell such that each annular cell has the same volume.
        """
        # Compute equally spaced points in r^2
        r2_faces = np.linspace(r0**2, r1**2, n + 1)
        # Convert back to r
        return np.sqrt(r2_faces)
    
    def grid_frustum_equivolume(x0, x1, r0, r1, n):
        """
        Returns axial faces x for a frustum such that each cell has the same volume.
    
        Parameters
        ----------
        x0, x1 : float
            Axial domain boundaries.
        n : int
            Number of cells.
        r0, r1 : float
            Radius at x0 and x1, respectively.
    
        Returns
        -------
        faces : ndarray
            Array of length n+1 containing face locations.
        """
        if abs(r1 - r0) < 1e-15:
            return np.linspace(x0, x1, n + 1)
    
        r_faces = (r0**3 + np.linspace(0.0, 1.0, n + 1) * (r1**3 - r0**3))**(1.0 / 3.0)
        faces = x0 + (x1 - x0) * (r_faces - r0) / (r1 - r0)
        
        return faces

    #%% Define settings and benchmarks

    cadet_configs = []
    cadet_config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    ax_methods = []
    ax_discs = []
    disc_refinement_functions = []
    
    time_integrator = {
        'ABSTOL' : 1e-8, 'RELTOL' : 1e-8, 'ALGTOL' : 1e-8,
        'USE_MODIFIED_NEWTON' : False,
        'init_step_size' : 1e-6,
        'max_steps' : 500
        }
    
    spatial_discretization_WENO3 = {
        'NCOL': 8,
        'SPATIAL_METHOD' : 'FV', 'RECONSTRUCTION' : 'WENO',
        'weno' : {'WENO_ORDER' : 3, 'WENO_EPS' : 1e-10,
        'BOUNDARY_MODEL' : 0},
        'USE_ANALYTIC_JACOBIAN': True, 'USE_MODIFIED_NEWTON' : False
        }
    
    # grid_function = partial(grid_equidistant)
    grid_function = partial(grid_sinusoidal_perturbation, alpha=0.3)
    # grid_function = partial(grid_tanh_mapping, alpha=3.0)
    
    spatial_discretization_WENO3NonEq = copy.deepcopy(spatial_discretization_WENO3)
    spatial_discretization_WENO3NonEq['NonEq'] = True
    spatial_discretization_WENO3NonEq['grid_function'] = grid_function

    spatial_discretization_WENO2 = copy.deepcopy(spatial_discretization_WENO3)
    spatial_discretization_WENO2['weno']['WENO_ORDER'] = 2
    spatial_discretization_WENO2NonEq = copy.deepcopy(spatial_discretization_WENO3NonEq)
    spatial_discretization_WENO2NonEq['weno']['WENO_ORDER'] = 2

    spatial_discretization_KOREN = {
        'NCOL': 8,
        'SPATIAL_METHOD' : 'FV', 'RECONSTRUCTION' : 'KOREN',
        'koren' : {'KOREN_EPS' : 1e-10},
        'USE_ANALYTIC_JACOBIAN': True, 'USE_MODIFIED_NEWTON' : False
        }

    spatial_discretization_KORENNonEq = copy.deepcopy(spatial_discretization_KOREN)
    spatial_discretization_KORENNonEq['NonEq'] = True
    spatial_discretization_KORENNonEq['grid_function'] = grid_function

    def refine_discretization(config_data, disc_idx, setting_name,
                              spatial_discretization,
                              time_integrator=None,
                              unit_id = '001', 
                              only_return_name=False,
                              **kwargs):
        """ Takes a CADET-configuration as dictionary, adjusts it and returns the 
            Cadet-Object. Optionally saves the corresponding h5 config file.

        Parameters
        ----------
        config_data : Dictionary
            Dictionary with CADET configuration
        disc_idx: int
            current index of refinement starting at zero
        Returns
        -------
        Cadet-Object
            Cadet object.
        """

        # Adjust configuration to desired numerical refinement
        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator
            
        nCol = 8 * 2** (disc_idx)
        
        unit_type = config_data['input']['model']['unit_' + unit_id]['unit_type']
        match = re.search(r"(FRUSTUM|RADIAL)", unit_type)
        unit_geometry = match.group(0) if match else "AXIAL"
        
        if 'NonEq' in spatial_discretization:
            if spatial_discretization['NonEq']:
                
                if unit_geometry == "FRUSTUM":
                    
                    if spatial_discretization['grid_function'].__name__ == "grid_frustum_equivolume":
                    
                        x0 = 0.0
                        x1 = config_data['input']['model']['unit_' + unit_id]['col_length']
                        r0 = config_data['input']['model']['unit_' + unit_id]['col_radius_inner']
                        r1 = config_data['input']['model']['unit_' + unit_id]['col_radius_outer']
                     
                        config_data['input']['model']['unit_' + unit_id]['discretization']['GRID_FACES'] = spatial_discretization['grid_function'](x0, x1, r0, r1, nCol)
                        
                    else:
                        config_data['input']['model']['unit_' + unit_id]['discretization']['GRID_FACES'] = spatial_discretization['grid_function'](x0, x1, nCol)
        
                elif unit_geometry in ["AXIAL", "RADIAL"]:
                
                    x0 = 0.0 if unit_geometry == "AXIAL" else config_data['input']['model']['unit_' + unit_id]['col_radius_inner']
                    x1 = config_data['input']['model']['unit_' + unit_id]['col_length'] if unit_geometry == "AXIAL" else config_data['input']['model']['unit_' + unit_id]['col_radius_outer']
                    
                    config_data['input']['model']['unit_' + unit_id]['discretization']['GRID_FACES'] = spatial_discretization['grid_function'](x0, x1, nCol)
        
                else:
                    raise Exception("Uknown unit geometry: " + unit_geometry)
        
        config_data['input']['model']['unit_' + unit_id]['discretization'].update(
            {k: v for k, v in spatial_discretization.items() if k not in {'NonEq', 'grid_function'}}
            )
        config_data['input']['model']['unit_' + unit_id]['discretization']['NCOL'] = nCol
        
        config_name = convergence.generate_1D_name(setting_name, 0, nCol)

        model = Cadet()
        model.root.input = copy.deepcopy(config_data['input'])
        
        if output_path is not None:
            
            model.filename = str(output_path) + '/' + config_name
            
            if only_return_name:
                return model.filename
            else:
                model.save()
                return model
    
    #%% Axial flow transport

    nNumMethods = 6
    
    addition = {
            'cadet_config_jsons': [
                setting_axial_transport.get_model()
            ],
            'cadet_config_names': [
                'COL1D_transport_1comp_benchmark1'
            ],
            'include_sens': [False],
            'ref_files': [[None] * nNumMethods],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'ax_methods': [[0] * nNumMethods],
            'ax_discs': [[
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3)
            ]],
            'disc_refinement_functions' : [[
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_WENO2_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO2),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_WENO2nonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO2NonEq),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_WENO3_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_WENO3nonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3NonEq),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_KOREN_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KOREN),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="COL1D_transport_1comp_KORENnonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KORENNonEq),
                         time_integrator=time_integrator
                         )
                ]]
        }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions = disc_refinement_functions)

    #%% Radial flow transport

    nNumMethods = 6
    
    spatial_discretization_WENO2NonEq['grid_function'] = grid_radial_equivolume
    spatial_discretization_WENO3NonEq['grid_function'] = grid_radial_equivolume
    spatial_discretization_KORENNonEq['grid_function'] = grid_radial_equivolume
    
    addition = {
            'cadet_config_jsons': [
                setting_radial_transport.get_model()
            ],
            'cadet_config_names': [
                'COL1D_radTransport_1comp_benchmark1'
            ],
            'include_sens': [False],
            'ref_files': [[None] * nNumMethods],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'ax_methods': [[0] * nNumMethods],
            'ax_discs': [[
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3),
                bench_func.disc_list(8, 10 if not small_test else 3)
            ]],
            'disc_refinement_functions' : [[
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_WENO2_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO2),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_WENO2nonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO2NonEq),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_WENO3_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_WENO3nonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3NonEq),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_KOREN_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KOREN),
                         time_integrator=time_integrator
                         ),
                partial(refine_discretization,
                         setting_name="radCOL1D_transport_1comp_KORENnonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KORENNonEq),
                         time_integrator=time_integrator
                         )
                ]]
        }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions = disc_refinement_functions)
    
    # %% Frustum flow transport

    nNumMethods = 6

    # reset non-equidistant grid choice
    spatial_discretization_WENO2NonEq['grid_function'] = grid_frustum_equivolume
    spatial_discretization_WENO3NonEq['grid_function'] = grid_frustum_equivolume
    spatial_discretization_KORENNonEq['grid_function'] = grid_frustum_equivolume

    addition = {
        'cadet_config_jsons': [
            setting_frustum_transport.get_model()
        ],
        'cadet_config_names': [
            'COL1D_frustumTransport_1comp_benchmark1'
        ],
        'include_sens': [False],
        'ref_files': [[None] * nNumMethods],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'ax_methods': [[0] * nNumMethods],
        'ax_discs': [[
            bench_func.disc_list(8, 10 if not small_test else 3),
            bench_func.disc_list(8, 10 if not small_test else 3),
            bench_func.disc_list(8, 10 if not small_test else 3),
            bench_func.disc_list(8, 10 if not small_test else 3),
            bench_func.disc_list(8, 10 if not small_test else 3),
            bench_func.disc_list(8, 10 if not small_test else 3)
        ]],
        'disc_refinement_functions': [[
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_WENO2_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_WENO2),
                    time_integrator=time_integrator
                    ),
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_WENO2nonEq_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_WENO2NonEq),
                    time_integrator=time_integrator
                    ),
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_WENO3_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_WENO3),
                    time_integrator=time_integrator
                    ),
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_WENO3nonEq_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_WENO3NonEq),
                    time_integrator=time_integrator
                    ),
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_KOREN_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_KOREN),
                    time_integrator=time_integrator
                    ),
            partial(refine_discretization,
                    setting_name="frustumCOL1D_transport_1comp_KORENnonEq_benchmark1",
                    spatial_discretization=copy.deepcopy(spatial_discretization_KORENNonEq),
                    time_integrator=time_integrator
                    )
        ]]
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions=disc_refinement_functions)
    

    #%% 2DGRM (General Rate Model 2D) - axial flow transport refinement

    # reset, 2D models must additionally provide radial discretization and must run their own convergence analysis
    cadet_configs = []
    cadet_config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    disc_refinement_functions = []

    def refine_discretization_col2d(config_data, disc_idx, setting_name,
                              spatial_discretization,
                              time_integrator=None,
                              unit_id = '001',
                              only_return_name=False,
                              **kwargs):
        """Refinement function for 2DGRM: uses AXIAL_GRID_FACES instead of GRID_FACES."""

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        nCol = 8 * 2** (disc_idx)

        if 'NonEq' in spatial_discretization:
            if spatial_discretization['NonEq']:
                x0 = 0.0
                x1 = config_data['input']['model']['unit_' + unit_id]['col_length']
                config_data['input']['model']['unit_' + unit_id]['discretization']['AXIAL_GRID_FACES'] = spatial_discretization['grid_function'](x0, x1, nCol)

        config_data['input']['model']['unit_' + unit_id]['discretization'].update(
            {k: v for k, v in spatial_discretization.items() if k not in {'NonEq', 'grid_function'}}
            )
        config_data['input']['model']['unit_' + unit_id]['discretization']['NCOL'] = nCol
        config_data['input']['model']['unit_' + unit_id]['discretization']['NRAD'] = 2 # remains constant

        config_name = convergence.generate_1D_name(setting_name, 0, nCol)

        model = Cadet()
        model.root.input = copy.deepcopy(config_data['input'])

        if output_path is not None:

            model.filename = str(output_path) + '/' + config_name

            if only_return_name:
                return model.filename
            else:
                model.save()
                return model

    time_integrator_2dgrm = {
        'ABSTOL' : 1e-10, 'RELTOL' : 1e-8, 'ALGTOL' : 1e-10,
        'USE_MODIFIED_NEWTON' : False,
        'init_step_size' : 1e-10,
        'max_steps' : 1000000
        }

    nNumMethods = 4
    numRefinements = 7 if not small_test else 3

    addition = {
            'cadet_config_jsons': [
                setting_COL2D_axTransport.get_model()
            ],
            'cadet_config_names': [
                'COL2D_transport_1comp_benchmark1'
            ],
            'include_sens': [False],
            'ref_files': [[None] * nNumMethods],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'ax_methods': [[0] * nNumMethods],
            'ax_discs': [[
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements)
            ]],
            'rad_methods': [[0] * nNumMethods],
            'rad_discs': [[
                [2] * numRefinements,
                [2] * numRefinements,
                [2] * numRefinements,
                [2] * numRefinements,
            ]],
            'disc_refinement_functions' : [[
                partial(refine_discretization_col2d,
                         setting_name="COL2D_transport_1comp_WENO3_axbenchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3),
                         time_integrator=time_integrator_2dgrm
                         ),
                partial(refine_discretization_col2d,
                         setting_name="COL2D_transport_1comp_WENO3nonEq_axbenchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3NonEq),
                         time_integrator=time_integrator_2dgrm
                         ),
                partial(refine_discretization_col2d,
                         setting_name="COL2D_transport_1comp_KOREN_axbenchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KOREN),
                         time_integrator=time_integrator_2dgrm
                         ),
                partial(refine_discretization_col2d,
                         setting_name="COL2D_transport_1comp_KORENnonEq_axbenchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KORENNonEq),
                         time_integrator=time_integrator_2dgrm
                         )
                ]]
        }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        rad_methods=rad_methods, rad_discs=rad_discs,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions = disc_refinement_functions)
    
    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=cadet_config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods,
        ax_discs=ax_discs,
        rad_methods=rad_methods,
        rad_discs=rad_discs,
        n_jobs=n_jobs,
        rerun_sims=True,
        disc_refinement_functions = disc_refinement_functions
    )
    
    
    #%% MCT (Multi-Channel Transport) - axial flow transport

    # reset, MCT must run its own convergence analysis to handle DOF calculation via transport model specification
    cadet_configs = []
    cadet_config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    ax_methods = []
    ax_discs = []
    disc_refinement_functions = []

    # Reset grid functions to axial (sinusoidal perturbation)
    spatial_discretization_WENO3NonEq['grid_function'] = partial(grid_sinusoidal_perturbation, alpha=0.3)
    spatial_discretization_KORENNonEq['grid_function'] = partial(grid_sinusoidal_perturbation, alpha=0.3)

    nNumMethods = 4
    numRefinements = 8 if not small_test else 3

    addition = {
            'cadet_config_jsons': [
                setting_MCT_transport_2channel.get_model()
            ],
            'cadet_config_names': [
                'MCT_transport_1comp_benchmark1'
            ],
            'include_sens': [False],
            'ref_files': [[None] * nNumMethods],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'ax_methods': [[0] * nNumMethods],
            'ax_discs': [[
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements),
                bench_func.disc_list(8, numRefinements)
            ]],
            'disc_refinement_functions' : [[
                partial(refine_discretization,
                         setting_name="MCT_transport_1comp_WENO3_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3),
                         time_integrator=time_integrator,

                         ),
                partial(refine_discretization,
                         setting_name="MCT_transport_1comp_WENO3nonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_WENO3NonEq),
                         time_integrator=time_integrator,

                         ),
                partial(refine_discretization,
                         setting_name="MCT_transport_1comp_KOREN_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KOREN),
                         time_integrator=time_integrator,

                         ),
                partial(refine_discretization,
                         setting_name="MCT_transport_1comp_KORENnonEq_benchmark1",
                         spatial_discretization=copy.deepcopy(spatial_discretization_KORENNonEq),
                         time_integrator=time_integrator,

                         )
                ]]
        }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions = disc_refinement_functions)

    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=cadet_config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods,
        ax_discs=ax_discs,
        n_jobs=n_jobs,
        rerun_sims=True,
        disc_refinement_functions = disc_refinement_functions,
        transport_model="MCT"
    )