import numpy as np
from cadet import Cadet
import multiple_reaction_configs as MRC
import pytest


def multiple_reactions_cstr_test_particle(output_path, cadet_path, plot=False):

    # new interface particle 1 reaction
    case_particle_one = MRC.get_dict_one_reaction_particle()
    
    model_particle_one = Cadet(cadet_path)
    model_particle_one = MRC.lrmp_setup(model_particle_one)
    model_particle_one = MRC.mal_setup_paricle(model_particle_one, case_particle_one)
    
    model_particle_one.filename = f"lrmp_{case_particle_one['num_reactions']}_reaction_particle.h5"
    model_particle_one.save()
    data_particle_one = model_particle_one.run_simulation()
    if data_particle_one.return_code != 0:
        print(data_particle_one.return_code)
    print(data_particle_one.error_message)

    # new interface particle 2 reactions
    case_particle_two = MRC.get_dict_two_reactions_particle()
    model_particle_two = Cadet(cadet_path)
    model_particle_two = MRC.lrmp_setup(model_particle_two)
    model_particle_two = MRC.mal_setup_paricle(model_particle_two, case_particle_two)
    model_particle_two.filename = f"lrmp_{case_particle_two['num_reactions']}_reaction_particle.h5"
    model_particle_two.save()
    data_particle_two = model_particle_two.run_simulation()
    if data_particle_two.return_code != 0:
            print(data_particle_two.return_code)
    print(data_particle_two.error_message)

    # new interface particle 3 reactions 
    case_particle_three = MRC.get_dict_three_reactions_particle()
    model_particle_three = Cadet(cadet_path)
    model_particle_three = MRC.lrmp_setup(model_particle_three)
    model_particle_three = MRC.mal_setup_paricle(model_particle_three, case_particle_three)
    model_particle_three.filename = f"lrmp_{case_particle_three['num_reactions']}_reaction_particle.h5"
    model_particle_three.save()
    data_particle_three = model_particle_three.run_simulation()
    if data_particle_three.return_code != 0:
        print(data_particle_three.return_code)
    print(data_particle_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_particle_one.root.output.solution.solution_times, model_particle_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 1 Reaction')
        plt.plot(model_particle_two.root.output.solution.solution_times, model_particle_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 2 Reactions')
        plt.plot(model_particle_three.root.output.solution.solution_times, model_particle_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('LRMP Multiple Particle Reactions Comparison')
        #plt.savefig(output_path + '/lrmp_multiple_reactions_particle_comparison.png')
        plt.show()
    
    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 particle reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 particle reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'
    
    max_error_2_3 = np.max(np.abs(model_particle_two.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 particle reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_lrmp_test_bulk(output_path, cadet_path, plot=False):

    # old interface bulk
    case_bulk_mal_old = MRC.get_dict_one_reaction_mal()
    
    model_bulk_mal_old = Cadet(cadet_path)
    model_bulk_mal_old = MRC.lrmp_setup(model_bulk_mal_old)
    model_bulk_mal_old = MRC.mal_setup_bulk_old(model_bulk_mal_old, case_bulk_mal_old)
    
    model_bulk_mal_old.filename = f"lrmp_{case_bulk_mal_old['num_reactions']}_reaction_bulk_old.h5"
    model_bulk_mal_old.save()
    data_bulk_mal_old = model_bulk_mal_old.run_simulation()
    if data_bulk_mal_old.return_code != 0:
        print(data_bulk_mal_old.return_code)
    print(data_bulk_mal_old.error_message)

    # new interface bulk 1 reaction
    case_bulk_mal_one = MRC.get_dict_one_reaction_mal()
    
    model_bulk_mal_one = Cadet(cadet_path)
    model_bulk_mal_one = MRC.lrmp_setup(model_bulk_mal_one)
    model_bulk_mal_one = MRC.mal_setup_bulk(model_bulk_mal_one, case_bulk_mal_one)
    model_bulk_mal_one.filename = f"lrmp_{case_bulk_mal_one['num_reactions']}_reaction_bulk_new.h5"
    model_bulk_mal_one.save()
    data_bulk_mal_one = model_bulk_mal_one.run_simulation()
    if data_bulk_mal_one.return_code != 0:
        print(data_bulk_mal_one.return_code)
    print(data_bulk_mal_one.error_message)

    # new interface bulk 2 reactions
    case_bulk_mal_two = MRC.get_dict_two_reaction_mal()
    model_bulk_mal_two = Cadet(cadet_path)
    model_bulk_mal_two = MRC.lrmp_setup(model_bulk_mal_two)
    model_bulk_mal_two = MRC.mal_setup_bulk(model_bulk_mal_two, case_bulk_mal_two)
    model_bulk_mal_two.filename = f"lrmp_{case_bulk_mal_two['num_reactions']}_reaction_bulk_new.h5"
    model_bulk_mal_two.save()
    data_bulk_mal_two = model_bulk_mal_two.run_simulation()
    if data_bulk_mal_two.return_code != 0:
        print(data_bulk_mal_two.return_code)
    print(data_bulk_mal_two.error_message)

    # new interface bulk 3 reactions 
    case_bulk_mal_three = MRC.get_dict_three_reaction_mal()
    model_bulk_mal_three = Cadet(cadet_path)
    model_bulk_mal_three = MRC.lrmp_setup(model_bulk_mal_three)
    model_bulk_mal_three = MRC.mal_setup_bulk(model_bulk_mal_three, case_bulk_mal_three)
    model_bulk_mal_three.filename = f"lrmp_{case_bulk_mal_three['num_reactions']}_reaction_bulk_new.h5"
    model_bulk_mal_three.save()
    data_bulk_mal_three = model_bulk_mal_three.run_simulation()
    if data_bulk_mal_three.return_code != 0:
        print(data_bulk_mal_three.return_code)
    print(data_bulk_mal_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_bulk_mal_old.root.output.solution.solution_times, model_bulk_mal_old.root.output.solution.unit_000.solution_bulk[:, 0], label='Old Interface - Reaction 1')
        plt.plot(model_bulk_mal_one.root.output.solution.solution_times, model_bulk_mal_one.root.output.solution.unit_000.solution_bulk[:, 0], label='New Interface - Reaction 1')
        plt.plot(model_bulk_mal_two.root.output.solution.solution_times, model_bulk_mal_two.root.output.solution.unit_000.solution_bulk[:, 0], label='New Interface - Reaction 2')
        plt.plot(model_bulk_mal_three.root.output.solution.solution_times, model_bulk_mal_three.root.output.solution.unit_000.solution_bulk[:, 0], label='New Interface - Reaction 3')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('LRMP Multiple Bulk Reactions Comparison')
        #plt.savefig(output_path + '/lrmp_multiple_reactions_bulk_comparison.png')
        plt.show()
    
    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1 = np.max(np.abs(model_bulk_mal_old.root.output.solution.unit_000.solution_bulk - model_bulk_mal_one.root.output.solution.unit_000.solution_bulk))
    assert max_error_1 < tolerance, f'  WARNING: Max error between old and new interface for 1 reaction ({max_error_1}) exceeds tolerance ({tolerance})'

    max_error_2 = np.max(np.abs(model_bulk_mal_old.root.output.solution.unit_000.solution_bulk - model_bulk_mal_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_2 < tolerance, f'  WARNING: Max error between old and new interface for 2 reactions ({max_error_2}) exceeds tolerance ({tolerance})'
    
    max_error_3 = np.max(np.abs(model_bulk_mal_old.root.output.solution.unit_000.solution_bulk - model_bulk_mal_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_3 < tolerance, f'  WARNING: Max error between old and new interface for 3 reactions ({max_error_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_lrmp_test_particle(output_path, cadet_path, plot=False):

    # new interface particle 1 reaction
    case_particle_one = MRC.get_dict_one_reaction_mal()
    
    model_particle_one = Cadet(cadet_path)
    model_particle_one = MRC.lrmp_setup(model_particle_one)
    model_particle_one = MRC.mal_setup_particle(model_particle_one, case_particle_one)
    
    model_particle_one.filename = f"lrmp_{case_particle_one['num_reactions']}_reaction_particle.h5"
    model_particle_one.save()
    data_particle_one = model_particle_one.run_simulation()
    if data_particle_one.return_code != 0:
        print(data_particle_one.return_code)
    print(data_particle_one.error_message)

    # new interface particle 2 reactions
    case_particle_two = MRC.get_dict_two_reaction_mal()
    model_particle_two = Cadet(cadet_path)
    model_particle_two = MRC.lrmp_setup(model_particle_two)
    model_particle_two = MRC.mal_setup_particle(model_particle_two, case_particle_two)
    model_particle_two.filename = f"lrmp_{case_particle_two['num_reactions']}_reaction_particle.h5"
    model_particle_two.save()
    data_particle_two = model_particle_two.run_simulation()
    if data_particle_two.return_code != 0:
        print(data_particle_two.return_code)
    print(data_particle_two.error_message)

    # new interface particle 3 reactions 
    case_particle_three = MRC.get_dict_three_reaction_mal()
    model_particle_three = Cadet(cadet_path)
    model_particle_three = MRC.lrmp_setup(model_particle_three)
    model_particle_three = MRC.mal_setup_particle(model_particle_three, case_particle_three)
    model_particle_three.filename = f"lrmp_{case_particle_three['num_reactions']}_reaction_particle.h5"
    model_particle_three.save()
    data_particle_three = model_particle_three.run_simulation()
    if data_particle_three.return_code != 0:
        print(data_particle_three.return_code)
    print(data_particle_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_particle_one.root.output.solution.solution_times, model_particle_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 1 Reaction')
        plt.plot(model_particle_two.root.output.solution.solution_times, model_particle_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 2 Reactions')
        plt.plot(model_particle_three.root.output.solution.solution_times, model_particle_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('LRMP Multiple Particle Reactions Comparison')
        #plt.savefig(output_path + '/lrmp_multiple_reactions_particle_comparison.png')
        plt.show()
    
    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 particle reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 particle reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'
    
    max_error_2_3 = np.max(np.abs(model_particle_two.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 particle reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_lrmp_test_cross_phase(output_path, cadet_path, plot=False):

     #old interface particle
    case_particle_mal_old = MRC.get_dict_one_reaction_cross_phase()
    model_particle_mal_old = Cadet(cadet_path)
    model_particle_mal_old = MRC.lrmp_setup(model_particle_mal_old)
    model_particle_mal_old = MRC.mal_setup_particle_old(model_particle_mal_old, case_particle_mal_old)
    model_particle_mal_old.filename = f"lrmp_{case_particle_mal_old['num_reactions']}_reaction_particle_old.h5"
    model_particle_mal_old.save()
    data_particle_mal_old = model_particle_mal_old.run_simulation()
    if data_particle_mal_old.return_code != 0:
        print(data_particle_mal_old.return_code)
    print(data_particle_mal_old.error_message)
    
    # new interface cross_phase 1 reaction
    case_cross_phase_one = MRC.get_dict_one_reaction_cross_phase()
    
    model_cross_phase_one = Cadet(cadet_path)
    model_cross_phase_one = MRC.lrmp_setup(model_cross_phase_one)
    model_cross_phase_one = MRC.mal_setup_cross_phase(model_cross_phase_one, case_cross_phase_one)
    
    model_cross_phase_one.filename = f"lrmp_{case_cross_phase_one['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_one.save()
    data_cross_phase_one = model_cross_phase_one.run_simulation()
    if data_cross_phase_one.return_code != 0:
        print(data_cross_phase_one.return_code)
    print(data_cross_phase_one.error_message)

    # new interface cross_phase 2 reactions
    case_cross_phase_two = MRC.get_dict_two_reactions_cross_phase()
    model_cross_phase_two = Cadet(cadet_path)
    model_cross_phase_two = MRC.lrmp_setup(model_cross_phase_two)
    model_cross_phase_two = MRC.mal_setup_cross_phase(model_cross_phase_two, case_cross_phase_two)
    model_cross_phase_two.filename = f"lrmp_{case_cross_phase_two['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_two.save()
    data_cross_phase_two = model_cross_phase_two.run_simulation()
    print(data_cross_phase_two.return_code)
    print(data_cross_phase_two.error_message)

    # new interface cross_phase 3 reactions 
    case_cross_phase_three = MRC.get_dict_three_reactions_cross_phase()
    model_cross_phase_three = Cadet(cadet_path)
    model_cross_phase_three = MRC.lrmp_setup(model_cross_phase_three)
    model_cross_phase_three = MRC.mal_setup_cross_phase(model_cross_phase_three, case_cross_phase_three)
    model_cross_phase_three.filename = f"lrmp_{case_cross_phase_three['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_three.save()
    data_cross_phase_three = model_cross_phase_three.run_simulation()
    print(data_cross_phase_three.return_code)
    print(data_cross_phase_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_cross_phase_one.root.output.solution.solution_times, model_cross_phase_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 1 Reaction')
        plt.plot(model_cross_phase_two.root.output.solution.solution_times, model_cross_phase_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 2 Reactions')
        plt.plot(model_cross_phase_three.root.output.solution.solution_times, model_cross_phase_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('LRMP Multiple Cross Phase Reactions Comparison')
        #plt.savefig(output_path + '/lrmp_multiple_reactions_cross_phase_comparison.png')
        plt.show()
    
    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 cross phase reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 cross phase reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'
    
    max_error_2_3 = np.max(np.abs(model_cross_phase_two.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 cross phase reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_grm_test_bulk(output_path, cadet_path, plot=False):

    #old interface bulk
    case_bulk_old = MRC.get_dict_one_reaction_mal()
    model_bulk_old = Cadet(cadet_path)
    model_bulk_old = MRC.grm_setup(model_bulk_old)
    model_bulk_old = MRC.mal_setup_bulk_old(model_bulk_old, case_bulk_old)
    model_bulk_old.filename = f"grm_{case_bulk_old['num_reactions']}_reaction_bulk_old.h5"
    model_bulk_old.save()
    data_bulk_old = model_bulk_old.run_simulation()
    print(data_bulk_old.return_code)
    print(data_bulk_old.error_message)

    
    # new interface bulk 1 reaction
    case_bulk_one = MRC.get_dict_one_reaction_mal()

    model_bulk_one = Cadet(cadet_path)
    model_bulk_one = MRC.grm_setup(model_bulk_one)
    model_bulk_one = MRC.mal_setup_bulk(model_bulk_one, case_bulk_one)
    model_bulk_one.filename = f"grm_{case_bulk_one['num_reactions']}_reaction_bulk.h5"
    model_bulk_one.save()
    data_bulk_one = model_bulk_one.run_simulation()
    print(data_bulk_one.return_code)
    print(data_bulk_one.error_message)

    # new interface bulk 2 reactions
    case_bulk_two = MRC.get_dict_two_reaction_mal()
    model_bulk_two = Cadet(cadet_path)
    model_bulk_two = MRC.grm_setup(model_bulk_two)
    model_bulk_two = MRC.mal_setup_bulk(model_bulk_two, case_bulk_two)
    model_bulk_two.filename = f"grm_{case_bulk_two['num_reactions']}_reaction_bulk.h5"
    model_bulk_two.save()
    data_bulk_two = model_bulk_two.run_simulation()
    print(data_bulk_two.return_code)
    print(data_bulk_two.error_message)

    # new interface bulk 3 reactions
    case_bulk_three = MRC.get_dict_three_reaction_mal()
    model_bulk_three = Cadet(cadet_path)
    model_bulk_three = MRC.grm_setup(model_bulk_three)
    model_bulk_three = MRC.mal_setup_bulk(model_bulk_three, case_bulk_three)
    model_bulk_three.filename = f"grm_{case_bulk_three['num_reactions']}_reaction_bulk.h5"
    model_bulk_three.save()
    data_bulk_three = model_bulk_three.run_simulation()
    print(data_bulk_three.return_code)
    print(data_bulk_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_bulk_one.root.output.solution.solution_times, model_bulk_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Bulk - 1 Reaction')
        plt.plot(model_bulk_two.root.output.solution.solution_times, model_bulk_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Bulk - 2 Reactions')
        plt.plot(model_bulk_three.root.output.solution.solution_times, model_bulk_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Bulk - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('GRM Multiple Bulk Reactions Comparison')
        #plt.savefig(output_path + '/grm_multiple_reactions_bulk_comparison.png')
        plt.show()

    # check max absolute error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_bulk_one.root.output.solution.unit_000.solution_bulk - model_bulk_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 bulk reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_bulk_one.root.output.solution.unit_000.solution_bulk - model_bulk_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 bulk reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'

    max_error_2_3 = np.max(np.abs(model_bulk_two.root.output.solution.unit_000.solution_bulk - model_bulk_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 bulk reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

    # compare with old interface
    max_error_old_1 = np.max(np.abs(model_bulk_old.root.output.solution.unit_000.solution_bulk - model_bulk_one.root.output.solution.unit_000.solution_bulk))
    assert max_error_old_1 < tolerance, f'  WARNING: Max error between old and new interface for 1 reaction ({max_error_old_1}) exceeds tolerance ({tolerance})'

def multiple_reactions_grm_test_particle(output_path, cadet_path, plot=False):

    # new interface particle 1 reaction
    case_particle_one = MRC.get_dict_one_reaction_mal()

    model_particle_one = Cadet(cadet_path)
    model_particle_one = MRC.grm_setup(model_particle_one)
    model_particle_one = MRC.mal_setup_particle(model_particle_one, case_particle_one)

    model_particle_one.filename = f"grm_{case_particle_one['num_reactions']}_reaction_particle.h5"
    model_particle_one.save()
    data_particle_one = model_particle_one.run_simulation()
    print(data_particle_one.return_code)
    print(data_particle_one.error_message)

    # new interface particle 2 reactions
    case_particle_two = MRC.get_dict_two_reaction_mal()
    model_particle_two = Cadet(cadet_path)
    model_particle_two = MRC.grm_setup(model_particle_two)
    model_particle_two = MRC.mal_setup_particle(model_particle_two, case_particle_two)
    model_particle_two.filename = f"grm_{case_particle_two['num_reactions']}_reaction_particle.h5"
    model_particle_two.save()
    data_particle_two = model_particle_two.run_simulation()
    print(data_particle_two.return_code)
    print(data_particle_two.error_message)

    # new interface particle 3 reactions
    case_particle_three = MRC.get_dict_three_reaction_mal()
    model_particle_three = Cadet(cadet_path)
    model_particle_three = MRC.grm_setup(model_particle_three)
    model_particle_three = MRC.mal_setup_particle(model_particle_three, case_particle_three)
    model_particle_three.filename = f"grm_{case_particle_three['num_reactions']}_reaction_particle.h5"
    model_particle_three.save()
    data_particle_three = model_particle_three.run_simulation()
    print(data_particle_three.return_code)
    print(data_particle_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_particle_one.root.output.solution.solution_times, model_particle_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 1 Reaction')
        plt.plot(model_particle_two.root.output.solution.solution_times, model_particle_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 2 Reactions')
        plt.plot(model_particle_three.root.output.solution.solution_times, model_particle_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Particle - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('GRM Multiple Particle Reactions Comparison')
        #plt.savefig(output_path + '/grm_multiple_reactions_particle_comparison.png')
        plt.show()

    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 particle reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_particle_one.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 particle reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'

    max_error_2_3 = np.max(np.abs(model_particle_two.root.output.solution.unit_000.solution_bulk - model_particle_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 particle reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_grm_test_cross_phase(output_path, cadet_path, plot=False):

    # new interface cross_phase 1 reaction
    case_cross_phase_one = MRC.get_dict_one_reaction_cross_phase()

    model_cross_phase_one = Cadet(cadet_path)
    model_cross_phase_one = MRC.grm_setup(model_cross_phase_one)
    model_cross_phase_one = MRC.mal_setup_cross_phase(model_cross_phase_one, case_cross_phase_one)

    model_cross_phase_one.filename = f"grm_{case_cross_phase_one['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_one.save()
    data_cross_phase_one = model_cross_phase_one.run_simulation()
    print(data_cross_phase_one.return_code)
    print(data_cross_phase_one.error_message)

    # new interface cross_phase 2 reactions
    case_cross_phase_two = MRC.get_dict_two_reactions_cross_phase()
    model_cross_phase_two = Cadet(cadet_path)
    model_cross_phase_two = MRC.grm_setup(model_cross_phase_two)
    model_cross_phase_two = MRC.mal_setup_cross_phase(model_cross_phase_two, case_cross_phase_two)
    model_cross_phase_two.filename = f"grm_{case_cross_phase_two['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_two.save()
    data_cross_phase_two = model_cross_phase_two.run_simulation()
    print(data_cross_phase_two.return_code)
    print(data_cross_phase_two.error_message)

    # new interface cross_phase 3 reactions
    case_cross_phase_three = MRC.get_dict_three_reactions_cross_phase()
    model_cross_phase_three = Cadet(cadet_path)
    model_cross_phase_three = MRC.grm_setup(model_cross_phase_three)
    model_cross_phase_three = MRC.mal_setup_cross_phase(model_cross_phase_three, case_cross_phase_three)
    model_cross_phase_three.filename = f"grm_{case_cross_phase_three['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_three.save()
    data_cross_phase_three = model_cross_phase_three.run_simulation()
    print(data_cross_phase_three.return_code)
    print(data_cross_phase_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_cross_phase_one.root.output.solution.solution_times, model_cross_phase_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 1 Reaction')
        plt.plot(model_cross_phase_two.root.output.solution.solution_times, model_cross_phase_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 2 Reactions')
        plt.plot(model_cross_phase_three.root.output.solution.solution_times, model_cross_phase_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('GRM Multiple Cross Phase Reactions Comparison')
        #plt.savefig(output_path + '/grm_multiple_reactions_cross_phase_comparison.png')
        plt.show()

    # print max absolut error
    max_error_1_2 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_two.root.output.solution.unit_000.solution_bulk))
    print(f'Max absolute error between 1 and 2 cross phase reactions: {max_error_1_2}')

    max_error_1_3 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    print(f'Max absolute error between 1 and 3 cross phase reactions: {max_error_1_3}')

    max_error_2_3 = np.max(np.abs(model_cross_phase_two.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    print(f'Max absolute error between 2 and 3 cross phase reactions: {max_error_2_3}')

def multiple_reactions_lrm_cross_phase_test(output_path, cadet_path, plot=False):

    
    # new interface cross_phase 1 reaction
    case_cross_phase_one = MRC.get_dict_one_reaction_cross_phase()

    model_cross_phase_one = Cadet(cadet_path)
    model_cross_phase_one = MRC.lrm_setup(model_cross_phase_one)
    model_cross_phase_one = MRC.mal_setup_cross_phase(model_cross_phase_one, case_cross_phase_one)

    model_cross_phase_one.filename = f"lrm_{case_cross_phase_one['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_one.save()
    data_cross_phase_one = model_cross_phase_one.run_simulation()
    print(data_cross_phase_one.return_code)
    print(data_cross_phase_one.error_message)

    # new interface cross_phase 2 reactions
    case_cross_phase_two = MRC.get_dict_two_reactions_cross_phase()
    model_cross_phase_two = Cadet(cadet_path)
    model_cross_phase_two = MRC.lrm_setup(model_cross_phase_two)
    model_cross_phase_two = MRC.mal_setup_cross_phase(model_cross_phase_two, case_cross_phase_two)
    model_cross_phase_two.filename = f"lrm_{case_cross_phase_two['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_two.save()
    data_cross_phase_two = model_cross_phase_two.run_simulation()
    print(data_cross_phase_two.return_code)
    print(data_cross_phase_two.error_message)

    # new interface cross_phase 3 reactions
    case_cross_phase_three = MRC.get_dict_three_reactions_cross_phase()
    model_cross_phase_three = Cadet(cadet_path)
    model_cross_phase_three = MRC.lrm_setup(model_cross_phase_three)
    model_cross_phase_three = MRC.mal_setup_cross_phase(model_cross_phase_three, case_cross_phase_three)
    model_cross_phase_three.filename = f"lrm_{case_cross_phase_three['num_reactions']}_reaction_cross_phase.h5"
    model_cross_phase_three.save()
    data_cross_phase_three = model_cross_phase_three.run_simulation()
    print(data_cross_phase_three.return_code)
    print(data_cross_phase_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_cross_phase_one.root.output.solution.solution_times, model_cross_phase_one.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 1 Reaction')
        plt.plot(model_cross_phase_two.root.output.solution.solution_times, model_cross_phase_two.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 2 Reactions')
        plt.plot(model_cross_phase_three.root.output.solution.solution_times, model_cross_phase_three.root.output.solution.unit_000.solution_bulk[:, 0], label='Cross Phase - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('LRM Multiple Cross Phase Reactions Comparison')
        #plt.savefig(output_path + '/lrm_multiple_reactions_cross_phase_comparison.png')
        plt.show()

    # check max absolut error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 cross phase reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_cross_phase_one.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 cross phase reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'

    max_error_2_3 = np.max(np.abs(model_cross_phase_two.root.output.solution.unit_000.solution_bulk - model_cross_phase_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 cross phase reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

def multiple_reactions_mct_test_bulk(output_path, cadet_path, plot=False):
    
    # old interface bulk 
    case_bulk_old = MRC.get_dict_one_reaction_mal()
    model_bulk_old = Cadet(cadet_path)
    model_bulk_old = MRC.mct_setup(model_bulk_old)
    model_bulk_old = MRC.mal_setup_bulk_old(model_bulk_old, case_bulk_old)
    model_bulk_old.filename = f"mct_{case_bulk_old['num_reactions']}_reaction_bulk_old.h5"
    model_bulk_old.save()
    data_bulk_old = model_bulk_old.run_simulation()
    print(data_bulk_old.return_code)
    print(data_bulk_old.error_message)
    
    case_bulk_one = MRC.get_dict_one_reaction_mal()

    model_bulk_one = Cadet(cadet_path)
    model_bulk_one = MRC.mct_setup(model_bulk_one)
    model_bulk_one = MRC.mal_setup_bulk(model_bulk_one, case_bulk_one)
    model_bulk_one.filename = f"mct_{case_bulk_one['num_reactions']}_reaction_bulk.h5"
    model_bulk_one.save()
    data_bulk_one = model_bulk_one.run_simulation()
    print(data_bulk_one.return_code)
    print(data_bulk_one.error_message)

    # new interface bulk 2 reactions
    case_bulk_two = MRC.get_dict_two_reaction_mal()
    model_bulk_two = Cadet(cadet_path)
    model_bulk_two = MRC.mct_setup(model_bulk_two)
    model_bulk_two = MRC.mal_setup_bulk(model_bulk_two, case_bulk_two)
    model_bulk_two.filename = f"mct_{case_bulk_two['num_reactions']}_reaction_bulk.h5"
    model_bulk_two.save()
    data_bulk_two = model_bulk_two.run_simulation()
    print(data_bulk_two.return_code)
    print(data_bulk_two.error_message)

    # new interface bulk 3 reactions
    case_bulk_three = MRC.get_dict_three_reaction_mal()
    model_bulk_three = Cadet(cadet_path)
    model_bulk_three = MRC.mct_setup(model_bulk_three)
    model_bulk_three = MRC.mal_setup_bulk(model_bulk_three, case_bulk_three)
    model_bulk_three.filename = f"mct_{case_bulk_three['num_reactions']}_reaction_bulk.h5"
    model_bulk_three.save()
    data_bulk_three = model_bulk_three.run_simulation()
    print(data_bulk_three.return_code)
    print(data_bulk_three.error_message)

    # compare results
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_bulk_one.root.output.solution.solution_times, model_bulk_one.root.output.solution.unit_000.solution_bulk[:, 0, 0], label='Bulk - 1 Reaction')
        plt.plot(model_bulk_two.root.output.solution.solution_times, model_bulk_two.root.output.solution.unit_000.solution_bulk[:,0, 0], label='Bulk - 2 Reactions')
        plt.plot(model_bulk_three.root.output.solution.solution_times, model_bulk_three.root.output.solution.unit_000.solution_bulk[:, 0,0], label='Bulk - 3 Reactions')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('MCT Multiple Bulk Reactions Comparison')
        #plt.savefig(output_path + '/mct_multiple_reactions_bulk_comparison.png')
        plt.show()

    # check max absolute error with tolerance
    tolerance = 1e-10
    max_error_1_2 = np.max(np.abs(model_bulk_one.root.output.solution.unit_000.solution_bulk - model_bulk_two.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_2 < tolerance, f'  WARNING: Max error between 1 and 2 bulk reactions ({max_error_1_2}) exceeds tolerance ({tolerance})'

    max_error_1_3 = np.max(np.abs(model_bulk_one.root.output.solution.unit_000.solution_bulk - model_bulk_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_1_3 < tolerance, f'  WARNING: Max error between 1 and 3 bulk reactions ({max_error_1_3}) exceeds tolerance ({tolerance})'

    max_error_2_3 = np.max(np.abs(model_bulk_two.root.output.solution.unit_000.solution_bulk - model_bulk_three.root.output.solution.unit_000.solution_bulk))
    assert max_error_2_3 < tolerance, f'  WARNING: Max error between 2 and 3 bulk reactions ({max_error_2_3}) exceeds tolerance ({tolerance})'

    # compare old and new interface
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(model_bulk_old.root.output.solution.solution_times, model_bulk_old.root.output.solution.unit_000.solution_bulk[:, 0, 0], label='Old Interface - Bulk - 1 Reaction')
        plt.plot(model_bulk_one.root.output.solution.solution_times, model_bulk_one.root.output.solution.unit_000.solution_bulk[:, 0, 0], label='New Interface - Bulk - 1 Reaction')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.title('MCT Old vs New Interface Bulk Reactions Comparison')
        #plt.savefig(output_path + '/mct_old_vs_new_interface_bulk_comparison.png')
        plt.show()

    # compare old and new interface with tolerance
    tolerance = 1e-10
    max_error_old_new = np.max(np.abs(model_bulk_old.root.output.solution.unit_000.solution_bulk - model_bulk_one.root.output.solution.unit_000.solution_bulk))
    assert max_error_old_new < tolerance, f'  WARNING: Max error between old and new interface for 1 reaction ({max_error_old_new}) exceeds tolerance ({tolerance})'
