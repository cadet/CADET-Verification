# -*- coding: utf-8 -*-
"""

This script implements helper functions to create json files
(as used in cadet-database) from cadet h5 input files.

"""

import numpy as np
import h5py
import json
import re
from cadetrdm import ProjectRepo

project_repo = ProjectRepo()
file_path = project_repo.output_path / "test_cadet-core"


def dataset_to_json(dataset):
    data = dataset[()]
    if isinstance(data, str):
        return data
    elif isinstance(data, (int, float)):
        return data
    elif isinstance(data, bytes):  # Handle the case where dataset is a scalar bytes
        return data.decode('utf-8')
    elif isinstance(data, np.ndarray):
        if data.dtype.kind == 'S':  # Check if data is an array of strings
            return data.astype(str).tolist()  # Convert bytes to strings
        elif data.ndim > 1:  # Check if data is multi-dimensional
            flat_list = data.flatten().tolist()  # Flatten the array
            return flat_list
        else:
            return data.tolist()
    else:
        return data.tolist()


def recursive_group_to_json(group, ignore_list=None):
    if ignore_list is None:
        ignore_list = []

    group_data = {}
    for name, item in group.items():
        if name in ignore_list:
            continue
        if isinstance(item, h5py.Dataset):
            group_data[name] = dataset_to_json(item)
        elif isinstance(item, h5py.Group):
            sub_ignore_list = [ignore_item.split(
                '/', 1)[1] for ignore_item in ignore_list if ignore_item.startswith(name + '/')]
            group_data[name] = recursive_group_to_json(item, sub_ignore_list)
        else:
            group_data[name] = None  # Handle other types
    return group_data

# the following definition additionally deletes the top level key "input" and promotes its nested keys
# def recursive_group_to_json(group, ignore_list=None):
#     if ignore_list is None:
#         ignore_list = []

#     group_data = {}
#     for name, item in group.items():
#         if name == 'input':
#             for sub_name, sub_item in item.items():
#                 if sub_name in ignore_list:
#                     continue
#                 if isinstance(sub_item, h5py.Dataset):
#                     group_data[sub_name] = dataset_to_json(sub_item)
#                 elif isinstance(sub_item, h5py.Group):
#                     group_data[sub_name] = recursive_group_to_json(sub_item, ignore_list)
#                 else:
#                     group_data[sub_name] = None  # Handle other types
#         elif name in ignore_list:
#             continue
#         else:
#             if isinstance(item, h5py.Dataset):
#                 group_data[name] = dataset_to_json(item)
#             elif isinstance(item, h5py.Group):
#                 group_data[name] = recursive_group_to_json(item, ignore_list)
#             else:
#                 group_data[name] = None  # Handle other types
#     return group_data


def promote_input(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Check if there is only one root folder named "input"
    if len(data.keys()) == 1 and 'input' in data:
        input_data = data['input']
        del data['input']  # Remove the top level key "input"

        # Promote nested keys of "input" to the top level
        data.update(input_data)

        # Save the modified data back to the JSON file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

        print("Successfully promoted nested keys of 'input'.")
    else:
        print("The JSON file does not have 'input' as the only root folder or it has multiple root folders.")


def h5_to_json(h5_filename, json_filename, ignore_list):
    with h5py.File(h5_filename, 'r') as h5_file:
        data_dict = recursive_group_to_json(h5_file, ignore_list)

    with open(json_filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)


# %% random single file

file_names = [
    "lrmp2d_debug"
]
json_filenames = [
    "lrmp2d_debug"
]


ignore_list = [
    # 'input/model/unit_001/discretization',
    # 'input/solver/time_integrator',
    # 'input/solver/CONSISTENT_INIT_MODE',
    # 'input/solver/CONSISTENT_INIT_MODE_SENS',
    # 'input/solver/NTHREADS',
    # 'input/solver/USER_SOLUTION_TIMES',
    # 'input/model/solver',
    # 'input/sensitivity',
    # 'input/return',
    'meta',
    'output'
]

file_path = r"C:\Users\jmbr\JupyterNotebooks/"
for file_name, json_filename in zip(file_names, json_filenames):

    h5_filename = file_path + file_name + ".h5"
    json_filename = file_path + json_filename + ".json"

    h5_to_json(h5_filename, json_filename, ignore_list)

    if not ignore_list == []:
        promote_input(json_filename)

    print(f"JSON data written to file: {json_filename}")

# %% chromatography (axial models)


file_path_chromatography = file_path / 'chromatography'

file_names = [
    "sens_LRM_reqSMA_4comp_FV_Z64", "sens_LRMP_reqSMA_4comp_FV_Z32",
    "sens_GRM_reqSMA_4comp_FV_Z16parZ2",
    "sens_LRM_dynLin_1comp_FV_Z256", "sens_LRMP_dynLin_1comp_FV_Z32",
    "sens_GRM_dynLin_1comp_FV_Z32parZ4",
]
# ignore list to get only the model; set empty to get the full configuration
ignore_list = [
    'input/model/unit_000/discretization',  # for LWE settings
    'input/model/unit_001/discretization',  # for linear settings
    'input/solver/time_integrator',
    'input/solver/CONSISTENT_INIT_MODE',
    'input/solver/CONSISTENT_INIT_MODE_SENS',
    'input/solver/NTHREADS',
    'input/solver/USER_SOLUTION_TIMES',
    'input/model/solver',
    'input/sensitivity',
    'input/return',
    'meta',
    'output'
]

# for file_name in file_names:

#     h5_filename = file_path_chromatography / file_name + ".h5"
#     if ignore_list == []:
#         json_filename = file_path_chromatography / "configuration_" + re.search(r"(?<=sens_)(.*?)(?=_FV)", file_name).group(
#             1) + "_"+re.search(r"sens", file_name).group()+"benchmark1" + re.search(r"_FV.*", file_name).group() + ".json"
#     else:
#         json_filename = file_path_chromatography / "model_" + \
#             re.search(r"(?<=sens_)(.*?)(?=_FV)",
#                       file_name).group(1) + "_benchmark1.json"
#     h5_to_json(h5_filename, json_filename, ignore_list)
#     if not ignore_list == []:
#         promote_input(json_filename)
#     print(f"JSON data written to file: {json_filename}")

# 
# %% radial chromatography

file_path_radial = file_path / "chromatography" / "radial"

file_names = [
    "radLRM_dynLin_1comp_sensbenchmark1_FV_Z32",
    "radLRMP_dynLin_1comp_sensbenchmark1_FV_Z32",
    "radGRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4"
]
# ignore list to get only the model; set empty to get the full configuration
ignore_list = [
    'input/model/unit_000/discretization',  # for LWE settings
    'input/model/unit_001/discretization',  # for linear settings
    'input/solver/time_integrator',
    'input/solver/CONSISTENT_INIT_MODE',
    'input/solver/CONSISTENT_INIT_MODE_SENS',
    'input/solver/NTHREADS',
    'input/solver/USER_SOLUTION_TIMES',
    'input/model/solver',
    'input/sensitivity',
    'input/return',
    'meta',
    'output'
]


# for file_name in file_names:

#     h5_filename = file_path_radial / str(file_name + ".h5")
#     if ignore_list == []:
#         json_filename = file_path_radial / \
#         str("configuration_" + re.search(r"(.*?)(?=_FV)", file_name).group(1) + ".json")
#     else:
#         json_filename = file_path_radial / \
#         str("model_" + re.search(r"(.*?)(?=_FV)", file_name).group(1) + ".json")
#     h5_to_json(h5_filename, json_filename, ignore_list)
#     if not ignore_list == []:
#         promote_input(json_filename)
#     print(f"JSON data written to file: {json_filename}")


# %% crystallization

file_path_crystallization = file_path / 'crystallization'

file_names = [
    "ref_PBM_CSTR_growth", "ref_PBM_CSTR_growthSizeDep",
    "ref_PBM_CSTR_PBM_CSTR_primarySecondaryNucleationAndGrowth",
    "ref_PBM_CSTR_primaryNucleationAndGrowth",
    "ref_PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion",
    "ref_PBM_DPFR_primarySecondaryNucleationGrowth",
]
# ignore list to get only the model; set empty to get the full configuration
ignore_list = [
    # 'input/model/unit_000/discretization',  # for LWE settings
    # 'input/model/unit_001/discretization',  # for linear settings
    # 'input/solver/time_integrator',
    # 'input/solver/CONSISTENT_INIT_MODE',
    # 'input/solver/CONSISTENT_INIT_MODE_SENS',
    # 'input/solver/NTHREADS',
    # 'input/solver/USER_SOLUTION_TIMES',
    # 'input/model/solver',
    # 'input/sensitivity',
    # 'input/return',
    'meta',
    'output'
]

for file_name in file_names:

    h5_filename = str(file_path_crystallization / file_name) + ".h5"
    json_filename = str(file_path_crystallization / "configuration_") + \
        re.search(r"ref_(.*)",
                  file_name).group(1) + "_benchmark1.json"
        
    print(json_filename)
        
    h5_to_json(h5_filename, json_filename, ignore_list)
    if not ignore_list == []:
        promote_input(json_filename)
    print(f"JSON data written to file: {json_filename}")


# %% MCT

file_names = [
    "ref_MCT1ch_noEx_noReac_benchmark1_FV_Z256",
    "ref_MCT1ch_noEx_reac_benchmark1_FV_Z256",
    "ref_MCT2ch_oneWayEx_reac_benchmark1_FV_Z256",
    "ref_MCT3ch_twoWayExc_reac_benchmark1_FV_Z256",
    "ref_LRM_noBnd_1comp_MCTbenchmark_FV_Z256"
]
# json_filenames = [
#     "model_MCT1ch_noEx_noReac_benchmark1",
#     "model_MCT1ch_noEx_reac_benchmark1",
#     "model_MCT2ch_oneWayEx_reac_benchmark1",
#     "model_MCT3ch_twoWayExc_reac_benchmark1",
#     "model_LRM_noBnd_1comp_MCTbenchmark"
# ]
json_filenames = [
    "configuration_MCT1ch_noEx_noReac_benchmark1",
    "configuration_MCT1ch_noEx_reac_benchmark1",
    "configuration_MCT2ch_oneWayEx_reac_benchmark1",
    "configuration_MCT3ch_twoWayExc_reac_benchmark1",
    "configuration_LRM_noBnd_1comp_MCTbenchmark"
]

ignore_list = [
    # 'input/model/unit_001/discretization',
    # 'input/solver/time_integrator',
    # 'input/solver/CONSISTENT_INIT_MODE',
    # 'input/solver/CONSISTENT_INIT_MODE_SENS',
    # 'input/solver/NTHREADS',
    # 'input/solver/USER_SOLUTION_TIMES',
    # 'input/model/solver',
    # 'input/sensitivity',
    # 'input/return',
    'meta',
    'output'
]

# file_path = "C:/Users/jmbr/Cadet_testBuild/CADET_MCT/test/data/"
# file_path = "C:/Users/jmbr/software/CADET-Database/cadet_config/test_cadet-core/mct/"

# for file_name, json_filename in zip(file_names, json_filenames):

#     h5_filename = file_path + file_name + ".h5"
#     json_filename = file_path + json_filename + ".json"

#     h5_to_json(h5_filename, json_filename, ignore_list)

#     if not ignore_list == []:
#         promote_input(json_filename)

#     print(f"JSON data written to file: {json_filename}")


# %% compare json files

def compare_json_objects(obj1, obj2, path="", differences=None):
    if differences is None:
        differences = {}

    # Check if the objects are dictionaries
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        # Iterate through keys in obj1
        for key in obj1:
            # Check if the key exists in obj2
            if key in obj2:
                # Recursively compare nested objects
                compare_json_objects(
                    obj1[key], obj2[key], path + key + ".", differences)
            else:
                differences[path + key] = "exists in dict1 but not in dict2"
        # Check for keys in obj2 but not in obj1
        for key in obj2:
            if key not in obj1:
                differences[path + key] = "exists in dict2 but not in dict1"
    else:
        # Compare values
        if obj1 != obj2:
            differences[path[:-1]
                        ] = "different values: {} != {}".format(obj1, obj2)

    return differences


def compare_json_files(data1, data2, print_values=True):
    differences = compare_json_objects(data1, data2)

    if differences:
        print(differences.items())
        print("Differences found between the two JSON files:")
        for key, value in differences.items():
            if value != "different values":
                print(key)
    else:
        print("Both JSON files are identical.")


# file_path = "C:/Users/jmbr/Cadet_testBuild/CADET_MCT/test/data"
# file1 = file_path+"/compare_MCT1ch_noEx_noReac_benchmark1.json"
# file2 = file_path+"/compare_configuration_MCT1ch_noEx_noReac_benchmark1_FV_Z256.json"


# with open(file1, 'r') as f1, open(file2, 'r') as f2:
#     data1 = json.load(f1)
#     data2 = json.load(f2)

# compare_json_files(data1, data2, print_values=False)
# compare_json_files(data1, data2, print_values=False)
# compare_json_files(data1, data2, print_values=False)


# %% Modify json and h5 files

def modify_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    NSEC = data['input']['solver']['sections']['NSEC']

    # Creating a vector of ones with length NSEC
    ones_vector = [1] * (NSEC - 1)

    # Overwriting sec_con with the ones vector
    data['input']['solver']['sections']['SECTION_CONTINUITY'] = ones_vector

    # Writing the modified JSON back to the file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def modify_h5(h5_file):
    with h5py.File(h5_file, 'r+') as f:
        # Extracting NESC from the input/solver subgroup
        NSEC = f['input/solver/sections/NSEC'][()]

        if 'SECTION_CONTINUITY' in f['input/solver/sections']:
            del f['input/solver/sections/SECTION_CONTINUITY']

        # Creating a vector of ones with length NESC
        ones_vector = [True] * (NSEC-1)

        # Overwriting sec_con with the ones vector
        f['input/solver/sections/SECTION_CONTINUITY'] = ones_vector


# json_file = "C:/Users/jmbr/Cadet_testBuild/CADET_MCT/test/data/configuration_MCT1ch_noEx_reac_benchmark1_FV_Z256.json"
# modify_json(json_file)

# h5_file = "C:/Users/jmbr/Cadet_testBuild/CADET_MCT/test/data/ref_LRM_noBnd_1comp_MCTbenchmark_FV_Z256.h5"
# modify_h5(h5_file)


# %% Compare an h5 and json file

# file_path = "C:/Users/jmbr/software/CADET-Reference/"

# h5File = file_path+"/ref_LRMP_reqSMA_4comp_sensbenchmark1_DG_P3Z8.h5"
# jsonFile = file_path+"/ref_LRMP_reqSMA_4comp_sensbenchmark1_DG_P3Z8.json"


# tmpFile = file_path+"jojo.json"

# h5_to_json(h5File, jsonFile, [])


# with open(jsonFile, 'r') as f1, open(tmpFile, 'r') as f2:
#     data1 = json.load(f1)
#     data2 = json.load(f2)

# compare_json_files(data1['input'], data2, print_values=False)


