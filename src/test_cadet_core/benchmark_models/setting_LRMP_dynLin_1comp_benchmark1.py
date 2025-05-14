import urllib
import json
import re
import numpy as np

def get_model(database_url):

    cadet_config_json_name = 'configuration_LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json'
        
    with urllib.request.urlopen(
            database_url + cadet_config_json_name) as url:

        config_data = json.loads(url.read().decode())

    setting_name = re.search(r'configuration_(.*?)(?:\.json|_FV|_DG)',
                             cadet_config_json_name).group(1)
    
    return config_data

def add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(0)},
        'FILM_DIFFUSION': {'sens_comp': np.int64(0)},
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


def get_sensbenchmark1(filename=None):
    
    model = get_model(filename)
    model['input'].pop('sensitivity')
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'FILM_DIFFUSION')
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'LIN_KA')
    
    return model
    
    
    
    
    