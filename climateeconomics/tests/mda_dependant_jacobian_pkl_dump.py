'''
Created on 10 May 2021

'''

import climateeconomics.tests as jacobian_target

from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from climateeconomics.tests.data.mda_coarse_data_generator import launch_data_pickle_generation
import os

if __name__ == '__main__':

    directory = 'data'
    launch_data_pickle_generation(directory)
    os.system(f'git add ./{directory}/*.pkl')
    os.system('git commit -m "regeneration of mda_coarse data pickles"')
    #os.system('git pull')
    #os.system('git push')

    AbstractJacobianUnittest.launch_all_pickle_generation(
        jacobian_target, 'l1_test_gradient_coarse.py', test_names=[
            'test_01_coarse_renewable_techno_discipline_jacobian',
            'test_02_coarse_fossil_techno_discipline_jacobian',
            'test_03_coarse_dac_techno_discipline_jacobian',
            'test_04_coarse_flue_gas_capture_techno_discipline_jacobian',
            'test_05_coarse_carbon_storage_techno_discipline_jacobian',
            'test_06_coarse_renewable_stream_discipline_jacobian',
            'test_07_coarse_fossil_stream_discipline_jacobian',
        ])

