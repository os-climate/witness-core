'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/07-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import pickle
from copy import deepcopy
from os.path import dirname, join

import numpy as np
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import (
    Study as EnergyMDA,
)
from sostrades_core.tools.compare_data_manager_tooling import compare_dict

from climateeconomics.glossarycore import GlossaryCore


class EnergyMix_co2emissions_nonreg_test():
    '''
    Test to compare the new CO2 emissions pyworld3 with the implementation out of EnergyMix
    '''

    def setUp(self):

        self.compare_list = [
            GlossaryEnergy.StreamsCO2EmissionsValue, 'energy_CO2_emissions_after_use', 'co2_emissions',
            'co2_emissions_by_energy', GlossaryCore.CO2EmissionsGtValue, 'CCS_price'
        ]

    def convert_types(self, value):
        # convert int32 to int64 to avoid issues between platform
        if type(value) is pd.DataFrame:
            dtypes = dict(value.dtypes)
            for column, dtype in dtypes.items():
                if dtype == np.int32:
                    value[column] = value[column].astype(np.int64)
        return value

    def run_non_reg(self, usecase, dm_ref, compare_list, usecase_name):
        # load and run usecase
        usecase.load_data()
        inputs_dict = {}
        inputs_dict[usecase.ee.dm.get_all_namespaces_from_var_name('inner_mda_name')[
            0]] = 'MDAGaussSeidel'
        usecase.ee.load_study_from_input_dict(inputs_dict)
        usecase.run(for_test=False)
        dm_run = deepcopy(usecase.execution_engine.get_anonimated_data_dict())

        dm_ref_to_compare = {}
        dm_run_to_compare = {}
        for param_name in compare_list:
            for key, metadata in dm_ref.items():
                if param_name in key.split('.') and metadata['io_type'] == 'out':
                    value_ref = self.convert_types(metadata.get('value', None))
                    dm_ref_to_compare[key] = value_ref
                    dm_run_to_compare[key] = None

                    run_metadata = dm_run.get(key, None)
                    if run_metadata is not None:
                        value_run = self.convert_types(
                            run_metadata.get('value', None))
                        dm_run_to_compare[key] = value_run

        dict_error = {}
        test_passed = True
        output_error = ''
        compare_dict(dm_ref_to_compare, dm_run_to_compare,
                     '', dict_error)
        if dict_error != {}:
            test_passed = False
            output_error += f'Non-regression failed on {usecase_name}:\n'
            for error in dict_error:
                output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                output_error += '\n---------------------------------------------------------\n'

        if not test_passed:
            raise Exception(f'{output_error}')
        else:
            print(f'Non-regression success on {usecase_name}')

    def test_non_reg_EnergyMDA_CO2_emissions(self):
        # non regression test on EnergyMDA CO2 emissions usecase results
        usecase = EnergyMDA()
        dm_ref = pickle.load(open(
            join(dirname(__file__), 'data', 'dm_energymda_co2emissionsnonreg.pkl'), 'rb'))

        self.run_non_reg(usecase=usecase, dm_ref=dm_ref, compare_list=self.compare_list,
                         usecase_name='Energy MDA nonreg - CO2 emissions')


if __name__ == "__main__":
    cls = EnergyMix_co2emissions_nonreg_test()
    cls.setUp()
    cls.test_non_reg_EnergyMDA_CO2_emissions()
