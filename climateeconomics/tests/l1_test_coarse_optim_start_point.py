'''
Copyright 2023 Capgemini

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
import logging
import unittest
from tempfile import gettempdir

from energy_models.database_witness_energy import DatabaseWitnessEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import (
    Study,
)


class StartPointOptimTest(unittest.TestCase):
    """
    This test is created to assert that first point of coarse optimization investment variables is not activated and
    is set to the correct value (invest at 2020
    """

    def setUp(self):
        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)
        logging.disable(logging.INFO)

    def test_start_point_optim(self):
        # import process
        repo = 'climateeconomics.sos_processes.iam.witness'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness_coarse_dev_optim_process')
        # set builder to coupling
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes(True)
        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()
        # we only need to configure and to set all the values
        self.ee.load_study_from_input_dict(values_dict)
        # get design space directly from data manager
        design_space_df = self.ee.dm.get_value('Test.WITNESS_MDO.design_space')
        design_variables_to_check = [f"{GlossaryCore.clean_energy}.{GlossaryCore.CleanEnergySimpleTechno}.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_array_mix",
                                     'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix',
                                     'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
                                     'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                                     'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix']

        invest_value_ccus_2020 = DatabaseWitnessEnergy.InvestCCUS2020.value / 3
        # values we expect for first point are the investments
        expected_values_dict = {
            f"{GlossaryCore.clean_energy}.{GlossaryCore.CleanEnergySimpleTechno}.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_array_mix": DatabaseWitnessEnergy.InvestCleanEnergy2020.value,
            'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': DatabaseWitnessEnergy.InvestFossil2020.value,
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': invest_value_ccus_2020,
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': invest_value_ccus_2020,
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': invest_value_ccus_2020
            }

        # get first value of each design variable value
        ds_values_from_dm = {var: design_space_df[design_space_df['variable'] == var]['value'].values[0][0] for var in
                             design_variables_to_check
                             }

        # get activation elem from dm to check first element is not activated
        ds_activation_elem_from_dm = {var: design_space_df[design_space_df['variable'] == var]['activated_elem'].values[0][0] for
                                      var in design_variables_to_check}

        expected_activated_elem = {var: False for var in design_variables_to_check}

        self.assertDictEqual(expected_values_dict, ds_values_from_dm, msg="Dictionnary of values for first point is "
                                                                          "not matching the expected values")
        self.assertDictEqual(ds_activation_elem_from_dm, expected_activated_elem, msg="Dictionnary of activated "
                                                                                      "elements is not matching the "
                                                                                      "expected: at least one "
                                                                                      "starting point is activated "
                                                                                      "but should not ")
