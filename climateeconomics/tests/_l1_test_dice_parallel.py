'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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
import unittest
from copy import deepcopy
from tempfile import gettempdir

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)
from gemseo.utils.compare_data_manager_tooling import (
    compare_dict,
    delete_keys_from_dict,
)
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import (
    DataStudy,
)


class DICEParallelTest(unittest.TestCase):

    RESSOURCE_CO2 = ResourceGlossary.CO2['name']

    def setUp(self):

        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)

        self.co2_emissions_gt = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.TotalCO2Emissions: np.linspace(34, 55, len(self.years))})

    def test_01_exec_parallel(self):
        """
        1 proc
        """
        n_proc = 1

        repo = 'climateeconomics.sos_processes.iam'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness_wo_energy')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()
        usecase = DataStudy()
        usecase.study_name = self.name
        values_dict = {}
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.energy_supply_df_all = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years))
        })

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(43, 76, len(self.years))
        })

        CCS_price = pd.DataFrame({
            GlossaryCore.Years: years,
            'ccs_price_per_tCO2': np.linspace(311, 515, len(years))
        })

        self.energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.EnergyPriceValue: np.arange(200, 200 + len(years))
        })

        self.co2_emissions_ccus_Gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'carbon_storage Limited by capture (Gt)': 0.02
        })

        self.CO2_emissions_by_use_sources = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'CO2 from energy mix (Gt)': 0.0,
            'carbon_capture from energy mix (Gt)': 0.0,
            'Total CO2 by use (Gt)': 20.0,
            'Total CO2 from Flue Gas (Gt)': 3.2
        })

        self.CO2_emissions_by_use_sinks = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'CO2 removed by energy mix (Gt)': 0.0
        })

        self.co2_emissions_needed_by_energy_mix = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'carbon_capture needed by energy mix (Gt)': 0.0
        })

        # put manually the index

        values_dict[f'{self.name}.{GlossaryCore.EnergyProductionValue}'] = self.energy_supply_df
        values_dict[f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}'] = self.co2_emissions_gt
        values_dict[f'{self.name}.{GlossaryCore.EnergyPriceValue}'] = self.energy_mean_price
        values_dict[f'{self.name}.CCS_price'] = CCS_price
        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.n_processes'] = n_proc
        values_dict[f'{self.name}.co2_emissions_ccus_Gt'] = self.co2_emissions_ccus_Gt
        values_dict[f'{self.name}.CO2_emissions_by_use_sources'] = self.CO2_emissions_by_use_sources
        values_dict[f'{self.name}.CO2_emissions_by_use_sinks'] = self.CO2_emissions_by_use_sinks
        values_dict[f'{self.name}.EnergyMix.co2_emissions_needed_by_energy_mix'] = self.co2_emissions_needed_by_energy_mix
        values_dict[f'{self.name}.{GlossaryCore.energy_list}'] = []
        values_dict[f'{self.name}.{GlossaryCore.ccs_list}'] = []
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.configure()
        self.ee.execute()
        dm_dict_1 = deepcopy(self.ee.get_anonimated_data_dict())
        residual_history = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.sub_mda_list[0].residual_history
        """
        8 proc
        """
        n_proc = 8
        self.ee8 = ExecutionEngine(self.name)
        builder = self.ee8.factory.get_builder_from_process(
            repo, 'witness_wo_energy')

        self.ee8.factory.set_builders_to_coupling_builder(builder)
        self.ee8.configure()
        self.ee8.display_treeview_nodes()
        usecase = DataStudy()
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        values_dict[f'{self.name}.{GlossaryCore.EnergyProductionValue}'] = self.energy_supply_df
        values_dict[f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}'] = self.co2_emissions_gt
        values_dict[f'{self.name}.{GlossaryCore.EnergyPriceValue}'] = self.energy_mean_price
        values_dict[f'{self.name}.CCS_price'] = CCS_price
        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.n_processes'] = n_proc
        values_dict[f'{self.name}.co2_emissions_ccus_Gt'] = self.co2_emissions_ccus_Gt
        values_dict[f'{self.name}.CO2_emissions_by_use_sources'] = self.CO2_emissions_by_use_sources
        values_dict[f'{self.name}.CO2_emissions_by_use_sinks'] = self.CO2_emissions_by_use_sinks
        values_dict[f'{self.name}.EnergyMix.co2_emissions_needed_by_energy_mix'] = self.co2_emissions_needed_by_energy_mix
        values_dict[f'{self.name}.{GlossaryCore.energy_list}'] = []
        values_dict[f'{self.name}.{GlossaryCore.ccs_list}'] = []

        self.ee8.load_study_from_input_dict(values_dict)

        self.ee8.execute()

        dm_dict_8 = deepcopy(self.ee8.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_8)
        compare_dict(dm_dict_1,
                     dm_dict_8, '', dict_error)

        residual_history8 = self.ee8.root_process.mdo_discipline_wrapp.mdo_discipline.sub_mda_list[0].residual_history
        self.assertListEqual(residual_history, residual_history8)

        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.n_processes.value': "1 and 8 don't match"})


if '__main__' == __name__:

    cls = DICEParallelTest()
    cls.setUp()
    cls.test_01_exec_parallel()
