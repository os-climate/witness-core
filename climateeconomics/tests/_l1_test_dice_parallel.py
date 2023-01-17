'''
Copyright 2022 Airbus SAS

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
import numpy as np
import pandas as pd
from os.path import join, dirname
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import DataStudy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from copy import deepcopy
from gemseo.utils.compare_data_manager_tooling import delete_keys_from_dict,\
    compare_dict
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from scipy.interpolate.interpolate import interp1d


class DICEParallelTest(unittest.TestCase):

    RESSOURCE_CO2 = ResourceGlossary.CO2['name']

    def setUp(self):

        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)

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
        years = np.arange(2020, 2101, 1)
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        data_dir = join(dirname(__file__), 'data')
        energy_supply_df_all = pd.read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        co2_emissions_gt.index = years

        energy_outlook = pd.DataFrame({
            'years': [2010, 2017, 2018, 2019, 2020, 2025, 2030, 2040, 2050, 2060, 2100],
            'energy_demand': [141057, 153513, 157366, 158839, 158839 * 0.94, 174058 * 0.91, 183234.136 * 0.91,
                              198699.708 * 0.91, 220000 * 0.91, 250000 * 0.91, 300000 * 0.91]})
        f2 = interp1d(energy_outlook['years'], energy_outlook['energy_demand'])
        energy_supply = f2(years)
        energy_supply_df = pd.DataFrame(
            {'years': years, 'Total production': energy_supply})
        energy_supply_df.index = years

        CCS_price = pd.DataFrame(
            {'years': years, 'ccs_price_per_tCO2': np.linspace(311, 515, len(years))})
        energy_price = np.arange(200, 200 + len(years))
        energy_mean_price = pd.DataFrame(
            {'years': years, 'energy_price': energy_price})

        co2_emissions_ccus_Gt = pd.DataFrame()
        co2_emissions_ccus_Gt['years'] = energy_supply_df_y["years"]
        co2_emissions_ccus_Gt['carbon_storage Limited by capture (Gt)'] = 0.02

        CO2_emissions_by_use_sources = pd.DataFrame()
        CO2_emissions_by_use_sources['years'] = energy_supply_df_y["years"]
        CO2_emissions_by_use_sources['CO2 from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['carbon_capture from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['Total CO2 by use (Gt)'] = 20.0
        CO2_emissions_by_use_sources['Total CO2 from Flue Gas (Gt)'] = 3.2

        CO2_emissions_by_use_sinks = pd.DataFrame()
        CO2_emissions_by_use_sinks['years'] = energy_supply_df_y["years"]
        CO2_emissions_by_use_sinks[
            f'{self.RESSOURCE_CO2} removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix['years'] = energy_supply_df_y["years"]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(2020, 2101)
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        values_dict[f'{self.name}.energy_production'] = energy_supply_df
        values_dict[f'{self.name}.co2_emissions_Gt'] = co2_emissions_gt
        values_dict[f'{self.name}.energy_mean_price'] = energy_mean_price
        values_dict[f'{self.name}.CCS_price'] = CCS_price
        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.n_processes'] = n_proc
        values_dict[f'{self.name}.co2_emissions_ccus_Gt'] = co2_emissions_ccus_Gt
        values_dict[f'{self.name}.CO2_emissions_by_use_sources'] = CO2_emissions_by_use_sources
        values_dict[f'{self.name}.CO2_emissions_by_use_sinks'] = CO2_emissions_by_use_sinks
        values_dict[f'{self.name}.EnergyMix.co2_emissions_needed_by_energy_mix'] = co2_emissions_needed_by_energy_mix
        values_dict[f'{self.name}.energy_list'] = []
        values_dict[f'{self.name}.ccs_list'] = []
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

        values_dict[f'{self.name}.energy_production'] = energy_supply_df
        values_dict[f'{self.name}.co2_emissions_Gt'] = co2_emissions_gt
        values_dict[f'{self.name}.energy_mean_price'] = energy_mean_price
        values_dict[f'{self.name}.CCS_price'] = CCS_price
        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.n_processes'] = n_proc
        values_dict[f'{self.name}.co2_emissions_ccus_Gt'] = co2_emissions_ccus_Gt
        values_dict[f'{self.name}.CO2_emissions_by_use_sources'] = CO2_emissions_by_use_sources
        values_dict[f'{self.name}.CO2_emissions_by_use_sinks'] = CO2_emissions_by_use_sinks
        values_dict[f'{self.name}.EnergyMix.co2_emissions_needed_by_energy_mix'] = co2_emissions_needed_by_energy_mix
        values_dict[f'{self.name}.energy_list'] = []
        values_dict[f'{self.name}.ccs_list'] = []

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
