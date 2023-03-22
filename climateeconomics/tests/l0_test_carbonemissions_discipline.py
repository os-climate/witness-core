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
from pandas import read_csv

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class CarbonEmissionDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'carbonemission'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_ccs': f'{self.name}',
                   'ns_energy': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all['years'] >= 2020]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})

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
        CO2_emissions_by_use_sinks['CO2 removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix['years'] = energy_supply_df_y["years"]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(2020, 2101)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(0.01, 0.10, len(years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        CO2_emitted_forest['years'] = years
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.co2_emissions_Gt': energy_supply_df_y,
                       f'{self.name}.CO2_land_emissions': CO2_emitted_forest,
                       f'{self.name}.co2_emissions_ccus_Gt': co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': co2_emissions_needed_by_energy_mix}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_carbon_cycle = self.ee.dm.get_value(
            f'{self.name}.CO2_emissions_df')
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_limit_co2_objective(self):
        # the limit is commented in th emodel we deaxctivate the test
        self.model_name = 'carbonemission'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_ccs': f'{self.name}',
                   'ns_energy': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all['years'] >= 2020]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
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
        CO2_emissions_by_use_sinks['CO2 removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix['years'] = energy_supply_df_y["years"]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(2020, 2101)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        min_co2_objective = -1000.0
        # put manually the index
        years = np.arange(2020, 2101)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        energy_supply_df_y['Total CO2 emissions'] = np.linspace(
            0, -100000, len(years))

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        CO2_emitted_forest['years'] = years
        CO2_emitted_forest['emitted_CO2_evol'] = emission_forest
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.co2_emissions_Gt': energy_supply_df_y,
                       f'{self.name}.CO2_land_emissions': CO2_emitted_forest,
                       f'{self.name}.{self.model_name}.min_co2_objective': min_co2_objective,
                       f'{self.name}.co2_emissions_ccus_Gt': co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': co2_emissions_needed_by_energy_mix}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        CO2_objective = self.ee.dm.get_value(f'{self.name}.CO2_objective')
        # If lower than min_co2_objective the objective is limited with an exp
        # until 10% of its limit (1100 il the value is 1000)
        self.assertLess(min_co2_objective * 1.1, CO2_objective)
