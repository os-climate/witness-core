'''
Copyright 2024 Capgemini

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
from os.path import join, dirname

import numpy as np
import pandas as pd
from pandas import read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class CarbonEmissionDiscTestCheckRange(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_check_range(self):
        """
        Test check range is correct
        """
        self.model_name = 'carbonemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
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

        economics_df_y = economics_df_all[economics_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})

        co2_emissions_ccus_Gt = pd.DataFrame()
        co2_emissions_ccus_Gt[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_ccus_Gt['carbon_storage Limited by capture (Gt)'] = 0.02

        CO2_emissions_by_use_sources = pd.DataFrame()
        CO2_emissions_by_use_sources[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sources['CO2 from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['carbon_capture from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['Total CO2 by use (Gt)'] = 20.0
        CO2_emissions_by_use_sources['Total CO2 from Flue Gas (Gt)'] = 3.2

        CO2_emissions_by_use_sinks = pd.DataFrame()
        CO2_emissions_by_use_sinks[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sinks['CO2 removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault + 1)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(0.01, 0.10, len(years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        CO2_emitted_forest[GlossaryCore.Years] = years
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': energy_supply_df_y,
                       f'{self.name}.CO2_land_emissions': CO2_emitted_forest,
                       f'{self.name}.co2_emissions_ccus_Gt': co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': co2_emissions_needed_by_energy_mix,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False} # activate check before run

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()


    def test_failing_check_range_input(self):
        """
        Test failing check range
        Put year of dataframe CO2_land_emissions outside of range
        """

        self.model_name = 'carbonemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
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

        economics_df_y = economics_df_all[economics_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        co2_emissions_ccus_Gt = pd.DataFrame()
        co2_emissions_ccus_Gt[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_ccus_Gt['carbon_storage Limited by capture (Gt)'] = 0.02

        CO2_emissions_by_use_sources = pd.DataFrame()
        CO2_emissions_by_use_sources[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sources['CO2 from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['carbon_capture from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['Total CO2 by use (Gt)'] = 20.0
        CO2_emissions_by_use_sources['Total CO2 from Flue Gas (Gt)'] = 3.2

        CO2_emissions_by_use_sinks = pd.DataFrame()
        CO2_emissions_by_use_sinks[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sinks['CO2 removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault + 1)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        min_co2_objective = -1000.0
        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault + 1)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        energy_supply_df_y[GlossaryCore.TotalCO2Emissions] = np.linspace(
            0, -100000, len(years))

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        # put incorrect year in column to check test will fail
        years[0] = 1950
        CO2_emitted_forest[GlossaryCore.Years] = years
        CO2_emitted_forest['emitted_CO2_evol'] = emission_forest
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': energy_supply_df_y,
                       f'{self.name}.CO2_land_emissions': CO2_emitted_forest,
                       f'{self.name}.{self.model_name}.min_co2_objective': min_co2_objective,
                       f'{self.name}.co2_emissions_ccus_Gt': co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': co2_emissions_needed_by_energy_mix,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': True}

        self.ee.load_study_from_input_dict(values_dict)
        # check test will fail because year of CO2_land_emissions is not in correct range
        with self.assertRaises(ValueError, msg="Expected ValueError due to incorrect range"):
            self.ee.execute()

    def test_failing_check_range_output(self):
        """
        Test failing check range
        Put very high emissions values as input so that output total emissions go beyond range
        """

        self.model_name = 'carbonemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
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

        economics_df_y = economics_df_all[economics_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        co2_emissions_ccus_Gt = pd.DataFrame()
        co2_emissions_ccus_Gt[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_ccus_Gt['carbon_storage Limited by capture (Gt)'] = 0.02

        CO2_emissions_by_use_sources = pd.DataFrame()
        CO2_emissions_by_use_sources[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sources['CO2 from energy mix (Gt)'] = 1.e9
        CO2_emissions_by_use_sources['carbon_capture from energy mix (Gt)'] = 0.0
        CO2_emissions_by_use_sources['Total CO2 by use (Gt)'] = 1.e9
        CO2_emissions_by_use_sources['Total CO2 from Flue Gas (Gt)'] = 3.2

        CO2_emissions_by_use_sinks = pd.DataFrame()
        CO2_emissions_by_use_sinks[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        CO2_emissions_by_use_sinks['CO2 removed by energy mix (Gt)'] = 0.0

        co2_emissions_needed_by_energy_mix = pd.DataFrame()
        co2_emissions_needed_by_energy_mix[GlossaryCore.Years] = energy_supply_df_y[GlossaryCore.Years]
        co2_emissions_needed_by_energy_mix[
            'carbon_capture needed by energy mix (Gt)'] = 0.0
        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault + 1)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        co2_emissions_ccus_Gt.index = years
        CO2_emissions_by_use_sources.index = years
        CO2_emissions_by_use_sinks.index = years
        co2_emissions_needed_by_energy_mix.index = years

        min_co2_objective = -1000.0
        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault + 1)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        energy_supply_df_y[GlossaryCore.TotalCO2Emissions] = np.linspace(
            0, -100000, len(years))

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(1.e9, 1.e9, len(years))
        cum_emission = np.cumsum(emission_forest) + 1.e9
        # put incorrect year in column to check test will fail
        CO2_emitted_forest[GlossaryCore.Years] = years
        CO2_emitted_forest['emitted_CO2_evol'] = emission_forest
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': energy_supply_df_y,
                       f'{self.name}.CO2_land_emissions': CO2_emitted_forest,
                       f'{self.name}.{self.model_name}.min_co2_objective': min_co2_objective,
                       f'{self.name}.co2_emissions_ccus_Gt': co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': co2_emissions_needed_by_energy_mix,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': True}

        self.ee.load_study_from_input_dict(values_dict)
        # check test will fail because year of CO2_land_emissions is not in correct range
        with self.assertRaises(ValueError, msg="Expected ValueError due to incorrect range"):
            self.ee.execute()

