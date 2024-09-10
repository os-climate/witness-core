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

import numpy as np
import pandas as pd
import scipy.interpolate as sc
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class AgricultureMixModelTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        self.years = np.arange(self.year_start, self.year_end + 1)
        year_range = self.year_end - self.year_start + 1
        self.technology_list = ['Crop', 'Forest']

        # ----------------------------------------------------
        # Crop related inputs
        population = np.array(np.linspace(7800, 9500, year_range))
        temperature = np.array(np.linspace(1.05, 4, year_range))

        self.temperature_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.TempAtmo: temperature})

        self.population_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.PopulationValue: population})

        red_meat_percentage = np.linspace(6, 1, year_range)
        white_meat_percentage = np.linspace(14, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
                            GlossaryCore.Years: self.years,
                            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                GlossaryCore.Years: self.years,
                                'white_meat_percentage': white_meat_percentage})

        self.diet_df = pd.DataFrame({'red meat': [11.02],
                                'white meat': [31.11],
                                'milk': [79.27],
                                'eggs': [9.68],
                                'rice and maize': [97.76],
                                'potatoes': [32.93],
                                'fruits and vegetables': [217.62],
                                })
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(1.15, 1.45, year_range)
        self.techno_production_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(17.03, 17.03, year_range)
        techno_prices_wotaxes = np.linspace(17.03, 17.03, year_range)
        self.techno_prices_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop': techno_prices, 'Crop_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop': CO2_emissions})

        land_use_required_energy = np.linspace(0.06, 0.13, year_range)
        land_use_required_food = np.linspace(5.17, 5.13, year_range)
        self.land_use_required_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop (Gha)': land_use_required_energy, 'Crop for Food (Gha)': land_use_required_food})
        # ----------------------------------------------------
        # Forest related inputs
        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, "deforested_surface": deforestation_surface})
        forest_invest = np.linspace(5, 8, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, "forest_investment": forest_invest})
        mw_invest = np.linspace(1, 4, year_range)
        uw_invest = np.linspace(0, 1, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.InvestmentsValue: mw_invest})
        self.uw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.InvestmentsValue: uw_invest})

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(15, 16, year_range)
        self.techno_production_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(9, 9, year_range)
        techno_prices_wotaxes = np.linspace(9, 9, year_range)
        self.techno_prices_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Forest': techno_prices, 'Forest_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Forest': CO2_emissions})

        land_use_required = np.linspace(4, 3.5, year_range)
        self.land_use_required_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Forest (Gha)': land_use_required})
        # -----------------------------------------------------
        # Agriculture mix related inputs
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'margin': np.ones(len(self.years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'transport': np.ones(len(self.years)) * 7.6})

        self.stream_co2_emissions = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryEnergy.biomass_dry: - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, GlossaryEnergy.electricity: 0.0, GlossaryEnergy.methane: 0.123 / 15.4, 'syngas': 0.0, f"{GlossaryEnergy.hydrogen}.{GlossaryEnergy.gaseous_hydrogen}": 0.0, 'crude oil': 0.02533})
        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01, 34.05, 39.08, 44.69, 50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        self.co2_taxes = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: func(self.years)})

    def test_agriculture_mix_discipline(self):
        '''
        Check discipline setup and run
        '''
        test_name = 'Test'
        disc_name = 'AgricultureMix'
        ee = ExecutionEngine(test_name)
        ns_dict = {'ns_agriculture': f'{test_name}',
                   'ns_energy_study': f'{test_name}',
                   'ns_public': f'{test_name}',
                   GlossaryCore.NS_WITNESS: f'{test_name}', }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_mix_disc.AgricultureMixDiscipline'
        builder = ee.factory.get_builder_from_module(disc_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000,
            GlossaryCore.NonUseCapital: 0.,
        })
        ee.configure()
        ee.display_treeview_nodes()
        inputs_dict = {f'{test_name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{test_name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{test_name}.{disc_name}.Crop.techno_consumption': self.techno_consumption_crop,
                       f'{test_name}.{disc_name}.Crop.{GlossaryCore.TechnoConsumptionWithoutRatioValue}': self.techno_consumption_woratio_crop,
                       f'{test_name}.{disc_name}.Crop.techno_production': self.techno_production_crop,
                       f'{test_name}.{disc_name}.Crop.techno_prices': self.techno_prices_crop,
                       f'{test_name}.{disc_name}.Crop.CO2_emissions': self.CO2_emissions_crop,
                       f'{test_name}.{disc_name}.Crop.land_use_required': self.land_use_required_crop,
                       f'{test_name}.{disc_name}.Crop.{GlossaryEnergy.TechnoCapitalValue}': techno_capital,
                       f'{test_name}.{disc_name}.Forest.{GlossaryEnergy.TechnoCapitalValue}': techno_capital,
                       f'{test_name}.{disc_name}.Forest.techno_consumption': self.techno_consumption_forest,
                       f'{test_name}.{disc_name}.Forest.{GlossaryCore.TechnoConsumptionWithoutRatioValue}': self.techno_consumption_woratio_forest,
                       f'{test_name}.{disc_name}.Forest.techno_production': self.techno_production_forest,
                       f'{test_name}.{disc_name}.Forest.techno_prices': self.techno_prices_forest,
                       f'{test_name}.{disc_name}.Forest.CO2_emissions': self.CO2_emissions_forest,
                       f'{test_name}.{disc_name}.Forest.land_use_required': self.land_use_required_forest,
                       f'{test_name}.{GlossaryCore.techno_list}': self.technology_list,
                       f'{test_name}.{GlossaryCore.CO2TaxesValue}': self.co2_taxes,
                      }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(
            f'{test_name}.{disc_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #   graph.to_plotly().show()
