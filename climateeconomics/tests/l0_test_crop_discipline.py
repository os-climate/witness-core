'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2023/11/03 Copyright 2023 Capgemini

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
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.core.core_agriculture.crop import Crop
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc import (
    CropDiscipline,
)


class CropTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2055
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7794.799, 7794.799, year_range))
        self.population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        self.population_df.index = years
        temperature = np.array(np.linspace(0.0,0.0, year_range))
        self.temperature_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.TempAtmo: temperature,
        })
        self.temperature_df.index = years

        lifetime = 50
    
        # Age distribution of forests in 2008 (
        initial_age_distribution = pd.DataFrame({'age': np.arange(1, lifetime),
                                             'distrib': [0.16, 0.24, 0.31, 0.39, 0.47, 0.55, 0.63, 0.71, 0.78, 0.86,
                                                         0.94, 1.02, 1.1, 1.18, 1.26, 1.33, 1.41, 1.49, 1.57, 1.65,
                                                         1.73, 1.81, 1.88, 1.96, 2.04, 2.12, 2.2, 2.28, 2.35, 2.43,
                                                         2.51, 2.59, 2.67, 2.75, 2.83, 2.9, 2.98, 3.06, 3.14, 3.22,
                                                         3.3, 3.38, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.92]})

        self.default_kg_to_m2 = {'red meat': 345.,
                            'white meat': 14.5,
                            'milk': 8.95,
                            'eggs': 6.27,
                            'rice and maize': 2.89,
                            'cereals': 4.5,
                            'fruits and vegetables': 0.8,
                            GlossaryCore.Fish: 0.,
                            GlossaryCore.OtherFood: 5.1041,
                            }
        # land use of other is provided in variable 'other_use_crop'

        self.default_kg_to_kcal = {'red meat': 1551.05,
                              'white meat': 2131.99,
                              'milk': 921.76,
                              'eggs': 1425.07,
                              'rice and maize': 2572.46,
                              'cereals': 2964.99,
                              'fruits and vegetables': 559.65,
                              GlossaryCore.Fish: 609.17,
                              GlossaryCore.OtherFood: 3061.06,
                              }

        self.diet_df = pd.DataFrame({"red meat": [11.02],
                                    "white meat": [31.11],
                                    "milk": [79.27],
                                    "eggs": [9.68],
                                    "rice and maize": [98.08],
                                    "cereals": [78],
                                    "fruits and vegetables": [293],
                                    GlossaryCore.Fish: [23.38],
                                    GlossaryCore.OtherFood: [77.24]
                                    })

        # investment: 1Mha of crop land each year
        self.crop_investment = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.ones(len(years)) * 0.381})
             
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        # From future of hydrogen
        self.transport_cost = pd.DataFrame(
            {GlossaryCore.Years: years, 'transport': np.ones(len(years)) * 7.6})

        # bioenergyeurope.org : Dedicated energy crops
        # represent 0.1% of the total biomass production in 2018
        energy_crop_percentage = 0.005
        # ourworldindata, average cereal yield: 4070kg/ha +
        # average yield of switchgrass on grazing lands: 2565,67kg/ha
        # residue is 0.25 more than that
        density_per_ha = 2903 * 1.25
        # available ha of crop: 4.9Gha, initial prod = crop energy + residue for
        # energy of all surfaces
        self.initial_production = 4.8 * density_per_ha * 3.6 * energy_crop_percentage   # in TWh

        # Our World in Data
        # https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions

        co2_gwp_100 = 1.0
        ch4_gwp_100 = 28.0
        n2o_gwp_100 = 265.0

        ghg_emissions_unit = 'kg/kg'  # in kgCo2eq per kg of food
        default_ghg_emissions = {'red meat': 21.56,
                                 'white meat': 4.41,
                                 'milk': 1.07,
                                 'eggs': 1.93,
                                 'rice and maize': 1.98,
                                 'cereals': 0.52,
                                 'fruits and vegetables': 0.51,
                                 GlossaryCore.Fish: 3.32,
                                 GlossaryCore.OtherFood: 0.93,
                                 }

        # Our World in Data
        # https://ourworldindata.org/carbon-footprint-food-methane
        ch4_emissions_unit = 'kg/kg'  # in kgCH4 per kg food
        default_ch4_emissions = {'red meat': 6.823e-1,
                                 'white meat': 1.25e-2,
                                 'milk': 3.58e-2,
                                 'eggs': 0.0,
                                 'rice and maize': 3.17e-2,
                                 # negligible methane in this category
                                 'cereals': 0.0,
                                 'fruits and vegetables': 0.0,
                                 # consider fish farm only
                                 GlossaryCore.Fish: 3.39e-2,
                                 GlossaryCore.OtherFood: 0.,
                                 }

        # FAO Stats
        # https://www.fao.org/faostat/en/#data/GT
        n2o_emissions_unit = 'kg/kg'  # in kgN2O per kg food$
        default_n2o_emissions = {'red meat': 9.268e-3,
                                 'white meat': 3.90e-4,
                                 'milk': 2.40e-4,
                                 'eggs': 1.68e-4,
                                 'rice and maize': 9.486e-4,
                                 'cereals': 1.477e-3,
                                 'fruits and vegetables': 2.63e-4,
                                 GlossaryCore.Fish: 0.,  # no crop or livestock related
                                 GlossaryCore.OtherFood: 1.68e-3,
                                 }

        co2_emissions_unit = 'kg/kg'  # in kgCO2 per kg food$
        # co2 emissions = (total_emissions - ch4_emissions * ch4_gwp_100 -
        # n2o_emissions * n2o_gwp_100)/co2_gwp_100
        # Difference method
        default_co2_emissions = {}
        for food in default_ghg_emissions:
            default_co2_emissions[food] = max(0., (default_ghg_emissions[food] - default_ch4_emissions[food] * ch4_gwp_100 -
                                           default_n2o_emissions[food] * n2o_gwp_100) / co2_gwp_100)

        # values taken from https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=OgMTTe
        # tab computekcalandkg for the design var to reach 2925.92 kcal/person/day
        red_meat_daily_cal = np.linspace(46.82, 46.82, year_range)
        white_meat_daily_cal = np.linspace(181.71, 181.71, year_range)
        vegetables_and_carbs_calories_per_day = np.linspace(1774.12, 1774.12, year_range)
        milk_and_eggs = np.linspace(237.98, 237.98, year_range)
        fish_daily_cal = np.linspace(39.02, 39.02, year_range)
        other_food_daily_cal = np.linspace(647.77, 647.77, year_range)
        self.red_meat_calories_per_day = pd.DataFrame({
                            GlossaryCore.Years: years,
                            'red_meat_calories_per_day': red_meat_daily_cal})
        self.white_meat_calories_per_day = pd.DataFrame({
                                GlossaryCore.Years: years,
                                'white_meat_calories_per_day': white_meat_daily_cal})
        self.veg_calories_per_day = pd.DataFrame({
                                GlossaryCore.Years: years,
                                'vegetables_and_carbs_calories_per_day': vegetables_and_carbs_calories_per_day})

        self.milk_eggs_calories_per_day = pd.DataFrame({
                                GlossaryCore.Years: years,
                                'milk_and_eggs_calories_per_day': milk_and_eggs})


        self.fish_calories_per_day = pd.DataFrame({
                                GlossaryCore.Years: years,
                                GlossaryCore.FishDailyCal: fish_daily_cal})
        self.other_food_calories_per_day = pd.DataFrame({
                                GlossaryCore.Years: years,
                                GlossaryCore.OtherDailyCal: other_food_daily_cal})


        self.param = {GlossaryCore.YearStart: self.year_start,
                      GlossaryCore.YearEnd: self.year_end,
                      'diet_df': self.diet_df,
                      'kg_to_kcal_dict': self.default_kg_to_kcal,
                      GlossaryCore.PopulationDfValue: self.population_df,
                      GlossaryCore.TemperatureDfValue: self.temperature_df,
                      'kg_to_m2_dict': self.default_kg_to_m2,
                      'param_a':  - 0.00833,
                      'param_b': - 0.04167,
                      'crop_investment': self.crop_investment,
                      'margin': self.margin,
                      'transport_margin': self.margin,
                      'transport_cost': self.transport_cost,
                      'data_fuel_dict': BiomassDry.data_energy_dict,
                      'techno_infos_dict': CropDiscipline.techno_infos_dict_default,
                      'scaling_factor_crop_investment': 1e3,
                      'scaling_factor_techno_consumption': 1e3,
                      'scaling_factor_techno_production': 1e3,
                      'initial_age_distrib': initial_age_distribution,
                      'initial_production': self.initial_production,
                      'red_meat_calories_per_day': self.red_meat_calories_per_day,
                      'white_meat_calories_per_day': self.white_meat_calories_per_day,
                      'vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                      'milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                      GlossaryCore.FishDailyCal: self.fish_calories_per_day,
                      GlossaryCore.OtherDailyCal: self.other_food_calories_per_day,
                      'co2_emissions_per_kg': default_co2_emissions,
                      'ch4_emissions_per_kg': default_ch4_emissions,
                      'n2o_emissions_per_kg': default_n2o_emissions,
                      'constraint_calories_limit': 1700.,
                      'constraint_calories_ref': 3400.,
                      GlossaryCore.CheckRangeBeforeRunBoolName: False,
                      }

    def test_crop_discipline(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'crop'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_REFERENCE: f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_biomass_dry': f'{name}.{model_name}',
                   'ns_land_use':f'{name}.{model_name}',
                   'ns_crop':f'{name}.{model_name}',
                   'ns_invest':f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Crop.DIET_DF}': self.diet_df,
                       f'{name}.{model_name}.{Crop.KG_TO_KCAL_DICT}': self.default_kg_to_kcal,
                       f'{name}.{model_name}.{Crop.KG_TO_M2_DICT}': self.default_kg_to_m2,
                       f'{name}.{model_name}.{Crop.POPULATION_DF}': self.population_df,
                       f'{name}.{model_name}.red_meat_calories_per_day': self.red_meat_calories_per_day,
                       f'{name}.{model_name}.white_meat_calories_per_day': self.white_meat_calories_per_day,
                       f'{name}.{model_name}.vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                       f'{name}.{model_name}.milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                       f'{name}.{model_name}.{GlossaryCore.FishDailyCal}': self.fish_calories_per_day,
                       f'{name}.{model_name}.{GlossaryCore.OtherDailyCal}': self.other_food_calories_per_day,
                       f'{name}.{model_name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{name}.{model_name}.crop_investment': self.crop_investment,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.transport_margin': self.margin,
                       f'{name}.{model_name}.transport_cost': self.transport_cost,
                       f'{name}.{model_name}.data_fuel_dict': BiomassDry.data_energy_dict,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        #outputs = disc.get_sosdisc_outputs() # to compare the emissions and land use with the ones computed on excel
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        #for graph in graph_list:
        #    graph.to_plotly().show()
