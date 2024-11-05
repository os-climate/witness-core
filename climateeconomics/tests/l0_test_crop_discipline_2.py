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
from energy_models.glossaryenergy import GlossaryEnergy


class CropFoodTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2055
        self.years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        self.crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.CropProductivityReductionName: np.linspace(1.0, 5.0, year_range),
        })

        self.damage_fraction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: np.linspace(3, 9, year_range) / 100.,
        })

        self.investments = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.InvestmentsValue: np.linspace(5.0, 7.0, year_range),
        })
        self.workforce_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Population1570: 30.,
        })

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(8000, 10000, year_range),
        })

        self.enegy_agri = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: 30.,
        })

        self.default_kg_to_m2 = {GlossaryCore.RedMeat: 345.,
                            GlossaryCore.WhiteMeat: 14.5,
                            GlossaryCore.Milk: 8.95,
                            GlossaryCore.Eggs: 6.27,
                            GlossaryCore.RiceAndMaize: 2.89,
                            GlossaryCore.Cereals: 4.5,
                            GlossaryCore.FruitsAndVegetables: 0.8,
                            GlossaryCore.Fish: 0.,
                            GlossaryCore.OtherFood: 5.1041,
                            }
        # land use of other is provided in variable 'other_use_crop'

        self.default_kg_to_kcal = {
            GlossaryCore.RedMeat: 1551.05,
            GlossaryCore.WhiteMeat: 2131.99,
             GlossaryCore.Milk: 921.76,
             GlossaryCore.Eggs: 1425.07,
             GlossaryCore.RiceAndMaize: 2572.46,
             GlossaryCore.Cereals: 2964.99,
             GlossaryCore.FruitsAndVegetables: 559.65,
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

        self.food_types = list(self.diet_df.keys())

    def test_crop_discipline_2(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'crop_food'
        ee = ExecutionEngine(name)
        ns_dict = {
            'ns_public': name,
            GlossaryCore.NS_WITNESS: name,
            'ns_crop': f'{name}.{model_name}',
            'ns_food': f'{name}.{model_name}',
            'ns_sectors': f'{name}',
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc_2.CropDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        default_dict_values = {
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CO2): {
                GlossaryCore.RedMeat: 0.0,
                GlossaryCore.WhiteMeat: 3.95,
                GlossaryCore.Milk: 0.0,
                GlossaryCore.Eggs: 1.88,
                GlossaryCore.RiceAndMaize: 0.84,
                GlossaryCore.Cereals: 0.12,
                GlossaryCore.FruitsAndVegetables: 0.44,
                GlossaryCore.Fish: 2.37,
                GlossaryCore.OtherFood: 0.48
            },
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CH4): {
                GlossaryCore.RedMeat: 6.823e-1,
                GlossaryCore.WhiteMeat: 1.25e-2,
                GlossaryCore.Milk: 3.58e-2,
                GlossaryCore.Eggs: 0.0,
                GlossaryCore.RiceAndMaize: 3.17e-2,
                # negligible methane in this category
                GlossaryCore.Cereals: 0.0,
                GlossaryCore.FruitsAndVegetables: 0.0,
                # consider fish farm only
                GlossaryCore.Fish: 3.39e-2,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.N2O): {
                GlossaryCore.RedMeat: 9.268e-3,
                GlossaryCore.WhiteMeat: 3.90e-4,
                GlossaryCore.Milk: 2.40e-4,
                GlossaryCore.Eggs: 1.68e-4,
                GlossaryCore.RiceAndMaize: 9.486e-4,
                GlossaryCore.Cereals: 1.477e-3,
                GlossaryCore.FruitsAndVegetables: 2.63e-4,
                GlossaryCore.Fish: 0.,  # no crop or livestock related
                GlossaryCore.OtherFood: 1.68e-3,
            },
            GlossaryCore.FoodTypeKcalByProdUnitName: {
                GlossaryCore.RedMeat: 1551.05,
                GlossaryCore.WhiteMeat: 2131.99,
                GlossaryCore.Milk: 921.76,
                GlossaryCore.Eggs: 1425.07,
                GlossaryCore.RiceAndMaize: 2572.46,
                GlossaryCore.Cereals: 2964.99,
                GlossaryCore.FruitsAndVegetables: 559.65,
                GlossaryCore.Fish: 609.17,
                GlossaryCore.OtherFood: 3061.06,
            },
            GlossaryCore.FoodTypeLandUseByProdUnitName: {
                GlossaryCore.RedMeat: 345.,
                GlossaryCore.WhiteMeat: 14.5,
                GlossaryCore.Milk: 8.95,
                GlossaryCore.Eggs: 6.27,
                GlossaryCore.RiceAndMaize: 2.89,
                GlossaryCore.Cereals: 4.5,
                GlossaryCore.FruitsAndVegetables: 0.8,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 5.1041,
            },
            GlossaryCore.FoodTypeWasteAtProductionShareName: {
                GlossaryCore.RedMeat: 3,
                GlossaryCore.WhiteMeat: 3,
                GlossaryCore.Milk: 8,
                GlossaryCore.Eggs: 7,
                GlossaryCore.RiceAndMaize: 8,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 15,
                GlossaryCore.Fish: 10,
                GlossaryCore.OtherFood: 5,
            },
            GlossaryCore.FoodTypeWasteByConsumersShareName: {
                GlossaryCore.RedMeat: 3,
                GlossaryCore.WhiteMeat: 3,
                GlossaryCore.Milk: 8,
                GlossaryCore.Eggs: 7,
                GlossaryCore.RiceAndMaize: 8,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 15,
                GlossaryCore.Fish: 10,
                GlossaryCore.OtherFood: 5,
            },
            GlossaryCore.FoodTypeEnergyNeedName: {
                GlossaryCore.RedMeat: 1,
                GlossaryCore.WhiteMeat: 1,
                GlossaryCore.Milk: 1,
                GlossaryCore.Eggs: 1,
                GlossaryCore.RiceAndMaize: 1,
                GlossaryCore.Cereals: 1,
                GlossaryCore.FruitsAndVegetables: 1,
                GlossaryCore.Fish: 1,
                GlossaryCore.OtherFood: 1,
            },
            GlossaryCore.FoodTypeWorkforceNeedName: {
                GlossaryCore.RedMeat: 3,
                GlossaryCore.WhiteMeat: 3,
                GlossaryCore.Milk: 8,
                GlossaryCore.Eggs: 7,
                GlossaryCore.RiceAndMaize: 8,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 15,
                GlossaryCore.Fish: 10,
                GlossaryCore.OtherFood: 5,
            },
            GlossaryCore.ShareInvestFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.ShareEnergyUsageFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.ShareWorkforceFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.FoodTypeCapexName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 3,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 20.,
                GlossaryCore.Cereals: 30.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 10.,
                GlossaryCore.Cereals: 10.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 0.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 5,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 1.,
            },
            GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 0.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 20.,
            },
            GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 10.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 5.,
            },
        }
        inputs_dict = {
            f'{name}.{GlossaryCore.YearStart}': self.year_start,
            f'{name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{name}.{GlossaryCore.CropProductivityReductionName}': self.crop_productivity_reduction,
            f'{name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
            f'{name}.{GlossaryCore.PopulationDfValue}': self.population_df,
            f'{name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction,
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}': self.enegy_agri,
            f'{name}.{model_name}.{GlossaryCore.FoodTypesName}': self.food_types,
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}': self.investments,
        }
        for varname, default_dict_values_var in default_dict_values.items():
            df = pd.DataFrame({
                GlossaryCore.Years: self.years,
                **default_dict_values_var
            })
            inputs_dict.update({f'{name}.{model_name}.{varname}': df})

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
