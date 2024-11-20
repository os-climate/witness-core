'''
Copyright 2024 Capgemini
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
from os.path import dirname

import numpy as np
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop_2.crop_disc_2 import (
    CropDiscipline,
)


class Crop2JacobianTestCase(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return []


    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test'
        self.model_name = 'crop_food'

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        self.crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.CropProductivityReductionName: np.linspace(3, 12, year_range),  # fake
        })

        self.damage_fraction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: np.linspace(0.43 /100., 12 / 100., year_range), # 2020 value
        })

        self.investments_food_types = pd.DataFrame({
            GlossaryCore.Years: self.years,  # 0.61 T$ (2020 value)
            **{food_type: 0.61 * GlossaryCore.crop_calibration_data['invest_food_type_share_start'][food_type] / 100. * 1000. for food_type in GlossaryCore.DefaultFoodTypes}  # convert to G$
        })
        self.workforce_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.SectorAgriculture: np.linspace(935., 935. * 1.2, year_range),  # millions of people (2020 value)
        })

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7870, 7870 * 1.2, year_range),  # millions of people (2020 value)
        })

        self.enegy_agri = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: 2591. /1000.,  # PWh, 2020 value
        })

        dict_inputs = {
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
        }
        dict_to_dataframes = {
            GlossaryCore.FoodTypeWasteAtProdAndDistribShareName: {
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
        food_types = list(list(dict_inputs.values())[0].keys())
        inputs_dict = {
            f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.name}.{GlossaryCore.CropProductivityReductionName}': self.crop_productivity_reduction,
            f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction,
            f'{self.name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}': self.enegy_agri,
            f'{self.name}.{self.model_name}.{GlossaryCore.FoodTypesName}': food_types,
            f'{self.name}.{GlossaryCore.FoodTypesInvestName}': self.investments_food_types,
        }
        for varname, default_dict_values_var in dict_to_dataframes.items():
            df = pd.DataFrame({
                GlossaryCore.Years: self.years,
                **default_dict_values_var
            })
            inputs_dict.update({f'{self.name}.{varname}': df})
            inputs_dict.update({f'{self.name}.{self.model_name}.{varname}': df})

        inputs_dict.update({f'{self.name}.{self.model_name}.{varname}': value for varname, value in dict_inputs.items()})



        self.inputs_dict = inputs_dict

        self.ee = ExecutionEngine(self.name)
        ns_dict = {
            'ns_public': self.name,
            GlossaryCore.NS_WITNESS: self.name,
            GlossaryCore.NS_CROP: f'{self.name}',
            'ns_sectors': f'{self.name}',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop_2.crop_disc_2.CropDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.coupling_inputs = [
            f'{self.name}.{GlossaryCore.CropProductivityReductionName}',
            f'{self.name}.{GlossaryCore.WorkforceDfValue}',
            f'{self.name}.{GlossaryCore.PopulationDfValue}',
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
            f'{self.name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}',
            f'{self.name}.{GlossaryCore.FoodTypesInvestName}',
        ]
        self.coupling_outputs = [
            f"{self.name}.{GlossaryCore.CropFoodLandUseName}",
            f"{self.name}.{GlossaryCore.CropFoodEmissionsName}",
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}",
            f"{self.name}.{self.model_name}.non_used_capital",
            f"{self.name}.{GlossaryCore.FoodTypeDeliveredToConsumersName}",
            f"{self.name}.{GlossaryCore.FoodTypeCapitalName}",
        ]
        self.coupling_outputs.extend(
            [f'{self.name}.{GlossaryCore.CropProdForEnergyName.format(stream)}' for stream in CropDiscipline.streams_energy_prod]
        )

    def test_crop_discipline_2(self):
        '''
        Check discipline setup and run
        '''
        self.ee.load_study_from_input_dict(self.inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
            pass

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_crop_discipline_2.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.coupling_inputs,
                            outputs=self.coupling_outputs)
