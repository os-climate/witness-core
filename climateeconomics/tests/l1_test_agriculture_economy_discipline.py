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
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class JacobianTestCaseAgricultureEconomy(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return []


    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test'
        self.model_name = 'model'

        self.year_start = 2021
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1, 1)

        energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: 50.
        })

        ft_waste_by_prod_loss = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 2.5 for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        food_type_waste_by_climate_change = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 4.5 for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        food_type_delivered_to_consumers = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 60.5 for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        crop_prod_for_all_streams = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 60.5 for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        food_type_capital_breakdown = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 400. for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        food_type_invest_breakdown = pd.DataFrame({
            GlossaryCore.Years: self.years,
            **{ft: 400. * 0.1 for ft in GlossaryCore.DefaultFoodTypesV2}
        })

        inputs_dict = {
            f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price,
            f'{self.name}.{GlossaryCore.FoodTypeDeliveredToConsumersName}': food_type_delivered_to_consumers,
            f'{self.name}.{GlossaryCore.CropProdForAllStreamName}': crop_prod_for_all_streams,
            f'{self.name}.{GlossaryCore.FoodTypeNotProducedDueToClimateChangeName}': ft_waste_by_prod_loss,
            f'{self.name}.{GlossaryCore.FoodTypeWasteByClimateDamagesName}': food_type_waste_by_climate_change,
            f'{self.name}.{GlossaryCore.FoodTypeCapitalName}': food_type_capital_breakdown,
            f'{self.name}.{GlossaryCore.FoodTypesInvestName}': food_type_invest_breakdown,
        }

        self.inputs_dict = inputs_dict

        self.ee = ExecutionEngine(self.name)
        ns_dict = {
            'ns_public': self.name,
            GlossaryCore.NS_WITNESS: self.name,
            GlossaryCore.NS_CROP: f'{self.name}',
            'ns_sectors': f'{self.name}',
            GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_economy_discipline.AgricultureEconomyDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.coupling_inputs = [
            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
            f'{self.name}.{GlossaryCore.FoodTypeDeliveredToConsumersName}',
            f'{self.name}.{GlossaryCore.CropProdForAllStreamName}',
            f'{self.name}.{GlossaryCore.FoodTypeNotProducedDueToClimateChangeName}',
            f'{self.name}.{GlossaryCore.FoodTypeWasteByClimateDamagesName}',
            f'{self.name}.{GlossaryCore.FoodTypeCapitalName}',
            f'{self.name}.{GlossaryCore.FoodTypesInvestName}',
        ]
        self.coupling_outputs = [
            f"{self.name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}",
            f"{self.name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}",
        ]


    def test_agriculture_economy_discipline(self):
        '''
        Check discipline setup and run
        '''
        self.override_dump_jacobian = 1
        self.ee.load_study_from_input_dict(self.inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass
        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobianc_agriculture_economy_disc.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.coupling_inputs,
                            outputs=self.coupling_outputs)
