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
from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.core.core_forest.forest_v2 import Forest
from climateeconomics.glossarycore import GlossaryCore


class ForestJacobianDiscTest(AbstractJacobianUnittest):
    # np.set_printoptions(threshold=np.inf)

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_forest_analytic_grad
        ]

    def test_forest_analytic_grad(self):
        # deforestation do not reach the limits
        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2035
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        self.CO2_per_ha = 13000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        deforest_invest = np.linspace(10, 1, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest})
        self.reforestation_cost_per_ha = 13800

        construction_delay = 3

        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: [1.135081, 1.135081, 1.135081]})

        mw_invest = np.linspace(1, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - \
                                                1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_forest_v2_discipline.pkl',
                            local_data=disc_techno.local_data,
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[
                                f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.managed_wood_investment',
                            ],
                            outputs=[
                                f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                                f'{self.name}.{model_name}.CO2_land_emission_df',
                                f'{self.name}.Forest.techno_production',
                                f'{self.name}.Forest.techno_prices',
                                f'{self.name}.Forest.techno_consumption',
                                f'{self.name}.Forest.{GlossaryCore.TechnoConsumptionWithoutRatioValue}',
                                f'{self.name}.Forest.land_use_required',
                                f'{self.name}.Forest.CO2_emissions',
                                f'{self.name}.Forest.forest_lost_capital',
                            ]
                            )

    def test_forest_analytic_grad_unmanaged_limit(self):
        # deforestation reaches the unmanaged limits
        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2030
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(4, 4, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.linspace(5000., 1., len(years))})
        self.reforestation_cost_per_ha = 13800

        construction_delay = 3

        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})

        mw_invest = np.linspace(1, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - \
                                                1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_forest_v2_discipline_2.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[
                                f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.managed_wood_investment',
                            ],
                            outputs=[f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                                     f'{self.name}.Forest.land_use_required',
                                     f'{self.name}.{model_name}.CO2_land_emission_df',
                                     f'{self.name}.Forest.CO2_emissions',
                                     f'{self.name}.Forest.techno_production',
                                     f'{self.name}.Forest.techno_consumption',
                                     f'{self.name}.Forest.{GlossaryCore.TechnoConsumptionWithoutRatioValue}',
                                     f'{self.name}.Forest.techno_prices',
                                     f'{self.name}.Forest.forest_lost_capital',
                                     ]
                            )

    def test_forest_analytic_grad_managed_limit(self):
        # deforestation reaches the unmanaged and managed limits
        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2030
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(4, 4, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.linspace(5000., 1., len(years))})
        self.reforestation_cost_per_ha = 13800
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})

        mw_invest = np.linspace(1, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - \
                                                1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_forest_v2_discipline_3.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[
                                f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.managed_wood_investment',
                            ],
                            outputs=[f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                                     f'{self.name}.Forest.land_use_required',
                                     f'{self.name}.{model_name}.CO2_land_emission_df',
                                     f'{self.name}.Forest.CO2_emissions',
                                     f'{self.name}.Forest.techno_production',
                                     f'{self.name}.Forest.techno_consumption',
                                     f'{self.name}.Forest.{GlossaryCore.TechnoConsumptionWithoutRatioValue}',
                                     f'{self.name}.Forest.techno_prices',
                                     f'{self.name}.Forest.forest_lost_capital',
                                     ]
                            )

    def test_forest_analytic_grad_bigmanaged_limit(self):
        # deforestation reaches the unmanaged and managed limits in one time
        # (not progressive)
        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2030
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(4, 4, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.linspace(1000., 1., len(years))})
        self.reforestation_cost_per_ha = 13800
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})

        mw_invest = np.linspace(1, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - \
                                                1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_forest_v2_discipline_4.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[
                                f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
                                f'{self.name}.{model_name}.managed_wood_investment',
                            ],
                            outputs=[f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                                     f'{self.name}.Forest.land_use_required',
                                     f'{self.name}.{model_name}.CO2_land_emission_df',
                                     f'{self.name}.Forest.CO2_emissions',
                                     f'{self.name}.Forest.techno_production',
                                     f'{self.name}.Forest.techno_consumption',
                                     f'{self.name}.Forest.{GlossaryCore.TechnoConsumptionWithoutRatioValue}',
                                     f'{self.name}.Forest.techno_prices',
                                     f'{self.name}.Forest.forest_lost_capital',

                                     ]
                            )
