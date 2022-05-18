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
import numpy as np
import pandas as pd
from os.path import join, dirname
from climateeconomics.core.core_forest.forest_v2 import Forest

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class ForestJacobianDiscTest(AbstractJacobianUnittest):

    AbstractJacobianUnittest.DUMP_JACOBIAN = True
    #     np.set_printoptions(threshold=np.inf)

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_forest_analytic_grad
        ]

    def test_forest_analytic_grad(self):

        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness': f'{self.name}',
                   'ns_functions': f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = 2020
        self.year_end = 2035
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(10, 1, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {"years": years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})
        deforest_invest = np.linspace(10, 1, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {"years": years, "investment": deforest_invest})
        self.reforestation_cost_per_ha = 13800

        wood_density = 600.0  # kg/m3
        residues_density = 200.0  # kg/m3
        residue_density_m3_per_ha = 46.5
        # average of 360 and 600 divided by 5
        wood_density_m3_per_ha = 96
        construction_delay = 3
        wood_residue_price_percent_dif = 0.34
        wood_percentage_for_energy = 0.48
        residue_percentage_for_energy = 0.48

        density_per_ha = residue_density_m3_per_ha + \
            wood_density_m3_per_ha

        wood_percentage = wood_density_m3_per_ha / density_per_ha
        residue_percentage = residue_density_m3_per_ha / density_per_ha

        mean_density = wood_percentage * wood_density + \
            residue_percentage * residues_density
        years_between_harvest = 20

        recycle_part = 0.52  # 52%
        self.managed_wood_techno_dict = {'maturity': 5,
                                         'wood_residues_moisture': 0.35,  # 35% moisture content
                                         'wood_residue_colorific_value': 4.356,
                                         'Opex_percentage': 0.045,
                                         'managed_wood_price_per_ha': 15000,  # 13047,
                                         'unmanaged_wood_price_per_ha': 11000,  # 10483,
                                         'Price_per_ha_unit': 'euro/ha',
                                         'full_load_hours': 8760.0,
                                         'euro_dollar': 1.1447,  # in 2019, date of the paper
                                         'percentage_production': 0.52,
                                         'residue_density_percentage': residue_percentage,
                                         'non_residue_density_percentage': wood_percentage,
                                         'density_per_ha': density_per_ha,
                                         'wood_percentage_for_energy': wood_percentage_for_energy,
                                         'residue_percentage_for_energy': residue_percentage_for_energy,
                                         'density': mean_density,
                                         'wood_density': wood_density,
                                         'residues_density': residues_density,
                                         'density_per_ha_unit': 'm^3/ha',
                                         'techno_evo_eff': 'no',  # yes or no
                                         'years_between_harvest': years_between_harvest,
                                         'wood_residue_price_percent_dif': wood_residue_price_percent_dif,
                                         'recycle_part': recycle_part,
                                         'construction_delay': construction_delay,
                                         'WACC': 0.07
                                         }
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0), 'investment': [1.135081, 1.135081, 1.135081]})
        self.mw_initial_production = 1.25 * 0.92 * \
            density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)  # in Twh

        mw_invest = np.linspace(1, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {"years": years, "investment": mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {"years": years, "transport": transport})
        self.margin = pd.DataFrame(
            {'years': years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unsused_forest_surface = 4 - \
            1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{model_name}.{Forest.LIMIT_DEFORESTATION_SURFACE}': self.limit_deforestation_surface,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_initial_prod': self.managed_wood_techno_dict,
                       f'{self.name}.{model_name}.wood_techno_dict': self.managed_wood_techno_dict,
                       f'{self.name}.{model_name}.managed_wood_initial_prod': self.mw_initial_production,
                       f'{self.name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unsused_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_forest_v2_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[
            f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
            f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
            f'{self.name}.{model_name}.managed_wood_investment',
        ],
            outputs=[
                f'{self.name}.{Forest.FOREST_SURFACE_DF}',
            f'{self.name}.{model_name}.{Forest.CO2_EMITTED_FOREST_DF}',
            f'{self.name}.Forest.techno_production',
                                f'{self.name}.Forest.techno_prices',
                                f'{self.name}.Forest.techno_consumption',
                                f'{self.name}.Forest.techno_consumption_woratio',
                                f'{self.name}.Forest.land_use_required',
                                f'{self.name}.Forest.CO2_emissions',
                                f'{self.name}.Forest.techno_capital',
                                f'{self.name}.Forest.non_use_capital'
        ]
        )

    def _test_forest_analytic_grad_unmanaged_limit(self):

        model_name = 'Forest'
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness': f'{self.name}',
                   'ns_functions': f'{self.name}.{model_name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_agriculture': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}.{model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = 2020
        self.year_end = 2035
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(4, 4, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {"years": years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})
        deforest_invest = np.linspace(10, 1, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {"years": years, "investment": deforest_invest})
        self.reforestation_cost_per_ha = 13800

        wood_density = 600.0  # kg/m3
        residues_density = 200.0  # kg/m3
        residue_density_m3_per_ha = 46.5
        # average of 360 and 600 divided by 5
        wood_density_m3_per_ha = 96
        construction_delay = 3
        wood_residue_price_percent_dif = 0.34
        wood_percentage_for_energy = 0.48
        residue_percentage_for_energy = 0.48

        density_per_ha = residue_density_m3_per_ha + \
            wood_density_m3_per_ha

        wood_percentage = wood_density_m3_per_ha / density_per_ha
        residue_percentage = residue_density_m3_per_ha / density_per_ha

        mean_density = wood_percentage * wood_density + \
            residue_percentage * residues_density
        years_between_harvest = 25

        recycle_part = 0.52  # 52%
        self.managed_wood_techno_dict = {'maturity': 5,
                                         'wood_residues_moisture': 0.35,  # 35% moisture content
                                         'wood_residue_colorific_value': 4.356,
                                         'Opex_percentage': 0.045,
                                         'managed_wood_price_per_ha': 15000,  # 13047,
                                         'unmanaged_wood_price_per_ha': 11000,  # 10483,
                                         'Price_per_ha_unit': 'euro/ha',
                                         'full_load_hours': 8760.0,
                                         'euro_dollar': 1.1447,  # in 2019, date of the paper
                                         'percentage_production': 0.52,
                                         'residue_density_percentage': residue_percentage,
                                         'non_residue_density_percentage': wood_percentage,
                                         'density_per_ha': density_per_ha,
                                         'wood_percentage_for_energy': wood_percentage_for_energy,
                                         'residue_percentage_for_energy': residue_percentage_for_energy,
                                         'density': mean_density,
                                         'wood_density': wood_density,
                                         'residues_density': residues_density,
                                         'density_per_ha_unit': 'm^3/ha',
                                         'techno_evo_eff': 'no',  # yes or no
                                         'years_between_harvest': years_between_harvest,
                                         'wood_residue_price_percent_dif': wood_residue_price_percent_dif,
                                         'recycle_part': recycle_part,
                                         'construction_delay': construction_delay,
                                         'WACC': 0.07
                                         }
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0), 'investment': [1.135081, 1.135081, 1.135081]})
        self.mw_initial_production = 1.25 * 0.92 * \
            density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)  # in Twh

        mw_invest = np.linspace(10000, 10000, year_range)
        self.mw_invest_df = pd.DataFrame(
            {"years": years, "investment": mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {"years": years, "transport": transport})
        self.margin = pd.DataFrame(
            {'years': years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unsused_forest_surface = 4 - \
            1.25 - self.initial_protected_forest_surface

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{model_name}.{Forest.LIMIT_DEFORESTATION_SURFACE}': self.limit_deforestation_surface,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{self.name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{self.name}.{model_name}.managed_wood_initial_prod': self.managed_wood_techno_dict,
                       f'{self.name}.{model_name}.wood_techno_dict': self.managed_wood_techno_dict,
                       f'{self.name}.{model_name}.managed_wood_initial_prod': self.mw_initial_production,
                       f'{self.name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{self.name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{self.name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{self.name}.transport_cost': self.transport_df,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unsused_forest_surface,
                       f'{self.name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_forest_v2_discipline_2.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[
            f'{self.name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}',
            # f'{self.name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}',
            f'{self.name}.{model_name}.managed_wood_investment',
        ],
            outputs=[f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                     f'{self.name}.{model_name}.{Forest.CO2_EMITTED_FOREST_DF}',
                     f'{self.name}.Forest.techno_production',
                     f'{self.name}.Forest.techno_prices',
                     f'{self.name}.Forest.techno_consumption',
                     f'{self.name}.Forest.techno_consumption_woratio',
                     f'{self.name}.Forest.land_use_required',
                     f'{self.name}.Forest.CO2_emissions',
                     f'{self.name}.Forest.techno_capital',
                     f'{self.name}.Forest.non_use_capital'
                     ]
        )
