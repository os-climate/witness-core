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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pandas import read_csv
from climateeconomics.core.core_forest.forest_v2 import Forest
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine

import numpy as np
import pandas as pd


class ForestTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        self.CO2_per_ha = 4000
        # Mha
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 15, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})
        self.reforestation_cost_per_ha = 3800

        wood_density = 600.0  # kg/m3
        residues_density = 200.0  # kg/m3
        residue_density_m3_per_ha = 46.5
        # average of 360 and 600 divided by 5
        wood_density_m3_per_ha = 96
        construction_delay = 10
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
                                         'managed_wood_price_per_ha': 14872,  # 13047,
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
                                         'WACC': 0.07,
                                         # 1 tonne of tree absorbs 1.8t of CO2 in one
                                         # year
                                         # for a tree of 50 year, for 6.2tCO2/ha/year
                                         # it should be 3.49
                                         'CO2_from_production': - 0.425 * 44.01 / 12.0,
                                         'CO2_from_production_unit': 'kg/kg'
                                         }
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0), 'investment': np.array([1.135081] * construction_delay)})
        self.mw_initial_production = 1.25 * 0.92 * \
            density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)  # in Twh

        mw_invest = np.linspace(10, 15, year_range)
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

        deforest_invest = np.linspace(10, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {"years": years, "investment": deforest_invest})

        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end,
                      'time_step': self.time_step,
                      Forest.DEFORESTATION_INVESTMENT: self.deforest_invest_df,
                      Forest.DEFORESTATION_COST_PER_HA: 8000,
                      Forest.LIMIT_DEFORESTATION_SURFACE: self.limit_deforestation_surface,
                      Forest.CO2_PER_HA: self.CO2_per_ha,
                      Forest.INITIAL_CO2_EMISSIONS: self.initial_emissions,
                      Forest.REFORESTATION_INVESTMENT:  self.forest_invest_df,
                      Forest.REFORESTATION_COST_PER_HA:  self.reforestation_cost_per_ha,
                      'wood_techno_dict': self.managed_wood_techno_dict,
                      'managed_wood_initial_prod': self.mw_initial_production,
                      'managed_wood_initial_surface': 1.25 * 0.92,
                      'managed_wood_invest_before_year_start': self.invest_before_year_start,
                      'managed_wood_investment': self.mw_invest_df,
                      'transport_cost': self.transport_df,
                      'margin': self.margin,
                      'initial_unmanaged_forest_surface': self.initial_unsused_forest_surface,
                      'protected_forest_surface': self.initial_protected_forest_surface,
                      'scaling_factor_techno_consumption': 1e3,
                      'scaling_factor_techno_production': 1e3,
                      }

    def test_forest_model(self):
        '''
        Basique test of forest model
        Mainly check the overal run without value checks (will be done in another test)
        '''

        forest = Forest(self.param)

        forest.compute(self.param)

    def test_forest_discipline(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness': f'{name}.{model_name}',
                   'ns_functions': f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.year_start': self.year_start,
                       f'{name}.year_end': self.year_end,
                       f'{name}.time_step': 1,
                       f'{name}.{model_name}.{Forest.LIMIT_DEFORESTATION_SURFACE}': self.limit_deforestation_surface,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_prod': self.managed_wood_techno_dict,
                       f'{name}.{model_name}.wood_techno_dict': self.managed_wood_techno_dict,
                       f'{name}.{model_name}.managed_wood_initial_prod': self.mw_initial_production,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unsused_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()
