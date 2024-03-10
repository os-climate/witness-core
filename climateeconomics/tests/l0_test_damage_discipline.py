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
from os.path import join, dirname

import numpy as np
import pandas as pd
from pandas import read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class DamageDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'damage'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')


        temperature_df_all = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))

        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        years = temperature_df_y[GlossaryCore.Years]
        damage_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.Damages: np.linspace(40, 60, len(temperature_df_y)),
            GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(temperature_df_y))
        })

        extra_co2_t_since_preindustrial = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(years))
        })

        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault +1, 1)
        temperature_df_y.index = years

        values_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.assumptions_dict': {'compute_gdp': True,
                                'compute_climate_impact_on_gdp': False,
                                'activate_climate_effect_population': True,
                                'activate_pandemic_effect_population': True,
                                'invest_co2_tax_in_renewables': True,
                                },
                       f'{self.name}.co2_damage_price_dev_formula': False,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': damage_df,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df_y,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': extra_co2_t_since_preindustrial,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate((np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15)))),
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False,
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

        for graph in graph_list:
           #graph.to_plotly().show()
           pass

    def test_execute_dev_formula(self):

        self.model_name = 'damage'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')


        temperature_df_all = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))

        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        years = temperature_df_y[GlossaryCore.Years]
        damage_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.Damages: np.linspace(40, 60, len(temperature_df_y)),
            GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(temperature_df_y))
        })

        extra_co2_t_since_preindustrial = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(years))
        })

        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault +1, 1)
        temperature_df_y.index = years

        values_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.assumptions_dict': {'compute_gdp': True,
                                'compute_climate_impact_on_gdp': False,
                                'activate_climate_effect_population': True,
                                'activate_pandemic_effect_population': True,
                                'invest_co2_tax_in_renewables': True,
                                },
                       f'{self.name}.co2_damage_price_dev_formula': True,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': damage_df,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df_y,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': extra_co2_t_since_preindustrial,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate((np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15))))
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

        for graph in graph_list:
           #graph.to_plotly().show()
           pass
