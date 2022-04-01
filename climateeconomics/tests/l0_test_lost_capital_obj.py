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
import unittest
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class LostCapitalObjDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'lost_capital'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy': f'{self.name}.EnergyMix',
                   'ns_ref': f'{self.name}',
                   'ns_ccs': f'{self.name}.CCUS',
                   'ns_forest': f'{self.name}.Forest', }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.lost_capital_objective.lost_capital_obj_discipline.LostCapitalObjectiveDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        year_end = 2100
        year_start = 2020
        loss_fg = 12
        loss_ct = 2
        loss_ub = 22
        loss_rf = 16
        loss_ft = 4
        loss_ref = 3
        lost_capital_fg = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                        'FossilGas': loss_fg})
        lost_capital_ub = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                        'UpgradingBiogas': loss_ub})
        lost_capital_rf = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                        'Refinery': loss_rf})
        lost_capital_ft = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                        'FischerTropsch': loss_ft})
        lost_capital_ct = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                        'CC_tech': loss_ct})
        lost_capital_ref = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                         'reforestation': loss_ref})
        lost_capital_obj_ref = 100.
        values_dict = {f'{self.name}.year_start': year_start,
                       f'{self.name}.year_end': year_end,
                       f'{self.name}.lost_capital_obj_ref': lost_capital_obj_ref,
                       f'{self.name}.energy_list': ['fuel.liquid_fuel', 'methane'],
                       f'{self.name}.ccs_list': ['carbon_capture'],
                       f'{self.name}.biomass_list': ['biomass_dry'],
                       f'{self.name}.EnergyMix.methane.technologies_list': ['FossilGas', 'UpgradingBiogas'],
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.technologies_list': ['Refinery', 'FischerTropsch'],
                       f'{self.name}.CCUS.carbon_capture.technologies_list': ['CC_tech'],
                       f'{self.name}.Forest.biomass_dry.technologies_list': ['reforestation'],
                       f'{self.name}.CCUS.carbon_capture.CC_tech.lost_capital': lost_capital_ct,
                       f'{self.name}.EnergyMix.methane.FossilGas.lost_capital': lost_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.lost_capital': lost_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.Refinery.lost_capital': lost_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.FischerTropsch.lost_capital': lost_capital_ft,
                       f'{self.name}.CCUS.carbon_capture.CC_tech.techno_capital': lost_capital_ct,
                       f'{self.name}.EnergyMix.methane.FossilGas.techno_capital': lost_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.techno_capital': lost_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.Refinery.techno_capital': lost_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.FischerTropsch.techno_capital': lost_capital_ft,
                       f'{self.name}.Forest.biomass_dry.reforestation.lost_capital': lost_capital_ref,
                       f'{self.name}.Forest.biomass_dry.reforestation.techno_capital': lost_capital_ref, }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        lost_capital_df = self.ee.dm.get_value(
            f'{self.name}.{self.model_name}.lost_capital_df')
        lost_capital_objective = self.ee.dm.get_value(
            f'{self.name}.lost_capital_objective')
        sum_lost_capital_th = loss_fg + loss_ub + \
            loss_rf + loss_ft + loss_ct + loss_ref
        self.assertListEqual([sum_lost_capital_th] * (year_end - year_start + 1),
                             lost_capital_df['Sum of lost capital'].values.tolist())

        self.assertEqual(lost_capital_objective,
                         sum_lost_capital_th * (year_end - year_start + 1) / lost_capital_obj_ref)
        print(lost_capital_objective)
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
