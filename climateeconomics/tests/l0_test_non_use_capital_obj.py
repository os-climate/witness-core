'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/06-2023/11/03 Copyright 2023 Capgemini

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
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class NonUseCapitalObjDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'usecase_1_witness_coarse_fixed_gdp_wo_damage_wo_co2_tax.WITNESS_MDO.WITNESS_Eval.WITNESS'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):
        self.model_name = 'non_use_capital_dev'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_energy': f'{self.name}.EnergyMix',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}.CCUS',
                   'ns_agriculture': f'{self.name}.AgricultureMix',
                   'ns_forest': f'{self.name}.AgricultureMix.Forest',
                   'ns_invest': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.non_use_capital_objective.non_use_capital_obj_discipline.NonUseCapitalObjectiveDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        year_end = GlossaryCore.YearEndDefault
        year_start = GlossaryCore.YearStartDefault
        loss_fg = 12
        loss_ct = 2
        loss_ub = 22
        loss_rf = 16
        loss_ft = 4
        loss_ref = 3
        loss_reforest = 3
        non_use_capital_fg = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                           GlossaryEnergy.FossilGas: loss_fg})
        non_use_capital_ub = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                           GlossaryEnergy.UpgradingBiogas: loss_ub})
        non_use_capital_rf = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                           GlossaryEnergy.Refinery: loss_rf})
        non_use_capital_ft = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                           GlossaryEnergy.FischerTropsch: loss_ft})
        non_use_capital_ct = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                           'direct_air_capture.AmineScrubbing': loss_ct})
        non_use_capital_ref = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                            'Forest': loss_ref})
        forest_lost_capital = pd.DataFrame({GlossaryCore.Years: np.arange(year_start, year_end + 1),
                                            'reforestation': loss_reforest,
                                            'managed_wood': loss_reforest,
                                            'deforestation': loss_reforest})
        forest_lost_capital_cons_ref = 1
        forest_lost_capital_cons_limit = 10

        alpha, gamma = 0.5, 0.5
        non_use_capital_obj_ref = 100.
        delta_years = year_end + 1 - year_start
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.is_dev': True,
                       f'{self.name}.non_use_capital_obj_ref': non_use_capital_obj_ref,
                       f'{self.name}.{GlossaryCore.energy_list}': ['fuel.liquid_fuel', GlossaryEnergy.methane],
                       f'{self.name}.{GlossaryCore.ccs_list}': [GlossaryEnergy.carbon_capture],
                       f'{self.name}.agri_capital_techno_list': ['Forest'],
                       f'{self.name}.EnergyMix.methane.{GlossaryCore.techno_list}': [GlossaryEnergy.FossilGas, GlossaryEnergy.UpgradingBiogas],
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.{GlossaryCore.techno_list}': [GlossaryEnergy.Refinery, GlossaryEnergy.FischerTropsch],
                       f'{self.name}.CCUS.carbon_capture.{GlossaryCore.techno_list}': ['direct_air_capture.AmineScrubbing'],
                       f'{self.name}.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.non_use_capital': non_use_capital_ct,
                       f'{self.name}.EnergyMix.methane.{GlossaryEnergy.FossilGas}.non_use_capital': non_use_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.non_use_capital': non_use_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.{GlossaryEnergy.Refinery}.non_use_capital': non_use_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.{GlossaryEnergy.FischerTropsch}.non_use_capital': non_use_capital_ft,
                       f'{self.name}.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.techno_capital': non_use_capital_ct,
                       f'{self.name}.EnergyMix.methane.{GlossaryEnergy.FossilGas}.techno_capital': non_use_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.techno_capital': non_use_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.{GlossaryEnergy.Refinery}.techno_capital': non_use_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.{GlossaryEnergy.FischerTropsch}.techno_capital': non_use_capital_ft,
                       f'{self.name}.AgricultureMix.Forest.non_use_capital': non_use_capital_ref,
                       f'{self.name}.AgricultureMix.Forest.techno_capital': non_use_capital_ref,
                       f'{self.name}.AgricultureMix.Forest.forest_lost_capital': forest_lost_capital,
                       f'{self.name}.forest_lost_capital_cons_limit': forest_lost_capital_cons_limit,
                       f'{self.name}.forest_lost_capital_cons_ref': forest_lost_capital_cons_ref}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
           #graph.to_plotly().show()
           pass
