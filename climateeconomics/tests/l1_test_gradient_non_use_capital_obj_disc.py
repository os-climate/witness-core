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
from os.path import dirname


from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class NonUseCapitalObjJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):
        self.name = 'Test'
        self.model_name = 'non_use_capital'
        self.year_start = 2020
        self.year_end = 2100
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.year_range = self.year_end - self.year_start

        self.ee = ExecutionEngine(self.name)

        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy': f'{self.name}.EnergyMix',
                   'ns_ref': f'{self.name}',
                   'ns_ccs': f'{self.name}.CCUS',
                   'ns_forest': f'{self.name}.Agriculture.Forest', }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.non_use_capital_objective.non_use_capital_obj_discipline.NonUseCapitalObjectiveDiscipline'
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
        non_use_capital_fg = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                           'FossilGas': loss_fg})
        non_use_capital_ub = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                           'UpgradingBiogas': loss_ub})
        non_use_capital_rf = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                           'Refinery': loss_rf})
        non_use_capital_ft = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                           'FischerTropsch': loss_ft})
        non_use_capital_ct = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                           'CC_tech': loss_ct})
        non_use_capital_ref = pd.DataFrame({'years': np.arange(year_start, year_end + 1),
                                            'Forest': loss_ref})
        non_use_capital_obj_ref = 100.
        values_dict = {f'{self.name}.year_start': year_start,
                       f'{self.name}.year_end': year_end,
                       f'{self.name}.non_use_capital_obj_ref': non_use_capital_obj_ref,
                       f'{self.name}.energy_list': ['fuel.liquid_fuel', 'methane'],
                       f'{self.name}.ccs_list': ['carbon_capture'],
                       f'{self.name}.agri_capital_techno_list': ['Forest'],
                       f'{self.name}.EnergyMix.methane.technologies_list': ['FossilGas', 'UpgradingBiogas'],
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.technologies_list': ['Refinery', 'FischerTropsch'],
                       f'{self.name}.CCUS.carbon_capture.technologies_list': ['CC_tech'],
                       f'{self.name}.CCUS.carbon_capture.CC_tech.non_use_capital': non_use_capital_ct,
                       f'{self.name}.EnergyMix.methane.FossilGas.non_use_capital': non_use_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.non_use_capital': non_use_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.Refinery.non_use_capital': non_use_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.FischerTropsch.non_use_capital': non_use_capital_ft,
                       f'{self.name}.CCUS.carbon_capture.CC_tech.techno_capital': non_use_capital_ct,
                       f'{self.name}.EnergyMix.methane.FossilGas.techno_capital': non_use_capital_fg,
                       f'{self.name}.EnergyMix.methane.UpgradingBiogas.techno_capital': non_use_capital_ub,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.Refinery.techno_capital': non_use_capital_rf,
                       f'{self.name}.EnergyMix.fuel.liquid_fuel.FischerTropsch.techno_capital': non_use_capital_ft,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.gamma': 0.5,
                       f'{self.name}.Agriculture.Forest.non_use_capital': non_use_capital_ref,
                       f'{self.name}.Agriculture.Forest.techno_capital': non_use_capital_ref, }

        self.ee.load_study_from_input_dict(values_dict)

        self.disc_techno = self.ee.root_process.sos_disciplines[0]

    def analytic_grad_entry(self):
        return [
            self.test_01_grad_non_use_capital_objective
        ]

    def test_01_grad_non_use_capital_objective(self):

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_non_use_capital_objective.pkl', discipline=self.disc_techno, step=1e-15,
                            inputs=[f'{self.name}.EnergyMix.methane.FossilGas.non_use_capital',
                                    f'{self.name}.EnergyMix.methane.UpgradingBiogas.non_use_capital',
                                    f'{self.name}.CCUS.carbon_capture.CC_tech.non_use_capital',
                                    f'{self.name}.EnergyMix.fuel.liquid_fuel.Refinery.non_use_capital',
                                    f'{self.name}.EnergyMix.fuel.liquid_fuel.FischerTropsch.non_use_capital',
                                    f'{self.name}.Agriculture.Forest.non_use_capital',
                                    f'{self.name}.Agriculture.Forest.techno_capital'],
                            outputs=[f'{self.name}.non_use_capital_objective',
                                     f'{self.name}.energy_capital'],
                            derr_approx='complex_step')
