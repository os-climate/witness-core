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
from climateeconomics.core.core_deforest.deforest import Deforest

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class ForestJacobianDiscTest(AbstractJacobianUnittest):

    AbstractJacobianUnittest.DUMP_JACOBIAN = False

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_damage_analytic_grad
        ]

    def test_deforestation_forests(self):

        model_name = 'Test'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_deforestation': f'{self.name}.{model_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_deforest.deforest.deforest_disc.DeforestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = 2020
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_surface = np.array(np.linspace(3.4, 2.8, year_range))
        self.forest_df = pd.DataFrame(
            {"years": years, "forest_surface": forest_surface})
        deforestation_rate = np.array(np.linspace(-1, 0.5, year_range))
        self.deforestation_rate_df = pd.DataFrame(
            {"years": years, "forest_evolution": deforestation_rate})
        self.CO2_per_ha = 5000

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.forest_df': self.forest_df,
                       f'{self.name}.{Deforest.DEFORESTATION_RATE_DF}': self.deforestation_rate_df,
                       f'{self.name}.{model_name}.{Deforest.CO2_PER_HA}': self.CO2_per_ha,
                       }
        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_deforestation_discipline.pkl',
                            discipline=disc_techno, step=1e-15,
                            inputs=[f'Test.forest_df',
                                    f'{self.name}.{Deforest.DEFORESTATION_RATE_DF}'],
                            outputs=[f'{self.name}.{Deforest.DEFORESTED_SURFACE_DF}', f'{self.name}.{Deforest.NON_CAPTURED_CO2_DF}'], derr_approx='complex_step')
