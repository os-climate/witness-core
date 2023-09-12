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

from os.path import join, dirname, exists
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import \
    Study as witness_sub_proc_usecase
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS


class OptimSubprocessJacobianDiscTest(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return [self.test_01_gradient_subprocess_objective_over_design_var(),
                ]

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_01_gradient_subprocess_objective_over_design_var(self):
        """
        """
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process('climateeconomics.sos_processes.iam.witness',
                                                           'witness_optim_sub_process',
                                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
                                                           invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
                                                           process_level='dev')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(bspline=True,
                                           execution_engine=self.ee,
                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
                                           process_level='dev',
                                           )
        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(AbstractJacobianUnittest.PICKLE_DIRECTORY, 'optim_check_gradient_dev')

        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        self.ee.load_study_from_input_dict(full_values_dict)

        self.ee.execute()

        # loop over all disciplines

        coupling_disc = self.ee.root_process.proxy_disciplines[0]

        outputs = self.ee.dm.get_all_namespaces_from_var_name(
            'objective_lagrangian')
        inputs_name = [f'{energy}_{techno}_array_mix' for energy, techno_dict in DEFAULT_COARSE_TECHNO_DICT.items() for
                       techno in techno_dict['value']]
        inputs_name = [name.replace('.', '_') for name in inputs_name]
        inputs = []
        for name in inputs_name:
            inputs.extend(self.ee.dm.get_all_namespaces_from_var_name(name))

        pkl_name = f'jacobian_obj_vs_design_var.pkl'

        AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-15, derr_approx='finite_differences', threshold=1e-5,
                            local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                            inputs=inputs,
                            outputs=outputs)
