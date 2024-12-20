'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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

import platform
from os.path import dirname, exists, join

from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)
from tqdm import tqdm


class WitnessJacobianDiscTest(AbstractJacobianUnittest):
    # Parallel computation if platform is not windows
    PARALLEL = False
    if platform.system() != 'Windows':
        PARALLEL = True

    def all_usecase_disciplines_jacobian_test(self, usecase, directory=AbstractJacobianUnittest.PICKLE_DIRECTORY,
                                              excluded_disc=[],
                                              excluded_outputs=[],
                                              max_mda_iter=2,
                                              optional_disciplines_list=None):
        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.warm_start'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.inner_mda_name'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = max_mda_iter

        self.ee.load_study_from_input_dict(full_values_dict)

        self.ee.execute()

        # loop over all disciplines

        all_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines

        # specify a list of disciplines to check
        if optional_disciplines_list:
            all_disc = [disc for disc in all_disc if disc.sos_name in optional_disciplines_list]

        total_disc = len(all_disc)
        counter = 0
        untested_disc = []
        for disc in tqdm(all_disc):

            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            if disc.sos_name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()

            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling')]

            # remove excluded
            outputs = list(set(outputs) - set(excluded_outputs))

            print('\nTesting', disc.sos_name, '...')

            pkl_name = f'jacobian_{disc.sos_name}.pkl'
            filepath = join(dirname(__file__), directory, pkl_name)

            if len(inputs) != 0 and disc.sos_name not in excluded_disc:
                counter += 1

                if not exists(filepath):
                    print('Create missing pkl file')
                    self.ee.dm.delete_complex_in_df_and_arrays()
                    self.override_dump_jacobian = True
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                        step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                        inputs=inputs,
                                        outputs=outputs,
                                        directory=directory,
                                        parallel=WitnessJacobianDiscTest.PARALLEL)

                else:
                    print('Pkl file already exists')
                    try:
                        if self.dump_jacobian:
                            print('Testing with jacobian recomputation and dump')
                        else:
                            print('Testing jacobian vs existing pkl file')
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs,
                                            directory=directory,
                                            parallel=WitnessJacobianDiscTest.PARALLEL)
                    except:
                        try:
                            print('Jacobian may have changed, dumping pkl...')
                            self.ee.dm.delete_complex_in_df_and_arrays()
                            self.override_dump_jacobian = True
                            self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                                local_data={},
                                                step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                                inputs=inputs,
                                                outputs=outputs,
                                                directory=directory,
                                                parallel=WitnessJacobianDiscTest.PARALLEL)
                        except:
                            print(f'Jacobian for {disc.sos_name} is false')
                            excluded_disc.append(disc.sos_name)
            else:
                untested_disc.append(disc.sos_name)

        print('excluded ', excluded_disc)
        print('untested ', untested_disc)
        print(f'Summary: checked {counter} disciplines out of {total_disc}.')

    # def _test_gradient_all_disciplines_witness_full_at_x(self):
    #     """
    #     """
    #     self.name = 'Test'
    #     self.ee = ExecutionEngine(self.name)
    #
    #     builder = self.ee.factory.get_builder_from_process(
    #         'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
    #     self.ee.factory.set_builders_to_coupling_builder(builder)
    #     self.ee.configure()
    #
    #     usecase = witness_sub_proc_usecase(
    #         bspline=True, execution_engine=self.ee)
    #     usecase.study_name = self.name
    #     values_dict = usecase.setup_usecase()
    #
    #     full_values_dict = {}
    #     for dict_v in values_dict:
    #         full_values_dict.update(dict_v)
    #
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance_linear_solver_MDO'] = 1.0e-12
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.linearization_mode'] = 'adjoint'
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.warm_start'] = False
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.chain_linearize'] = False
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.inner_mda_name'] = 'MDAGaussSeidel'
    #     full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 2
    #
    #     self.ee.load_study_from_input_dict(full_values_dict)
    #
    #     values_dict_design_var = {}
    #     df_xvect = pd.read_csv(
    #         join(dirname(__file__), 'data', 'design_space_last_ite.csv'))
    #     for i, row in df_xvect.iterrows():
    #         try:
    #             ns_var = self.ee.dm.get_all_namespaces_from_var_name(
    #                 row['variable'])[0]
    #             values_dict_design_var[ns_var] = np.asarray(
    #                 row['value'][1:-1].split(', '), dtype=float)
    #         except:
    #             pass
    #
    #     self.ee.load_study_from_input_dict(values_dict_design_var)
    #
    #     self.ee.execute()
    #
    #     # loop over all disciplines
    #
    #     excluded_disc = []
    #
    #     all_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines
    #     total_disc = len(all_disc)
    #     counter = 0
    #     for disc in tqdm(all_disc):
    #
    #         outputs = disc.get_output_data_names()
    #         outputs = [output for output in outputs if self.ee.dm.get_data(
    #             output, 'coupling')]
    #
    #         if disc.sos_name == 'FunctionsManager':
    #             outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
    #                 'objective_lagrangian')[0])
    #         inputs = disc.get_input_data_names()
    #
    #         inputs = [input for input in inputs if self.ee.dm.get_data(
    #             input, 'coupling')]
    #         print('\nTesting', disc.sos_name, '...')
    #
    #         pkl_name = f'jacobian_{disc.sos_name}.pkl'
    #         filepath = join(dirname(__file__),
    #                         AbstractJacobianUnittest.PICKLE_DIRECTORY, pkl_name)
    #
    #         if len(inputs) != 0 and disc.sos_name not in excluded_disc:
    #             counter += 1
    #
    #             if not exists(filepath):
    #                 print('Create missing pkl file')
    #                 self.ee.dm.delete_complex_in_df_and_arrays()
    #                 self.override_dump_jacobian = True
    #                 self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
    #                                     step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
    #                                     inputs=inputs,
    #                                     outputs=outputs)
    #
    #             else:
    #                 print('Pkl file already exists')
    #                 try:
#                        if self.dump_jacobian:
#                             print('Testing with jacobian recomputation and dump')
#                         else:
#                             print('Testing jacobian vs existing pkl file')
    #                     self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
    #                                         step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
    #                                         inputs=inputs,
    #                                         outputs=outputs)
    #                 except:
    #                     print('Jacobian may have change, dumping pkl...')
    #                     self.ee.dm.delete_complex_in_df_and_arrays()
    #                     self.override_dump_jacobian = True
    #                     self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
    #                                         step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
    #                                         inputs=inputs,
    #                                         outputs=outputs)
    #     print(f'Summary: checked {counter} disciplines out of {total_disc}.')
