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
from os.path import dirname, exists, join

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_dev_optim_process.usecase_witness_optim_invest_distrib import (
    Study as witness_usecase,
)
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)


class WitnessDevJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        '''

        '''
        return [
            self.test_01_gradient_dev_specific_disciplines,
        ]

    def test_01_gradient_dev_specific_disciplines(self):
        '''
        Test all the couplings for the disciplines in the "specific_disc" list on the witness_dev process
        '''

        specific_disciplines = ['Resources.coal_resource', 'Resources.oil_resource',
                                'Resources.natural_gas_resource',
                                'Resources.uranium_resource', 'Resources',
                                'Resources', 'Land.Land_Use', 'Land.Agriculture',
                                'Population', 'Land.Forest', ]

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_dev_optim_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_usecase(
            bspline=True, execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.warm_start'] = False
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.chain_linearize'] = False
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.sub_mda_class'] = 'MDANewtonRaphson'
        full_values_dict[f'{self.name}.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)

        for i in range(10):
            self.ee = ExecutionEngine(self.name)
            builder = self.ee.factory.get_builder_from_process(
                'climateeconomics.sos_processes.iam.witness', 'witness_dev_optim_process')
            self.ee.factory.set_builders_to_coupling_builder(builder)
            self.ee.configure()
            full_values_dict[f'{self.name}.max_mda_iter'] = i
            self.ee.load_study_from_input_dict(full_values_dict)
            self.ee.execute()
            for j, disc in enumerate(self.ee.root_process.proxy_disciplines):
                inputs = disc.get_input_data_names()
                inputs = [input for input in inputs if self.ee.dm.get_data(
                    input, 'coupling') and not input.endswith(GlossaryCore.ResourcesPriceValue) and not input.endswith(
                    'resources_CO2_emissions')]
                outputs = disc.get_output_data_names()
                outputs = [output for output in outputs if self.ee.dm.get_data(
                    output, 'coupling')]
                print('*************************************************')
                print(f'For discipline {disc.name} [{j}], check gradients of:')
                print('-------------------------------------------------')
                print('Inputs:')
                print(inputs)
                print('-------------------------------------------------')
                print('Outputs:')
                print(outputs)
                print('-------------------------------------------------')
                disc_underscored_name = disc.name.replace('.', '_')
                pkl_name = f'pickle_{disc_underscored_name}_ite{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_dev',
                                pkl_name)
                if (len(inputs) != 0) and (len(outputs) != 0):
                    self.ee.dm.delete_complex_in_df_and_arrays()
                    self.check_jacobian(location=dirname(__file__), filename='l2_witness_dev/' + pkl_name,
                                        discipline=disc,
                                        step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                        inputs=inputs,
                                        outputs=outputs)

    def test_02_gradient_dev_root_process(self):
        '''
        Test all the couplings for the root coupling discipline on the witness_dev process
        '''

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_v1')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_usecase(
            bspline=True, execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.warm_start'] = False
        full_values_dict[f'{self.name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)
        self.ee.execute()

        disc = self.ee.root_process
        inputs = disc.get_input_data_names()
        inputs = [input for input in inputs if self.ee.dm.get_data(
            input, 'coupling') and not input.endswith(GlossaryCore.ResourcesPriceValue) and not input.endswith(
            'resources_CO2_emissions')]
        outputs = disc.get_output_data_names()
        outputs = [output for output in outputs if self.ee.dm.get_data(
            output, 'coupling')]
        print('*************************************************')
        print(f'For discipline {disc.name}, check gradients of:')
        print('-------------------------------------------------')
        print('Inputs:')
        print(inputs)
        print('-------------------------------------------------')
        print('Outputs:')
        print(outputs)
        print('-------------------------------------------------')
        disc_underscored_name = disc.name.replace('.', '_')
        pkl_name = f'pickle_{disc_underscored_name}.pkl'
        filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_dev',
                        pkl_name)
        if (len(inputs) != 0) and (len(outputs) != 0):
            self.ee.dm.delete_complex_in_df_and_arrays()
            self.check_jacobian(location=dirname(__file__), filename='l2_witness_dev/' + pkl_name, discipline=disc,
                                step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                inputs=inputs,
                                outputs=outputs)

    def test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_v1')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_usecase(
            bspline=True, execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.warm_start'] = False
        full_values_dict[f'{self.name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        # df_xvect = pd.read_csv(
        #     join(dirname(__file__), 'data', 'design_space_last_ite_crash.csv'))
        # for i, row in df_xvect.iterrows():
        #     try:
        #         ns_var = self.ee.dm.get_all_namespaces_from_var_name(
        #             row['variable'])[0]
        #         values_dict_design_var[ns_var] = np.asarray(
        #             row['value'][1:-1].split(', '), dtype=float)
        #     except:
        #         pass
        # dspace_df = df_xvect

        # self.ee.load_study_from_input_dict(values_dict_design_var)
        self.ee.execute()
        i = 0

        for disc in self.ee.root_process.proxy_disciplines:
            #         disc = self.ee.dm.get_disciplines_with_name(
            #             f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix')[0]
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(output, 'coupling')
                       and not output.endswith('all_streams_demand_ratio')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(input, 'coupling')
                      and not input.endswith(GlossaryCore.ResourcesPriceValue)
                      and not input.endswith('resources_CO2_emissions')

                      and not input.endswith('all_streams_demand_ratio')]
            print(disc.name)
            print(i)
            if i in [0, 1, 2, 3, 4, 5, 6, 11, 81, 82, 83]:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_dev_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_dev',
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
            i += 1

    def test_06_gradient_each_discipline_on_dm_pkl(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_v1')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        # pkl_dict = pd.read_pickle(
        #     join(dirname(__file__), 'data', 'dm_3_ite.pkl'))
        # inp_dict = {key.replace('<study_ph>.WITNESS_MDO',
        # self.name): value for key, value in pkl_dict.items()}

        # self.ee.load_study_from_dict(inp_dict)
        i = 0

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            #         disc = self.ee.dm.get_disciplines_with_name(
            #             f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix')[0]
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling') and not input.endswith(GlossaryCore.ResourcesPriceValue) and not input.endswith(
                'resources_CO2_emissions')]
            print(disc.name)
            print(i)
            if i not in [6, 27, 53, 58, 62]:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_withx0_ite3nR_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_dev',
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):

                        self.ee.dm.delete_complex_in_df_and_arrays()

                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
            i += 1


if '__main__' == __name__:
    cls = WitnessDevJacobianDiscTest()
    cls.test_01_gradient_dev_specific_disciplines()
    # self.test_02_gradient_dev_root_process()
    # self.test_06_gradient_each_discipline_on_dm_pkl()
