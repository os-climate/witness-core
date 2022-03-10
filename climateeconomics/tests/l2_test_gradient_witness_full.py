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
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as witness_usecase
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import Study as witness_sub_proc_usecase
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT, DEFAULT_COARSE_TECHNO_DICT_ccs_2, DEFAULT_COARSE_TECHNO_DICT_ccs_3
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from tqdm import tqdm


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):

    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):

        return [self.test_01_gradient_all_disciplines_witness_full(),
                self._test_02_gradient_all_disciplines_witness_full_at_x(),
                ]

    def test_01_gradient_all_disciplines_witness_full(self):
        """
        """
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.warm_start'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 2
        full_values_dict[f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.carbon_capture.flue_gas_effect'] = False

        self.ee.load_study_from_input_dict(full_values_dict)

        self.ee.execute()

        # loop over all disciplines

        excluded_disc = ['WITNESS.Resources.uranium_resource',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE',
                         'WITNESS.EnergyMix.biogas.AnaerobicDigestion',
                         'WITNESS.EnergyMix.liquid_fuel.Refinery',
                         'WITNESS.EnergyMix.liquid_fuel.FischerTropsch',
                         'WITNESS.EnergyMix.hydrotreated_oil_fuel.HefaDecarboxylation',
                         'WITNESS.EnergyMix.hydrotreated_oil_fuel.HefaDeoxygenation',
                         'WITNESS.EnergyMix.solid_fuel.CoalExtraction',
                         'WITNESS.EnergyMix.electricity.Nuclear',
                         'WITNESS.EnergyMix.electricity.CoalGen',
                         'WITNESS.EnergyMix.biodiesel.Transesterification',
                         'WITNESS.EnergyMix.hydrogen.liquid_hydrogen',
                         'WITNESS.EnergyMix',
                         'WITNESS.consumptionco2',
                         'WITNESS.CCUS.carbon_capture.direct_air_capture.AmineScrubbing',
                         'WITNESS.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing',
                         'WITNESS.CCUS.carbon_capture.flue_gas_capture.CalciumLooping',
                         'WITNESS.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine',
                         ]

        all_disc = self.ee.root_process.sos_disciplines[0].sos_disciplines
        total_disc = len(all_disc)
        counter = 0
        for disc in tqdm(all_disc):

            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(output, 'coupling')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()

            inputs = [input for input in inputs if self.ee.dm.get_data(input, 'coupling')]
            print('\nTesting', disc.name, '...')

            pkl_name = f'jacobian_{disc.name}.pkl'
            filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, pkl_name)

            if len(inputs) != 0 and disc.name not in excluded_disc:
                counter += 1

                if not exists(filepath):
                    print('Create missing pkl file')
                    self.ee.dm.delete_complex_in_df_and_arrays()
                    AbstractJacobianUnittest.DUMP_JACOBIAN = True
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                        step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                        inputs=inputs,
                                        outputs=outputs)

                else:
                    print('Pkl file already exists')
                    try:
                        print('Testing jacobian vs existing pkl file')
                        AbstractJacobianUnittest.DUMP_JACOBIAN = False
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                            inputs=inputs,
                                            outputs=outputs)
                    except:
                        print('Jacobian may have changed, redumping pkl...')
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        AbstractJacobianUnittest.DUMP_JACOBIAN = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                            inputs=inputs,
                                            outputs=outputs)

        print(f'Summary: checked {counter} disciplines out of {total_disc}.')

    def _test_02_gradient_all_disciplines_witness_full_at_x(self):
        """
        """
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.warm_start'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 2
        full_values_dict[f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.carbon_capture.flue_gas_effect'] = False

        self.ee.load_study_from_input_dict(full_values_dict)

        values_dict_design_var = {}
        df_xvect = pd.read_csv(join(dirname(__file__), 'data', 'design_space_last_ite.csv'))
        for i, row in df_xvect.iterrows():
            try:
                ns_var = self.ee.dm.get_all_namespaces_from_var_name(
                    row['variable'])[0]
                values_dict_design_var[ns_var] = np.asarray(
                    row['value'][1:-1].split(', '), dtype=float)
            except:
                pass

        self.ee.load_study_from_input_dict(values_dict_design_var)

        self.ee.execute()

        # loop over all disciplines

        excluded_disc = ['WITNESS.Resources.uranium_resource',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM',
                         'WITNESS.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE',
                         'WITNESS.EnergyMix.biogas.AnaerobicDigestion',
                         'WITNESS.EnergyMix.liquid_fuel.Refinery',
                         'WITNESS.EnergyMix.liquid_fuel.FischerTropsch',
                         'WITNESS.EnergyMix.hydrotreated_oil_fuel.HefaDecarboxylation',
                         'WITNESS.EnergyMix.hydrotreated_oil_fuel.HefaDeoxygenation',
                         'WITNESS.EnergyMix.solid_fuel.CoalExtraction',
                         'WITNESS.EnergyMix.electricity.Nuclear',
                         'WITNESS.EnergyMix.electricity.CoalGen',
                         'WITNESS.EnergyMix.biodiesel.Transesterification',
                         'WITNESS.EnergyMix.hydrogen.liquid_hydrogen',
                         'WITNESS.EnergyMix',
                         'WITNESS.consumptionco2',
                         'WITNESS.CCUS.carbon_capture.direct_air_capture.AmineScrubbing',
                         'WITNESS.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing',
                         'WITNESS.CCUS.carbon_capture.flue_gas_capture.CalciumLooping',
                         'WITNESS.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine',
                         ]

        all_disc = self.ee.root_process.sos_disciplines[0].sos_disciplines
        total_disc = len(all_disc)
        counter = 0
        for disc in tqdm(all_disc):

            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(output, 'coupling')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()

            inputs = [input for input in inputs if self.ee.dm.get_data(input, 'coupling')]
            print('\nTesting', disc.name, '...')

            pkl_name = f'jacobian_{disc.name}.pkl'
            filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, pkl_name)

            if len(inputs) != 0 and disc.name not in excluded_disc:
                counter += 1

                if not exists(filepath):
                    print('Create missing pkl file')
                    self.ee.dm.delete_complex_in_df_and_arrays()
                    AbstractJacobianUnittest.DUMP_JACOBIAN = True
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                        step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                        inputs=inputs,
                                        outputs=outputs)

                else:
                    print('Pkl file already exists')
                    try:
                        print('Testing jacobian vs existing pkl file')
                        AbstractJacobianUnittest.DUMP_JACOBIAN = False
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                            inputs=inputs,
                                            outputs=outputs)
                    except:
                        print('Jacobian may have changed, redumping pkl...')
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        AbstractJacobianUnittest.DUMP_JACOBIAN = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                                            inputs=inputs,
                                            outputs=outputs)

        print(f'Summary: checked {counter} disciplines out of {total_disc}.')


if '__main__' == __name__:
    AbstractJacobianUnittest.DUMP_JACOBIAN = True
    cls = WitnessFullJacobianDiscTest()
    cls.setUp()
    cls.test_01_gradient_all_disciplines_witness_full()
