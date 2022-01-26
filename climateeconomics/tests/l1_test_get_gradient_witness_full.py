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
from os.path import join, dirname
import pandas as pd
import numpy as np

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import Study as witness_proc_usecase


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):

    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    obj_const = ['welfare_objective', 'min_utility_objective', 'temperature_objective', 'CO2_objective', 'ppm_objective', 'co2_emissions_objective', 'CO2_tax_minus_CO2_damage_constraint_df', 'EnergyMix.methane.demand_violation', 'EnergyMix.hydrogen.gaseous_hydrogen.demand_violation', 'EnergyMix.biogas.demand_violation', 'EnergyMix.syngas.demand_violation', 'EnergyMix.liquid_fuel.demand_violation',
                 'EnergyMix.solid_fuel.demand_violation', 'EnergyMix.biomass_dry.demand_violation', 'EnergyMix.electricity.demand_violation', 'EnergyMix.biodiesel.demand_violation', 'EnergyMix.hydrogen.liquid_hydrogen.demand_violation', 'primary_energies_production', 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return []

    def test_01_gradient_objective_lagrangian_wrt_design_var_on_witness_full(self):

        self.name = 'usecase_witness_optim_sub'
        self.ee = ExecutionEngine(self.name)

        repo = 'climateeconomics.sos_processes.iam.witness'
        chain_builders = self.ee.factory.get_builder_from_process(
            repo, 'witness_optim_sub_process')

        self.ee.factory.set_builders_to_coupling_builder(
            chain_builders)

        self.ee.configure()

        usecase = witness_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        values_dict[f'{self.name}.epsilon0'] = 1.0e-6
        values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        values_dict[f'{self.name}.linear_solver_MDA'] = 'GMRES_PETSC'
        values_dict[f'{self.name}.linear_solver_MDA_preconditioner'] = 'gasm'
        values_dict[f'{self.name}.linear_solver_MDO'] = 'GMRES_PETSC'
        values_dict[f'{self.name}.linear_solver_MDO_preconditioner'] = 'gasm'
        values_dict[f'{self.name}.max_mda_iter'] = 200
        values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.name}.tolerance'] = 1.0e-10
        values_dict[f'{self.name}.sub_mda_class'] = 'MDANewtonRaphson'

        func_df = values_dict[f'{self.name}.{usecase.coupling_name}.FunctionsManager.function_df']
        constraints = ['CO2_tax_minus_CO2_damage_constraint_df',
                       'methane.demand_violation',
                       'hydrogen.gaseous_hydrogen.demand_violation',
                       'biogas.demand_violation', 'syngas.demand_violation',
                       'liquid_fuel.demand_violation', 'solid_fuel.demand_violation',
                       'biomass_dry.demand_violation', 'electricity.demand_violation',
                       'biodiesel.demand_violation',
                       'hydrogen.liquid_hydrogen.demand_violation',
                       'primary_energies_production', 'CO2_tax_minus_CCS_constraint_df',
                       'carbon_to_be_stored_constraint',
                       'total_prod_minus_min_prod_constraint_df',
                       'prod_hydropower_constraint', 'total_prod_solid_fuel_elec',
                       'total_prod_h2_liquid', 'land_demand_constraint_df']
        for variable in constraints:
            func_df.loc[func_df['variable'] == variable, 'weight'] = -1.0
        values_dict[f'{self.name}.{usecase.coupling_name}.FunctionsManager.function_df'] = func_df

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.configure()
        # self.ee.execute()

        dm = self.ee.dm
#         self.ee.root_process.sos_disciplines[0].check_jacobian(linearization_mode='adjoint', inputs=[
#             'usecase_witness_optim_sub.WITNESS_Eval.WITNESS.EnergyMix.biodiesel.biodiesel_array_mix'], outputs=['usecase_witness_optim_sub.WITNESS_Eval.WITNESS.welfare_objective'])
        input_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']
        for energy in values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.energy_list']:
            energy_wo_dot = energy.replace('.', '_')
#             input_full_names.append(
#                 f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{energy_wo_dot}_array_mix')

            for technology in values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.technologies_list']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        a = self.ee.root_process.sos_disciplines[0].get_infos_gradient(
            ['usecase_witness_optim_sub.objective_lagrangian'], input_full_names)
        #output_var_list = usecase.func_df['variable'].values
#         output_var_full_list = [dm.get_all_namespaces_from_var_name(
#             var_name)[0] for var_name in output_var_list]
        print(a)
        # self.ee.execute()
        # linearize_res = self.ee.root_process.get_infos_gradient(output_var_full_list,
        #                                                        input_dv_full_list)
        df = pd.DataFrame(
            columns=['out_name', 'in_name', 'min', 'max', 'mean'])
        for k, v in a.items():
            for k_k, v_v in v.items():
                df = df.append({'out_name': k.split('.')[-1], 'in_name': k_k.split('.')[-1],
                                'min': v_v['min'], 'max': v_v['max'], 'mean': v_v['mean']}, ignore_index=True)
        df.to_csv('gradients_lagrangian_wrt_dv_allconstraints.csv')


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_01_gradient_objective_lagrangian_wrt_design_var_on_witness_full()
