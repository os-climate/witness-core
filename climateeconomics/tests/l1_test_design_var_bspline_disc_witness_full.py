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
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from energy_models.core.stream_type.carbon_models.carbon_capture import CarbonCapture
from energy_models.core.stream_type.carbon_models.carbon_storage import CarbonStorage


class DesignVarDisc(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return [
            self.test_derivative
        ]

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'DesignVar'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_optim': f'{self.name}',
                   'ns_ccs': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.core.design_variables_translation.witness_bspline.design_var_disc.Design_Var_Discipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        co2_tax_array = list(np.linspace(50.0, 150.0, 8))
        energy_mix_array = list(np.linspace(0.0, 100.0, 8))
        livestock_usage_factor_array = list(np.linspace(0.0, 100.0, 8))
        ccs_percentage_array = list(np.linspace(0.0, 10.0, 8))
        years = np.arange(2020, 2101, 1)
        self.energy_list = ['methane', 'liquid_fuel', 'electricity']
        self.ccs_list = [CarbonCapture.name, CarbonStorage.name]

        values_dict = {f'{self.name}.CO2_taxes_array': co2_tax_array,
                       f'{self.name}.ccs_percentage_array': ccs_percentage_array,
                       f'{self.name}.livestock_usage_factor_array': livestock_usage_factor_array,
                       f'{self.name}.energy_list': self.energy_list,
                       f'{self.name}.ccs_list': self.ccs_list,
                       f'{self.name}.methane.technologies_list': ['FossilGas', 'UpgradingBioGas'],
                       f'{self.name}.liquid_fuel.technologies_list': ['Refinery', 'FischerTropsch'],
                       f'{self.name}.carbon_capture.technologies_list': ['Capture1', 'Capture2'],
                       f'{self.name}.carbon_storage.technologies_list': ['Storage1', 'Storage2'],
                       f'{self.name}.electricity.technologies_list': ['CoalGen', 'Nuclear', 'SolarPV']}
        self.input_names = []
        ddict = {}
        for energy in self.energy_list + self.ccs_list:
            energy_wo_dot = energy.replace('.', '_')
            invest_mix_name = f'{self.name}.{energy}.{energy_wo_dot}_array_mix'
            invest_mix_name_wo = f'{energy_wo_dot}_array_mix'
            values_dict[invest_mix_name] = energy_mix_array
            ddict[invest_mix_name_wo] = {'value': energy_mix_array,
                                         'lower_bnd': 1, 'upper_bnd': 100, 'enable_variable': True, 'activated_elem': [True, True, True, True, True, True, True]}
            self.input_names.append(invest_mix_name)
            for techno in values_dict[f'{self.name}.{energy}.technologies_list']:
                techno_wo_dot = techno.replace('.', '_')
                invest_mix_name = f'{self.name}.{energy}.{techno}.{energy_wo_dot}_{techno_wo_dot}_array_mix'
                invest_mix_name_wo = f'{energy_wo_dot}_{techno_wo_dot}_array_mix'

                ddict[invest_mix_name_wo] = {'value': energy_mix_array,
                                             'lower_bnd': 1, 'upper_bnd': 100, 'enable_variable': True, 'activated_elem': [True, True, True, True, True, True, True]}
                values_dict[invest_mix_name] = energy_mix_array
                self.input_names.append(invest_mix_name)

        ddict['CO2_taxes_array'] = {'value': co2_tax_array,
                                    'lower_bnd': 50.0, 'upper_bnd': 100.0, 'enable_variable': True, 'activated_elem': [True, True, True, True, True, True, True]}
        ddict['ccs_percentage_array'] = {'value': ccs_percentage_array,
                                         'lower_bnd': 0.0, 'upper_bnd': 20.0, 'enable_variable': True, 'activated_elem': [True, True, True, True, True, True, True]}

        ddict['livestock_usage_factor_array'] = {'value': livestock_usage_factor_array,
                                                 'lower_bnd': 50.0, 'upper_bnd': 100.0, 'enable_variable': True, 'activated_elem': [True, True, True, True, True, True, True]}

        dspace_df_columns = ['variable', 'value', 'lower_bnd',
                             'upper_bnd', 'enable_variable', 'activated_elem']
        dspace_df = pd.DataFrame(columns=dspace_df_columns)

        for key, elem in ddict.items():
            dict_var = {'variable': key}
            dict_var.update(elem)
            dspace_df = dspace_df.append(dict_var, ignore_index=True)

        values_dict.update({f'{self.name}.design_space': dspace_df})

        self.ee.load_study_from_input_dict(values_dict)

    def test_execute(self):

        self.ee.execute()

    def test_derivative(self):
        disc_techno = self.ee.root_process.sos_disciplines[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        output_names = [f'{self.name}.invest_energy_mix',
                        f'{self.name}.livestock_usage_factor_df']
        output_names.extend(
            [f'{self.name}.{energy}.invest_techno_mix' for energy in self.energy_list])
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_design_var_bspline_full.pkl', discipline=disc_techno, step=1e-15, inputs=self.input_names,
                            outputs=output_names, derr_approx='complex_step')
