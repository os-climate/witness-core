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


class MacroEconomicsJacobianDiscTest(AbstractJacobianUnittest):
    AbstractJacobianUnittest.DUMP_JACOBIAN = False

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.data_dir = join(dirname(__file__), 'data')
        energy_supply_df_all = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep.csv'))
        damage_df_all = read_csv(
            join(self.data_dir, 'damage_data_onestep.csv'))

        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020]
        self.energy_supply_df = energy_supply_df_y[['years',
                                                    'cumulative_total_energy_supply']]

        self.energy_supply_df = self.energy_supply_df.rename(
            columns={'cumulative_total_energy_supply': 'Total production'})
        damage_df_y = damage_df_all[damage_df_all['years'] >= 2020]
        self.damage_df = damage_df_y[['years', 'damage_frac_output']]

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        self.population_df = read_csv(
            join(global_data_dir, 'population_df.csv'))

        # put manually the index
        self.years = np.arange(2020, 2101, 1)
        self.energy_supply_df.index = self.years
        self.damage_df.index = self.years
        self.population_df.index = self.years

        self.year_start = 2020
        self.year_end = 2100
        self.time_step = 1
        self.nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)

        energy_invest = np.asarray([2.6] * self.nb_per)
        self.total_invest = np.asarray([25.0] * self.nb_per)
        self.total_invest = DataFrame(
            {'years': self.years, 'share_investment': self.total_invest})
        self.share_energy_investment = DataFrame(
            {'years': self.years, 'share_investment': energy_invest})

        #- default CO2 tax
        self.default_CO2_tax = pd.DataFrame(
            {'years': self.years, 'CO2_tax': 50.0}, index=self.years)

        #- retrieve co2_emissions_gt input
        energy_supply_df_all = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        self.co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        self.co2_emissions_gt.index = self.years
        self.default_co2_efficiency = pd.DataFrame(
            {'years': self.years, 'CO2_tax_efficiency': 40.0}, index=self.years)

    def analytic_grad_entry(self):
        return [
            self.test_macro_economics_analytic_grad,
            self.test_macro_economics_analytic_grad_damageproductivity,
            self.test_macro_economics_energy_supply_negative_damageproductivity,
            self.test_macro_economics_analytic_grad_max_damage,
            self.test_macro_economics_analytic_grad_gigantic_invest,
            self.test_macro_economics_very_high_emissions
        ]

    def test_macro_economics_analytic_grad(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline.pkl', discipline=disc_techno, inputs=[
                            f'{self.name}.energy_production', f'{self.name}.damage_df', f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                            f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df', f'{self.name}.energy_investment'], step=1e-15, derr_approx='complex_step')

        # self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_temp1.pkl',
        #                    discipline=disc_techno, inputs=[f'{self.name}.damage_df'],
        #                                                     outputs=[f'{self.name}.economics_df'],
        #                                                     step=1e-15, derr_approx='complex_step',
        #                                                     )

    def test_macro_economics_energy_supply_negative_damageproductivity(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.energy_supply_df['Total production'] = - \
            self.energy_supply_df['Total production']

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # np.set_printoptions(threshold=1000000)
        # AbstractJacobianUnittest.DUMP_JACOBIAN=True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_energy_supply_negative_damageproductivity.pkl',
                            discipline=disc_techno, inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                                            f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                                            f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment'],
                            step=1e-15, derr_approx='complex_step')

    def test_macro_economics_analytic_grad_damageproductivity(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # AbstractJacobianUnittest.DUMP_JACOBIAN=True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_damageproductivity.pkl',
                            discipline=disc_techno,
                            inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df', f'{self.name}.energy_investment'], step=1e-15, derr_approx='complex_step')

    def test_macro_economics_analytic_grad_max_damage(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.damage_df['damage_frac_output'] = 0.9

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # AbstractJacobianUnittest.DUMP_JACOBIAN=True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_max_damage.pkl',
                            discipline=disc_techno,
                            inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df', f'{self.name}.energy_investment'], step=1e-15, derr_approx='complex_step')

    def test_macro_economics_analytic_grad_gigantic_invest(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_invest = np.asarray([60.0] * self.nb_per)
        total_invest = np.asarray([80.0] * self.nb_per)
        total_invest = DataFrame(
            {'years': self.years, 'share_investment': total_invest})
        share_energy_investment = DataFrame(
            {'years': self.years, 'share_investment': energy_invest})

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # AbstractJacobianUnittest.DUMP_JACOBIAN=True
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_gigantic_invest.pkl',
                            discipline=disc_techno,
                            inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df', f'{self.name}.energy_investment'], step=1e-15, derr_approx='complex_step')

    def test_macro_economics_very_high_emissions(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_df_all = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_high_CO2.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        co2_emissions_gt.index = self.years
        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_very_high_emissions.pkl',
                            discipline=disc_techno, inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                                            f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                                            f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment'],
                            step=1e-15, derr_approx='complex_step')

    def test_macro_economics_negativeco2_emissions(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_df_all = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_high_CO2.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        co2_emissions_gt.index = self.years

        co2_emissions_gt['Total CO2 emissions'] = [45.143527103270934, 50.42228659644606, 55.779659824483375, 56.41547837017959, 56.53784255467195, 56.43900544088292, 56.42533522117556, 56.46554778468026, 56.35804371384892, 56.11296510902577, 55.57946600291536, 54.96618807135573, 54.19580090328581, 52.92583405530353, 52.15743204279378, 51.32390362690408, 50.24118420614764, 48.89743826258721, 47.939851830263045, 46.43457938505807, 43.87127445423289, 41.746434584109956, 39.44846265515009, 36.0498788838304, 32.56621522334439, 30.041083140219786, 28.766286011420252, 26.3007071618495, 24.195099055884093, 23.373231504420716, 22.142688692652005, 21.26819920547946, 20.399276689891234, 19.46727578343429, 18.226517347776273, 13.595565011635488, 9.302786417948244, 5.709786437524023, 4.536494014265404, 3.556123644938279, 2.7456697070217224,
                                                   1.9793394190218219, 1.3390435521141348, 0.8035357151520224, 0.3833734375504325, 0.04081942984597913, -0.22163341759622449, -0.42412091679832215, -0.5815715311786763, -0.739517766011626, -0.8500944291138354, -0.8540443811445986, -0.8532941419675569, -0.8503259690387479, -0.8463610093221002, -0.8416809286282907, -0.8363833302600822, -0.8305585805441522, -0.8242899327300016, -0.8176534736662388, -0.8107185827839539, -0.8035477213519983, -0.7961961542703018, -0.7887118701170115, -0.7811216451531053, -0.7734832247683551, -0.765818912961112, -0.7581487271033238, -0.7504930108277099, -0.742873191681412, -0.7353118090766286, -0.7278323101366646, -0.7204589158636198, -0.7132164529855264, -0.7061301850411289, -0.6992255105343219, -0.6925272678358535, -0.6860595352682831, -0.6798455881466655, -0.6739078597930552, -0.6682679062364281]

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_negative_emissions.pkl',
                            discipline=disc_techno, inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                                            f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                                            f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment'],
                            step=1e-15, derr_approx='complex_step')

    def _test_macro_economics_negativeco2_tax(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.default_CO2_tax = pd.DataFrame(
            {'years': self.years, 'CO2_tax': np.linspace(50, -50, len(self.years))}, index=self.years)
        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.share_energy_investment': self.share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest,
                       f'{self.name}.CO2_taxes': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': self.default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': self.co2_emissions_gt,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        # AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_negative_co2_tax.pkl',
                            discipline=disc_techno, inputs=[f'{self.name}.energy_production', f'{self.name}.damage_df',
                                                            f'{self.name}.share_energy_investment', f'{self.name}.total_investment_share_of_gdp',
                                                            f'{self.name}.co2_emissions_Gt',  f'{self.name}.CO2_taxes', f'{self.name}.population_df'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment'],
                            step=1e-15, derr_approx='complex_step')


if '__main__' == __name__:
    cls = MacroEconomicsJacobianDiscTest()
    cls.setUp()
    cls._test_macro_economics_negativeco2_tax()
