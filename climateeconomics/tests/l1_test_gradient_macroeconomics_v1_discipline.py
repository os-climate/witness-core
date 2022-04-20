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
from scipy.interpolate import interp1d

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class MacroEconomicsJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2050
        self.time_step = 1
        self.years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        # -------------------------
        # csv data
        # energy production
        self.data_dir = join(dirname(__file__), 'data')
        brut_net = 1/1.45
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(self.year_start, self.year_end+1))
        energy_supply_values = energy_supply * brut_net 
        energy_supply_df = pd.DataFrame({'years': self.years, 'Total production': energy_supply_values})
        energy_supply_df.index = self.years
        energy_supply_df.loc[2021, 'Total production'] = 116.1036348
        self.energy_supply_df = energy_supply_df

        # -------------------------
        # csv data
        # co2 emissions
        energy_supply_csv = read_csv(join(self.data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv['years'] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv['years'] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        # energy production divided by 1e3 (scaling factor production)
        energy_supply_csv['cumulative_total_energy_supply'] = energy_supply_csv['cumulative_total_energy_supply'] / 1e3
        self.co2_emissions_gt = energy_supply_df.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        self.co2_emissions_gt.index = self.years
        for i in np.arange(2021, self.year_end+1): 
            emission_vefore =  self.co2_emissions_gt.loc[i-1, 'Total CO2 emissions']
            self.co2_emissions_gt.loc[i,'Total CO2 emissions'] = emission_vefore*(1.02)
        self.default_co2_efficiency = pd.DataFrame(
            {'years': self.years, 'CO2_tax_efficiency': 40.0}, index=self.years)
        # -------------------------
        # csv data
        # damage
        damage_csv = read_csv(join(self.data_dir, 'damage_data_onestep.csv'))
        # adapt lenght to the year range
        damage_df_start = damage_csv.loc[damage_csv['years'] >= self.year_start]
        damage_df_end = damage_csv.loc[damage_csv['years'] <= self.year_end]
        damage_df = pd.merge(damage_df_start, damage_df_end)
        self.damage_df = damage_df[['years', 'damage_frac_output']]
        self.damage_df.index = self.years
        # -------------------------
        # csv data
        # population
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        population_csv = read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df_start = population_csv.loc[population_csv['years'] >= self.year_start]
        population_df_end = population_csv.loc[population_csv['years'] <= self.year_end]
        self.population_df = pd.merge(population_df_start, population_df_end)
        self.population_df.index = self.years

        # energy invest divided by 1e2 (scaling factor invest)
        energy_invest = np.asarray([2.6] * self.nb_per)
        self.total_invest = np.asarray([27.0] * self.nb_per)
        self.total_invest = DataFrame(
            {'years': self.years, 'share_investment': self.total_invest})
        self.share_energy_investment = DataFrame(
            {'years': self.years, 'share_investment': energy_invest})

        # default CO2 tax
        self.default_CO2_tax = pd.DataFrame(
            {'years': self.years, 'CO2_tax': 50.0}, index = self.years)
        self.default_CO2_tax.loc[2020, 'CO2_tax'] = 5000.0
        self.default_CO2_tax.loc[2021, 'CO2_tax'] = 120.0
        
        #Population workforce
        self.working_age_population_df = pd.DataFrame(
            {'years': self.years, 'population_1570': 6300}, index=self.years)
        
        # energy_capital
        nb_per = len(self.years)
        energy_capital_year_start = 16.09
        energy_capital = []
        energy_capital.append(energy_capital_year_start)
        for year in np.arange(1, nb_per):
            energy_capital.append(energy_capital[year - 1] * 1.02)
        self.energy_capital = pd.DataFrame({'years': self.years, 'energy_capital': energy_capital})

    def analytic_grad_entry(self):
        return [
            self.test_macro_economics_analytic_grad,
            self.test_macro_economics_analytic_grad_damageproductivity,
            self.test_macro_economics_analytic_grad_max_damage,
            self.test_macro_economics_analytic_grad_gigantic_invest,
            self.test_macro_economics_very_high_emissions,
            self.test_macro_economics_negativeco2_emissions,
            self.test_macro_economics_negativeco2_tax
        ]

    def test_macro_economics_analytic_grad(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df, 
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_v1_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df',
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint', 
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_analytic_grad_damageproductivity(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_v1_discipline_grad_damageproductivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df',
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint',
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_analytic_grad_max_damage(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_v1_discipline_grad_max_damage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df',
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint',
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_analytic_grad_gigantic_invest(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_v1_discipline_grad_gigantic_invest.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df', 
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint',
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_very_high_emissions(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_csv = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_high_CO2.csv'))
        # adapt lenght to the year range
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv['years'] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv['years'] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        energy_supply_df["years"] = energy_supply_df['years']
        co2_emissions_gt = energy_supply_df.rename(
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_v1_discipline_very_high_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df',
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint',
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_negativeco2_emissions(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_csv = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_negative_CO2.csv'))
        # adapt lenght to the year range
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv['years'] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv['years'] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        energy_supply_df["years"] = energy_supply_df['years']
        co2_emissions_gt = energy_supply_df.rename(
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_v1_discipline_negative_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.share_energy_investment',
                                    f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.co2_emissions_Gt',
                                    f'{self.name}.CO2_taxes',
                                    f'{self.name}.population_df',
                                    f'{self.name}.working_age_population_df',
                                    f'{self.name}.energy_capital'],
                            outputs=[f'{self.name}.economics_df',
                                     f'{self.name}.energy_investment',
                                     f'{self.name}.pc_consumption_constraint',
                                     f'{self.name}.global_investment_constraint',
                                     f'{self.name}.emax_enet_constraint',
                                     f'{self.name}.delta_capital_objective',
                                     f'{self.name}.delta_capital_objective_weighted',
                                     f'{self.name}.delta_capital_constraint'])

    def test_macro_economics_negativeco2_tax(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics_v1.macroeconomics_v1_discipline.MacroeconomicsDiscipline'
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
                       f'{self.name}.working_age_population_df' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_v1_discipline_negative_co2_tax.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                             inputs=[f'{self.name}.energy_production',
                                     f'{self.name}.damage_df',
                                     f'{self.name}.share_energy_investment',
                                     f'{self.name}.total_investment_share_of_gdp',
                                     f'{self.name}.co2_emissions_Gt',
                                     f'{self.name}.CO2_taxes',
                                     f'{self.name}.population_df',
                                     f'{self.name}.working_age_population_df', 
                                     f'{self.name}.energy_capital'],
                             outputs=[f'{self.name}.economics_df',
                                      f'{self.name}.energy_investment',
                                      f'{self.name}.pc_consumption_constraint',
                                      f'{self.name}.global_investment_constraint',
                                      f'{self.name}.emax_enet_constraint',
                                      f'{self.name}.delta_capital_objective',
                                      f'{self.name}.delta_capital_objective_weighted',
                                      f'{self.name}.delta_capital_constraint'])


if '__main__' == __name__:
    cls = MacroEconomicsJacobianDiscTest()
    cls.setUp()
    cls.test_macro_economics_negativeco2_tax()
