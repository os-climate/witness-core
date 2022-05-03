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
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
from os.path import join, dirname

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from scipy.interpolate import interp1d


class MacroDiscTest(unittest.TestCase):
    '''
    Economic Manufacturer static model test case
    '''

    def setUp(self):
        '''
        Set up function
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        # put manually the index
        years = np.arange(2020, 2101, 1)
        self.years = years

        year_start = 2020
        year_end = 2100
        time_step = 1
        nb_per = round(
            (year_end - year_start) / time_step + 1)
        self.nb_per = nb_per
        # Energy invest divided by 1e2 (scaling factor invest)
        energy_invest = np.asarray([2.6] * nb_per)

        total_invest = np.asarray([27.0] * nb_per)
        total_invest = DataFrame(
            {'years': years, 'share_investment': total_invest})
        share_energy_investment = DataFrame(
            {'years': years, 'share_investment': energy_invest})

        # Our world in data Direct primary energy conso data until 2019, then for 2020 drop in 6% according to IEA
        # then IEA data*0.91 (WEO 2020 stated) until 2040 then invented. 0.91 =
        # ratio net brut in 2020
        # Energy production divided by 1e3 (scaling factor production)
        brut_net = 1/1.45
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(year_start, year_end+1))
        energy_supply_values = energy_supply * brut_net 
        energy_supply_df = pd.DataFrame({'years': self.years, 'Total production': energy_supply_values})
        energy_supply_df.index = self.years
        energy_supply_df.loc[2021, 'Total production'] = 116.1036348

        self.damage_df = pd.DataFrame({'years': self.years, 'damages': np.zeros(self.nb_per), 'damage_frac_output': np.zeros(self.nb_per),
                                       'base_carbon_price': np.zeros(self.nb_per)})
        self.damage_df.index = self.years

        default_CO2_tax = pd.DataFrame(
            {'years': years, 'CO2_tax': 50.0}, index=years)
        
        # energy_capital
        nb_per = len(self.years)
        energy_capital_year_start = 16.09
        energy_capital = []
        energy_capital.append(energy_capital_year_start)
        for year in np.arange(1, nb_per):
            energy_capital.append(energy_capital[year - 1] * 1.02)
        self.energy_capital_df = pd.DataFrame({'years': self.years, 'energy_capital': energy_capital})

        # retrieve co2_emissions_gt input
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        population_df = read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        working_age_pop_df = read_csv(
            join(data_dir, 'workingage_population_df.csv'))
        working_age_pop_df.index = years
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})
        co2_emissions_gt.index = years
        default_co2_efficiency = pd.DataFrame(
            {'years': years, 'CO2_tax_efficiency': 40.0}, index=years)
    

        # out dict definition
        values_dict = {f'{self.name}.year_start': year_start,
                       f'{self.name}.year_end': year_end,
                       f'{self.name}.time_step': time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       # f'{self.name}.{self.model_name}.total_energy_capacity':
                       # 0.0,
                       f'{self.name}.share_energy_investment': share_energy_investment,
                       # f'{self.name}.share_non_energy_investment':
                       # share_non_energy_investment,
                       f'{self.name}.total_investment_share_of_gdp': total_invest,
                       f'{self.name}.energy_production': energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.population_df': population_df,
                       f'{self.name}.CO2_taxes': default_CO2_tax,
                       f'{self.name}.{self.model_name}.CO2_tax_efficiency': default_co2_efficiency,
                       f'{self.name}.co2_emissions_Gt': co2_emissions_gt,
                       f'{self.name}.working_age_population_df': working_age_pop_df, 
                       f'{self.name}.energy_capital': self.energy_capital_df
                       }

        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
#         for graph in graph_list:
#             graph.to_plotly().show()
