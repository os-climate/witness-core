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


class ServicesDiscTest(unittest.TestCase):
    '''
    Economic Manufacturer static model test case
    '''

    def setUp(self):
        '''
        Set up function
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        
        self.model_name = 'Services'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_macro': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)
        
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        # put manually the index
        years = np.arange(2020, 2101, 1)
        self.years = years

        year_start = 2020
        self.year_start = year_start
        year_end = 2100
        self.year_end = year_end
        time_step = 1
        self.time_step = time_step
        nb_per = round((year_end - year_start) / time_step + 1)
        self.nb_per = nb_per
       
        # input
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        total_workforce_df = read_csv(join(data_dir, 'workingage_population_df.csv'))
        total_workforce_df.index = years
        #multiply ageworking pop by employment rate and by % in services
        workforce = total_workforce_df['population_1570']* 0.659 * 0.509
        self.workforce_df = pd.DataFrame({'years': years, 'Services': workforce})

        #Energy_supply
        brut_net = 1/1.45
        share_indus = 0.37
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(year_start, year_end+1))
        energy_supply_values = energy_supply * brut_net * share_indus
        self.energy_supply_df = pd.DataFrame({'years': self.years, 'Total production': energy_supply_values})
        self.energy_supply_df.index = self.years
        #energy_supply_df.loc[2020, 'Total production'] = 91.936

        #Investment growth at 2% 
        init_value = 25
        invest_serie = []
        invest_serie.append(init_value)
        for year in np.arange(1, nb_per):
            invest_serie.append(invest_serie[year - 1] * 1.02)
        self.total_invest = pd.DataFrame({'years': years, 'investment': invest_serie})
        
        #damage
        self.damage_df = pd.DataFrame({'years': self.years, 'damages': np.zeros(self.nb_per), 'damage_frac_output': np.zeros(self.nb_per),
                                       'base_carbon_price': np.zeros(self.nb_per)})
        self.damage_df.index = self.years

    def test_execute(self):
        
        # out dict definition
        values_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.damage_to_productivity': True,
                       f'{self.name}.{self.model_name}.sector_investment': self.total_invest,
                       f'{self.name}.{self.model_name}.energy_production': self.energy_supply_df,
                       f'{self.name}.{self.model_name}.damage_df': self.damage_df,
                       f'{self.name}.workforce_df': self.workforce_df, 
                       f'{self.name}.{self.model_name}.capital_start': 273.1805902, #2019 value for test 
                       f'{self.name}.prod_function_fitting': False}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_execute_forfitting(self):
        
        # out dict definition
        values_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.damage_to_productivity': True,
                       f'{self.name}.{self.model_name}.sector_investment': self.total_invest,
                       f'{self.name}.{self.model_name}.energy_production': self.energy_supply_df,
                       f'{self.name}.{self.model_name}.damage_df': self.damage_df,
                       f'{self.name}.workforce_df': self.workforce_df, 
                       f'{self.name}.{self.model_name}.capital_start': 273.1805902, #2019 value for test 
                       f'{self.name}.prod_function_fitting': True,
                       f'{self.name}.{self.model_name}.energy_eff_max_range_ref' : 15
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
#         for graph in graph_list:
#             graph.to_plotly().show()

