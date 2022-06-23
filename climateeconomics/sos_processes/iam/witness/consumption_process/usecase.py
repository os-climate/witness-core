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

from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.study_manager.study_manager import StudyManager

from os.path import join, dirname
from pandas import read_csv
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def update_dspace_with(dspace_dict, name, value, lower, upper):
    ''' type(value) has to be ndarray
    '''
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)
    dspace_dict['variable'].append(name)
    dspace_dict['value'].append(value.tolist())
    dspace_dict['lower_bnd'].append(lower)
    dspace_dict['upper_bnd'].append(upper)
    dspace_dict['dspace_size'] += len(value)


def update_dspace_dict_with(dspace_dict, name, value, lower, upper, activated_elem=None, enable_variable=True):
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)

    if activated_elem is None:
        activated_elem = [True] * len(value)
    dspace_dict[name] = {'value': value,
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable, 'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.macro_name = '.Macroeconomics'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):

        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        self.nb_per = round(self.year_end - self.year_start + 1)
       
        # data dir 
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')
        
        if self.year_start == 2000 and self.year_end == 2020: 
            data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data/sectorization_fitting')
            #Invest
            hist_invest = pd.read_csv(join(data_dir, 'hist_invest_sectors.csv'))
            agri_invest = pd.DataFrame({'years': hist_invest['years'], 'investment': hist_invest['Agriculture']})
            services_invest = pd.DataFrame({'years': hist_invest['years'], 'investment': hist_invest['Services']})
            indus_invest = pd.DataFrame({'years': hist_invest['years'], 'investment': hist_invest['Industry']})
            #Energy
            hist_energy = pd.read_csv(join(data_dir, 'hist_energy_sect.csv'))
            agri_energy = pd.DataFrame({'years': hist_energy['years'], 'Total production': hist_energy['Agriculture']})
            services_energy = pd.DataFrame({'years': hist_energy['years'], 'Total production': hist_energy['Services']})
            indus_energy = pd.DataFrame({'years': hist_energy['years'], 'Total production': hist_energy['Industry']})
            #Workforce
            hist_workforce = pd.read_csv(join(data_dir, 'hist_workforce_sect.csv'))
            agri_workforce = pd.DataFrame({'years': hist_workforce['years'], 'workforce': hist_workforce['Agriculture']})
            services_workforce = pd.DataFrame({'years': hist_workforce['years'], 'workforce': hist_workforce['Services']})
            indus_workforce = pd.DataFrame({'years': hist_workforce['years'], 'workforce': hist_workforce['Industry']})
            
        else:
            invest_init = 31.489
            invest_serie = np.zeros(self.nb_per)
            invest_serie[0] = invest_init
            for year in np.arange(1, self.nb_per):
                invest_serie[year] = invest_serie[year - 1] * 1.02
                agri_invest = pd.DataFrame({'years': years, 'investment': invest_serie * 0.0187})
                indus_invest = pd.DataFrame({'years': years, 'investment': invest_serie * 0.18737})
                services_invest = pd.DataFrame({'years': years, 'investment': invest_serie * 0.7939})
            #Energy
            brut_net = 1/1.45
            energy_outlook = pd.DataFrame({
                 'year': [2000, 2005, 2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
                 'energy': [118.112,134.122 ,149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
            f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
            #Find values for 2020, 2050 and concat dfs 
            energy_supply = f2(np.arange(self.year_start, self.year_end+1))
            energy_supply_values = energy_supply * brut_net 
            indus_energy = pd.DataFrame({'years': years, 'Total production': energy_supply_values * 0.2894})
            agri_energy = pd.DataFrame({'years': years, 'Total production': energy_supply_values *  0.2136})
            services_energy = pd.DataFrame({'years': years, 'Total production': energy_supply_values * 0.37})
            
            total_workforce_df = pd.read_csv(join(data_dir, 'workingage_population_df.csv'))
            #multiply ageworking pop by employment rate 
            workforce = total_workforce_df['population_1570']* 0.659 
            workforce = workforce[:self.nb_per]
            agri_workforce = pd.DataFrame({'years': years, 'workforce': workforce * 0.274})
            services_workforce = pd.DataFrame({'years': years, 'workforce': workforce * 0.509})
            indus_workforce = pd.DataFrame({'years': years, 'workforce': workforce * 0.217})
       
        #Damage
        damage_df = pd.DataFrame({'years': years, 'damages': np.zeros(self.nb_per), 'damage_frac_output': np.zeros(self.nb_per),
                                       'base_carbon_price': np.zeros(self.nb_per)})
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')
        temperature_df = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))
        residential_energy = np.linspace(21, 15, len(years))
        residential_energy_df = pd.DataFrame(
            {'years': years, 'residential_energy': residential_energy})
        energy_price = np.arange(200, 200 + len(years))
        energy_mean_price = pd.DataFrame(
            {'years': years, 'energy_price': energy_price})
        #Share invest
        share_invest = np.asarray([27.0] * self.nb_per)
        share_invest = pd.DataFrame({'years':years, 'share_investment': share_invest})
        share_invest_df = share_invest

        cons_input = {}
        cons_input[self.study_name + '.year_start'] = self.year_start
        cons_input[self.study_name + '.year_end'] = self.year_end
        
        cons_input[self.study_name + self.macro_name +'.Agriculture.workforce_df'] = agri_workforce
        cons_input[self.study_name + self.macro_name +'.Services.workforce_df'] = services_workforce
        cons_input[self.study_name + self.macro_name +'.Industry.workforce_df'] = indus_workforce
        
        cons_input[self.study_name + self.macro_name +'.Agriculture.sector_investment'] = agri_invest
        cons_input[self.study_name + self.macro_name +'.Services.sector_investment'] = services_invest
        cons_input[self.study_name + self.macro_name +'.Industry.sector_investment'] = indus_invest
        
        cons_input[self.study_name + self.macro_name +'.Industry.energy_production'] = indus_energy
        cons_input[self.study_name + self.macro_name +'.Agriculture.energy_production'] = agri_energy
        cons_input[self.study_name + self.macro_name +'.Services.energy_production'] = services_energy
    
        cons_input[self.study_name + self.macro_name +'.Industry.damage_df'] = damage_df
        cons_input[self.study_name + self.macro_name +'.Agriculture.damage_df'] = damage_df
        cons_input[self.study_name + self.macro_name +'.Services.damage_df'] = damage_df
        
        cons_input[self.study_name + '.total_investment_share_of_gdp'] = share_invest_df

        cons_input[self.study_name + '.temperature_df'] = temperature_df
        cons_input[self.study_name + '.residential_energy'] = residential_energy_df
        cons_input[self.study_name + '.energy_mean_price'] = energy_mean_price
        
        if self.year_start == 2000:
            cons_input[self.study_name + self.macro_name +'.Industry.capital_start'] = 37.15058 
            cons_input[self.study_name + self.macro_name +'.Agriculture.capital_start'] = 4.035565
            cons_input[self.study_name + self.macro_name +'.Services.capital_start'] = 139.1369
            cons_input[self.study_name +'.damage_to_productivity'] = False
            cons_input[self.study_name + self.macro_name +'.Services.init_output_growth'] = 0
            cons_input[self.study_name + self.macro_name +'.Agriculture.init_output_growth'] = 0
            cons_input[self.study_name + self.macro_name +'.Industry.init_output_growth'] = 0
            
            
        setup_data_list.append(cons_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()
    
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        # for graph in graph_list:
        #     graph.to_plotly().show()
