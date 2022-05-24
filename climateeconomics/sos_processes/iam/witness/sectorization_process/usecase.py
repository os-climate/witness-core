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

    def __init__(self, year_start=2000, year_end=2020, time_step=1, name='', execution_engine=None):
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
       
        # Workforce
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')

        total_workforce_df = pd.read_csv(join(data_dir, 'workingage_population_df.csv'))
        #multiply ageworking pop by employment rate 
        workforce = total_workforce_df['population_1570']* 0.659 
        workforce = workforce[:self.nb_per]
        workforce_df_agri = pd.DataFrame({'years': years, 'workforce': workforce * 0.274})
        workforce_df_services = pd.DataFrame({'years': years, 'workforce': workforce * 0.509})
        workforce_df_indus = pd.DataFrame({'years': years, 'workforce': workforce * 0.217})
        
        #Invest
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
        energy_df_indus = pd.DataFrame({'years': years, 'Total production': energy_supply_values * 0.2894})
        energy_df_agri = pd.DataFrame({'years': years, 'Total production': energy_supply_values *  0.2136})
        energy_df_services = pd.DataFrame({'years': years, 'Total production': energy_supply_values * 0.37})
        
        #Damage
        damage_df = pd.DataFrame({'years': years, 'damages': np.zeros(self.nb_per), 'damage_frac_output': np.zeros(self.nb_per),
                                       'base_carbon_price': np.zeros(self.nb_per)})
        
        #Share invest
        share_invest = np.asarray([27.0] * self.nb_per)
        share_invest = pd.DataFrame({'years':years, 'share_investment': share_invest})
        share_invest_df = share_invest

        sect_input = {}
        sect_input[self.study_name + '.year_start'] = self.year_start
        sect_input[self.study_name + '.year_end'] = self.year_end
        
        sect_input[self.study_name + self.macro_name +'.Agriculture.workforce_df'] = workforce_df_agri
        sect_input[self.study_name + self.macro_name +'.Services.workforce_df'] = workforce_df_services
        sect_input[self.study_name + self.macro_name +'.Industry.workforce_df'] = workforce_df_indus
        
        sect_input[self.study_name + self.macro_name +'.Agriculture.sector_investment'] = agri_invest
        sect_input[self.study_name + self.macro_name +'.Services.sector_investment'] = services_invest
        sect_input[self.study_name + self.macro_name +'.Industry.sector_investment'] = indus_invest
        
        sect_input[self.study_name + self.macro_name +'.Industry.energy_production'] = energy_df_indus
        sect_input[self.study_name + self.macro_name +'.Agriculture.energy_production'] = energy_df_agri 
        sect_input[self.study_name + self.macro_name +'.Services.energy_production'] = energy_df_services
    
        sect_input[self.study_name + self.macro_name +'.Industry.damage_df'] = damage_df
        sect_input[self.study_name + self.macro_name +'.Agriculture.damage_df'] = damage_df
        sect_input[self.study_name + self.macro_name +'.Services.damage_df'] = damage_df
        
        sect_input[self.study_name + '.total_investment_share_of_gdp'] = share_invest_df

        setup_data_list.append(sect_input)

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

#         for graph in graph_list:
#             graph.to_plotly().show()
