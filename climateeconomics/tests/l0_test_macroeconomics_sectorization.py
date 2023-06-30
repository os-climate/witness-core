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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pandas import read_csv
from pandas import DataFrame 
import numpy as np
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class MacroeconomicsTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2100
        nb_per = round(self.year_end - self.year_start + 1)
        self.nb_per = nb_per
        self.years = np.arange(self.year_start, self.year_end+1)

        total_invest = np.asarray([27.0] * nb_per)
        total_invest = DataFrame({'years':self. years, 'share_investment': total_invest})
        self.total_invest = total_invest

        indus_invest = np.asarray([6.8998] * nb_per)
        agri_invest = np.asarray([0.4522] * nb_per)
        services_invest = np.asarray([19.1818] * nb_per)
        share_sector_invest = DataFrame({'years': self.years, 'Industry': indus_invest, 'Agriculture': agri_invest, 'Services': services_invest})
        self.share_sector_invest = share_sector_invest

        # Test With a GDP and capital that grows at 2%
        gdp_year_start = 130.187
        capital_year_start = 376.6387
        gdp_serie = np.zeros(self.nb_per)
        capital_serie = np.zeros(self.nb_per)
        gdp_serie[0] =gdp_year_start
        capital_serie[0] = capital_year_start
        for year in np.arange(1, self.nb_per):
            gdp_serie[year] = gdp_serie[year - 1] * 1.02
            capital_serie[year] = capital_serie[year - 1] * 1.02
        #for each sector share of total gdp 2020
        gdp_agri = gdp_serie * 6.775773/100
        gdp_indus = gdp_serie * 28.4336/100
        gdp_service = gdp_serie * 64.79/100
        self.prod_agri = DataFrame({'years':self. years,'output': gdp_agri, 'output_net_of_damage': gdp_agri*0.995})
        self.prod_indus = DataFrame({'years':self. years,'output': gdp_indus, 'output_net_of_damage': gdp_indus*0.995})
        self.prod_service = DataFrame({'years':self. years,'output': gdp_service, 'output_net_of_damage': gdp_service*0.995})
        cap_agri = capital_serie * 0.018385
        cap_indus = capital_serie * 0.234987
        cap_service = capital_serie * 0.74662
        self.cap_agri_df = DataFrame({'years':self. years,'capital': cap_agri, 'usable_capital': cap_agri*0.8})
        self.cap_indus_df = DataFrame({'years':self. years,'capital': cap_indus, 'usable_capital': cap_indus*0.8})
        self.cap_service_df = DataFrame({'years':self. years,'capital': cap_service, 'usable_capital': cap_service*0.8})
        
#         self.param = {'year_start': self.year_start,
#                       'year_end': self.year_end,
#                       'agriculture.production_df':,
#                       'services.production_df': ,
#                       'industry.production_df': ,
#                       'industry.capital_df':,
#                       'services.capital_df':,
#                       'agriculture;capital_df':  
#                       }
        

    def test_macroeconomics_discipline(self):
        '''
        Check discipline setup and run
        '''
        name = 'Test'
        model_name = 'Macreconomics'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness':  f'{name}', 
                   'ns_macro': f'{name}.{model_name}' }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()
        inputs_dict = {f'{name}.year_start': self.year_start,
                       f'{name}.year_end': self.year_end,
                       f'{name}.total_investment_share_of_gdp': self.total_invest,
                       f'{name}.sectors_investment_share': self.share_sector_invest,
                       f'{name}.{model_name}.Agriculture.production_df': self.prod_agri,
                       f'{name}.{model_name}.Services.production_df': self.prod_service,
                       f'{name}.{model_name}.Industry.production_df': self.prod_indus,
                       f'{name}.{model_name}.Industry.capital_df': self.cap_indus_df,
                       f'{name}.{model_name}.Services.capital_df': self.cap_service_df,
                       f'{name}.{model_name}.Agriculture.capital_df':self.cap_agri_df,
                       }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#        for graph in graph_list:
#            graph.to_plotly().show()
