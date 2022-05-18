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
from pandas import DataFrame
from os.path import join, dirname


from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class MacroeconomicsJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2050
        nb_per = round(self.year_end - self.year_start + 1)
        self.nb_per = nb_per
        self.years = np.arange(self.year_start, self.year_end+1)

        total_invest = np.asarray([27.0] * nb_per)
        total_invest = DataFrame({'years':self. years, 'share_investment': total_invest})
        self.total_invest = total_invest

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
        
        
    def analytic_grad_entry(self):
        return [
            self.test_macro_analytic_grad
        ]

    def test_macro_analytic_grad(self):
        
        model_name = 'Macroeconomics'
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness':  f'{self.name}', 
                   'ns_macro': f'{self.name}.{model_name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        
        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.total_investment_share_of_gdp': self.total_invest, 
                       f'{self.name}.{model_name}.Agriculture.production_df': self.prod_agri,
                       f'{self.name}.{model_name}.Services.production_df': self.prod_service,
                       f'{self.name}.{model_name}.Industry.production_df': self.prod_indus,
                       f'{self.name}.{model_name}.Industry.capital_df': self.cap_indus_df,
                       f'{self.name}.{model_name}.Services.capital_df': self.cap_service_df,
                       f'{self.name}.{model_name}.Agriculture.capital_df':self.cap_agri_df,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macro_sectorization_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.total_investment_share_of_gdp',
                                    f'{self.name}.{model_name}.Agriculture.production_df',
                                    f'{self.name}.{model_name}.Services.production_df',
                                    f'{self.name}.{model_name}.Industry.production_df',
                                    f'{self.name}.{model_name}.Industry.capital_df',
                                    f'{self.name}.{model_name}.Services.capital_df',
                                    f'{self.name}.{model_name}.Agriculture.capital_df'],
                            outputs=[f'{self.name}.economics_df', 
                                     f'{self.name}.investment_df'])
        
   
