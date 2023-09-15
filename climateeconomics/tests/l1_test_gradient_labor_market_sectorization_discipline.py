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
from pandas import read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class LaborMarketJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2040
        nb_per = round(self.year_end - self.year_start + 1)
        self.nb_per = nb_per
        self.years = np.arange(self.year_start, self.year_end+1)
        
        indusshare = 21.7
        agri_year_start = 27.4
        agri = []
        agri.append(agri_year_start)
        for year in np.arange(1, nb_per):
            agri.append(agri[year - 1] * 0.99)
        service = np.array([100.0]*nb_per) - agri - indusshare

        #service = np.substract(total, agri)
        workforce_share = DataFrame({GlossaryCore.Years:self. years, 'Agriculture': agri,
                                     'Industry': indusshare, 'Services': service})
        self.workforce_share = workforce_share
        data_dir = join(dirname(__file__), 'data')
        working_age_pop_df = read_csv(
                join(data_dir, 'workingage_population_df.csv'))
        self.working_age_pop_df = working_age_pop_df[(working_age_pop_df[GlossaryCore.Years]<=self.year_end) & (working_age_pop_df[GlossaryCore.Years]>=self.year_start)]
        
        
    def analytic_grad_entry(self):
        return [
            self.test_macro_analytic_grad
        ]

    def test_labor_analytic_grad(self):
        
        model_name = 'LaborMarket'
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness':  f'{self.name}'}
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.labor_market.labor_market_discipline.LaborMarketDiscipline'
        builder = self.ee.factory.get_builder_from_module(model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{model_name}.workforce_share_per_sector': self.workforce_share, 
                       f'{self.name}.working_age_population_df': self.working_age_pop_df
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_labormarket_sectorization_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.working_age_population_df'],
                            outputs=[f'{self.name}.workforce_df'])
        
   
