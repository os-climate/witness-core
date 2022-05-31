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


class ServicesJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2050
        self.time_step = 1
        self.years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_per = round((self.year_end - self.year_start) / self.time_step + 1)
        # -------------------------
        # input
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        total_workforce_df = read_csv(join(data_dir, 'workingage_population_df.csv'))
        total_workforce_df = total_workforce_df[total_workforce_df['years']<=self.year_end]
        #multiply ageworking pop by employment rate and by % in services
        workforce = total_workforce_df['population_1570']* 0.659 * 0.509
        self.workforce_df = pd.DataFrame({'years': self.years, 'workforce': workforce})

        #Energy_supply
        brut_net = 1/1.45
        share_indus = 0.37
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(self.year_start, self.year_end+1))
        energy_supply_values = energy_supply * brut_net * share_indus
        energy_supply_df = pd.DataFrame({'years': self.years, 'Total production': energy_supply_values})
        energy_supply_df.index = self.years
        self.energy_supply_df = energy_supply_df
        #energy_supply_df.loc[2020, 'Total production'] = 91.936

        #Investment growth at 2% 
        init_value = 25
        invest_serie = []
        invest_serie.append(init_value)
        for year in np.arange(1, self.nb_per):
            invest_serie.append(invest_serie[year - 1] * 1.02)
        self.total_invest = pd.DataFrame({'years': self.years, 'investment': invest_serie})
        
        #damage
        self.damage_df = pd.DataFrame({'years': self.years, 'damages': np.zeros(self.nb_per), 'damage_frac_output': np.zeros(self.nb_per),
                                       'base_carbon_price': np.zeros(self.nb_per)})
        self.damage_df.index = self.years
        self.damage_df['damage_frac_output'] = 1e-2 
        
        
    def analytic_grad_entry(self):
        return [
            self.test_services_analytic_grad,
            self.test_services_withotudamagetoproductivity
        ]

    def test_services_analytic_grad(self):

        self.model_name = 'Services'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}' }
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.workforce_df': self.workforce_df,
                       f'{self.name}.sector_investment': self.total_invest,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_services_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.workforce_df',
                                    f'{self.name}.sector_investment'],
                            outputs=[f'{self.name}.production_df', 
                                     f'{self.name}.capital_df',
                                     f'{self.name}.emax_enet_constraint'])
        
    def test_services_withotudamagetoproductivity(self):

        self.model_name = 'Services'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}' }
        
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.time_step': self.time_step,
                       f'{self.name}.damage_to_productivity': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.energy_production': self.energy_supply_df,
                       f'{self.name}.damage_df': self.damage_df,
                       f'{self.name}.workforce_df': self.workforce_df,
                       f'{self.name}.sector_investment': self.total_invest,
                       f'{self.name}.alpha': 0.5
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_services_discipline_withoutdamage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.energy_production',
                                    f'{self.name}.damage_df',
                                    f'{self.name}.workforce_df',
                                    f'{self.name}.sector_investment'],
                            outputs=[f'{self.name}.production_df', 
                                     f'{self.name}.capital_df',
                                     f'{self.name}.emax_enet_constraint'])
