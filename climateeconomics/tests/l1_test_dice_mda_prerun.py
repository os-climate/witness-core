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
from climateeconomics.sos_processes.iam.dice.dice_model.usecase import Study
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from time import time
from pandas.core.frame import DataFrame
from tempfile import gettempdir


class DICEMDAPrerunTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        repo = 'climateeconomics.sos_processes.iam.dice'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'dice_model')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes(True)
        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        print('first execution with eco_df as input')
        self.ee.load_study_from_input_dict(values_dict)

        t0 = time()
        self.ee.execute()
        t1 = time() - t0
        print('time for first exec : ', t1, 's')
        residual_history_1 = self.ee.root_process.sub_mda_list[0].residual_history
        # print(residual_history_1)
        ee2 = ExecutionEngine(self.name)
        builder = ee2.factory.get_builder_from_process(
            repo, 'dice_model')

        ee2.factory.set_builders_to_coupling_builder(builder)

        ee2.configure()
        ee2.display_treeview_nodes(True)
        usecase = Study(execution_engine=ee2)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict.pop(usecase.study_name + '.economics_df')

        dice_input = {}
        years = np.arange(usecase.year_start,
                          usecase.year_end + 1, usecase.time_step)
        data = np.zeros(len(years))
        df = DataFrame({'year': years,
                        'damages': data,
                        'damage_frac_output': data,
                        'backstop_price': data,
                        'adj_backstop_cost': data,
                        'abatecost': data,
                        'marg_abatecost': data,
                        'carbon_price': data,
                        'base_carbon_price': data},
                       index=np.arange(usecase.year_start, usecase.year_end + 1, usecase.time_step))
        dice_input[usecase.study_name + '.damage_df'] = df

        values_dict.update(dice_input)

        print('second execution with damage_df as input')

        ee2.load_study_from_input_dict(values_dict)
        t0 = time()
        ee2.execute()
        t1 = time() - t0
        print('time for second exec : ', t1, 's')
        residual_history_2 = ee2.root_process.sub_mda_list[0].residual_history
        # print(residual_history_2)

        CO2_emissions_df = DataFrame({
            'year': years,
            'gr_sigma': data,
            'sigma': data,
            'land_emissions': data,
            'cum_land_emissions': data,
            'indus_emissions': data,
            'cum_indus_emissions': data,
            'total_emissions': data,
            'cum_total_emissions': data,
            'emissions_control_rate': data}, index=np.arange(usecase.year_start, usecase.year_end + 1, usecase.time_step))

        carboncycle_df = DataFrame({
            'year': years,
            'atmo_conc': data,
            'lower_ocean_conc': data,
            'shallow_ocean_conc': data,
            'ppm': data,
            'atmo_share_since1850': data,
            'atmo_share_sinceystart': data}, index=np.arange(usecase.year_start, usecase.year_end + 1, usecase.time_step))

        temperature_df = DataFrame({'year': years,
                                    'exog_forcing': data,
                                    'forcing': data,
                                    'temp_atmo': data,
                                    'temp_ocean': data}, index=np.arange(usecase.year_start, usecase.year_end + 1, usecase.time_step))
        ee2 = ExecutionEngine(self.name)
        builder = ee2.factory.get_builder_from_process(
            repo, 'dice_model')

        ee2.factory.set_builders_to_coupling_builder(builder)

        ee2.configure()
        ee2.display_treeview_nodes(True)
        usecase = Study(execution_engine=ee2)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        dice_input = {}
        dice_input[usecase.study_name + '.CO2_emissions_df'] = CO2_emissions_df
        dice_input[usecase.study_name + '.carboncycle_df'] = carboncycle_df
        dice_input[usecase.study_name + '.temperature_df'] = temperature_df
        dice_input[usecase.study_name + '.damage_df'] = df
        values_dict.update(dice_input)

        print('all inputs execution with all inputs')
        ee2.load_study_from_input_dict(values_dict)
        t0 = time()
        ee2.execute()
        t1 = time() - t0
        print('time for all inputs exec : ', t1, 's')
        residual_history_5 = ee2.root_process.sub_mda_list[0].residual_history

        ee2 = ExecutionEngine(self.name)
        builder = ee2.factory.get_builder_from_process(
            repo, 'dice_model')

        ee2.factory.set_builders_to_coupling_builder(builder)

        ee2.configure()
        ee2.display_treeview_nodes(True)
        usecase = Study(execution_engine=ee2)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        dice_input = {}
        dice_input[usecase.study_name + '.carboncycle_df'] = carboncycle_df

        values_dict.update(dice_input)

        values_dict.pop(usecase.study_name + '.economics_df')

        print('only carbon cycle execution will crash')
        ee2.load_study_from_input_dict(values_dict)

        self.assertRaises(Exception, ee2.execute)
