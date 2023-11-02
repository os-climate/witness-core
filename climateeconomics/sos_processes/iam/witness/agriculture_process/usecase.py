'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/02 Copyright 2023 Capgemini

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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sostrades_core.study_manager.study_manager import StudyManager

from pathlib import Path
from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


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
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable,
                         'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='Land.Agriculture', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.agriculture_name = name
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):
        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7900, 8500, year_range))

        temperature = np.array(np.linspace(1.05, 5, year_range))
        #         temperature = np.array(np.linspace(1.05, 1.05, year_range))

        temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
        temperature_df.index = years

        population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        population_df.index = years
        red_meat_percentage = np.linspace(6.82, 1, year_range)
        white_meat_percentage = np.linspace(13.95, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
            GlossaryCore.Years: years,
            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
            GlossaryCore.Years: years,
            'white_meat_percentage': white_meat_percentage})
        diet_df = pd.DataFrame({'red meat': [11.02],
                                'white meat': [31.11],
                                'milk': [79.27],
                                'eggs': [9.68],
                                'rice and maize': [97.76],
                                'potatoes': [32.93],
                                'fruits and vegetables': [217.62],
                                })
        other = np.array(np.linspace(0.102, 0.102, year_range))

        # private values economics operator pyworld3
        agriculture_input = {}
        agriculture_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        agriculture_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        agriculture_input[f"{self.study_name}.{self.agriculture_name}.{'diet_df'}"] = diet_df
        agriculture_input[f"{self.study_name}.{'red_meat_percentage'}"] = self.red_meat_percentage
        agriculture_input[f"{self.study_name}.{'white_meat_percentage'}"] = self.white_meat_percentage
        agriculture_input[f"{self.study_name}.{self.agriculture_name}.{'other_use_agriculture'}"] = other
        agriculture_input[f"{self.study_name}.{GlossaryCore.PopulationDfValue}"] = population_df
        agriculture_input[f"{self.study_name}.{GlossaryCore.TemperatureDfValue}"] = temperature_df

        setup_data_list.append(agriculture_input)

        red_meat_percentage_ctrl = np.linspace(300, 300, self.nb_poles)
        white_meat_percentage_ctrl = np.linspace(200, 200, self.nb_poles)

        design_space_ctrl_dict = {}
        design_space_ctrl_dict['red_meat_percentage_ctrl'] = red_meat_percentage_ctrl
        design_space_ctrl_dict['white_meat_percentage_ctrl'] = white_meat_percentage_ctrl

        design_space_ctrl = pd.DataFrame(design_space_ctrl_dict)
        self.design_space_ctrl = design_space_ctrl
        self.dspace = self.setup_design_space_ctrl_new()
        return setup_data_list

    def setup_design_space_ctrl_new(self):
        # Design Space
        # header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables:
        update_dspace_dict_with(ddict, 'red_meat_percentage_ctrl',
                                list(self.design_space_ctrl['red_meat_percentage_ctrl'].values), [1.0] * self.nb_poles,
                                [10.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        update_dspace_dict_with(ddict, 'white_meat_percentage_ctrl',
                                list(self.design_space_ctrl['white_meat_percentage_ctrl'].values),
                                [5.0] * self.nb_poles, [20.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)

        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.set_debug_mode()

