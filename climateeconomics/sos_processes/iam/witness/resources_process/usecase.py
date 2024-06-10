'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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
from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)
from numpy import arange
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.core_resources.resource_mix.resource_mix import (
    ResourceMixModel,
)
from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
OBJECTIVE = FunctionManagerDisc.OBJECTIVE
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM

RESOURCE_LIST=ResourceMixModel.RESOURCE_LIST

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


class Study(ClimateEconomicsStudyManager):

    def __init__(self,
                 year_start=GlossaryCore.YearStartDefault,
                 year_end=GlossaryCore.YearEndDefault,
                 time_step=1,
                 execution_engine=None,
                 main_study: bool = True,
                 ):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.main_study = main_study

        self.all_resource_name = '.Resources'

        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 5

    def setup_usecase(self, study_folder_path=None):

        setup_data_list = []


        year_range = self.year_end - self.year_start
        years = arange(self.year_start, self.year_end + 1, 1)

        global_data_dir = join(Path(__file__).parents[4], 'data')

        # ALL_RESOURCE
        resource_input = {}
        modeled_resources = ResourceMixModel.RESOURCE_LIST
        non_modeled_resource_price = pd.DataFrame({GlossaryCore.Years: years})
        resources_CO2_emissions = pd.DataFrame({GlossaryCore.Years: years})
        for resource in ResourceGlossary.GlossaryDict.values():
            if resource['name'] not in modeled_resources:
                non_modeled_resource_price[resource['name']
                                           ] = resource['price']
            resources_CO2_emissions[resource['name']
                                    ] = resource['CO2_emissions']
        resources_CO2_emissions.index = resources_CO2_emissions[GlossaryCore.Years]
        resource_input[self.study_name + self.all_resource_name +
                       '.non_modeled_resource_price'] = non_modeled_resource_price
        resource_input[f'{self.study_name}.{GlossaryCore.YearStart}'] = self.year_start
        resource_input[f'{self.study_name}.{GlossaryCore.YearEnd}'] = self.year_end
        setup_data_list.append(resource_input)
        data_dir_resource = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')
        resource_demand = pd.read_csv(
            join(data_dir_resource, 'all_demand_from_energy_mix.csv'))

        resource_demand = resource_demand.loc[resource_demand[GlossaryCore.Years]
                                              >= self.year_start]
        resource_demand = resource_demand.loc[resource_demand[GlossaryCore.Years]
                                              <= self.year_end]
        if self.main_study:
            resource_input[self.study_name + self.all_resource_name + '.resources_CO2_emissions'] = resources_CO2_emissions
            resource_input[self.study_name + self.all_resource_name + '.resources_demand'] = resource_demand
            resource_input[self.study_name + self.all_resource_name + '.resources_demand_woratio'] = resource_demand
        setup_data_list.append(resource_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(main_study=True)
    uc_cls.test()
