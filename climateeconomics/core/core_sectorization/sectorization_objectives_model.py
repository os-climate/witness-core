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

import numpy as np
import pandas as pd
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import AgricultureDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import ServicesDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import IndustrialDiscipline
from gemseo.third_party.prettytable.prettytable import NONE


class ObjectivesModel():
    """
    Objectives model for sectorisation optimisation fitting process 
    """

    #Units conversion
    conversion_factor=1.0
    SECTORS_DISC_LIST = [AgricultureDiscipline, ServicesDiscipline, IndustrialDiscipline]
    SECTORS_LIST = [disc.sector_name for disc in SECTORS_DISC_LIST]
    SECTORS_OUT_UNIT = {disc.sector_name: disc.prod_cap_unit for disc in SECTORS_DISC_LIST}

    def __init__(self, inputs_dict):
        '''
        Constructor
        '''
        self.economics_df = None
        self.configure_parameters(inputs_dict)

    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.year_start = inputs_dict['year_start']  # year start
        self.year_end = inputs_dict['year_end']  # year end
        self.time_step = inputs_dict['time_step']
        self.years_range = np.arange(self.year_start,self.year_end + 1,self.time_step)
        self.nb_years = len(self.years_range)
        self.historical_gdp = inputs_dict['historical_gdp']
        self.historical_capital = inputs_dict['historical_capital']
#         self.data_energy_df = inputs_dict['data_energy_df']
#         self.data_investments_df = inputs_dict['data_investments_df']
#         self.data_workforce_df = inputs_dict['data_workforce_df']
   
    def set_coupling_inputs(self, inputs):
        self.economics_df = inputs['economics_df']
        self.economics_df.index = self.economics_df['years'].values
        #Put all inputs in dictionary and check if complex
        sectors_capital_dfs = {}
        sectors_production_dfs ={}
        for sector in self.SECTORS_LIST:
            sectors_capital_dfs[sector] = inputs[f'{sector}.capital_df']
            sectors_production_dfs[sector] = inputs[f'{sector}.production_df']
        self.sectors_capital_dfs = sectors_capital_dfs
        self.sectors_production_dfs = sectors_production_dfs
            
    def compute_all_errors(self, inputs):
        """ For all variables takes predicted values and reference and compute the quadratic error
        """
        self.set_coupling_inputs(inputs)
        
        error_pib_total = self.compute_quadratic_error(self.historical_gdp['total'].values, self.economics_df['net_output'].values)
        error_cap_total = self.compute_quadratic_error(self.historical_capital['total'].values, self.economics_df['capital'].values)
        #Per sector
        sectors_cap_errors = {}
        sectors_gdp_errors = {}
        for sector in self.SECTORS_LIST:
            capital_df = self.sectors_capital_dfs[sector]
            production_df = self.sectors_production_dfs[sector]
            sectors_cap_errors[sector] = self.compute_quadratic_error(self.historical_capital[sector].values, capital_df['capital'].values)
            sectors_gdp_errors[sector] = self.compute_quadratic_error(self.historical_capital[sector].values, production_df['output_net_of_damage'].values)
        
        return error_pib_total, error_cap_total, sectors_cap_errors, sectors_gdp_errors
    
    def compute_quadratic_error(self, ref, pred):
        """
        Compute quadratic error. Inputs: ref and pred are arrays
        """
        delta = np.subtract(pred, ref)
        delta_squared = np.square(delta)
        error = np.mean(delta_squared)
        return error
        