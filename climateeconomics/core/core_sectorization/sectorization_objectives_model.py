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
        self.historical_energy = inputs_dict['historical_energy']
   
    def set_coupling_inputs(self, inputs):
        self.economics_df = inputs['economics_df']
        self.economics_df.index = self.economics_df['years'].values
        #Put all inputs in dictionary and check if complex
        sectors_capital_dfs = {}
        sectors_production_dfs ={}
        for sector in self.SECTORS_LIST:
            sectors_capital_dfs[sector] = inputs[f'{sector}.detailed_capital_df']
            sectors_production_dfs[sector] = inputs[f'{sector}.production_df']
        self.sectors_capital_dfs = sectors_capital_dfs
        self.sectors_production_dfs = sectors_production_dfs
            
    def compute_all_errors(self, inputs):
        """ For all variables takes predicted values and reference and compute the quadratic error
        """
        self.set_coupling_inputs(inputs)
        #Initialise a dataframe to store hsitorical energy efficiency per sector
        self.hist_energy_eff = pd.DataFrame({'years': self.years_range})
        self.hist_energy_eff['years'] = self.years_range
       
        #compute total errors  
        error_pib_total = self.compute_quadratic_error(self.historical_gdp['total'].values, self.economics_df['output_net_of_d'].values)
        #Per sector
        sectors_gdp_errors = {}
        sectors_energy_eff_errors = {}
        
        for sector in self.SECTORS_LIST:
            capital_df = self.sectors_capital_dfs[sector]
            production_df = self.sectors_production_dfs[sector]
            sectors_gdp_errors[sector] = self.compute_quadratic_error(self.historical_gdp[sector].values, production_df['output_net_of_damage'].values)
            self.compute_hist_energy_efficiency(sector)
            sectors_energy_eff_errors[sector] = self.compute_quadratic_error(self.hist_energy_eff[sector].values, capital_df['energy_efficiency'].values )

        return error_pib_total, sectors_gdp_errors, sectors_energy_eff_errors, self.hist_energy_eff
    
    def compute_quadratic_error(self, ref, pred):
        """
        Compute quadratic error. Inputs: ref and pred are arrays
        """
        #Find maximum value in data to normalise objective
        norm_value = np.amax(ref)
        delta = np.subtract(pred, ref)
        #And normalise delta
        delta_norm = delta / norm_value
        delta_squared = np.square(delta_norm)
        error = np.mean(delta_squared)
        return error
    
    def compute_hist_energy_efficiency(self, sector):
        """
        Compute historical energy efficiency value: energy in 1e3Twh and capital in T$
        """
        energy = self.historical_energy[sector].values 
        capital = self.historical_capital[sector].values 
        #compute 
        energy_eff = capital/energy
        #and store
        self.hist_energy_eff[sector] = energy_eff
        