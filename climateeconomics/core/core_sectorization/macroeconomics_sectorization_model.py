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


class MacroeconomicsModel():
    """
    Sector model
    General implementation of sector model 
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
        self.investment_df = None
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
        share_invest_df = inputs_dict['total_investment_share_of_gdp']
        self.share_invest = share_invest_df['share_investment'].values
        #self.scaling_factor_invest = inputs_dict['scaling_factor_investment']      
        
    def set_coupling_inputs(self, inputs):
        share_invest_df = inputs['total_investment_share_of_gdp']
        self.share_invest = share_invest_df['share_investment'].values
        arr_type_output = 'float64'
        arr_type_netoutput = 'float64'
        arr_type_capital = 'float64'
        arr_type_ucapital = 'float64'
        #Put all inputs in dictionary and check if complex
        capital_dfs = {}
        production_dfs ={}
        for sector in self.SECTORS_LIST:
            capital_dfs[sector] = inputs[f'{sector}.capital_df']
            production_dfs[sector] = inputs[f'{sector}.production_df']
            if 'complex128' in [inputs[f'{sector}.capital_df']['capital'].dtype]:
                arr_type_capital = 'complex128'
            if 'complex128' in [inputs[f'{sector}.capital_df']['usable_capital'].dtype]:
                arr_type_ucapital = 'complex128'
            if 'complex128' in [inputs[f'{sector}.production_df']['output'].dtype]:
                arr_type_output = 'complex128'
            if 'complex128' in [inputs[f'{sector}.production_df']['net_output'].dtype]:
                arr_type_netoutput = 'complex128'
                
        self.sum_output = np.zeros(self.nb_years, dtype= arr_type_output)
        self.sum_net_output = np.zeros(self.nb_years, dtype= arr_type_netoutput)
        self.sum_capital = np.zeros(self.nb_years, dtype= arr_type_capital)
        self.sum_u_capital = np.zeros(self.nb_years, dtype= arr_type_ucapital)
        self.output_growth = np.zeros(self.nb_years, dtype= arr_type_netoutput)
        self.capital_dfs = capital_dfs
        self.production_dfs = production_dfs

    def create_dataframes(self):
        '''
        Create dataframes with years
        '''
        economics_df = pd.DataFrame({'years': self.years_range, 'capital': self.sum_capital, 'usable_capital': self.sum_u_capital,
                                     'output': self.sum_output, 'net_output': self.sum_net_output, 
                                     'output_growth': self.output_growth})
        investment_df = pd.DataFrame({'years': self.years_range, 'investment': self.investment})
        investment_df.index = self.years_range
        economics_df.index = self.years_range  
        self.economics_df = economics_df 
        self.investment_df = investment_df 

    def sum_all(self):
        """ Sum output, net output, capital, usable capital and invest from all sectors. Unit: 1e12$ 
        """

        for sector in self.SECTORS_LIST:
            capital_df = self.capital_dfs[sector]
            production_df = self.production_dfs[sector]
            self.sum_capital += capital_df['capital'].values
            self.sum_u_capital += capital_df['usable_capital'].values
            self.sum_output += production_df['output'].values
            self.sum_net_output += production_df['net_output'].values
            
    def compute_investment(self):
        """ Compute total investement available 
        Investment = net_output * share_invest 
        """
        self.investment = self.sum_net_output * self.share_invest/100
    
    def compute_output_growth(self):
        """
        Compute the output growth between year t and year t+1
        """
        #Loop over every years except last one 
        for period in np.arange(0, self.nb_years-1):
            output_a = self.sum_net_output[period+1]
            output = self.sum_net_output[period]
            output = max(1e-6, output)
            self.output_growth[period] = ((output_a -output) / output)
        #For last year put the previous year value to avoid a 0 
        self.output_growth[self.nb_years-1] = self.output_growth[self.nb_years-2]
        return self.output_growth
    
    #RUN
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.inputs = inputs
        self.set_coupling_inputs(inputs)
        self.sum_all()
        self.compute_investment()
        self.compute_output_growth()
        self.create_dataframes()

        return self.economics_df, self.investment_df
    
    ### GRADIENTS ###
    def get_derivative_sectors(self):
        """ 
        Compute gradient for netoutput and invest wrt net output from each sector
        """
        grad_netoutput = np.identity(self.nb_years)
        #Invest = net_output * share_invest (share invest in%)
        grad_invest = grad_netoutput * self.share_invest/100
        return grad_netoutput, grad_invest
    
    def get_derivative_dinvest_dshare(self):
        """
        Compute the derivative of investment wrt invest_share_gdp 
        """
        #Invest = net_output * share_invest
        dinvest = np.identity(self.nb_years)/100 * self.sum_net_output
        return dinvest
