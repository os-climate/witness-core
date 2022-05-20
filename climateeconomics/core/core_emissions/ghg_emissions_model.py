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
from energy_models.core.stream_type.carbon_models.nitrous_oxide import N2O


class GHGEmissions():
    '''
    Used to compute ghg emissions from different sectors
    '''
    GHG_TYPE_LIST = [N2O.name, 'CO2', 'CH4']

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.configure_parameters()
        self.CO2_objective = None
        self.create_dataframe()
        self.sector_list = ['energy', 'land', 'industry']

    def configure_parameters(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.time_step = self.param['time_step']

        self.CO2_land_emissions = self.param['CO2_land_emissions']
        self.CO2_indus_emissions_df = self.param['CO2_indus_emissions_df']
        self.GHG_total_energy_emissions = self.param['GHG_total_energy_emissions']
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12

        self.gwp_20 = self.param['GHG_global_warming_potential20']
        self.gwp_100 = self.param['GHG_global_warming_potential100']

    def configure_parameters_update(self, inputs_dict):

        self.CO2_land_emissions = inputs_dict['CO2_land_emissions']
        self.CO2_indus_emissions_df = inputs_dict['CO2_indus_emissions_df']
        self.GHG_total_energy_emissions = inputs_dict['GHG_total_energy_emissions']
        self.create_dataframe()

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        # declare class variable as local variable
        year_start = self.year_start
        year_end = self.year_end

        self.years_range = np.arange(
            year_start, year_end + 1, self.time_step)

        self.ghg_emissions_df = pd.DataFrame({'years': self.years_range})
        self.gwp_emissions = pd.DataFrame({'years': self.years_range})

    def compute_land_emissions(self):
        '''
        Compute emissions from land
        '''

        self.ghg_emissions_df['CO2 land_emissions'] = self.CO2_land_emissions.drop(
            'years', axis=1).sum(axis=1).values
        self.ghg_emissions_df['CH4 land_emissions'] = 0.
        self.ghg_emissions_df['N2O land_emissions'] = 0.

    def compute_total_emissions(self):
        '''
        Total emissions taking energy emissions as inputs
        '''
        self.ghg_emissions_df['CO2 industry_emissions'] = self.CO2_indus_emissions_df['indus_emissions'].values
        self.ghg_emissions_df['CH4 industry_emissions'] = 0.
        self.ghg_emissions_df['N2O industry_emissions'] = 0.

        for ghg in self.GHG_TYPE_LIST:
            self.ghg_emissions_df[f'{ghg} energy_emissions'] = self.GHG_total_energy_emissions[f'Total {ghg} emissions'].values

            self.ghg_emissions_df[f'Total {ghg} emissions'] = self.ghg_emissions_df[f'{ghg} land_emissions'].values + \
                self.ghg_emissions_df[f'{ghg} industry_emissions'].values + \
                self.ghg_emissions_df[f'{ghg} energy_emissions'].values

    def compute_gwp(self):

        for ghg in self.GHG_TYPE_LIST:

            self.gwp_emissions[f'{ghg}_20'] = self.ghg_emissions_df[f'Total {ghg} emissions'] * self.gwp_20[ghg]
            self.gwp_emissions[f'{ghg}_100'] = self.ghg_emissions_df[f'Total {ghg} emissions'] * self.gwp_100[ghg]

    def compute_co2_emissions_for_carbon_cycle(self):
        co2_emissions_df = self.ghg_emissions_df[['years', 'Total CO2 emissions']].rename(
            {'Total CO2 emissions': 'total_emissions'}, axis=1)

        co2_emissions_df['cum_total_emissions'] = co2_emissions_df['total_emissions'].cumsum(
        )
        return co2_emissions_df

    def compute(self):
        """
        Compute outputs of the model
        """

        self.compute_land_emissions()
        self.compute_total_emissions()
        self.compute_gwp()
