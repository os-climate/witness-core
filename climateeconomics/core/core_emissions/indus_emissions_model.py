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
import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class IndusEmissions():
    '''
    Used to compute industrial CO2 emissions
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.init_gr_sigma = self.param['init_gr_sigma']
        self.decline_rate_decarbo = self.param['decline_rate_decarbo']
        self.init_indus_emissions = self.param['init_indus_emissions']
        self.init_gross_output = self.param[GlossaryCore.InitialGrossOutput['var_name']]
        self.init_cum_indus_emissions = self.param['init_cum_indus_emissions']
        self.energy_emis_share = self.param['energy_emis_share']
        self.land_emis_share = self.param['land_emis_share']
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        # declare class variable as local variable
        year_start = self.year_start
        year_end = self.year_end
        init_gr_sigma = self.init_gr_sigma
        init_indus_emissions = self.init_indus_emissions
        init_cum_indus_emissions = self.init_cum_indus_emissions

        years_range = np.arange(
            year_start, year_end + 1)
        self.years_range = years_range
        indus_emissions_df = pd.DataFrame(index=years_range, columns=[GlossaryCore.Years,
                                                                      'gr_sigma', 'sigma', 'indus_emissions',
                                                                      'cum_indus_emissions'])

        for key in indus_emissions_df.keys():
            indus_emissions_df[key] = 0
        indus_emissions_df[GlossaryCore.Years] = years_range
        indus_emissions_df.loc[year_start, 'gr_sigma'] = init_gr_sigma
        indus_emissions_df.loc[year_start,
                               'indus_emissions'] = init_indus_emissions
        indus_emissions_df.loc[year_start,
                               'cum_indus_emissions'] = init_cum_indus_emissions
        self.indus_emissions_df = indus_emissions_df

    def compute_sigma(self, year):
        '''
        Compute CO2-equivalent-emissions output ratio at t 
        using sigma t-1 and growht_rate sigma  t-1
        '''

        if year == self.year_start:
            sigma = self.init_indus_emissions / \
                self.init_gross_output
        else:
            p_gr_sigma = self.indus_emissions_df.at[year - 1, 'gr_sigma']
            p_sigma = self.indus_emissions_df.at[year - 1, 'sigma']
            sigma = p_sigma * np.exp(p_gr_sigma)
        self.indus_emissions_df.loc[year, 'sigma'] = sigma
        return sigma

    def compute_change_sigma(self, year):
        """
        Compute change in sigma growth rate at t 
        using sigma grouwth rate t-1
        """

        if year == self.year_start:
            pass
        else:
            p_gr_sigma = self.indus_emissions_df.at[year - 1, 'gr_sigma']
            gr_sigma = p_gr_sigma * \
                ((1.0 + self.decline_rate_decarbo))
            self.indus_emissions_df.loc[year, 'gr_sigma'] = gr_sigma
            return gr_sigma

    def compute_indus_emissions(self, year):
        """
        Compute industrial emissions at t 
        using gross output (t)
        emissions control rate (t)
        emissions not coming from land change or energy 
        """
        sigma = self.indus_emissions_df.at[year, 'sigma']
        gross_output_ter = self.economics_df.at[year, GlossaryCore.GrossOutput]

        indus_emissions = sigma * gross_output_ter * \
            (1 - self.energy_emis_share - self.land_emis_share)
        self.indus_emissions_df.loc[year, 'indus_emissions'] = indus_emissions
        return indus_emissions

    def compute_cum_indus_emissions(self, year):
        """
        Compute cumulative industrial emissions at t
        using emissions indus at t- 1 
        and cumulative indus emissions at t-1
        """

        if year == self.year_start:
            pass
        else:
            p_cum_indus_emissions = self.indus_emissions_df.at[year - 1, 'cum_indus_emissions']
            indus_emissions = self.indus_emissions_df.at[year, 'indus_emissions']
            cum_indus_emissions = p_cum_indus_emissions + \
                indus_emissions / self.gtco2_to_gtc
            self.indus_emissions_df.loc[year,
                                        'cum_indus_emissions'] = cum_indus_emissions
            return cum_indus_emissions

    ######### GRADIENTS ########

    def compute_d_indus_emissions(self):
        """
        Compute gradient d_indus_emissions/d_gross_output, 
        d_cum_indus_emissions/d_gross_output, 
        d_cum_indus_emissions/d_total_CO2_emitted
        """
        years = np.arange(self.year_start,
                          self.year_end + 1)
        nb_years = len(years)

        # derivative matrix initialization
        d_indus_emissions_d_gross_output = np.identity(nb_years) * 0
        d_cum_indus_emissions_d_gross_output = np.identity(nb_years) * 0
        d_cum_indus_emissions_d_total_CO2_emitted = np.identity(nb_years) * 0

        i = 0
        line = 0
        for i in range(nb_years):
            for line in range(nb_years):
                if i > 0 and i <= line:  # fill triangular descendant
                    d_cum_indus_emissions_d_total_CO2_emitted[line, i] = 1 / self.gtco2_to_gtc

                    d_cum_indus_emissions_d_gross_output[line, i] = 1 / self.gtco2_to_gtc *\
                        self.indus_emissions_df.at[years[i], 'sigma'] *\
                        (1.0 - self.energy_emis_share - self.land_emis_share)
                if i == line:  # fill diagonal
                    d_indus_emissions_d_gross_output[line, i] = self.indus_emissions_df.at[years[line], 'sigma'] \
                        * (1 - self.energy_emis_share - self.land_emis_share)

        return d_indus_emissions_d_gross_output, d_cum_indus_emissions_d_gross_output, d_cum_indus_emissions_d_total_CO2_emitted

    def compute(self, inputs_models):
        """
        Compute outputs of the pyworld3
        """
        self.inputs_models = inputs_models
        self.economics_df = self.inputs_models[GlossaryCore.EconomicsDfValue]
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values

        # Iterate over years
        for year in self.years_range:
            self.compute_change_sigma(year)
            self.compute_sigma(year)
            self.compute_indus_emissions(year)
            self.compute_cum_indus_emissions(year)

        return self.indus_emissions_df
