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
from copy import deepcopy

from climateeconomics.glossarycore import GlossaryCore


class DamageModel():
    '''
    Damage from climate change
    '''

    CO2_TAX_MINUS_CO2_DAMAGE_CONSTRAINT_DF = 'CO2_tax_minus_CO2_damage_constraint_df'
    CO2_TAX_MINUS_CO2_DAMAGE_CONSTRAINT = 'CO2_tax_minus_CO2_damage_constraint'

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.time_step = self.param['time_step']
        self.init_damag_int = self.param["init_damag_int"]
        self.damag_int = self.param['damag_int']
        self.damag_quad = self.param['damag_quad']
        self.damag_expo = self.param['damag_expo']
        self.tipping_point = self.param['tipping_point']
        self.tp_a1 = self.param['tp_a1']
        self.tp_a2 = self.param['tp_a2']
        self.tp_a3 = self.param['tp_a3']
        self.tp_a4 = self.param['tp_a4']
        self.frac_damage_prod = self.param['frac_damage_prod']
        self.total_emissions_ref = self.param['total_emissions_damage_ref']
        self.damage_constraint_factor = self.param['damage_constraint_factor']
        self.co2_damage_price_df = None
        self.CO2_TAX_MINUS_CO2_DAMAGE_CONSTRAINT_DF = None
        self.compute_climate_impact_on_gdp = self.param['assumptions_dict']['compute_climate_impact_on_gdp']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        damage_df = pd.DataFrame(
            index=years_range,
            columns=[GlossaryCore.Years,
                     'damages',
                     'damage_frac_output',
                     ])

        if 'complex128' in [self.temperature_df['temp_atmo'].dtype]:
            arr_type = 'complex128'
        else:
            arr_type = 'float64'

        damage_df[GlossaryCore.Years] = self.years_range
        damage_df['damages'] = np.zeros(
            len(self.years_range), dtype=arr_type)
        damage_df['damage_frac_output'] = np.zeros(
            len(self.years_range), dtype=arr_type)

        return damage_df

    def compute_damage_fraction_on_gdp(self, year):
        """
        Compute damages fraction of output at t
        using variables at t
        If tipping point = True : Martin Weitzman damage function.
        """
        temp_atmo = self.temperature_df.at[year, 'temp_atmo']
        if self.tipping_point == True:
            if temp_atmo < 0:
                dam = 0.0
            else:
                dam = (temp_atmo / self.tp_a1)**self.tp_a2 + \
                    (temp_atmo / self.tp_a3)**self.tp_a4
            damage_frac_output = 1 - (1 / (1 + dam))
        else:
            damage_frac_output = self.damag_int * temp_atmo + \
                self.damag_quad * temp_atmo**self.damag_expo
        self.expected_damage_df.loc[year, 'damage_frac_output'] = damage_frac_output
        return damage_frac_output

    def compute_damages_on_gdp(self, year) -> pd.DataFrame:
        """
        Compute damages (t) (trillions 2005 USD per year) on GDP
        using variables at t
        """
        gross_output = self.economics_df.at[year, 'gross_output']
        damage_frac_output = self.expected_damage_df.at[year, 'damage_frac_output']
        damages = gross_output * damage_frac_output
        self.expected_damage_df.loc[year, 'damages'] = damages

        return damages

    def compute_CO2_tax_minus_CO2_damage_constraint(self):
        """
        Compute CO2 tax - CO2 damage constraint:
                 CO2 tax - fact * CO2_damage_price  > 0  
            with CO2_damage_price[year] = 1e3 * 1.01**(year_start-year) * mean(damage_df[year:year+25] (T$)) / total_emissions_ref (Gt)
        """
        if 'complex128' in [self.expected_damage_df['damages'].values.dtype]:
            arr_type = 'complex128'
        else:
            arr_type = 'float64'

        co2_damage_price = np.zeros(
            len(self.expected_damage_df.index), dtype=arr_type)

        damages = self.expected_damage_df['damages'].values.tolist()

        for i, year in enumerate(self.expected_damage_df.index):

            if year == self.year_end:
                co2_damage_price[i] = 1e3 * 1.01**i * \
                    damages[i] / self.total_emissions_ref
            else:
                k = max(year + 25 - self.year_end, 0)
                co2_damage_price[i] = 1e3 * 1.01**i * \
                    np.mean(damages[i:i + 25 - k]) / self.total_emissions_ref

        self.co2_damage_price_df = pd.DataFrame(
            {GlossaryCore.Years: self.expected_damage_df.index, 'CO2_damage_price': co2_damage_price})

    def compute_gradient(self):
        """
        Compute gradient
        d_damage_frac_output/d_temp_atmo, 
        d_damages/d_temp_atmo, 
        d_damages/d_gross_output, 
        d_constraint/d_CO2_taxes, 
        d_constraint/d_temp_atmo, 
        d_constraint_economics
        """
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        ddamage_frac_output_temp_atmo = np.zeros((nb_years, nb_years))
        ddamages_temp_atmo = np.zeros((nb_years, nb_years))
        ddamages_gross_output = np.zeros((nb_years, nb_years))
        for i in range(nb_years):
            for line in range(nb_years):
                if i == line:
                    temp_atmo = self.temperature_df.at[years[line],
                                                    'temp_atmo']
                    if self.tipping_point == True:
                        if temp_atmo < 0:
                            ddamage_frac_output_temp_atmo[line, i] = 0.0
                        else:
                            ddamage_frac_output_temp_atmo[line, i] = ((self.tp_a4 * (temp_atmo / self.tp_a3)**self.tp_a4) +
                                                                    (self.tp_a2 * (temp_atmo / self.tp_a1)**self.tp_a2)) / \
                                (temp_atmo * (
                                    ((temp_atmo / self.tp_a1)**self.tp_a2)
                                    + ((temp_atmo / self.tp_a3)**self.tp_a4)
                                    + 1.0) ** 2.0)
                    else:
                        ddamage_frac_output_temp_atmo[line, i] = self.damag_int + \
                            self.damag_quad * self.damag_expo * \
                            temp_atmo ** (self.damag_expo - 1)

                    ddamages_temp_atmo[line, i] = ddamage_frac_output_temp_atmo[line,
                                                                                i] * self.economics_df.at[years[line], 'gross_output']
                    ddamages_gross_output[line,
                                        i] = self.expected_damage_df.at[years[line], 'damage_frac_output']

        dconstraint_temp_atmo, dconstraint_economics = self.compute_dconstraint(
            ddamages_temp_atmo, ddamages_gross_output)
        dconstraint_CO2_taxes = np.identity(
            nb_years)

        return ddamage_frac_output_temp_atmo, ddamages_temp_atmo, ddamages_gross_output, dconstraint_CO2_taxes, dconstraint_temp_atmo, dconstraint_economics

    def compute_dconstraint(self, ddamages_temp_atmo, ddamages_gross_output):
        '''
        Compute gradient of constraint wrt temp_atmo and economics
        '''
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        dconstraint_temp_atmo = np.zeros((nb_years, nb_years))
        dconstraint_economics = np.zeros((nb_years, nb_years))
        damage_constraint_factor = self.damage_constraint_factor

        for i, year in enumerate(years):

            if year == self.year_end:
                dconstraint_temp_atmo[i, i] = 1e3 * 1.01**i * ddamages_temp_atmo[i][i] / \
                    self.total_emissions_ref

                dconstraint_economics[i, i] = 1e3 * 1.01**i * ddamages_gross_output[i][i] / \
                    self.total_emissions_ref

            else:
                k = max(year + 25 - self.year_end, 0)
                for j in range(0, 25 - k):
                    dconstraint_temp_atmo[i, i + j] = 1e3 * 1.01**i * ddamages_temp_atmo[i + j][i + j] / (
                        25 - k) / self.total_emissions_ref

                    dconstraint_economics[i, i + j] = 1e3 * 1.01**i * ddamages_gross_output[i + j][i + j] / (
                        25 - k) / self.total_emissions_ref

        return dconstraint_temp_atmo, dconstraint_economics

    def compute(self, economics_df, temperature_df):
        """
        Compute the outputs of the pyworld3
        """
        self.economics_df = economics_df
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values

        self.temperature_df = temperature_df
        self.temperature_df.index = self.temperature_df[GlossaryCore.Years].values

        self.expected_damage_df = self.create_dataframe()
        for year in self.years_range:
            self.compute_damage_fraction_on_gdp(year)
            self.compute_damages_on_gdp(year)

        self.expected_damage_df = self.expected_damage_df.replace([np.inf, -np.inf], np.nan)
        self.compute_CO2_tax_minus_CO2_damage_constraint()
        self.expected_damage_df.fillna(0.0, inplace=True)

        damage_df = pd.DataFrame.copy(self.expected_damage_df)
        if not self.compute_climate_impact_on_gdp:
            damage_df['damages'] = 0
            damage_df['damage_frac_output'] = 0
            damage_df['base_carbon_price'] = 0

        return damage_df, self.expected_damage_df, self.co2_damage_price_df
