'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/28-2023/11/03 Copyright 2023 Capgemini

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


class DamageModel:
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
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.init_damag_int = self.param["init_damag_int"]
        self.damag_int = self.param['damag_int']
        self.damag_quad = self.param['damag_quad']
        self.damag_expo = self.param['damag_expo']
        self.tipping_point_model = self.param['tipping_point']
        self.tp_a1 = self.param['tp_a1']
        self.tp_a2 = self.param['tp_a2']
        self.tp_a3 = self.param['tp_a3']
        self.tp_a4 = self.param['tp_a4']
        self.frac_damage_prod = self.param[GlossaryCore.FractionDamageToProductivityValue]
        self.total_emissions_ref = self.param['total_emissions_damage_ref']
        self.damage_constraint_factor = self.param['damage_constraint_factor']
        self.co2_damage_price_df = None
        self.CO2_TAX_MINUS_CO2_DAMAGE_CONSTRAINT_DF = None

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range

        self.damage_fraction_df = pd.DataFrame(index=self.years_range, data={
            GlossaryCore.Years: self.years_range,
            GlossaryCore.DamageFractionOutput: 0.,
            GlossaryCore.BaseCarbonPrice: 0.,
        })

    def compute_damage_fraction_of_gdp(self,):
        """
        Compute damages fraction of output at t
        using variables at t
        If tipping point = True : Martin Weitzman damage function.
        """
        temp_atmo = self.temperature_df[GlossaryCore.TempAtmo]
        temp_atmo = temp_atmo.clip(0.0)
        if self.tipping_point_model:
            dam = (temp_atmo / self.tp_a1)**self.tp_a2 + (temp_atmo / self.tp_a3)**self.tp_a4
            damage_frac_output = 1 - (1 / (1 + dam))
        else:
            damage_frac_output = self.damag_int * temp_atmo + self.damag_quad * temp_atmo**self.damag_expo
        self.damage_fraction_df[GlossaryCore.DamageFractionOutput] = damage_frac_output.values

    def compute_CO2_damage_price(self):
        """
        Compute CO2 tax - CO2 damage constraint:
                 CO2 tax - fact * CO2_damage_price  > 0  
            with CO2_damage_price[year] = 1e3 * 1.01**(year_start-year) * mean(damage_df[year:year+25] (T$)) / total_emissions_ref (Gt)
        """
        if 'complex128' in [self.damage_fraction_df[GlossaryCore.DamageFractionOutput].values.dtype]:
            arr_type = 'complex128'
        else:
            arr_type = 'float64'

        co2_damage_price = np.zeros(
            len(self.damage_fraction_df.index), dtype=arr_type)

        damages = self.damage_df[GlossaryCore.Damages].values.tolist()

        for i, year in enumerate(self.damage_fraction_df.index):

            if year == self.year_end:
                co2_damage_price[i] = 1e3 * 1.01**i * \
                    damages[i] / self.total_emissions_ref
            else:
                k = max(year + 25 - self.year_end, 0)
                co2_damage_price[i] = 1e3 * 1.01**i * \
                    np.mean(damages[i:i + 25 - k]) / self.total_emissions_ref

        self.co2_damage_price_df = pd.DataFrame(
            {GlossaryCore.Years: self.damage_fraction_df.index,
             'CO2_damage_price': co2_damage_price})

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
        for i in range(nb_years):
            for line in range(nb_years):
                if i == line:
                    temp_atmo = self.temperature_df.at[years[line],
                                                    GlossaryCore.TempAtmo]
                    if self.tipping_point_model:
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

        return ddamage_frac_output_temp_atmo

    def d_co2_damage_price_d_damages(self):
        '''
        Compute gradient of constraint wrt temp_atmo and economics
        '''
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        d_co2_damage_price_d_damages = np.zeros((nb_years, nb_years))

        for i, year in enumerate(years):
            if year == self.year_end:
                d_co2_damage_price_d_damages[i, i] = 1e3 * 1.01**i / self.total_emissions_ref
            else:
                k = max(year + 25 - self.year_end, 0)
                for j in range(0, 25 - k):
                    d_co2_damage_price_d_damages[i, i + j] = 1e3 * 1.01**i / (25 - k) / self.total_emissions_ref

        return d_co2_damage_price_d_damages

    def compute(self, damage_df, temperature_df):
        """
        Compute the outputs of the pyworld3
        """
        self.damage_df = damage_df
        self.damage_df.index = self.damage_df[GlossaryCore.Years].values

        self.temperature_df = temperature_df
        self.temperature_df.index = self.temperature_df[GlossaryCore.Years].values

        self.create_dataframe()
        self.compute_damage_fraction_of_gdp()

        self.damage_fraction_df = self.damage_fraction_df.replace([np.inf, -np.inf], np.nan)
        self.compute_CO2_damage_price()
        self.damage_fraction_df.fillna(0.0, inplace=True)

        return self.damage_fraction_df, self.co2_damage_price_df
