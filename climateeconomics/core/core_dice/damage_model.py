'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/07-2023/11/03 Copyright 2023 Capgemini

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


class DamageModel():
    '''
    Damage from climate change
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.init_damag_int = self.param["init_damag_int"]
        self.damag_int = self.param['damag_int']
        self.damag_quad = self.param['damag_quad']
        self.damag_expo = self.param['damag_expo']
        #self.init_damag_quad = self.param['init_damag_quad']
        self.exp_cont_f = self.param['exp_cont_f']
        self.cost_backtsop = self.param['cost_backstop']
        self.init_cost_bacsktop = self.param['init_cost_backstop']
        self.gr_base_carbonprice = self.param['gr_base_carbonprice']
        self.init_base_carbonprice = self.param['init_base_carbonprice']
        self.tipping_point = self.param['tipping_point']
        self.tp_a1 = self.param['tp_a1']
        self.tp_a2 = self.param['tp_a2']
        self.tp_a3 = self.param['tp_a3']
        self.tp_a4 = self.param['tp_a4']
        self.damage_to_productivity = self.param[GlossaryCore.DamageToProductivity]
        self.frac_damage_prod = self.param[GlossaryCore.FractionDamageToProductivityValue]

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1)
        self.years_range = years_range
        damage_df = pd.DataFrame(
            index=years_range,
            columns=[GlossaryCore.Years,
                     GlossaryCore.Damages,
                     GlossaryCore.DamageFractionOutput,
                     'backstop_price',
                     'adj_backstop_cost',
                     'abatecost',
                     'marg_abatecost',
                     'carbon_price',
                     GlossaryCore.BaseCarbonPrice])
        damage_df[GlossaryCore.Years] = years_range
        self.damage_df = damage_df
        return damage_df

    def compute_base_carbon_price(self, year):
        """
        Compute base case carbon price
        """
        t = (year - self.year_start) + 1
        base_carbon_price = self.init_base_carbonprice * \
            (1 + self.gr_base_carbonprice)**(t - 1)
        self.damage_df.loc[year, GlossaryCore.BaseCarbonPrice] = base_carbon_price
        return base_carbon_price

    def compute_backstop_price(self, year):
        """
        Compute backstop_price(t)
        """
        t = (year - self.year_start) + 1
        backstop_price = self.cost_backtsop * \
            (1 - self.init_cost_bacsktop)**(t - 1)
        self.damage_df.loc[year, 'backstop_price'] = backstop_price
        return backstop_price

    def compute_adj_backstop_cost(self, year):
        """
        Compute adjusted backstop cost at t
        using variables at t
        """
        backstop_price = self.damage_df.loc[year, 'backstop_price']
        sigma = self.emissions_df.loc[year, 'sigma']
        adj_backstop_cost = backstop_price * sigma / self.exp_cont_f / 1000
        self.damage_df.loc[year, 'adj_backstop_cost'] = adj_backstop_cost
        return adj_backstop_cost

    def compute_damage_fraction(self, year):
        """
        Compute damages fraction of output at t
        using variables at t
        If tipping point = True : Martin Weitzman damage function.
        """
        temp_atmo = self.temperature_df.loc[year, GlossaryCore.TempAtmo]
        if self.tipping_point:
            dam = (temp_atmo / self.tp_a1)**self.tp_a2 + \
                (temp_atmo / self.tp_a3)**self.tp_a4
            damage_frac_output = 1 - (1 / (1 + dam))
        else:
            damage_frac_output = self.damag_int * temp_atmo + \
                self.damag_quad * temp_atmo**self.damag_expo
        self.damage_df.loc[year, GlossaryCore.DamageFractionOutput] = damage_frac_output
        return damage_frac_output

    def compute_damages(self, year):
        """
        Compute damages (t) (trillions 2005 USD per year)
        using variables at t
        """
        gross_output = self.economics_df.loc[year, GlossaryCore.GrossOutput]
        damage_frac_output = self.damage_df.loc[year, GlossaryCore.DamageFractionOutput]
        damages = gross_output * damage_frac_output
        self.damage_df.loc[year, GlossaryCore.Damages] = damages
        return damages

    def compute_abatecost(self, year):
        """
        Compute abatement cost (t)  (trillions 2005 USD per year)
        using variables at t
        """
        gross_output = self.economics_df.loc[year, GlossaryCore.GrossOutput]
        adj_backstop_cost = self.damage_df.loc[year, 'adj_backstop_cost']
        emission_control_rate = self.emissions_control_rate[year]
        abatecost = gross_output * adj_backstop_cost * \
            emission_control_rate ** self.exp_cont_f
        self.damage_df.loc[year, 'abatecost'] = abatecost
        return abatecost

    def compute_marg_abatecost(self, year):
        """
        marginal abatement cost at t, (2005$ per ton CO2)
        using variables at t
        """
        backstop_price = self.damage_df.loc[year, 'backstop_price']
        emissions_control_rate = self.emissions_control_rate[year]
        marg_abatecost = backstop_price * \
            emissions_control_rate ** (self.exp_cont_f - 1)
        self.damage_df.loc[year, 'marg_abatecost'] = marg_abatecost
        return marg_abatecost

    def compute_carbon_price(self, year):
        """
        Carbon price (2005$ per ton of CO2)
        """
        backstop_price = self.damage_df.loc[year, 'backstop_price']
        emissions_control_rate = self.emissions_control_rate[year]
        carbon_price = backstop_price * \
            emissions_control_rate ** (self.exp_cont_f - 1)
        self.damage_df.loc[year, 'carbon_price'] = carbon_price
        return carbon_price

    def compute(self, economics_df, emissions_df,
                temperature_df, emissions_control_rate):

        self.create_dataframe()
        emissions_control_rate = emissions_control_rate.set_index(
            self.years_range)
        self.emissions_control_rate = emissions_control_rate['value']
        self.economics_df = economics_df.set_index(
            self.years_range)
        self.emissions_df = emissions_df.set_index(
            self.years_range)
        self.temperature_df = temperature_df.set_index(
            self.years_range)
        for year in self.years_range:
            self.compute_base_carbon_price(year)
            self.compute_backstop_price(year)
            self.compute_adj_backstop_cost(year)
            self.compute_damage_fraction(year)
            self.compute_damages(year)
            self.compute_abatecost(year)
            self.compute_marg_abatecost(year)
            self.compute_carbon_price(year)

        self.damage_df = self.damage_df.replace([np.inf, -np.inf], np.nan)
        return self.damage_df.fillna(0.0)
