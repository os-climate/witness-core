"""
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
"""

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class UtilityModel:
    """
    Used to compute damage from climate change
    """

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]  # time_step
        self.conso_elasticity = self.param["conso_elasticity"]  # elasmu
        self.init_rate_time_pref = self.param["init_rate_time_pref"]  # prstp
        self.scaleone = self.param["scaleone"]  # scaleone
        self.scaletwo = self.param["scaletwo"]  # scale2

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        utility_df = pd.DataFrame(
            index=years_range,
            columns=[
                "year",
                GlossaryCore.UtilityDiscountRate,
                "period_utility",
                GlossaryCore.DiscountedUtility,
                GlossaryCore.Welfare,
            ],
        )
        utility_df["year"] = self.years_range
        self.utility_df = utility_df
        return utility_df

    def compute__u_discount_rate(self, year):
        """
        Compute Average utility social discount rate
         rr(t) = 1/((1+prstp)**(tstep*(t.val-1)));
        """
        t = ((year - self.year_start) / self.time_step) + 1
        u_discount_rate = 1 / ((1 + self.init_rate_time_pref) ** (self.time_step * (t - 1)))
        self.utility_df.loc[year, GlossaryCore.UtilityDiscountRate] = u_discount_rate
        return u_discount_rate

    def compute_period_utility(self, year):
        """
        Compute utility for period t
         Args:
            :param consumption_pc: per capita consumption
             :type consumption_pc: float
        (percapitaconso**(1-elasmu)-1)/(1-elasmu)-1
        """
        pc_consumption = self.economics_df.loc[year, GlossaryCore.PerCapitaConsumption]
        period_utility = (pc_consumption ** (1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
        self.utility_df.loc[year, "period_utility"] = period_utility
        return period_utility

    def compute_discounted_utility(self, year):
        """
        period Utility
        PERIODU(t) * L(t) * rr(t)
        """
        period_utility = self.utility_df.loc[year, "period_utility"]
        population = self.economics_df.loc[year, GlossaryCore.PopulationValue]
        u_discount_rate = self.utility_df.loc[year, GlossaryCore.UtilityDiscountRate]
        discounted_utility = period_utility * population * u_discount_rate
        self.utility_df.loc[year, GlossaryCore.DiscountedUtility] = discounted_utility
        return discounted_utility

    def compute_welfare(self):  # rescale
        """
        Compute welfare
        tstep * scale1 * sum(t,  CEMUTOTPER(t)) + scale2
        """
        sum_u = sum(self.utility_df[GlossaryCore.DiscountedUtility])
        #         if rescale:
        #             welfare = self.time_step * self.scaleone * sum_u + self.scaletwo
        #         else:
        #             welfare = self.time_step * self.scaleone * sum_u
        #        return welfare
        self.utility_df.loc[self.year_end, GlossaryCore.Welfare] = sum_u
        return sum_u

    def compute(self, economics_df, emissions_df, temperature_df):
        """pyworld3 execution"""

        self.create_dataframe()

        self.economics_df = economics_df.set_index(self.years_range)
        self.emissions_df = emissions_df.set_index(self.years_range)
        self.temperature_df = temperature_df.set_index(self.years_range)
        for year in self.years_range:
            self.compute__u_discount_rate(year)
            self.compute_period_utility(year)
            self.compute_discounted_utility(year)
        self.compute_welfare()
        return self.utility_df
