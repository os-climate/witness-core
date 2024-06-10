"""
Copyright 2022 Airbus SAS
Modifications on 2023/08/23-2023/11/03 Copyright 2023 Capgemini

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


class ConsumptionModel:
    """
    Used to compute population welfare, utility and consumption. Based on utility pyworld3
    """

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]  # time_step

        self.conso_elasticity = self.param["conso_elasticity"]  # elasmu
        self.init_rate_time_pref = self.param["init_rate_time_pref"]  # prstp
        self.alpha = self.param["alpha"]
        self.gamma = self.param["gamma"]
        self.initial_raw_energy_price = self.param["initial_raw_energy_price"]
        self.obj_option = self.param["welfare_obj_option"]
        self.init_discounted_utility = self.param["init_discounted_utility"]
        self.init_period_utility_pc = self.param["init_period_utility_pc"]
        self.discounted_utility_ref = self.param["discounted_utility_ref"]
        self.min_period_utility = 0.01
        self.lo_conso = self.param["lo_conso"]  # lower limit for conso
        self.lo_per_capita_conso = self.param["lo_per_capita_conso"]  # lower limit for conso per capita
        self.residential_energy_conso_ref = self.param[
            "residential_energy_conso_ref"
        ]  # residential energy consumption in 2019

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        utility_df = pd.DataFrame(
            index=years_range,
            columns=[
                GlossaryCore.Years,
                GlossaryCore.UtilityDiscountRate,
                GlossaryCore.PeriodUtilityPerCapita,
                GlossaryCore.DiscountedUtility,
                GlossaryCore.Welfare,
            ],
        )

        for key in utility_df.keys():
            utility_df[key] = 0
        utility_df[GlossaryCore.Years] = years_range
        self.utility_df = utility_df
        return utility_df

    def set_coupling_inputs(self):
        """
        Set couplings inputs with right index, scaling...
        """
        self.economics_df = self.inputs[GlossaryCore.EconomicsDfValue]
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values
        self.energy_mean_price = pd.DataFrame(
            {
                GlossaryCore.Years: self.inputs[GlossaryCore.EnergyMeanPriceValue][GlossaryCore.Years].values,
                GlossaryCore.EnergyPriceValue: self.inputs[GlossaryCore.EnergyMeanPriceValue][
                    GlossaryCore.EnergyPriceValue
                ].values,
            }
        )
        self.energy_mean_price.index = self.energy_mean_price[GlossaryCore.Years].values
        self.energy_price_ref = self.initial_raw_energy_price
        self.population_df = self.inputs[GlossaryCore.PopulationDfValue]
        self.population_df.index = self.population_df[GlossaryCore.Years].values
        self.investment_df = self.inputs[GlossaryCore.InvestmentDfValue]
        self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        self.residential_energy = self.inputs[GlossaryCore.ResidentialEnergyConsumptionDfValue]
        self.residential_energy.index = self.residential_energy[GlossaryCore.Years].values

    def compute_consumption(self, year):
        """Equation for consumption
        C, Consumption, trillions $USD
        Args:
            output: Utility output at t
            savings: Savings rate at t
        """
        net_output = self.economics_df.at[year, GlossaryCore.OutputNetOfDamage]
        investment = self.investment_df.at[year, GlossaryCore.InvestmentsValue]
        consumption = net_output - investment
        # lower bound for conso
        self.utility_df.loc[year, GlossaryCore.Consumption] = max(consumption, self.lo_conso)
        return consumption

    def compute_consumption_pc(self, year):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        consumption = self.utility_df.at[year, GlossaryCore.Consumption]
        population = self.population_df.at[year, GlossaryCore.PopulationValue]
        consumption_pc = consumption / population * 1000
        # Lower bound for pc conso
        self.utility_df.loc[year, GlossaryCore.PerCapitaConsumption] = max(consumption_pc, self.lo_per_capita_conso)
        return consumption_pc

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
        Compute utility for period t per capita
         Args:
            :param consumption_pc: per capita consumption
                    energy_price: energy price
                    energy_price_ref
             :type float

        ((percapitaconso**(1-elasmu)-1)/(1-elasmu)-1)*\
        energy_price_ref/energy_price

        """

        # Compute energy price ratio
        energy_price = self.energy_mean_price.at[year, GlossaryCore.EnergyPriceValue]
        energy_price_ratio = self.energy_price_ref / energy_price
        residential_energy = self.residential_energy.at[year, GlossaryCore.TotalProductionValue]
        residential_energy_ratio = residential_energy / self.residential_energy_conso_ref
        pc_consumption = self.utility_df.at[year, GlossaryCore.PerCapitaConsumption]
        period_utility = (pc_consumption ** (1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
        # need a limit for period utility because negative period utility is
        # not coherent and reverse gradient of utility vs energy price
        if period_utility < self.min_period_utility:
            period_utility = (
                self.min_period_utility / 10.0 * (9.0 + np.exp(period_utility / self.min_period_utility) * np.exp(-1))
            )

        adjusted_period_utility = period_utility * energy_price_ratio * residential_energy_ratio
        self.utility_df.loc[year, GlossaryCore.PeriodUtilityPerCapita] = adjusted_period_utility
        return period_utility

    def compute_discounted_utility(self, year):
        """
        period Utility
        PERIODU_pc(t) * rr(t) * L(t)
        """
        period_utility = self.utility_df.at[year, GlossaryCore.PeriodUtilityPerCapita]
        u_discount_rate = self.utility_df.at[year, GlossaryCore.UtilityDiscountRate]
        population = self.population_df.at[year, GlossaryCore.PopulationValue]
        discounted_utility = period_utility * u_discount_rate * population
        self.utility_df.loc[year, GlossaryCore.DiscountedUtility] = discounted_utility
        return discounted_utility

    def compute_welfare(self):  # rescalenose
        """
        Compute welfare
        tstep * scale1 * sum(t,  CEMUTOTPER(t)) + scale2
        """
        sum_u = sum(self.utility_df[GlossaryCore.DiscountedUtility])
        self.utility_df.loc[self.year_end, GlossaryCore.Welfare] = sum_u
        return sum_u

    def compute_welfare_objective(self):
        """
        Objective function: inputs : alpha, gamma and obj_option
        """
        obj_option = self.obj_option

        n_years = len(self.years_range)
        if obj_option == "last_utility":
            init_utility = self.init_period_utility_pc
            last_utility = self.utility_df.at[self.year_end, GlossaryCore.PeriodUtilityPerCapita]
            welfare_objective = np.asarray(
                [
                    self.alpha * init_utility / last_utility,
                ]
            )
            return welfare_objective

        elif obj_option == GlossaryCore.Welfare:
            init_discounted_utility = self.init_discounted_utility
            welfare = self.utility_df[GlossaryCore.Welfare][self.year_end]
            # To avoid pb during convergence
            if welfare / (init_discounted_utility * n_years) < 0.01:
                welfare = 0.01 + np.exp(welfare / (init_discounted_utility * n_years)) * np.exp(-0.02005033585350133)
            self.welfare = welfare
            welfare_objective = np.asarray(
                [
                    self.alpha * self.gamma * init_discounted_utility * n_years / welfare,
                ]
            )
            return welfare_objective

        else:
            # exception if objective option not in expected list
            raise Exception("unhandled objective option")

    def compute_negative_welfare_objective(self):
        """
        Compute welfare objective as - welfare / init_discounted_utility * n_years
        """
        n_years = len(self.years_range)

        init_discounted_utility = self.init_discounted_utility
        welfare = self.utility_df[GlossaryCore.Welfare][self.year_end]

        self.welfare = welfare
        welfare_objective = np.asarray([-welfare / (init_discounted_utility * n_years)])
        return welfare_objective

    def compute_min_utility_objective(self):
        """
        Objective function: inputs : alpha, gamma and discounted_utility_ref
        """
        init_discounted_utility = self.init_discounted_utility
        min_utility = min(list(self.utility_df[GlossaryCore.DiscountedUtility].values))
        # To avoid pb during convergence
        if min_utility / init_discounted_utility < 0.01:
            min_utility = 0.01 + np.exp(min_utility / init_discounted_utility) * np.exp(-0.02005033585350133)
        min_utility_objective = np.asarray(
            [
                self.alpha * (1 - self.gamma) * self.discounted_utility_ref / min_utility,
            ]
        )
        return min_utility_objective

    ######### GRADIENTS ########

    def compute_gradient(self):

        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        population = self.population_df[GlossaryCore.PopulationValue].values
        consumption = self.utility_df[GlossaryCore.Consumption].values
        pc_consumption = self.utility_df[GlossaryCore.PerCapitaConsumption].values
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values
        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        period_utility_pc = self.utility_df[GlossaryCore.PeriodUtilityPerCapita].values
        residential_energy = self.residential_energy[GlossaryCore.TotalProductionValue].values
        # compute gradient
        d_welfare_d_output_net_of_d = np.zeros((nb_years, nb_years))
        d_welfare_d_investment = np.zeros((nb_years, nb_years))
        d_welfare_d_population = np.zeros((nb_years, nb_years))

        d_consumption_d_output_net_of_d = np.identity(nb_years)
        # find index where lower bound reached
        theyears = np.where(consumption == self.lo_conso)[0]
        # Then for these years derivative = 0
        d_consumption_d_output_net_of_d[theyears] = 0

        d_pc_consumption_d_output_net_of_d = d_consumption_d_output_net_of_d / population * 1000
        theyears = np.where(pc_consumption == self.lo_per_capita_conso)[0]
        d_pc_consumption_d_output_net_of_d[theyears] = 0

        # For investment
        d_consumption_d_invest = -np.identity(nb_years)
        d_consumption_d_invest[theyears] = 0

        d_pc_consumption_d_investment = d_consumption_d_invest / population * 1000
        theyears = np.where(pc_consumption == self.lo_per_capita_conso)[0]
        d_pc_consumption_d_output_net_of_d[theyears] = 0

        d_pc_consumption_d_population = -1 * consumption / (population * population) * 1000 * np.identity(nb_years)
        theyears = np.where(pc_consumption == self.lo_per_capita_conso)[0]
        d_pc_consumption_d_population[theyears] = 0

        period_utility = (pc_consumption ** (1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
        theyears = np.where(period_utility < self.min_period_utility)[0]

        d_period_utility_pc_d_output_net_of_d = (
            d_pc_consumption_d_output_net_of_d
            * pc_consumption ** (-self.conso_elasticity)
            * self.energy_price_ref
            / energy_price
            * residential_energy
            / self.residential_energy_conso_ref
        )
        # limit min period utility
        d_period_utility_pc_d_output_net_of_d[theyears] = (
            d_period_utility_pc_d_output_net_of_d[theyears]
            * self.min_period_utility
            / 10.0
            * (np.exp(period_utility / self.min_period_utility) * np.exp(-1))
            / self.min_period_utility
        )

        d_period_utility_pc_d_investment = (
            d_pc_consumption_d_investment
            * pc_consumption ** (-self.conso_elasticity)
            * self.energy_price_ref
            / energy_price
            * residential_energy
            / self.residential_energy_conso_ref
        )
        d_period_utility_pc_d_investment[theyears] = (
            d_period_utility_pc_d_investment[theyears]
            * self.min_period_utility
            / 10.0
            * (np.exp(period_utility / self.min_period_utility) * np.exp(-1))
            / self.min_period_utility
        )

        d_period_utility_d_population = (
            d_pc_consumption_d_population
            * pc_consumption ** (-self.conso_elasticity)
            * self.energy_price_ref
            / energy_price
            * residential_energy
            / self.residential_energy_conso_ref
        )
        d_period_utility_d_population[theyears] = (
            d_period_utility_d_population[theyears]
            * self.min_period_utility
            / 10.0
            * (np.exp(period_utility / self.min_period_utility) * np.exp(-1))
            / self.min_period_utility
        )

        d_discounted_utility_d_output_net_of_d = d_period_utility_pc_d_output_net_of_d * u_discount_rate * population
        d_discounted_utility_d_investment = d_period_utility_pc_d_investment * u_discount_rate * population
        d_discounted_utility_d_population = (
            d_period_utility_d_population * u_discount_rate * population
            + period_utility_pc * u_discount_rate * np.identity(nb_years)
        )
        d_welfare_d_output_net_of_d[nb_years - 1] = d_discounted_utility_d_output_net_of_d.diagonal()
        d_welfare_d_investment[nb_years - 1] = d_discounted_utility_d_investment.diagonal()
        d_welfare_d_population[nb_years - 1] = d_discounted_utility_d_population.diagonal()

        return (
            d_pc_consumption_d_output_net_of_d,
            d_pc_consumption_d_investment,
            d_pc_consumption_d_population,
            d_period_utility_pc_d_output_net_of_d,
            d_period_utility_pc_d_investment,
            d_period_utility_d_population,
            d_discounted_utility_d_output_net_of_d,
            d_discounted_utility_d_investment,
            d_discounted_utility_d_population,
            d_welfare_d_output_net_of_d,
            d_welfare_d_investment,
            d_welfare_d_population,
        )

    def compute_gradient_energy_mean_price(self):
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        d_period_utility_d_energy_price = np.zeros((nb_years, nb_years))
        d_discounted_utility_d_energy_price = np.zeros((nb_years, nb_years))
        d_welfare_d_energy_price = np.zeros((nb_years, nb_years))

        population = self.population_df[GlossaryCore.PopulationValue].values
        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values

        d_period_utility_d_energy_price = (
            -1.0 * np.identity(nb_years) * self.utility_df[GlossaryCore.PeriodUtilityPerCapita].values / energy_price
        )

        d_discounted_utility_d_energy_price = d_period_utility_d_energy_price * u_discount_rate * population

        d_welfare_d_energy_price[nb_years - 1,] = d_discounted_utility_d_energy_price.diagonal()

        return d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, d_welfare_d_energy_price

    def compute_gradient_residential_energy(self):
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        d_period_utility_d_residential_energy = np.zeros((nb_years, nb_years))
        d_discounted_utility_d_residential_energy = np.zeros((nb_years, nb_years))
        d_welfare_d_residential_energy = np.zeros((nb_years, nb_years))

        population = self.population_df[GlossaryCore.PopulationValue].values
        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        residential_energy = self.residential_energy[GlossaryCore.TotalProductionValue].values

        d_period_utility_d_residential_energy = (
            self.utility_df[GlossaryCore.PeriodUtilityPerCapita].values / residential_energy * np.identity(nb_years)
        )

        d_discounted_utility_d_residential_energy = d_period_utility_d_residential_energy * u_discount_rate * population

        d_welfare_d_residential_energy[nb_years - 1,] = d_discounted_utility_d_residential_energy.diagonal()

        return (
            d_period_utility_d_residential_energy,
            d_discounted_utility_d_residential_energy,
            d_welfare_d_residential_energy,
        )

    def compute_gradient_objective(self):
        """
        if obj_option = 'last_utility': .alpha*init_utility/last_utility . with utility = period_utility_pc
        if obj_option = 'welfare :  alpha*init_discounted_utility*n_years/welfare
            if welfare < 1 : : alpha*initdiscounted_utility * n_years/
            (0.01+ np.exp(welfare/init_discounted_utility*n_years)*np.exp(-0.02005033585350133))
        """
        years = self.years_range
        period_utility_pc_0 = self.init_period_utility_pc
        period_utility_pc_end = self.utility_df.at[self.year_end, GlossaryCore.PeriodUtilityPerCapita]
        init_discounted_utility = self.init_discounted_utility

        n_years = len(years)

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_welfare = np.zeros(len(years))

        if self.obj_option == "last_utility":
            d_obj_d_period_utility_pc[-1] = -1.0 * self.alpha * period_utility_pc_0 / (period_utility_pc_end) ** 2

        elif self.obj_option == GlossaryCore.Welfare:
            welfare = self.utility_df[GlossaryCore.Welfare][self.year_end]
            if welfare / (init_discounted_utility * n_years) < 0.01:
                mask = np.append(np.zeros(len(years) - 1), np.array(1))
                f_prime = (
                    (1 / (init_discounted_utility * n_years))
                    * np.exp(welfare / (init_discounted_utility * n_years))
                    * np.exp(-0.02005033585350133)
                )
                f_squared = (
                    0.01 + np.exp(welfare / (init_discounted_utility * n_years)) * np.exp(-0.02005033585350133)
                ) ** 2
                d_obj_d_welfare = (
                    mask * self.alpha * self.gamma * init_discounted_utility * n_years * (-f_prime / f_squared)
                )
            else:
                mask = np.append(np.zeros(len(years) - 1), np.array(1))
                d_obj_d_welfare = -1.0 * mask * self.alpha * self.gamma * init_discounted_utility * n_years / welfare**2

        else:
            pass

        return d_obj_d_welfare, d_obj_d_period_utility_pc

    def compute_gradient_negative_objective(self):
        """

        welfare = welfare / init_discounted_utility*n_years

        """
        years = self.years_range
        init_discounted_utility = self.init_discounted_utility

        n_years = len(years)

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_welfare = np.zeros(len(years))

        mask = np.append(np.zeros(len(years) - 1), np.array(1))
        d_obj_d_welfare = -1.0 * mask / (init_discounted_utility * n_years)

        return d_obj_d_welfare, d_obj_d_period_utility_pc

    def compute_gradient_min_utility_objective(self):
        """ """
        years = self.years_range
        init_discounted_utility = self.init_discounted_utility

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_discounted_utility = np.zeros(len(years))

        min_utility = min(list(self.utility_df[GlossaryCore.DiscountedUtility].values))
        d_min_utility_d_discounted_utility = np.asarray(
            [
                1.0 if (val == min_utility) else 0.0
                for val in list(self.utility_df[GlossaryCore.DiscountedUtility].values)
            ],
            dtype=float,
        )
        if min_utility / init_discounted_utility < 0.01:

            f_prime = (
                d_min_utility_d_discounted_utility
                * (1 / init_discounted_utility)
                * np.exp(min_utility / init_discounted_utility)
                * np.exp(-0.02005033585350133)
            )
            f_squared = (0.01 + np.exp(min_utility / init_discounted_utility) * np.exp(-0.02005033585350133)) ** 2
            d_obj_d_discounted_utility = (
                self.alpha * (1 - self.gamma) * self.discounted_utility_ref * (-f_prime / f_squared)
            )
        else:
            d_obj_d_discounted_utility = (
                -1.0
                * d_min_utility_d_discounted_utility
                * self.alpha
                * (1 - self.gamma)
                * self.discounted_utility_ref
                / min_utility**2
            )

        return d_obj_d_discounted_utility, d_obj_d_period_utility_pc

    def compute(self, inputs):
        """pyworld3 execution"""
        self.inputs = inputs
        self.set_coupling_inputs()

        for year in self.years_range:
            self.compute_consumption(year)
            self.compute_consumption_pc(year)
            self.compute__u_discount_rate(year)
            self.compute_period_utility(year)
            self.compute_discounted_utility(year)
        self.compute_welfare()

        return self.utility_df
