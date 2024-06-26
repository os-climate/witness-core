"""
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/06 Copyright 2023 Capgemini

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


class SectorModel:
    """
    Sector pyworld3
    General implementation of sector pyworld3
    """

    # Units conversion
    conversion_factor = 1.0

    def __init__(self):
        """
        Constructor
        """
        self.energy_production = None
        self.productivity_df = None
        self.capital_df = None
        self.production_df = None
        self.damage_df = None
        self.workforce_df = None
        self.growth_rate_df = None
        self.lt_energy_eff = None
        self.emax_enet_constraint = None
        self.gdp_percentage_per_section_df = None
        self.section_list = []
        self.section_gdp_df = None
        self.energy_consumption_percentage_per_section_df = None
        self.section_energy_consumption_df = None
        self.range_energy_eff_cstrt = None
        self.energy_eff_xzero_constraint = None

    def configure_parameters(self, inputs_dict, sector_name):
        """
        Configure with inputs_dict from the discipline
        """
        # years range for long term energy efficiency
        self.years_lt_energy_eff = np.arange(1950, 2120)
        self.prod_function_fitting = inputs_dict["prod_function_fitting"]
        if self.prod_function_fitting:
            self.energy_eff_max_range_ref = inputs_dict["energy_eff_max_range_ref"]
            self.hist_sector_invest = inputs_dict["hist_sector_investment"]

        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
        self.time_step = inputs_dict[GlossaryCore.TimeStep]
        self.years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_years = len(self.years_range)
        self.sector_name = sector_name
        self.section_list = GlossaryCore.SectionDictSectors[self.sector_name]
        self.gdp_percentage_per_section_df = inputs_dict[
            f"{self.sector_name}.{GlossaryCore.SectionGdpPercentageDfValue}"
        ]
        self.energy_consumption_percentage_per_section_df = inputs_dict[
            f"{self.sector_name}.{GlossaryCore.SectionEnergyConsumptionPercentageDfValue}"
        ]

        def correct_years(df):

            input_dict = {GlossaryCore.Years: self.years_range}
            input_dict.update({section: df[section].values[0] for section in self.section_list})
            return pd.DataFrame(input_dict)

        self.gdp_percentage_per_section_df = correct_years(self.gdp_percentage_per_section_df)
        self.energy_consumption_percentage_per_section_df = correct_years(
            self.energy_consumption_percentage_per_section_df
        )
        self.productivity_start = inputs_dict["productivity_start"]
        # self.init_gross_output = inputs_dict[GlossaryCore.InitialGrossOutput['var_name']]
        self.capital_start = inputs_dict["capital_start"]
        self.productivity_gr_start = inputs_dict["productivity_gr_start"]
        self.decline_rate_tfp = inputs_dict["decline_rate_tfp"]
        self.depreciation_capital = inputs_dict["depreciation_capital"]
        self.frac_damage_prod = inputs_dict[GlossaryCore.FractionDamageToProductivityValue]
        self.damage_to_productivity = inputs_dict[GlossaryCore.DamageToProductivity]
        self.init_output_growth = inputs_dict["init_output_growth"]
        self.output_alpha = inputs_dict["output_alpha"]
        self.output_gamma = inputs_dict["output_gamma"]
        self.energy_eff_k = inputs_dict["energy_eff_k"]
        self.energy_eff_cst = inputs_dict["energy_eff_cst"]
        self.energy_eff_xzero = inputs_dict["energy_eff_xzero"]
        self.energy_eff_max = inputs_dict["energy_eff_max"]
        self.capital_utilisation_ratio = inputs_dict["capital_utilisation_ratio"]
        self.max_capital_utilisation_ratio = inputs_dict["max_capital_utilisation_ratio"]
        self.ref_emax_enet_constraint = inputs_dict["ref_emax_enet_constraint"]
        self.compute_climate_impact_on_gdp = inputs_dict["assumptions_dict"]["compute_climate_impact_on_gdp"]
        if not self.compute_climate_impact_on_gdp:
            self.damage_to_productivity = False

        self.sector_name = sector_name

        self.init_dataframes()

    def init_dataframes(self):
        """
        Init dataframes with years
        """
        self.years = np.arange(self.year_start, self.year_end + 1)
        default_index = self.years
        self.capital_df = pd.DataFrame(
            index=default_index, columns=GlossaryCore.CapitalDf["dataframe_descriptor"].keys(), dtype=float
        )
        self.production_df = pd.DataFrame(
            index=default_index, columns=GlossaryCore.ProductionDf["dataframe_descriptor"].keys(), dtype=float
        )
        self.section_gdp_df = pd.DataFrame(
            index=default_index, columns=GlossaryCore.SectionGdpDf["dataframe_descriptor"].keys(), dtype=float
        )
        self.damage_df = pd.DataFrame(
            index=default_index, columns=GlossaryCore.DamageDetailedDf["dataframe_descriptor"].keys(), dtype=float
        )
        self.productivity_df = pd.DataFrame(
            index=default_index, columns=GlossaryCore.ProductivityDf["dataframe_descriptor"].keys(), dtype=float
        )
        self.growth_rate_df = pd.DataFrame(index=default_index, columns=[GlossaryCore.Years, "net_output_growth_rate"])
        self.production_df[GlossaryCore.Years] = self.years
        self.section_gdp_df[GlossaryCore.Years] = self.years
        self.damage_df[GlossaryCore.Years] = self.years
        self.capital_df[GlossaryCore.Years] = self.years
        self.productivity_df[GlossaryCore.Years] = self.years
        self.growth_rate_df[GlossaryCore.Years] = self.years
        self.capital_df.loc[self.year_start, GlossaryCore.Capital] = self.capital_start

    def set_coupling_inputs(self, inputs):
        """
        Set couplings inputs with right index, scaling...
        """
        # If fitting takes investment from historical input not coupling
        if self.prod_function_fitting:
            self.investment_df = self.hist_sector_invest
            self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        else:
            self.investment_df = inputs[f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}"]
            self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        # scale energy production
        self.energy_production = inputs[GlossaryCore.EnergyProductionValue]
        self.workforce_df = inputs[GlossaryCore.WorkforceDfValue]
        self.workforce_df.index = self.workforce_df[GlossaryCore.Years].values
        self.damage_fraction_df = inputs[GlossaryCore.DamageFractionDfValue]
        self.damage_fraction_df.index = self.damage_fraction_df[GlossaryCore.Years].values

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        prod_growth_rate = self.productivity_gr_start * np.exp(
            -self.decline_rate_tfp * (self.years_range - self.year_start)
        )
        self.productivity_df[GlossaryCore.ProductivityGrowthRate] = prod_growth_rate

    def compute_productivity(self, year):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        if year == self.year_start:
            self.productivity_df.loc[year, GlossaryCore.Productivity] = self.productivity_start
            self.productivity_df.loc[year, GlossaryCore.ProductivityWithDamage] = self.productivity_start
            self.productivity_df.loc[year, GlossaryCore.ProductivityWithoutDamage] = self.productivity_start
        else:
            p_productivity_gr = self.productivity_df.at[year - self.time_step, GlossaryCore.ProductivityGrowthRate]

            p_productivity_w_damage = self.productivity_df.at[
                year - self.time_step, GlossaryCore.ProductivityWithDamage
            ]
            damefrac = self.damage_fraction_df.at[year, GlossaryCore.DamageFractionOutput]
            productivity_w_damage = (1 - self.frac_damage_prod * damefrac) * (
                p_productivity_w_damage / (1 - p_productivity_gr)
            )

            p_productivity_wo_damage = self.productivity_df.at[
                year - self.time_step, GlossaryCore.ProductivityWithoutDamage
            ]
            productivity_wo_damage = p_productivity_wo_damage / (1 - p_productivity_gr)

            self.productivity_df.loc[year, GlossaryCore.ProductivityWithDamage] = productivity_w_damage
            self.productivity_df.loc[year, GlossaryCore.ProductivityWithoutDamage] = productivity_wo_damage
            if self.damage_to_productivity:
                self.productivity_df.loc[year, GlossaryCore.Productivity] = productivity_w_damage
            else:
                self.productivity_df.loc[year, GlossaryCore.Productivity] = productivity_wo_damage

    def compute_capital(self, year):
        """
        K(t), Capital for time period, trillions $USD
        Args:
            :param capital: capital
            :param depreciation: depreciation rate
            :param investment: investment
            K(t) = K(t-1)*(1-depre_rate) + investment(t-1)
        """
        if year > self.year_end:
            pass
        else:
            # Capital
            investment = self.investment_df.loc[year - self.time_step, GlossaryCore.InvestmentsValue]
            capital = self.capital_df.at[year - self.time_step, GlossaryCore.Capital]
            capital_a = capital * (1 - self.depreciation_capital) + investment
            self.capital_df.loc[year, GlossaryCore.Capital] = capital_a

            return capital_a

    def compute_usable_capital(self, year):
        """Usable capital is the part of the capital stock that can be used in the production process.
        To be usable the capital needs enough energy.
        K_u = K*(E/E_max)
        E is energy in Twh and K is capital in trill dollars constant 2020
        Output: usable capital in trill dollars constant 2020
        """
        capital = self.capital_df.loc[year, GlossaryCore.Capital]

        usable_capital_unbounded = self.capital_df.loc[year, GlossaryCore.UsableCapitalUnbounded]
        upper_bound = self.max_capital_utilisation_ratio * capital

        usable_capital = (
            upper_bound if np.real(usable_capital_unbounded) > np.real(upper_bound) else usable_capital_unbounded
        )

        self.capital_df.loc[year, GlossaryCore.UsableCapital] = usable_capital

    def compute_gross_output(self, year):
        """Compute the gdp
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1)
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df.loc[year, GlossaryCore.Productivity]
        working_pop = self.workforce_df.loc[year, self.sector_name]
        capital_u = self.capital_df.loc[year, GlossaryCore.UsableCapital]
        # If gamma == 1/2 use sqrt but same formula
        if gamma == 1 / 2:
            output = productivity * (alpha * np.sqrt(capital_u) + (1 - alpha) * np.sqrt(working_pop)) ** 2
        else:
            output = productivity * (alpha * capital_u**gamma + (1 - alpha) * (working_pop) ** gamma) ** (1 / gamma)
        self.production_df.loc[year, GlossaryCore.GrossOutput] = output

        return output

    def compute_output_net_of_damage(self, year):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damage_fraction_df.at[year, GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df.at[year, GlossaryCore.GrossOutput]

        if not self.compute_climate_impact_on_gdp:
            output_net_of_d = gross_output
        else:
            if damage_to_productivity:
                damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
                output_net_of_d = (1 - damage) * gross_output
            else:
                output_net_of_d = gross_output * (1 - damefrac)
        self.production_df.loc[year, GlossaryCore.OutputNetOfDamage] = output_net_of_d
        return output_net_of_d

    def compute_output_net_of_damage_per_section(self):
        """
        Splitting output net of damages between sections of the sector
        """
        section_gdp_df = {GlossaryCore.Years: self.years}
        for section in self.section_list:
            section_gdp_df[section] = (
                self.production_df[GlossaryCore.OutputNetOfDamage].values
                / 100.0
                * self.gdp_percentage_per_section_df[section]
            )

        self.section_gdp_df = pd.DataFrame(section_gdp_df)

    def compute_output_growth_rate(self, year):
        """Compute output growth rate for every year for the year before:
        output_growth_rate(t-1) = (output(t) - output(t-1))/output(t-1)
        for the last year we put the value of the previous year to avoid a 0
        """
        if year == self.year_start:
            pass
        else:
            output = self.production_df.at[year - self.time_step, GlossaryCore.OutputNetOfDamage]
            output_a = self.production_df.at[year, GlossaryCore.OutputNetOfDamage]
            output = max(1e-6, output)
            output_growth = ((output_a - output) / output) / self.time_step
            self.growth_rate_df.loc[year - self.time_step, "net_output_growth_rate"] = output_growth
        # For the last year put the vale of the year before
        if year == self.year_end:
            self.growth_rate_df.loc[year, "net_output_growth_rate"] = output_growth

    # For production fitting optim  only
    def compute_long_term_energy_efficiency(self):
        """Compute energy efficiency function on a longer time scale to analyse shape
        of the function.
        """
        # period
        years = self.years_lt_energy_eff
        # param
        k = self.energy_eff_k
        cst = self.energy_eff_cst
        xo = self.energy_eff_xzero
        max_e = self.energy_eff_max
        # compute energy_efficiency
        energy_efficiency = cst + max_e / (1 + np.exp(-k * (years - xo)))
        self.lt_energy_eff = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.EnergyEfficiency: energy_efficiency})
        return self.lt_energy_eff

    def compute_energy_eff_constraints(self):
        """
        Compute constraints for energy efficiency fitting
        One constraint to limit the range of variation of the energy efficiency max/min < some chosen value
        One constraint to limit the value of the sigmoid midpoint (year)
        """
        # constraint for diff between min and max value
        self.range_energy_eff_cstrt = (
            self.energy_eff_cst + self.energy_eff_max
        ) / self.energy_eff_cst - self.energy_eff_max_range_ref
        self.range_energy_eff_cstrt = np.array([self.range_energy_eff_cstrt])

        return self.range_energy_eff_cstrt

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        years = self.capital_df[GlossaryCore.Years].values
        energy_efficiency = self.energy_eff_cst + self.energy_eff_max / (
            1 + np.exp(-self.energy_eff_k * (years - self.energy_eff_xzero))
        )
        self.capital_df[GlossaryCore.EnergyEfficiency] = energy_efficiency

    def compute_unbounded_usable_capital(self):
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        usable_capital_unbounded = self.capital_utilisation_ratio * net_energy_production * energy_efficiency
        self.capital_df[GlossaryCore.UsableCapitalUnbounded] = usable_capital_unbounded

    def compute_energy_usage(self):
        """Wasted energy is the overshoot of energy production not used by usable capital"""
        capital = self.capital_df[GlossaryCore.Capital].values
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values * 1e3  # PWh to TWh
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        optimal_energy_production = (
            self.max_capital_utilisation_ratio * capital / self.capital_utilisation_ratio / energy_efficiency * 1e3
        )
        self.productivity_df[GlossaryCore.OptimalEnergyProduction] = optimal_energy_production
        used_energy = net_energy_production * 0.0
        index_inf = np.real(net_energy_production) < np.real(optimal_energy_production)
        used_energy[index_inf] = net_energy_production[index_inf]
        used_energy[~index_inf] = optimal_energy_production[~index_inf]
        self.productivity_df[GlossaryCore.UsedEnergy] = used_energy
        unused_energy = net_energy_production * 0
        index_sup = np.real(net_energy_production) - np.real(optimal_energy_production) > 0.0
        unused_energy[index_sup] = (net_energy_production - optimal_energy_production)[index_sup]
        self.productivity_df[GlossaryCore.UnusedEnergy] = unused_energy

    def compute_energy_wasted_objective(self):
        """Computes normalized energy wasted constraint. Ewasted=max(Enet - Eoptimal, 0)
        Normalize by total energy since Energy wasted is around 10% of total energy => have constraint around 0.1
        which can be compared to the negative welfare objective (same order of magnitude)
        """
        # total energy is supposed to be > 0.
        energy_wasted_objective = (
            self.productivity_df[GlossaryCore.UnusedEnergy].values.sum()
            / self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        )

        self.energy_wasted_objective = np.array([energy_wasted_objective])

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor), and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        applied_productivity = self.productivity_df[GlossaryCore.Productivity].values

        estimated_damage_from_productivity_loss = (
            (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output
        )
        if self.damage_to_productivity:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = np.zeros_like(estimated_damage_from_productivity_loss)

        self.damage_df[GlossaryCore.DamagesFromProductivityLoss] = damage_from_productivity_loss
        self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.damage_fraction_df[GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        net_output = self.production_df[GlossaryCore.OutputNetOfDamage].values

        damage_from_climate = np.zeros_like(gross_output)
        if self.compute_climate_impact_on_gdp:
            damage_from_climate = gross_output - net_output
            estimated_damage_from_climate = damage_from_climate
        else:
            if self.damage_to_productivity:
                estimated_damage_from_climate = (
                    gross_output * damefrac * (1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac)
                )
            else:
                estimated_damage_from_climate = gross_output * damefrac

        self.damage_df[GlossaryCore.DamagesFromClimate] = damage_from_climate
        self.damage_df[GlossaryCore.EstimatedDamagesFromClimate] = estimated_damage_from_climate

    def compute_total_damages(self):
        """Damages are the sum of damages from climate + damges from loss of productivity"""

        self.damage_df[GlossaryCore.EstimatedDamages] = (
            self.damage_df[GlossaryCore.EstimatedDamagesFromClimate]
            + self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss]
        )
        self.damage_df[GlossaryCore.Damages] = (
            self.damage_df[GlossaryCore.DamagesFromClimate] + self.damage_df[GlossaryCore.DamagesFromProductivityLoss]
        )

    def compute_energy_consumption_per_section(self):
        """
        Computing the energy consumption for each section of the sector

        section_energy_consumption (PWh) = sector_energy_production (Pwh) x section_energy_consumption_percentage (%)
        """
        section_energy_consumption = {GlossaryCore.Years: self.years}
        sector_energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values
        for section in self.section_list:
            section_energy_consumption[section] = (
                sector_energy_production * self.energy_consumption_percentage_per_section_df[section].values / 100.0
            )
        self.section_energy_consumption_df = pd.DataFrame(section_energy_consumption)

    # RUN
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.init_dataframes()
        self.set_coupling_inputs(inputs)
        self.compute_productivity_growthrate()
        self.compute_energy_efficiency()
        self.compute_unbounded_usable_capital()

        # iterate over years
        for year in self.years_range:
            self.compute_productivity(year)
            self.compute_usable_capital(year)
            self.compute_gross_output(year)
            self.compute_output_net_of_damage(year)
            self.compute_output_growth_rate(year)
            # capital t+1 :
            self.compute_capital(year + 1)

        if self.prod_function_fitting:
            self.compute_long_term_energy_efficiency()
            self.compute_energy_eff_constraints()

        self.compute_output_net_of_damage_per_section()

        self.compute_energy_consumption_per_section()

        self.compute_energy_usage()
        self.compute_energy_wasted_objective()
        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()

        self.output_types_to_float()

    ### GRADIENTS ###

    def _null_derivative(self):
        nb_years = len(self.years_range)
        return np.zeros((nb_years, nb_years))

    def _identity_derivative(self):
        nb_years = len(self.years_range)
        return np.identity(nb_years)

    def compute_doutput_dworkforce(self):
        """Gradient for output output wrt workforce
        output = productivity * (alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma)**(1/gamma)
        """
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(workingpop) = (1-alpha)*gamma*workingpop**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        # first line stays at zero since derivatives of initial values are zero
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop) ** gamma
        g_prime = (1 - alpha) * gamma * working_pop ** (gamma - 1)
        f_prime = productivity * (1 / gamma) * g * g_prime
        doutput = doutput @ np.diag(f_prime)
        return doutput

    def doutput_denergy(self, dcapitalu_denergy):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput_dcap = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        # Derivative of output wrt capital
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(capital) = alpha*gamma*capital**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop) ** gamma
        g_prime = alpha * gamma * capital_u ** (gamma - 1)
        f_prime = productivity * (1 / gamma) * g * g_prime
        doutput_dcap *= f_prime
        # Then doutput = doutput_d_prod * dproductivity
        doutput = np.dot(dcapitalu_denergy, doutput_dcap)
        return doutput

    def doutput_ddamage(self, dproductivity):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        # Derivative of output wrt productivity
        doutput_dprod = np.diag((alpha * capital_u**gamma + (1 - alpha) * working_pop**gamma) ** (1 / gamma))
        # Then doutput = doutput_d_prod * dproductivity
        doutput = np.dot(doutput_dprod, dproductivity)
        return doutput

    def dcapital_dinvest(self):
        """Compute derivative of capital wrt investments."""
        nb_years = self.nb_years
        # capital depends on invest from year before. diagonal k-1
        dcapital = np.eye(nb_years, k=-1)
        d_Ku_d_invests = self._null_derivative()
        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.0
        for i in range(0, nb_years - 1):
            for j in range(0, i + 1):
                dcapital[i + 1, j] += dcapital[i, j] * (1 - self.depreciation_capital)
                d_Ku_d_invests[i + 1, j] += index_zeros[i + 1] * dcapital[i + 1, j] * self.max_capital_utilisation_ratio

        return dcapital, d_Ku_d_invests

    def d_enegy_wasted_obj_d_invest(self, d_capital_d_invest):
        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.0
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_Ew_d_invest = (
            -np.diag(
                index_zeros
                * self.max_capital_utilisation_ratio
                * 1e3
                / self.capital_utilisation_ratio
                / energy_efficiency
            )
            @ d_capital_d_invest
        )

        sum_energy_prod = self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        d_EWO_d_EW = np.ones_like(self.years) / sum_energy_prod
        d_EWO_d_invests = d_EWO_d_EW @ d_Ew_d_invest
        return d_Ew_d_invest, d_EWO_d_invests

    def doutput_dinvest(self, d_usable_capital_d_invest):
        alpha = self.output_alpha
        gamma = self.output_gamma
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        # Derivative of output wrt usable capital
        doutput_dusable_capital = np.diag(
            productivity
            * alpha
            * capital_u ** (gamma - 1)
            * (alpha * capital_u**gamma + (1 - alpha) * (working_pop) ** gamma) ** (1 / gamma - 1)
        )
        # Then doutput = doutput_d_prod * dproductivity
        doutput_dinvests = doutput_dusable_capital @ d_usable_capital_d_invest
        return doutput_dinvests

    def demaxconstraint(self, dcapital):
        """Compute derivative of e_max and emax constraint using derivative of capital.
        For all inputs that impacts e_max through capital
        """
        # e_max = capital*1e3/ (capital_utilisation_ratio * energy_efficiency)
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        demax = np.identity(self.nb_years)
        demax *= 1e3 / (self.capital_utilisation_ratio * energy_efficiency)
        demax = np.dot(demax, dcapital)
        demaxconstraint_demax = demax * self.max_capital_utilisation_ratio / self.ref_emax_enet_constraint
        return demaxconstraint_demax

    def dnetoutput(self, doutput):
        """Compute the derivatives of net output using derivatives of gross output
         if damage_to_productivity:
            damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        """
        if not self.compute_climate_impact_on_gdp:
            return doutput
        damefrac = self.damage_fraction_df[GlossaryCore.DamageFractionOutput].values
        if self.damage_to_productivity:
            dnet_output = (1 - damefrac) / (1 - self.frac_damage_prod * damefrac) * doutput
        else:
            dnet_output = (1 - damefrac) * doutput
        return dnet_output

    def dnetoutput_ddamage(self, doutput):
        """Compute the derivatives of net output wrt damage using derivatives of gross output
         if damage_to_productivity:
            damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        """
        frac = self.frac_damage_prod
        years = self.years_range
        nb_years = len(years)
        dnet_output = np.zeros((nb_years, nb_years))
        if self.compute_climate_impact_on_gdp:
            for i in range(0, nb_years):
                output = self.production_df.at[years[i], GlossaryCore.GrossOutput]
                damefrac = self.damage_fraction_df.at[years[i], GlossaryCore.DamageFractionOutput]
                for j in range(0, i + 1):
                    if i == j:
                        if self.damage_to_productivity:
                            dnet_output[i, j] = (frac - 1) / ((frac * damefrac - 1) ** 2) * output + (1 - damefrac) / (
                                1 - frac * damefrac
                            ) * doutput[i, j]
                        else:
                            dnet_output[i, j] = -output + (1 - damefrac) * doutput[i, j]
                    else:
                        if self.damage_to_productivity:
                            dnet_output[i, j] = (1 - damefrac) / (1 - frac * damefrac) * doutput[i, j]
                        else:
                            dnet_output[i, j] = (1 - damefrac) * doutput[i, j]
        return dnet_output

    def d_Y_Ku_Ew_Constraint_d_energy(self):
        """
        Derivative of :
        - usable capital
        - Non-energy capital
        - gross output
        - lower bound constraint
        - Energy_wasted
        wrt energy
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[self.sector_name].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values

        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_UKu_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)
        d_Ku_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)

        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.0

        d_Ku_d_E[index_zeros, index_zeros] = 0.0

        damefrac = self.damage_fraction_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = (
            1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
        )
        d_K_d_E = self._null_derivative()
        dY_dE = np.diag(
            productivity
            * alpha
            * usable_capital ** (gamma - 1)
            * np.diag(d_Ku_d_E)
            * (alpha * usable_capital**gamma + (1 - alpha) * working_pop**gamma) ** (1.0 / gamma - 1.0)
        )
        # dY_dE[0, 0] = 0.

        for i in range(1, self.nb_years + 1):
            for j in range(1, i):
                dY_dE[i - 1, j] = (
                    productivity[i - 1]
                    * alpha
                    * usable_capital[i - 1] ** (gamma - 1)
                    * d_Ku_d_E[i - 1, j]
                    * (alpha * usable_capital[i - 1] ** gamma + (1 - alpha) * working_pop[i - 1] ** gamma)
                    ** (1.0 / gamma - 1.0)
                )
                if i < self.nb_years:
                    d_K_d_E[i, j] = (1 - self.depreciation_capital) * d_K_d_E[i - 1, j]
                    d_Ku_d_E[i, j] = index_zeros[i] * self.max_capital_utilisation_ratio * d_K_d_E[i, j]

        # Energy_wasted Ew = E - KNE * k where k = max_capital_utilisation_ratio/capital_utilisation_ratio/energy_efficiency*1.e3
        # energy_efficiency is function of the years. Eoptimal in TWh
        k = np.diag(self.max_capital_utilisation_ratio / self.capital_utilisation_ratio / energy_efficiency * 1.0e3)
        d_energy_wasted_d_energy = self._identity_derivative() * 1.0e3 - np.matmul(
            k, d_K_d_E
        )  # Enet converted from PWh to TWh
        # Since Ewasted = max(Enet - Eoptimal, 0.), gradient should be 0 when Enet - Eoptimal <=0, ie when Ewasted =0
        # => put to 0 the lines of the gradient matrix corresponding to the years where Ewasted=0
        matrix_of_years_E_is_wasted = (self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.0).astype(int)
        d_energy_wasted_d_energy = np.transpose(
            np.multiply(matrix_of_years_E_is_wasted, np.transpose(d_energy_wasted_d_energy))
        )
        d_sum_energy_wasted_d_energy_total = np.ones(self.nb_years) @ d_energy_wasted_d_energy
        d_sum_energy_total_d_energy_total = np.ones(self.nb_years) @ self._identity_derivative()

        sum_ewasted = self.productivity_df[GlossaryCore.UnusedEnergy].values.sum()
        sum_etotal = self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        # sumetotal is supposed > 0 otherwise no energy in the system => cannot work
        grad_energy_wasted_obj = (
            sum_etotal * d_sum_energy_wasted_d_energy_total - sum_ewasted * d_sum_energy_total_d_energy_total
        ) / sum_etotal**2

        return dY_dE, d_UKu_d_E, d_Ku_d_E, grad_energy_wasted_obj

    def d_productivity_w_damage_d_damage_frac_output(self):
        """derivative of productivity with damage wrt damage frac output"""
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)
        p_productivity_gr = self.productivity_df[GlossaryCore.ProductivityGrowthRate].values
        p_productivity = self.productivity_df[GlossaryCore.ProductivityWithDamage].values

        # derivative matrix initialization
        d_productivity_w_damage_d_damage_frac_output = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are
        # zero
        for i in range(1, nb_years):
            d_productivity_w_damage_d_damage_frac_output[i, i] = (
                1 - self.frac_damage_prod * self.damage_fraction_df.at[years[i], GlossaryCore.DamageFractionOutput]
            ) * d_productivity_w_damage_d_damage_frac_output[i - 1, i] / (
                1 - p_productivity_gr[i - 1]
            ) - self.frac_damage_prod * p_productivity[
                i - 1
            ] / (
                1 - p_productivity_gr[i - 1]
            )
            for j in range(1, i):
                d_productivity_w_damage_d_damage_frac_output[i, j] = (
                    (
                        1
                        - self.frac_damage_prod
                        * self.damage_fraction_df.at[years[i], GlossaryCore.DamageFractionOutput]
                    )
                    * d_productivity_w_damage_d_damage_frac_output[i - 1, j]
                    / (1 - p_productivity_gr[i - 1])
                )

        return d_productivity_w_damage_d_damage_frac_output

    def d_productivity_d_damage_frac_output(self):
        """gradient for productivity for damage_df"""
        d_productivity_d_damage_frac_output = self._null_derivative()

        if self.damage_to_productivity:
            d_productivity_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        return d_productivity_d_damage_frac_output

    def d_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        return derivative

    def d_estimated_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_df[GlossaryCore.DamageFractionOutput]

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = (
                    np.diag(damefrac * (1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac))
                    @ d_gross_output_d_user_input
                )
            else:
                derivative = np.diag(damefrac) @ d_gross_output_d_user_input

        return derivative

    def d_estimated_damages_from_climate_d_damage_frac_output(
        self, d_gross_output_d_user_input, d_net_output_d_user_input
    ):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_df[GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df[GlossaryCore.GrossOutput].values

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = d_gross_output_d_user_input @ np.diag(
                    damefrac * (1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac)
                ) + np.diag(gross_output) * np.diag(
                    (1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac) ** 2
                )

            else:
                derivative = d_gross_output_d_user_input @ np.diag(damefrac) + np.diag(gross_output)

        return derivative

    def d_damages_from_productivity_loss_d_damage_fraction_output(self, d_gross_output_d_damage_fraction_output):
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values

        d_productivity_w_damage_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        nb_years = len(self.years_range)
        d_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        d_estimated_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        if self.damage_to_productivity:
            for i in range(nb_years):
                d_estimated_damages_from_productivity_loss_d_damage_fraction_output[i] = (
                    d_gross_output_d_damage_fraction_output[i]
                    * (productivity_wo_damage[i] / productivity_w_damage[i] - 1)
                    + -gross_output[i]
                    * productivity_wo_damage[i]
                    / productivity_w_damage[i] ** 2
                    * d_productivity_w_damage_d_damage_frac_output[i]
                )
        else:
            d_estimated_damages_from_productivity_loss_d_damage_fraction_output = (
                np.diag((productivity_wo_damage - productivity_w_damage) / productivity_wo_damage)
                @ d_gross_output_d_damage_fraction_output
                - np.diag(gross_output / productivity_wo_damage) @ d_productivity_w_damage_d_damage_frac_output
            )
        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_damage_fraction_output = (
                d_estimated_damages_from_productivity_loss_d_damage_fraction_output
            )

        return (
            d_damages_from_productivity_loss_d_damage_fraction_output,
            d_estimated_damages_from_productivity_loss_d_damage_fraction_output,
        )

    def d_damages_from_productivity_loss_d_user_input(self, d_gross_output_d_user_input):
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values

        d_damages_from_productivity_loss_d_user_input = self._null_derivative()
        applied_productivity = self.productivity_df[GlossaryCore.Productivity].values
        d_estimated_damages_from_prod_loss_d_user_input = (
            np.diag((productivity_wo_damage - productivity_w_damage) / (applied_productivity))
            @ d_gross_output_d_user_input
        )

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_user_input = d_estimated_damages_from_prod_loss_d_user_input

        return d_damages_from_productivity_loss_d_user_input, d_estimated_damages_from_prod_loss_d_user_input

    def d_damages_d_user_input(
        self, d_damages_from_climate_d_user_input, d_damages_from_productivity_loss_d_user_input
    ):
        return d_damages_from_climate_d_user_input + d_damages_from_productivity_loss_d_user_input

    def d_estimated_damages_d_user_input(
        self, d_estimated_damages_from_climate_d_user_input, d_estimated_damages_from_productivity_loss_d_user_input
    ):
        return d_estimated_damages_from_climate_d_user_input + d_estimated_damages_from_productivity_loss_d_user_input

    def d_section_energy_consumption_d_energy_production(self, section_name: str):
        return np.diag(self.energy_consumption_percentage_per_section_df[section_name].values / 100.0)

    def output_types_to_float(self):
        """make sure these dataframes columns have type float instead of object to avoid errors during
        seting of partial derivatives"""
        dataframes = [self.production_df, self.section_gdp_df, self.damage_df, self.capital_df, self.productivity_df]

        for df in dataframes:
            df.fillna(0.0, inplace=True)
