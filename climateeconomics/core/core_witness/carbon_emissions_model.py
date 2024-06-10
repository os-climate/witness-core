"""
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
"""

import numpy as np
import pandas as pd
from energy_models.core.stream_type.carbon_models.nitrous_oxide import N2O

from climateeconomics.glossarycore import GlossaryCore


class CarbonEmissions:
    """
    Used to compute carbon emissions from gross output
    """

    GHG_TYPE_LIST = [N2O.name, GlossaryCore.CO2, GlossaryCore.CH4]

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()
        self.CO2_objective = None
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.init_gr_sigma = self.param["init_gr_sigma"]
        self.decline_rate_decarbo = self.param["decline_rate_decarbo"]
        self.init_indus_emissions = self.param["init_indus_emissions"]
        self.init_gross_output = self.param[GlossaryCore.InitialGrossOutput["var_name"]]
        self.init_cum_indus_emissions = self.param["init_cum_indus_emissions"]
        self.energy_emis_share = self.param["energy_emis_share"]
        self.land_emis_share = self.param["land_emis_share"]
        self.alpha = self.param["alpha"]
        self.beta = self.param["beta"]
        self.total_emissions_ref = self.param["total_emissions_ref"]
        self.min_co2_objective = self.param["min_co2_objective"]
        self.CO2_land_emissions = self.param[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)]
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        # declare class variable as local variable
        year_start = self.year_start
        year_end = self.year_end
        init_gr_sigma = self.init_gr_sigma
        init_indus_emissions = self.init_indus_emissions
        init_cum_indus_emissions = self.init_cum_indus_emissions

        years_range = np.arange(year_start, year_end + 1, self.time_step)
        self.years_range = years_range
        CO2_emissions_df = pd.DataFrame(
            index=years_range,
            columns=[
                GlossaryCore.Years,
                "gr_sigma",
                "sigma",
                "land_emissions",
                "cum_land_emissions",
                "indus_emissions",
                "cum_indus_emissions",
                "total_emissions",
                "cum_total_emissions",
            ],
        )

        for key in CO2_emissions_df.keys():
            CO2_emissions_df[key] = 0
        CO2_emissions_df[GlossaryCore.Years] = years_range
        CO2_emissions_df.loc[year_start, "gr_sigma"] = init_gr_sigma
        CO2_emissions_df.loc[year_start, "indus_emissions"] = init_indus_emissions
        CO2_emissions_df.loc[year_start, "cum_indus_emissions"] = init_cum_indus_emissions
        self.CO2_emissions_df = CO2_emissions_df
        return CO2_emissions_df

    def compute_total_CO2_emissions(self):
        """
        Compute total CO2 emissions
        """

        self.co2_emissions_Gt = pd.DataFrame(
            {GlossaryCore.Years: self.CO2_emissions_by_use_sources[GlossaryCore.Years]}
        )
        self.co2_emissions_Gt.index = self.co2_emissions_Gt[GlossaryCore.Years].values

        # drop years column in new dataframe
        co2_emissions_by_use_wo_years = self.CO2_emissions_by_use_sources.drop(GlossaryCore.Years, axis=1)

        # sum all co2 sources using the wo years dataframe
        sum_sources = co2_emissions_by_use_wo_years.sum(axis=1)

        # get unique column in serie format
        limited_by_capture_wo_years = self.co2_emissions_ccus_Gt.drop(GlossaryCore.Years, axis=1).iloc[:, 0]
        needed_by_energy_mix_wo_years = self.co2_emissions_needed_by_energy_mix.drop(GlossaryCore.Years, axis=1).iloc[
            :, 0
        ]
        sinks_df = self.CO2_emissions_by_use_sinks.drop(GlossaryCore.Years, axis=1).iloc[:, 0]

        sum_sinks = limited_by_capture_wo_years + needed_by_energy_mix_wo_years + sinks_df

        total_CO2_emissions = sum_sources - sum_sinks

        self.co2_emissions[GlossaryCore.TotalCO2Emissions] = total_CO2_emissions

    def compute_sigma(self, year):
        """
        Compute CO2-equivalent-emissions output ratio at t
        using sigma t-1 and growht_rate sigma  t-1
        """
        # declare class variable as local ones
        year_start = self.year_start
        time_step = self.time_step
        init_indus_emissions = self.init_indus_emissions
        init_gross_output = self.init_gross_output

        if year == year_start:
            sigma = init_indus_emissions / init_gross_output
        # Old version with emission control rate
        #             sigma = self.init_indus_emissions / \
        #                 (self.init_gross_output *
        #                  (1 - self.emissions_control_rate[self.year_start]))
        else:
            p_gr_sigma = self.CO2_emissions_df.at[year - time_step, "gr_sigma"]
            p_sigma = self.CO2_emissions_df.at[year - time_step, "sigma"]
            sigma = p_sigma * np.exp(p_gr_sigma * time_step)
        self.CO2_emissions_df.loc[year, "sigma"] = sigma
        return sigma

    def compute_change_sigma(self, year):
        """
        Compute change in sigma growth rate at t
        using sigma grouwth rate t-1
        """
        # declare class variable as local ones
        year_start = self.year_start
        time_step = self.time_step
        decline_rate_decarbo = self.decline_rate_decarbo

        if year == year_start:
            pass
        else:
            p_gr_sigma = self.CO2_emissions_df.at[year - time_step, "gr_sigma"]
            gr_sigma = p_gr_sigma * ((1.0 + decline_rate_decarbo) ** time_step)
            self.CO2_emissions_df.loc[year, "gr_sigma"] = gr_sigma
            return gr_sigma

    def compute_land_emissions(self, year):
        """
        Compute emissions from land for t
        """
        # sum all sectors emissions
        land_emissions = 0
        for column in self.CO2_land_emissions.columns:
            if column != GlossaryCore.Years:
                land_emissions += self.CO2_land_emissions.loc[year, column]

        self.CO2_emissions_df.loc[year, "land_emissions"] = land_emissions
        return land_emissions

    def compute_cum_land_emissions(self, year):
        """
        compute cumulative emissions from land for t
        """
        # declare class variable as local ones
        year_start = self.year_start
        time_step = self.time_step

        if year == year_start:
            cum_land_emissions = self.CO2_emissions_df.at[year_start, "land_emissions"] / self.gtco2_to_gtc
        else:
            p_cum_land_emissions = self.CO2_emissions_df.at[year - time_step, "cum_land_emissions"]
            p_land_emissions = self.CO2_emissions_df.at[year, "land_emissions"]
            cum_land_emissions = p_cum_land_emissions + p_land_emissions * float(time_step) / self.gtco2_to_gtc
        self.CO2_emissions_df.loc[year, "cum_land_emissions"] = cum_land_emissions
        return cum_land_emissions

    def compute_indus_emissions(self, year):
        """
        Compute industrial emissions at t
        using gross output (t)
        emissions control rate (t)
        emissions not coming from land change or energy
        """
        sigma = self.CO2_emissions_df.at[year, "sigma"]
        gross_output_ter = self.economics_df.at[year, GlossaryCore.GrossOutput]
        energy_emis_share = self.energy_emis_share
        share_land_emis = self.land_emis_share
        energy_emissions = self.co2_emissions.at[year, GlossaryCore.TotalCO2Emissions]
        # emissions_control_rate = self.emissions_control_rate[year]
        # Version with emission control rate
        #         indus_emissions = sigma * gross_output_ter * \
        #             (1.0 - emissions_control_rate)
        indus_emissions = sigma * gross_output_ter * (1 - energy_emis_share - share_land_emis) + energy_emissions
        self.CO2_emissions_df.loc[year, "indus_emissions"] = indus_emissions
        return indus_emissions

    def compute_cum_indus_emissions(self, year):
        """
        Compute cumulative industrial emissions at t
        using emissions indus at t- 1
        and cumulative indus emissions at t-1
        """
        # declare class variable as local ones
        year_start = self.year_start
        time_step = self.time_step

        if year == year_start:
            pass
        else:
            p_cum_indus_emissions = self.CO2_emissions_df.at[year - time_step, "cum_indus_emissions"]
            indus_emissions = self.CO2_emissions_df.at[year, "indus_emissions"]
            cum_indus_emissions = p_cum_indus_emissions + indus_emissions * float(time_step) / self.gtco2_to_gtc
            self.CO2_emissions_df.loc[year, "cum_indus_emissions"] = cum_indus_emissions
            return cum_indus_emissions

    def compute_total_emissions(self, year):
        """
        Total emissions taking energy emissions as inputs
        """
        land_emissions = self.CO2_emissions_df.at[year, "land_emissions"]
        indus_emissions = self.CO2_emissions_df.at[year, "indus_emissions"]
        total_emissions = indus_emissions + land_emissions
        self.CO2_emissions_df.loc[year, "total_emissions"] = total_emissions
        return total_emissions

    def compute_cum_total_emissions(self, year):
        """
        Compute cumulative total emissions at t :
            cum_indus emissions (t) + cum deforetation emissions (t)
        """
        cum_land_emissions = self.CO2_emissions_df.at[year, "cum_land_emissions"]
        cum_indus_emissions = self.CO2_emissions_df.at[year, "cum_indus_emissions"]
        cum_total_emissions = cum_land_emissions + cum_indus_emissions
        self.CO2_emissions_df.loc[year, "cum_total_emissions"] = cum_total_emissions
        return cum_total_emissions

    ######### GRADIENTS ########

    def compute_d_indus_emissions(self):
        """
        Compute gradient d_indus_emissions/d_gross_output,
        d_cum_indus_emissions/d_gross_output,
        d_cum_indus_emissions/d_total_CO2_emitted
        """
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
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
                    d_cum_indus_emissions_d_total_CO2_emitted[line, i] = float(self.time_step) / self.gtco2_to_gtc

                    d_cum_indus_emissions_d_gross_output[line, i] = (
                        float(self.time_step)
                        / self.gtco2_to_gtc
                        * self.CO2_emissions_df.at[years[i], "sigma"]
                        * (1.0 - self.energy_emis_share - self.land_emis_share)
                    )
                if i == line:  # fill diagonal
                    d_indus_emissions_d_gross_output[line, i] = self.CO2_emissions_df.at[years[line], "sigma"] * (
                        1 - self.energy_emis_share - self.land_emis_share
                    )

        return (
            d_indus_emissions_d_gross_output,
            d_cum_indus_emissions_d_gross_output,
            d_cum_indus_emissions_d_total_CO2_emitted,
        )

    def compute_d_land_emissions(self):

        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        d_cum_land_emissions_d_total_CO2_emitted = np.identity(nb_years) * 0

        i = 0
        line = 0
        for i in range(nb_years):
            for line in range(nb_years):
                if i >= 0 and i <= line:  # fill triangular descendant
                    d_cum_land_emissions_d_total_CO2_emitted[line, i] = 1 / self.gtco2_to_gtc

        return d_cum_land_emissions_d_total_CO2_emitted

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = self.year_end - self.year_start + 1
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if derivative[i][i] != 0:
                if i > 0:
                    d_cum[i] += d_cum[i - 1]
        return d_cum

    def compute_d_CO2_objective(self):
        total_emissions_values = self.CO2_emissions_df["total_emissions"].values
        delta_years = len(total_emissions_values)
        result = np.zeros(len(total_emissions_values))

        #         dn1 = -1.0 * (self.beta * (1 - self.alpha) * (self.CO2_emissions_df['total_emissions'].sum() - total_emissions_values[0])) / (
        #             (total_emissions_values[0] ** 2) * delta_years)

        dnn = self.beta * (1 - self.alpha) / (self.total_emissions_ref * delta_years)

        for index in range(len(total_emissions_values)):
            #             if index == 0:
            #                 result[index] = dn1
            #             else:
            result[index] = dnn

        return result

    def compute(self, inputs_models):
        """
        Compute outputs of the pyworld3
        """
        self.inputs_models = inputs_models
        self.economics_df = self.inputs_models[GlossaryCore.EconomicsDfValue]
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values
        self.co2_emissions = pd.DataFrame({GlossaryCore.Years: self.economics_df[GlossaryCore.Years]})
        self.co2_emissions.index = self.co2_emissions[GlossaryCore.Years].values

        self.co2_emissions_ccus = self.inputs_models["co2_emissions_ccus_Gt"].copy(deep=True)
        self.co2_emissions_ccus.index = self.co2_emissions_ccus[GlossaryCore.Years].values

        self.CO2_emissions_by_use_sources = self.inputs_models["CO2_emissions_by_use_sources"].copy(deep=True)
        self.CO2_emissions_by_use_sources.index = self.CO2_emissions_by_use_sources[GlossaryCore.Years].values
        self.CO2_emissions_by_use_sinks = self.inputs_models["CO2_emissions_by_use_sinks"].copy(deep=True)
        self.CO2_emissions_by_use_sinks.index = self.CO2_emissions_by_use_sinks[GlossaryCore.Years].values

        self.co2_emissions_needed_by_energy_mix = self.inputs_models["co2_emissions_needed_by_energy_mix"].copy(
            deep=True
        )
        self.co2_emissions_needed_by_energy_mix.index = self.CO2_emissions_by_use_sinks[GlossaryCore.Years].values

        self.co2_emissions_ccus_Gt = self.inputs_models["co2_emissions_ccus_Gt"].copy(deep=True)
        self.co2_emissions_ccus_Gt.index = self.co2_emissions_ccus_Gt[GlossaryCore.Years].values

        self.CO2_land_emissions = self.inputs_models[
            GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)
        ].copy(deep=True)
        self.CO2_land_emissions.index = self.co2_emissions_ccus_Gt[GlossaryCore.Years].values
        self.compute_total_CO2_emissions()
        # Iterate over years
        for year in self.years_range:
            self.compute_change_sigma(year)
            self.compute_sigma(year)
            self.compute_land_emissions(year)
            self.compute_cum_land_emissions(year)
            self.compute_indus_emissions(year)
            self.compute_total_emissions(year)
            self.compute_cum_indus_emissions(year)
            self.compute_cum_total_emissions(year)

        # -- Compute CO2 objective with alpha trade and beta weight with temperature objective

        delta_years = len(self.years_range)
        self.CO2_objective = np.asarray(
            [
                self.beta
                * (1 - self.alpha)
                * self.CO2_emissions_df["total_emissions"].sum()
                / (self.total_emissions_ref * delta_years)
            ]
        )

        self.compute_objective_with_exp_min()

        return self.CO2_emissions_df, self.CO2_objective

    def compute_objective_with_exp_min(self):
        """
        Compute the production of each element by minimizing them with and exponential function to reach min prod
        Objective is to decrease gradients when prod are very low
        Be careful the objective is to increase the total production to
        decrease the gradient then we have to modify the sum also
        """

        if self.CO2_objective.min() < self.min_co2_objective:
            # if some values are below min_prod
            # We use the exp smoothing only on values below self.min_prod (np.minimum(prod_element, self.min_prod))
            # Then we take the maximum to take prod_element if it is higher
            # than min_prod
            # Avoid the underflow if higher than 200
            values_copy = self.CO2_objective.copy()
            values_copy[values_copy > 200.0 * self.min_co2_objective] = 200.0 * self.min_co2_objective
            self.CO2_objective = np.asarray(
                [
                    self.min_co2_objective
                    / 10.0
                    * (11.0 - np.exp(-1.0 * values_copy / self.min_co2_objective) * np.exp(1))
                ]
            )

    def compute_dobjective_with_exp_min(self):
        dCO2_objective = 1.0
        if self.CO2_objective.min() < self.min_co2_objective:
            values_copy = self.CO2_objective.copy()
            values_copy[values_copy > 200.0 * self.min_co2_objective] = 200.0 * self.min_co2_objective

            dCO2_objective = np.exp(-1.0 * values_copy / self.min_co2_objective) * np.exp(1) / 10.0

        return dCO2_objective
