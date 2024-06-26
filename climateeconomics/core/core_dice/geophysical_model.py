"""
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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


class CarbonEmissions:
    """
    Used to compute carbon emissions from gross output
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
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.init_land_emissions = self.param["init_land_emissions"]
        self.decline_rate_land_emissions = self.param["decline_rate_land_emissions"]
        self.init_cum_land_emisisons = self.param["init_cum_land_emisisons"]
        self.init_gr_sigma = self.param["init_gr_sigma"]
        self.decline_rate_decarbo = self.param["decline_rate_decarbo"]
        self.init_indus_emissions = self.param["init_indus_emissions"]
        self.init_gross_output = self.param[GlossaryCore.InitialGrossOutput["var_name"]]
        self.init_cum_indus_emissions = self.param["init_cum_indus_emissions"]

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        emissions_df = pd.DataFrame(
            index=years_range,
            columns=[
                "year",
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
        emissions_df.loc[self.year_start, "cum_land_emissions"] = self.init_cum_land_emisisons
        emissions_df.loc[self.year_start, "gr_sigma"] = self.init_gr_sigma
        emissions_df.loc[self.year_start, "indus_emissions"] = self.init_indus_emissions
        emissions_df.loc[self.year_start, "land_emissions"] = self.init_land_emissions
        emissions_df.loc[self.year_start, "cum_indus_emissions"] = self.init_cum_indus_emissions
        emissions_df["year"] = years_range
        self.emissions_df = emissions_df
        return emissions_df

    def compute_sigma(self, year):
        """
        Compute CO2-equivalent-emissions output ratio at t
        using sigma t-1 and growht_rate sigma  t-1
        """
        if year == self.year_start:
            sigma = self.init_indus_emissions / (
                self.init_gross_output * (1 - self.emissions_control_rate[self.year_start])
            )
        else:
            p_gr_sigma = self.emissions_df.loc[year - self.time_step, "gr_sigma"]
            p_sigma = self.emissions_df.loc[year - self.time_step, "sigma"]
            sigma = p_sigma * np.exp(p_gr_sigma * self.time_step)
        self.emissions_df.loc[year, "sigma"] = sigma
        return sigma

    def compute_change_sigma(self, year):
        """
        Compute change in sigma growth rate at t
        using sigma grouwth rate t-1
        """
        if year == self.year_start:
            pass
        else:
            p_gr_sigma = self.emissions_df.loc[year - self.time_step, "gr_sigma"]
            gr_sigma = p_gr_sigma * ((1.0 + self.decline_rate_decarbo) ** self.time_step)
            self.emissions_df.loc[year, "gr_sigma"] = gr_sigma
            return gr_sigma

    def compute_land_emissions(self, year):
        """
        compute emissions from land
        """
        if year == self.year_start:
            pass
        else:
            t = ((year - self.year_start) / self.time_step) + 1
            land_emissions = self.init_land_emissions * (1.0 - self.decline_rate_land_emissions) ** (t - 1)
            self.emissions_df.loc[year, "land_emissions"] = land_emissions
            return land_emissions

    def compute_cum_land_emissions(self, year):
        """
        compute cumulative emissions from land for t
        """
        if year == self.year_start:
            pass
        else:
            p_cum_land_emissions = self.emissions_df.loc[year - self.time_step, "cum_land_emissions"]
            p_land_emissions = self.emissions_df.loc[year - self.time_step, "land_emissions"]
            cum_land_emissions = p_cum_land_emissions + p_land_emissions * (5.0 / 3.666)
            self.emissions_df.loc[year, "cum_land_emissions"] = cum_land_emissions
            return cum_land_emissions

    def compute_indus_emissions(self, year):
        """
        Compute industrial emissions at t
        using gross output (t)
        emissions control rate (t)
        """
        sigma = self.emissions_df.loc[year, "sigma"]
        gross_output = self.economics_df.loc[year, GlossaryCore.GrossOutput]
        emissions_control_rate = self.emissions_control_rate[year]
        indus_emissions = sigma * gross_output * (1.0 - emissions_control_rate)
        self.emissions_df.loc[year, "indus_emissions"] = indus_emissions
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
            p_cum_indus_emissions = self.emissions_df.loc[year - self.time_step, "cum_indus_emissions"]
            indus_emissions = self.emissions_df.loc[year, "indus_emissions"]
            cum_indus_emissions = p_cum_indus_emissions + indus_emissions * float(self.time_step) / 3.666
            self.emissions_df.loc[year, "cum_indus_emissions"] = cum_indus_emissions
            return cum_indus_emissions

    def compute_total_emissions(self, year):
        """Compute total emissions at t,
        = emissions indus (t) + land emissions (t)
        """
        land_emissions = self.emissions_df.loc[year, "land_emissions"]
        indus_emissions = self.emissions_df.loc[year, "indus_emissions"]
        total_emissions = indus_emissions + land_emissions
        self.emissions_df.loc[year, "total_emissions"] = total_emissions
        return total_emissions

    def compute_cum_total_emissions(self, year):
        """
        Compute cumulative total emissions at t :
            cum_indus emissions (t) + cum deforetation emissions (t)
        """
        cum_land_emissions = self.emissions_df.loc[year, "cum_land_emissions"]
        cum_indus_emissions = self.emissions_df.loc[year, "cum_indus_emissions"]
        cum_total_emissions = cum_land_emissions + cum_indus_emissions
        self.emissions_df.loc[year, "cum_total_emissions"] = cum_total_emissions
        return cum_total_emissions

    def compute(self, inputs_models, emissions_control_rate):
        self.inputs_models = inputs_models

        self.create_dataframe()
        self.economics_df = self.inputs_models[GlossaryCore.EconomicsDfValue].set_index(self.years_range)
        emissions_control_rate = emissions_control_rate.set_index(self.years_range)
        self.emissions_control_rate = emissions_control_rate["value"]
        #         self.emissions_control_rate = pd.Series(emissions_control_rate, index=(np.arange(
        # self.param[GlossaryCore.YearStart], self.param[GlossaryCore.YearEnd] + 1,
        # self.param[GlossaryCore.TimeStep])))
        self.emissions_df["emissions_control_rate"] = self.emissions_control_rate
        # Iterate over years
        for year in self.years_range:
            self.compute_change_sigma(year)
            self.compute_sigma(year)
            self.compute_indus_emissions(year)
            self.compute_land_emissions(year)
            self.compute_total_emissions(year)
            self.compute_cum_indus_emissions(year)
            self.compute_cum_land_emissions(year)
            self.compute_cum_total_emissions(year)
        return self.emissions_df


class CarbonCycle:
    """
    Carbon cycle
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
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.conc_lower_strata = self.param["conc_lower_strata"]
        self.conc_upper_strata = self.param["conc_upper_strata"]
        self.conc_atmo = self.param["conc_atmo"]
        self.init_conc_atmo = self.param["init_conc_atmo"]
        self.init_upper_strata = self.param["init_upper_strata"]
        self.init_lower_strata = self.param["init_lower_strata"]
        self.b_twelve = self.param["b_twelve"]
        self.b_twentythree = self.param["b_twentythree"]
        self.b_eleven = 1.0 - self.b_twelve
        self.b_twentyone = self.b_twelve * self.conc_atmo / self.conc_upper_strata
        self.b_twentytwo = 1.0 - self.b_twentyone - self.b_twentythree
        self.b_thirtytwo = self.b_twentythree * self.conc_upper_strata / self.conc_lower_strata
        self.b_thirtythree = 1.0 - self.b_thirtytwo
        self.lo_mat = self.param["lo_mat"]
        self.lo_mu = self.param["lo_mu"]
        self.lo_ml = self.param["lo_ml"]

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        carboncycle_df = pd.DataFrame(
            index=years_range,
            columns=[
                "year",
                "atmo_conc",
                "lower_ocean_conc",
                "shallow_ocean_conc",
                "ppm",
                "atmo_share_since1850",
                "atmo_share_sinceystart",
            ],
        )
        carboncycle_df.loc[self.year_start, "atmo_conc"] = self.init_conc_atmo
        carboncycle_df.loc[self.year_start, "lower_ocean_conc"] = self.init_lower_strata
        carboncycle_df.loc[self.year_start, "shallow_ocean_conc"] = self.init_upper_strata
        carboncycle_df["year"] = years_range
        self.carboncycle_df = carboncycle_df

        return carboncycle_df

    def compute_atmo_conc(self, year):
        """
        compute atmo conc for t using value at t-1 (MAT in DICE)
        """
        p_atmo_conc = self.carboncycle_df.loc[year - self.time_step, "atmo_conc"]
        p_shallow_ocean_conc = self.carboncycle_df.loc[year - self.time_step, "shallow_ocean_conc"]
        p_emissions = self.emissions_df.loc[year - self.time_step, "total_emissions"]
        atmo_conc = p_atmo_conc * self.b_eleven + p_shallow_ocean_conc * self.b_twentyone + p_emissions * 5.0 / 3.666
        # Lower bound
        self.carboncycle_df.loc[year, "atmo_conc"] = max(atmo_conc, self.lo_mat)
        return atmo_conc

    def compute_lower_ocean_conc(self, year):
        """
        Compute lower ocean conc at t using values at t-1
        """
        p_lower_ocean_conc = self.carboncycle_df.loc[year - self.time_step, "lower_ocean_conc"]
        p_shallow_ocean_conc = self.carboncycle_df.loc[year - self.time_step, "shallow_ocean_conc"]
        lower_ocean_conc = p_lower_ocean_conc * self.b_thirtythree + p_shallow_ocean_conc * self.b_twentythree
        # Lower bound
        self.carboncycle_df.loc[year, "lower_ocean_conc"] = max(lower_ocean_conc, self.lo_ml)
        return lower_ocean_conc

    def compute_upper_ocean_conc(self, year):
        """
        Compute upper ocean conc at t using values at t-1
        """
        p_lower_ocean_conc = self.carboncycle_df.loc[year - self.time_step, "lower_ocean_conc"]
        p_shallow_ocean_conc = self.carboncycle_df.loc[year - self.time_step, "shallow_ocean_conc"]
        p_atmo_conc = self.carboncycle_df.loc[year - self.time_step, "atmo_conc"]
        shallow_ocean_conc = (
            p_atmo_conc * self.b_twelve
            + p_shallow_ocean_conc * self.b_twentytwo
            + p_lower_ocean_conc * self.b_thirtytwo
        )
        # Lower Bound
        self.carboncycle_df.loc[year, "shallow_ocean_conc"] = max(shallow_ocean_conc, self.lo_mu)

    def compute_ppm(self, year):
        """
        Compute Atmospheric concentrations parts per million at t
        """
        atmo_conc = self.carboncycle_df.loc[year, "atmo_conc"]
        ppm = atmo_conc / 2.13
        self.carboncycle_df.loc[year, "ppm"] = ppm
        return ppm

    def compute_atmo_share(self, year):
        """
        Compute atmo share since 1850 and since 2010
        """
        atmo_conc = self.carboncycle_df.loc[year, "atmo_conc"]
        init_atmo_conc = self.carboncycle_df.loc[self.year_start, "atmo_conc"]
        init_cum_total_emissions = self.emissions_df.loc[self.year_start, "cum_total_emissions"]
        cum_total_emissions = self.emissions_df.loc[year, "cum_total_emissions"]

        atmo_share1850 = (atmo_conc - 588.0) / (cum_total_emissions + 0.000001)
        atmo_shareystart = (atmo_conc - init_atmo_conc) / (cum_total_emissions - init_cum_total_emissions)

        self.carboncycle_df.loc[year, "atmo_share_since1850"] = atmo_share1850
        self.carboncycle_df.loc[year, "atmo_share_sinceystart"] = atmo_shareystart
        return atmo_share1850

    def compute(self, inputs_models):
        self.inputs_models = inputs_models

        self.create_dataframe()
        self.emissions_df = self.inputs_models["emissions_df"].set_index(self.years_range)
        self.compute_ppm(self.year_start)
        for year in self.years_range[1:]:

            self.compute_atmo_conc(year)
            self.compute_lower_ocean_conc(year)
            self.compute_upper_ocean_conc(year)
            self.compute_ppm(year)
            self.compute_atmo_share(year)
        self.carboncycle_df = self.carboncycle_df.replace([np.inf, -np.inf], np.nan)
        return self.carboncycle_df.fillna(0.0)


class TempChange:
    """
    Temperature evolution
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
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.init_temp_ocean = self.param["tocean0"]
        self.init_temp_atmo = self.param["tatm0"]
        self.eq_temp_impact = self.param["t2xco2"]
        self.init_forcing_nonco = self.param["fex0"]
        self.hundred_forcing_nonco = self.param["fex1"]
        self.climate_upper = self.param["c1"]
        self.transfer_upper = self.param["c3"]
        self.transfer_lower = self.param["c4"]
        self.forcing_eq_co2 = self.param["fco22x"]
        self.lo_tocean = self.param["lo_tocean"]
        self.up_tatmo = self.param["up_tatmo"]
        self.up_tocean = self.param["up_tocean"]

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        temperature_df = pd.DataFrame(
            index=years_range,
            columns=[
                "year",
                GlossaryCore.ExoGForcing,
                GlossaryCore.Forcing,
                GlossaryCore.TempAtmo,
                GlossaryCore.TempOcean,
            ],
        )
        temperature_df.loc[self.year_start, GlossaryCore.TempOcean] = self.init_temp_ocean
        temperature_df.loc[self.year_start, GlossaryCore.TempAtmo] = self.init_temp_atmo
        temperature_df["year"] = years_range
        self.temperature_df = temperature_df
        return temperature_df

    def compute_exog_forcing(self, year):
        """
        Compute exogenous forcing for other greenhouse gases
        """
        t = ((year - self.year_start) / self.time_step) + 1
        exog_forcing = None  # initialize exog_forcing variable defined in either if or else statement
        if t < 18:
            exog_forcing = self.init_forcing_nonco + (1 / 17) * (
                self.hundred_forcing_nonco - self.init_forcing_nonco
            ) * (t - 1)
        elif t >= 18:
            exog_forcing = self.init_forcing_nonco + (self.hundred_forcing_nonco - self.init_forcing_nonco)
        self.temperature_df.loc[year, GlossaryCore.ExoGForcing] = exog_forcing
        return exog_forcing

    def compute_forcing(self, year):
        """
        Compute increase in radiative forcing for t using values at t-1
        (watts per m2 from 1900)
        """
        atmo_conc = self.carboncycle_df.loc[year, "atmo_conc"]
        exog_forcing = self.temperature_df.loc[year, GlossaryCore.ExoGForcing]
        forcing = self.forcing_eq_co2 * ((np.log((atmo_conc) / 588)) / np.log(2)) + exog_forcing
        self.temperature_df.loc[year, GlossaryCore.Forcing] = forcing
        return forcing

    def compute_temp_atmo(self, year):
        """
        Compute temperature of atmosphere (t) using t-1 values

        """
        p_temp_atmo = self.temperature_df.loc[year - self.time_step, GlossaryCore.TempAtmo]
        p_temp_ocean = self.temperature_df.loc[year - self.time_step, GlossaryCore.TempOcean]
        forcing = self.temperature_df.loc[year, GlossaryCore.Forcing]
        temp_atmo = p_temp_atmo + self.climate_upper * (
            (forcing - (self.forcing_eq_co2 / self.eq_temp_impact) * p_temp_atmo)
            - (self.transfer_upper * (p_temp_atmo - p_temp_ocean))
        )
        # Lower bound
        self.temperature_df.loc[year, GlossaryCore.TempAtmo] = min(temp_atmo, self.up_tatmo)
        return temp_atmo

    def compute_temp_ocean(self, year):
        """
        Compute temperature of lower ocean  at t using t-1 values
        """
        p_temp_ocean = self.temperature_df.loc[year - self.time_step, GlossaryCore.TempOcean]
        p_temp_atmo = self.temperature_df.loc[year - self.time_step, GlossaryCore.TempAtmo]
        temp_ocean = p_temp_ocean + self.transfer_lower * (p_temp_atmo - p_temp_ocean)
        # Bounds
        temp_ocean = max(temp_ocean, self.lo_tocean)
        self.temperature_df.loc[year, GlossaryCore.TempOcean] = min(temp_ocean, self.up_tocean)
        return temp_ocean

    def compute(self, inputs_models):
        """
        Compute all
        """
        self.inputs_models = inputs_models
        self.carboncycle_df = self.inputs_models[GlossaryCore.CarbonCycleDfValue]
        self.carboncycle_df = self.carboncycle_df.set_index(self.years_range)
        self.create_dataframe()
        self.compute_exog_forcing(self.year_start)
        self.compute_forcing(self.year_start)
        for year in self.years_range[1:]:
            self.compute_exog_forcing(year)
            self.compute_forcing(year)
            self.compute_temp_atmo(year)
            self.compute_temp_ocean(year)
        self.temperature_df = self.temperature_df.replace([np.inf, -np.inf], np.nan)
        return self.temperature_df.fillna(0.0)
