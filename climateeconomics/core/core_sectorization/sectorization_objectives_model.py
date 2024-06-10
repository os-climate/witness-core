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

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import (
    AgricultureDiscipline,
)
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import (
    IndustrialDiscipline,
)
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import (
    ServicesDiscipline,
)


class ObjectivesModel:
    """
    Objectives pyworld3 for sectorisation optimisation fitting process
    """

    # Units conversion
    conversion_factor = 1.0
    SECTORS_DISC_LIST = [AgricultureDiscipline, ServicesDiscipline, IndustrialDiscipline]
    SECTORS_LIST = [disc.sector_name for disc in SECTORS_DISC_LIST]
    SECTORS_OUT_UNIT = {disc.sector_name: disc.prod_cap_unit for disc in SECTORS_DISC_LIST}

    def __init__(self, inputs_dict):
        """
        Constructor
        """
        self.economics_df = None
        self.configure_parameters(inputs_dict)

    def configure_parameters(self, inputs_dict):
        """
        Configure with inputs_dict from the discipline
        """

        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
        self.time_step = inputs_dict[GlossaryCore.TimeStep]
        self.years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_years = len(self.years_range)
        self.historical_gdp = inputs_dict["historical_gdp"]
        self.historical_capital = inputs_dict["historical_capital"]
        self.historical_energy = inputs_dict["historical_energy"]
        self.extra_hist_data = inputs_dict["data_for_earlier_energy_eff"]
        self.default_weight = inputs_dict["weights_df"]["weight"].values
        self.delta_max_gdp = inputs_dict["delta_max_gdp"]
        self.delta_max_energy_eff = inputs_dict["delta_max_energy_eff"]

    def set_coupling_inputs(self, inputs):
        self.economics_df = inputs[GlossaryCore.EconomicsDfValue]
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values
        # Put all inputs in dictionary
        sectors_capital_dfs = {}
        sectors_production_dfs = {}
        sectors_long_term_energy_eff_df = {}
        for sector in self.SECTORS_LIST:
            sectors_capital_dfs[sector] = inputs[f"{sector}.{GlossaryCore.DetailedCapitalDfValue}"]
            sectors_production_dfs[sector] = inputs[f"{sector}.{GlossaryCore.ProductionDfValue}"]
            sectors_long_term_energy_eff_df[sector] = inputs[f"{sector}.longterm_energy_efficiency"]
        self.sectors_capital_dfs = sectors_capital_dfs
        self.sectors_production_dfs = sectors_production_dfs
        self.sectors_long_term_energy_eff_df = sectors_long_term_energy_eff_df

    def compute_all_errors(self, inputs):
        """For all variables takes predicted values and reference and compute the quadratic error"""
        self.set_coupling_inputs(inputs)

        # compute total errors
        error_pib_total = self.compute_quadratic_error(
            self.historical_gdp["total"].values,
            self.economics_df[GlossaryCore.OutputNetOfDamage].values,
            self.default_weight,
            self.delta_max_gdp,
        )
        # Per sector
        sectors_gdp_errors = {}
        sectors_energy_eff_errors = {}
        hist_energy_eff_dfs = {}
        self.year_min_energy_eff = {}

        for sector in self.SECTORS_LIST:
            self.year_min_energy_eff[sector] = self.year_start
            capital_df = self.sectors_capital_dfs[sector]
            production_df = self.sectors_production_dfs[sector]
            sectors_gdp_errors[sector] = self.compute_quadratic_error(
                self.historical_gdp[sector].values,
                production_df[GlossaryCore.OutputNetOfDamage].values,
                self.default_weight,
                self.delta_max_energy_eff,
            )
            self.sim_energy_eff = capital_df[GlossaryCore.EnergyEfficiency].values

            # for energy efficiency: it depends if we add extra years
            if not self.extra_hist_data.empty:
                # If we have extra data, add it to compute error
                hist_energy_eff_dfs[sector], extra_weight = self.compute_extra_hist_energy_efficiency(sector)
                weight = np.append(extra_weight, self.default_weight)
            else:
                hist_energy_eff = self.compute_hist_energy_efficiency(
                    self.historical_energy[sector].values, self.historical_capital[sector].values
                )
                hist_energy_eff_dfs[sector] = pd.DataFrame(
                    {GlossaryCore.Years: self.years_range, GlossaryCore.EnergyEfficiency: hist_energy_eff}
                )
                weight = self.default_weight

            # Energy eff errors
            sectors_energy_eff_errors[sector] = self.compute_quadratic_error(
                hist_energy_eff_dfs[sector][GlossaryCore.EnergyEfficiency].values,
                self.sim_energy_eff,
                weight,
                self.delta_max_energy_eff,
            )

        return (
            error_pib_total,
            sectors_gdp_errors,
            sectors_energy_eff_errors,
            hist_energy_eff_dfs,
            self.year_min_energy_eff,
        )

    def compute_quadratic_error(self, ref, pred, weight, delta_max):
        """
        Compute quadratic error. Inputs: ref and pred are arrays
        """
        # Find maximum value in data to normalise objective
        delta = np.subtract(pred, ref)
        # And normalise delta
        delta_norm = delta / delta_max
        delta_squared = np.square(delta_norm)
        # Add weight
        with_weight = delta_squared * weight
        # and mean
        error = sum(with_weight) / sum(weight)
        # error = np.mean(delta_squared)
        return error

    def compute_hist_energy_efficiency(self, historical_energy, historical_capital):
        """
        Compute historical energy efficiency value: energy in 1e3Twh and capital in T$
        """
        # compute
        energy_eff = historical_capital / historical_energy
        return energy_eff

    def compute_extra_hist_energy_efficiency(self, sector):
        """
        Add extra years to compute errors for energy efficiency for a sector
        using extra_hist_data dataframe
        return energy effiency for a sector and the associated weight for fitting
        """
        extra_hist_data = self.extra_hist_data

        # check that for this sector data exist
        capital = extra_hist_data[f"{sector}.capital"]
        energy = extra_hist_data[f"{sector}.energy"]
        # find first year of extra data
        year_min_series = extra_hist_data[GlossaryCore.Years][(capital > 0) & (energy > 0)]
        year_min = year_min_series.min()

        if year_min < self.year_start:
            self.year_min_energy_eff[sector] = year_min
            # get extra data
            extra_hist_capital = extra_hist_data[f"{sector}.capital"][
                (extra_hist_data[GlossaryCore.Years] >= year_min)
                & (extra_hist_data[GlossaryCore.Years] < self.year_start)
            ]
            extra_hist_energy = extra_hist_data[f"{sector}.energy"][
                (extra_hist_data[GlossaryCore.Years] >= year_min)
                & (extra_hist_data[GlossaryCore.Years] < self.year_start)
            ]
            # and append original data
            hist_capital = np.append([extra_hist_capital], [self.historical_capital[sector].values])
            hist_energy = np.append([extra_hist_energy], [self.historical_energy[sector].values])
            # prepare data to compute error
            lt_ene_eff_df = self.sectors_long_term_energy_eff_df[sector]
            self.sim_energy_eff = lt_ene_eff_df[GlossaryCore.EnergyEfficiency][
                (lt_ene_eff_df[GlossaryCore.Years] <= self.year_end) & (lt_ene_eff_df[GlossaryCore.Years] >= year_min)
            ].values
            # get weight for fitting
            extra_weight = extra_hist_data["weight"][
                (extra_hist_data[GlossaryCore.Years] >= year_min)
                & (extra_hist_data[GlossaryCore.Years] < self.year_start)
            ].values
            year_range = np.arange(year_min, self.year_end + 1, self.time_step)
        else:
            # if extra data not coherent use original data: eg not same year start for capital and energy
            print("Using " + f"{self.year_start}" + "-" + f"{self.year_end}" + " data only for " + f"{sector}")
            hist_energy = self.historical_energy[sector].values
            hist_capital = self.historical_capital[sector].values
            year_range = self.years_range
            extra_weight = []

        # compute energy efficiency
        hist_energy_eff = self.compute_hist_energy_efficiency(hist_energy, hist_capital)
        hist_energy_eff_df = pd.DataFrame(
            {GlossaryCore.Years: year_range, GlossaryCore.EnergyEfficiency: hist_energy_eff}
        )

        return hist_energy_eff_df, extra_weight
