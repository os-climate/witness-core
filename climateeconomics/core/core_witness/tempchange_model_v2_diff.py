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

import autograd.numpy as np

from climateeconomics.core.tools.differentiable_model import (
    DifferentiableModel,
)
from climateeconomics.glossarycore import GlossaryCore


class TempChange(DifferentiableModel):
    """
    Temperature evolution
    """

    # Constant
    LAST_TEMPERATURE_OBJECTIVE = "last_temperature"
    INTEGRAL_OBJECTIVE = "integral"

    def __init__(self, inputs):
        """
        Constructor
        """
        super().__init__()

        self.ghg_cycle_df = None
        self.forcing_df = None
        self.temperature_objective = None
        self.temperature_end_constraint = None
        self.ppm_to_gtc = 2.13

        if inputs is not None:
            self.set_inputs(inputs)

    def set_data(self, inputs):
        self.year_start = inputs[GlossaryCore.YearStart]
        self.year_end = inputs[GlossaryCore.YearEnd]
        self.years_range = np.arange(self.year_start, self.year_end + 1)
        self.init_temp_ocean = inputs["init_temp_ocean"]
        self.init_temp_atmo = inputs["init_temp_atmo"]
        self.eq_temp_impact = inputs["eq_temp_impact"]
        self.temperature_model = inputs["temperature_model"]
        self.forcing_model = inputs["forcing_model"]
        self.ghg_cycle_df = inputs[GlossaryCore.GHGCycleDfValue]

        if self.forcing_model == "DICE":
            self.init_forcing_nonco = inputs["init_forcing_nonco"]
            self.hundred_forcing_nonco = inputs["hundred_forcing_nonco"]
        else:
            self.ch4_conc_init_ppm = inputs["pre_indus_ch4_concentration_ppm"]
            self.n2o_conc_init_ppm = inputs["pre_indus_n2o_concentration_ppm"]

        self.climate_upper = inputs["climate_upper"]
        self.transfer_upper = inputs["transfer_upper"]
        self.transfer_lower = inputs["transfer_lower"]
        self.forcing_eq_co2 = inputs["forcing_eq_co2"]
        self.c0_ppm = inputs["pre_indus_co2_concentration_ppm"]
        self.lo_tocean = inputs["lo_tocean"]
        self.up_tatmo = inputs["up_tatmo"]
        self.up_tocean = inputs["up_tocean"]

        self.alpha = inputs["alpha"]
        self.beta = inputs["beta"]
        self.temperature_obj_option = inputs["temperature_obj_option"]

        self.temperature_change_ref = inputs["temperature_change_ref"]

        self.temperature_end_constraint_limit = inputs[
            "temperature_end_constraint_limit"
        ]
        self.temperature_end_constraint_ref = inputs["temperature_end_constraint_ref"]

        # FUND
        self.climate_sensitivity = 3.0

    def compute_exog_forcing_dice(self, init_forcing_nonco, hundred_forcing_nonco):
        """
        Compute exogenous forcing for other greenhouse gases following DICE pyworld3
        linear increase of exogenous forcing following a given scenario
        """
        exog_forcing = np.linspace(
            init_forcing_nonco, hundred_forcing_nonco, len(self.years_range)
        )
        return exog_forcing

    def compute_exog_forcing_myhre(
        self, ch4_ppm, n2o_ppm, ch4_conc_init_ppm, n2o_conc_init_ppm
    ):
        """
        Compute exogenous forcing for CH4 and N2O gases following Myhre pyworld3
        Myhre et al, 1998, JGR, doi: 10.1029/98GL01908
        Myhre pyworld3 can be found in FUND, MAGICC and FAIR IAM models

        in FUND 0.036 * 1.4(np.sqrt(ch4_conc) - np.sqrt(ch4_conc_init))
        in FAIR 0.036 (np.sqrt(ch4_conc) -*np.sqrt(ch4_conc_init))

        We use ppm unit to compute this forcing as in FAIR and FUND
        """

        def MN(c1, c2):
            return 0.47 * np.log(
                1.0
                + 2.01e-5 * (c1 * c2) ** (0.75)
                + 5.31e-15 * c1 * (c1 * c2) ** (1.52)
            )

        exog_forcing = (
            0.036 * (np.sqrt(ch4_ppm) - np.sqrt(ch4_conc_init_ppm))
            - MN(ch4_ppm, n2o_conc_init_ppm)
            + 0.12 * (np.sqrt(n2o_ppm) - np.sqrt(n2o_conc_init_ppm))
            - MN(ch4_conc_init_ppm, n2o_ppm)
            + 2.0 * MN(ch4_conc_init_ppm, n2o_conc_init_ppm)
        )

        return exog_forcing

    def compute_forcing_etminan(
        self,
        co2_ppm,
        ch4_ppm,
        n2o_ppm,
        c0_ppm,
        ch4_conc_init_ppm,
        n2o_conc_init_ppm,
        forcing_eq_co2,
    ):
        """
        Compute radiative forcing following Etminan pyworld3 (found in FAIR)
        Etminan, M., Myhre, G., Highwood, E., and Shine, K.: Radiative forcing of carbon dioxide, methane, and nitrous oxide: A
        significant revision of the methane radiative forcing, Geophysical Research Letters, 43, 2016.
        """
        co2mean = 0.5 * (co2_ppm + c0_ppm)
        ch4mean = 0.5 * (ch4_ppm + ch4_conc_init_ppm)
        n2omean = 0.5 * (n2o_ppm + n2o_conc_init_ppm)

        sign_values = np.ones(len(co2_ppm))
        sign_values[co2_ppm.real < c0_ppm] = -1.0

        co2_forcing = (
            -2.4e-7 * (co2_ppm - c0_ppm) ** 2.0
            + 7.2e-4 * sign_values * (co2_ppm - c0_ppm)
            - 2.1e-4 * n2omean
            + forcing_eq_co2 / np.log(2.0)
        ) * np.log(co2_ppm / c0_ppm)
        ch4_forcing = (-1.3e-6 * ch4mean - 8.2e-6 * n2omean + 0.043) * (
            np.sqrt(ch4_ppm) - np.sqrt(ch4_conc_init_ppm)
        )
        n2o_forcing = (
            -8.0e-6 * co2mean + 4.2e-6 * n2omean - 4.9e-6 * ch4mean + 0.117
        ) * (np.sqrt(n2o_ppm) - np.sqrt(n2o_conc_init_ppm))

        return co2_forcing, ch4_forcing, n2o_forcing

    def compute_forcing_meinshausen(
        self, co2_ppm, ch4_ppm, n2o_ppm, c0_ppm, ch4_conc_init_ppm, n2o_conc_init_ppm
    ):
        """
        Compute radiative forcing following MeinsHausen pyworld3 (found in FAIR)
        Meinshausen, M., Nicholls, Z.R., Lewis, J., Gidden, M.J., Vogel, E., Freund,
        M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N. and Canadell, J.G., 2020.
        The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500.
        Geoscientific Model Development, 13(8), pp.3571-3605.
        """
        a1, b1, c1, d1 = -2.4785e-07, 0.00075906, -0.0021492, 5.2488
        a2, b2, c2, d2 = -0.00034197, 0.00025455, -0.00024357, 0.12173
        a3, b3, d3 = -8.9603e-05, -0.00012462, 0.045194

        Camax = c0_ppm - b1 / (2.0 * a1)
        alphap = d1 + a1 * (co2_ppm - c0_ppm) ** 2.0 + b1 * (co2_ppm - c0_ppm)

        # alphap[co2_ppm <= c0_ppm] = d1
        # alphap[co2_ppm >= Camax] = d1 - b1**2.0 / (4.0 * a1)

        # First condition: co2_ppm <= c0_ppm
        alphap = np.where(co2_ppm <= c0_ppm, d1, alphap)

        # Second condition: co2_ppm >= Camax
        alphap = np.where(co2_ppm >= Camax, d1 - b1**2.0 / (4.0 * a1), alphap)

        alpha_n2o = c1 * np.sqrt(n2o_ppm)
        co2_forcing = (alphap + alpha_n2o) * np.log(co2_ppm / c0_ppm)

        ch4_forcing = (a3 * np.sqrt(ch4_ppm) + b3 * np.sqrt(n2o_ppm) + d3) * (
            np.sqrt(ch4_ppm) - np.sqrt(ch4_conc_init_ppm)
        )

        n2o_forcing = (
            a2 * np.sqrt(co2_ppm) + b2 * np.sqrt(n2o_ppm) + c2 * np.sqrt(ch4_ppm) + d2
        ) * (np.sqrt(n2o_ppm) - np.sqrt(n2o_conc_init_ppm))

        return co2_forcing, ch4_forcing, n2o_forcing

    def compute_log_co2_forcing(self, co2_ppm, c0_ppm, forcing_eq_co2):
        co2_forcing = forcing_eq_co2 / np.log(2.0) * np.log(co2_ppm / c0_ppm)
        return co2_forcing

    def compute_temp_atmo_ocean_dice(
        self,
        init_temp_atmo,
        init_temp_ocean,
        forcing,
        climate_upper,
        transfer_upper,
        transfer_lower,
        forcing_eq_co2,
        eq_temp_impact,
        lo_tocean,
        up_tocean,
        up_tatmo,
    ):
        # Initialize temperature arrays
        temp_atmo = np.full(len(self.years_range), init_temp_atmo)
        temp_ocean = np.full(len(self.years_range), init_temp_ocean)

        # Calculate temperature differences
        temp_diff = temp_atmo - temp_ocean

        # Create arrays for the changes in ocean and atmosphere temperatures
        delta_ocean = (transfer_lower / 5.0) * temp_diff
        delta_atmo = (climate_upper / 5.0) * (
            (forcing - (forcing_eq_co2 / eq_temp_impact) * temp_atmo)
            - ((transfer_upper / 5.0) * temp_diff)
        )

        # Use np.cumsum to accumulate the changes
        temp_ocean = temp_ocean[0] + np.cumsum(np.concatenate(([0], delta_ocean[:-1])))
        temp_atmo = temp_atmo[0] + np.cumsum(np.concatenate(([0], delta_atmo[:-1])))

        return temp_ocean, temp_atmo

    def compute_temp_fund(self, init_temp_atmo, climate_sensitivity, forcing):
        alpha, beta_l, beta_q = -42.7, 29.1, 0.001
        e_folding_time = max(
            alpha
            + beta_l * climate_sensitivity
            + beta_q * climate_sensitivity * climate_sensitivity,
            1,
        )
        temperature = init_temp_atmo
        temperature_list = [temperature]

        for radiative_forcing in forcing[1:]:
            temperature = (
                1.0 - 1.0 / e_folding_time
            ) * temperature + climate_sensitivity / (
                5.35 * np.log(2) * e_folding_time
            ) * radiative_forcing
            temperature_list.append(temperature)

        return np.array(temperature_list)

    def compute_sea_level_fund(self, temp_atmo):
        rho, gamma = 500.0, 2.0
        initial_sea_level = 0.0
        sea_level = (1.0 - 1.0 / rho) * initial_sea_level + gamma * temp_atmo / rho
        return sea_level

    def compute_temperature_year_end_constraint(
        self,
        temp_atmo_year_end,
        temperature_end_constraint_limit,
        temperature_end_constraint_ref,
    ):
        return np.array(
            [
                (temperature_end_constraint_limit - temp_atmo_year_end)
                / temperature_end_constraint_ref
            ]
        )

    def get_ppm_inputs(self, inputs):
        co2_ppm = inputs[
            f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}"
        ]
        ch4_ppm = inputs[
            f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}"
        ]
        n2o_ppm = inputs[
            f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}"
        ]

        return co2_ppm, ch4_ppm, n2o_ppm

    def compute(self) -> None:
        """
        Compute all
        """

        co2_ppm, ch4_ppm, n2o_ppm = self.get_ppm_inputs(self.inputs)

        # A dataframe has been converted into a dictionary of arrays

        forcing_df = {GlossaryCore.Years: self.years_range}
        temperature_df = {GlossaryCore.Years: self.years_range}

        # Prepare Inputs
        init_temp_ocean = self.inputs["init_temp_ocean"]
        init_temp_atmo = self.inputs["init_temp_atmo"]
        eq_temp_impact = self.inputs["eq_temp_impact"]

        if self.forcing_model == "DICE":
            init_forcing_nonco = self.inputs["init_forcing_nonco"]
            hundred_forcing_nonco = self.inputs["hundred_forcing_nonco"]
        else:
            ch4_conc_init_ppm = self.inputs["pre_indus_ch4_concentration_ppm"]
            n2o_conc_init_ppm = self.inputs["pre_indus_n2o_concentration_ppm"]

        climate_upper = self.inputs["climate_upper"]
        transfer_upper = self.inputs["transfer_upper"]
        transfer_lower = self.inputs["transfer_lower"]
        forcing_eq_co2 = self.inputs["forcing_eq_co2"]
        c0_ppm = self.inputs["pre_indus_co2_concentration_ppm"]
        lo_tocean = self.inputs["lo_tocean"]
        up_tatmo = self.inputs["up_tatmo"]
        up_tocean = self.inputs["up_tocean"]

        alpha = self.inputs["alpha"]
        beta = self.inputs["beta"]
        temperature_obj_option = self.inputs["temperature_obj_option"]

        temperature_change_ref = self.inputs["temperature_change_ref"]

        temperature_end_constraint_limit = self.inputs[
            "temperature_end_constraint_limit"
        ]
        temperature_end_constraint_ref = self.inputs["temperature_end_constraint_ref"]

        # FUND
        climate_sensitivity = 3.0

        if self.forcing_model == "DICE":
            exog_forcing = self.compute_exog_forcing_dice(
                init_forcing_nonco, hundred_forcing_nonco
            )
            co2_forcing = self.compute_log_co2_forcing(co2_ppm, c0_ppm, forcing_eq_co2)
            forcing_df["CO2 forcing"] = co2_forcing
            forcing_df["CH4 and N20 forcing"] = exog_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == "Myhre":
            exog_forcing = self.compute_exog_forcing_myhre(
                ch4_ppm, n2o_ppm, ch4_conc_init_ppm, n2o_conc_init_ppm
            )
            co2_forcing = self.compute_log_co2_forcing(co2_ppm, c0_ppm, forcing_eq_co2)
            forcing_df["CO2 forcing"] = co2_forcing
            forcing_df["CH4 and N2O forcing"] = exog_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == "Etminan":
            co2_forcing, ch4_forcing, n2o_forcing = self.compute_forcing_etminan(
                co2_ppm,
                ch4_ppm,
                n2o_ppm,
                c0_ppm,
                ch4_conc_init_ppm,
                n2o_conc_init_ppm,
                forcing_eq_co2,
            )
            forcing_df["CO2 forcing"] = co2_forcing
            forcing_df["CH4 forcing"] = ch4_forcing
            forcing_df["N2O forcing"] = n2o_forcing
            forcing = co2_forcing + ch4_forcing + n2o_forcing

        elif self.forcing_model == "Meinshausen":
            co2_forcing, ch4_forcing, n2o_forcing = self.compute_forcing_meinshausen(
                co2_ppm, ch4_ppm, n2o_ppm, c0_ppm, ch4_conc_init_ppm, n2o_conc_init_ppm
            )
            forcing_df["CO2 forcing"] = co2_forcing
            forcing_df["CH4 forcing"] = ch4_forcing
            forcing_df["N2O forcing"] = n2o_forcing
            forcing = co2_forcing + ch4_forcing + n2o_forcing

        else:
            raise Exception("forcing model not in available models")

        temperature_df[GlossaryCore.Forcing] = forcing

        if self.temperature_model == "DICE":
            temp_ocean_list, temp_atmo_list = self.compute_temp_atmo_ocean_dice(
                init_temp_atmo,
                init_temp_ocean,
                forcing,
                climate_upper,
                transfer_upper,
                transfer_lower,
                forcing_eq_co2,
                eq_temp_impact,
                lo_tocean,
                up_tocean,
                up_tatmo,
            )
            temperature_df[GlossaryCore.TempOcean] = temp_ocean_list
            temperature_df[GlossaryCore.TempAtmo] = temp_atmo_list

        elif self.temperature_model == "FUND":
            temp_atmo_list = self.compute_temp_fund(
                init_temp_atmo, climate_sensitivity, forcing
            )
            temperature_df[GlossaryCore.TempAtmo] = temp_atmo_list
            sea_level = self.compute_sea_level_fund(temp_atmo_list)
            temperature_df["sea_level"] = sea_level

        elif self.temperature_model == "FAIR":
            raise NotImplementedError("FAIR Not implemented yet")

        temperature_end_constraint = self.compute_temperature_year_end_constraint(
            temperature_df[GlossaryCore.TempAtmo][-1],
            temperature_end_constraint_limit,
            temperature_end_constraint_ref,
        )

        # self.temperature_df = temperature_df.fillna(0.0)
        for k in forcing_df:
            self.outputs[f"forcing_df:{k}"] = forcing_df[k]
        self.outputs["forcing_df"] = forcing_df

        for k in temperature_df:
            self.outputs[f"temperature_df:{k}"] = temperature_df[k]
        self.outputs["temperature_df"] = temperature_df

        self.outputs["temperature_end_constraint"] = temperature_end_constraint


if __name__ == "__main__":
    from cProfile import Profile
    from os.path import dirname, join
    from pstats import SortKey, Stats

    from pandas import read_csv

    from climateeconomics.core.tools.differentiable_model import (
        DifferentiableModel,
        timer,
    )
    from climateeconomics.database import DatabaseWitnessCore
    from climateeconomics.glossarycore import GlossaryCore

    data_dir = join(dirname(__file__), "../../tests/data")
    carboncycle_df_ally = read_csv(join(data_dir, "carbon_cycle_data_onestep.csv"))

    ghg_cycle_df = carboncycle_df_ally[
        carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault
    ]

    ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df["ppm"]
    ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df["ppm"] * 1222 / 296
    ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df["ppm"] * 296 / 296
    ghg_cycle_df = ghg_cycle_df[
        [
            GlossaryCore.Years,
            GlossaryCore.CO2Concentration,
            GlossaryCore.CH4Concentration,
            GlossaryCore.N2OConcentration,
        ]
    ]

    # put manually the index
    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
    ghg_cycle_df.index = years

    # Dictionary containing all inputs for the TempChangeDiscipline model
    temp_change_inputs = {
        GlossaryCore.YearStart: GlossaryCore.YearStartDefault,
        GlossaryCore.YearEnd: GlossaryCore.YearEndDefault,
        "init_temp_ocean": DatabaseWitnessCore.OceanWarmingAnomalySincePreindustrial.get_value_at_year(
            GlossaryCore.YearStartDefault
        ),
        "init_temp_atmo": DatabaseWitnessCore.TemperatureAnomalyPreIndustrialYearStart.get_value_at_year(
            GlossaryCore.YearStartDefault
        ),
        "eq_temp_impact": 3.1,
        "temperature_model": "FUND",
        "climate_upper": 0.1005,
        "transfer_upper": 0.088,
        "transfer_lower": 0.025,
        "forcing_eq_co2": 3.74,
        "pre_indus_co2_concentration_ppm": DatabaseWitnessCore.CO2PreIndustrialConcentration.value,
        "lo_tocean": -1.0,
        "up_tatmo": 12.0,
        "up_tocean": 20.0,
        GlossaryCore.GHGCycleDfValue: ghg_cycle_df,  # This should be filled with actual dataframe
        "alpha": 0.5,  # This should be filled based on ClimateEcoDiscipline.ALPHA_DESC_IN
        "beta": 0.5,
        "temperature_obj_option": "INTEGRAL_OBJECTIVE",
        "temperature_change_ref": 0.2,
        "scale_factor_atmo_conc": 1e-2,
        "temperature_end_constraint_limit": 1.5,
        "temperature_end_constraint_ref": 3.0,
        "forcing_model": "Meinshausen",
        "pre_indus_ch4_concentration_ppm": 722.0,
        "pre_indus_n2o_concentration_ppm": 273.0,
    }

    def compute_jacobian(temperature_model, forcing_model, model: TempChange):
        # Used by all models

        if forcing_model == "DICE":
            model.compute_partial(
                "forcing_df:CO2 forcing",
                [f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}"],
            )

        elif forcing_model == "Myhre":
            d_forcing_co2 = model.compute_partial(
                "forcing_df:CO2 forcing",
                [f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}"],
            )

            d_forcing_n2o_ch4 = model.compute_partial(
                "forcing_df:CH4 and N2O forcing",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

        elif forcing_model == "Etminan" or forcing_model == "Meinshausen":
            d_forcing_co2 = model.compute_partial(
                "forcing_df:CO2 forcing",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

            d_forcing_ch4 = model.compute_partial(
                "forcing_df:CH4 forcing",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

            d_forcing_n2o = model.compute_partial(
                "forcing_df:N2O forcing",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

        if temperature_model == "DICE":
            d_temperature = model.compute_partial(
                f"temperature_df:{GlossaryCore.TempAtmo}",
                [f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}"],
            )

            d_temp_constraint = model.compute_partial(
                "temperature_end_constraint",
                [f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}"],
            )

        elif temperature_model == "FUND":
            d_temperature = model.compute_partial(
                f"temperature_df:{GlossaryCore.TempAtmo}",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

            d_temp_constraint = model.compute_partial(
                "temperature_end_constraint",
                [
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CO2Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.CH4Concentration}",
                    f"{GlossaryCore.GHGCycleDfValue}:{GlossaryCore.N2OConcentration}",
                ],
            )

    model = TempChange(temp_change_inputs)
    model.set_inputs(temp_change_inputs)
    model.set_data(temp_change_inputs)  # to remove

    with Profile() as profile:
        model.compute()

        (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CALLS)
        .print_stats()
    )

    with Profile() as profile:
        compute_jacobian( 
        temp_change_inputs["temperature_model"],
        temp_change_inputs["forcing_model"],
        model)
        (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CALLS)
        .print_stats()
    )
        


    with timer("Sum calculation", runs=10) as t:
        result = t.run(compute_jacobian, 
        temp_change_inputs["temperature_model"],
        temp_change_inputs["forcing_model"],
        model,
    )
