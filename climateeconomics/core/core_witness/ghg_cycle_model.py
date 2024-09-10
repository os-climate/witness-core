'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/07-2023/11/03 Copyright 2023 Capgemini

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
from typing import Union

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class GHGCycle():
    """
    GHG cycle
    """
    rockstrom_limit = 450

    def __init__(self, param):
        """
        Constructor
        """
        self.ghg_cycle_df = None
        self.param = param
        self.set_data()
        self.create_dataframe()
        self.global_warming_potential_df = None
        self.ghg_emissions_df = None
        self.ppm_co2_negative_indexes = None

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]

        self.gwp100_obj = 0.
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12
        # conversion factor 1ppm= 2.13 Gtc
        self.gtc_to_ppm = 2.13
        self.rockstrom_constraint_ref = self.param['rockstrom_constraint_ref']
        self.minimum_ppm_constraint_ref = self.param['minimum_ppm_constraint_ref']
        self.minimum_ppm_limit = self.param['minimum_ppm_limit']

        # FUND parameters
        self.em_ratios = self.param['co2_emissions_fractions']
        self.decays = self.param['co2_boxes_decays']
        self.boxes_conc = self.param['co2_boxes_init_conc']

        self.decay_ch4 = self.param['ch4_decay_rate']
        self.pre_indus_conc_co2 = self.param['co2_pre_indus_conc']
        self.pre_indus_conc_ch4 = self.param['ch4_pre_indus_conc']
        self.init_conc_ch4 = self.param['ch4_init_conc']
        self.decay_n2o = self.param['n2o_decay_rate']
        self.pre_indus_conc_n2o = self.param['n2o_pre_indus_conc']
        self.init_conc_n2o = self.param['n2o_init_conc']

        self.ghg_list = GlossaryCore.GreenHouseGases
        self.gwp_20 = self.param['GHG_global_warming_potential20']
        self.gwp_100 = self.param['GHG_global_warming_potential100']

        atmosphere_total_mass_kg = 5.1480 * 10 ** 18
        molar_mass_atmosphere = 0.02897  # kg/mol
        n_moles_in_atmosphere = atmosphere_total_mass_kg / molar_mass_atmosphere
        kg_to_gt = 10 ** (-12)
        molar_mass_co2, molar_mass_ch4, molar_mass_n2o = 0.04401, 0.016_04, 0.044_013  # kg/mol

        self.pp_to_gt = {
            GlossaryCore.CO2: n_moles_in_atmosphere * molar_mass_co2 * kg_to_gt * 10 ** -6,  # ppm
            GlossaryCore.CH4: n_moles_in_atmosphere * molar_mass_ch4 * kg_to_gt * 10 ** -6 * 1e-3,  # ppb
            GlossaryCore.N2O: n_moles_in_atmosphere * molar_mass_n2o * kg_to_gt * 10 ** -6 * 1e-3,  # ppb
        }
        self.gt_to_pp = {ghg: 1 / self.pp_to_gt[ghg] for ghg in GlossaryCore.GreenHouseGases}

        self.pred_indus_gwp20 = self.total_co2_equivalent(co2_conc=self.pre_indus_conc_co2,
                                                          ch4_conc=self.pre_indus_conc_ch4,
                                                          n2o_conc=self.pre_indus_conc_n2o,
                                                          gwp=self.gwp_20)
        self.pred_indus_gwp100 = self.total_co2_equivalent(co2_conc=self.pre_indus_conc_co2,
                                                           ch4_conc=self.pre_indus_conc_ch4,
                                                           n2o_conc=self.pre_indus_conc_n2o,
                                                           gwp=self.gwp_100)

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        self.years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.ghg_cycle_df = pd.DataFrame({GlossaryCore.Years: self.years_range})

    def compute_dco2_ppm_d_emissions(self):
        """
        computes derivative of co2_ppm with respect to CO2 emissions
        """

        coeff = 0.000471 * self.em_ratios[0] * 1e3
        decay = self.decays[0]
        mat = np.diag(coeff * np.ones(len(self.years_range)))

        for i in np.arange(1, len(self.years_range - 1)):
            coeff = coeff * decay
            mat += np.diag(coeff * np.ones(len(self.years_range) - i), -i)

        # first year is from initial data and is fixed ==> grad is zero
        mat[:, 0] = 0.0
        # gradient is null where the clip to 1e-10 was used on ppm co2
        mat[self.ppm_co2_negative_indexes, :] = 0.0
        return mat

    def d_gwp100_objective_d_ppm(self, d_ppm: pd.Series, specie: str) -> float:
        """
        Computes the derivative of the gwp100 objective w.r.t to the concentration of a species, which can be
        CO2, CH4 or N2O
        """
        return self.d_total_co2_equivalent_d_conc(d_conc=d_ppm, specie=specie, gwp=self.gwp_100).mean(axis=0) / self.pred_indus_gwp100

    def d_gwp20_objective_d_ppm(self, d_ppm: pd.Series, specie: str) -> float:
        """
        Computes the derivative of the gwp20 objective w.r.t to the concentration of a species, which can be
        CO2, CH4 or N2O
        """
        return self.d_total_co2_equivalent_d_conc(d_conc=d_ppm, specie=specie, gwp=self.gwp_20).mean(axis=0) / self.pred_indus_gwp20

    def compute_objectives(self):
        """
        Compute objective,
        defined as :
        - the average global warming potential over all years of the optimization
        normalized by :
        - the global warming potential at pre-industrial level


        """
        gwp100_all_years: pd.Series = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df[GlossaryCore.CO2Concentration],
                                                                ch4_conc=self.ghg_cycle_df[GlossaryCore.CH4Concentration],
                                                                n2o_conc=self.ghg_cycle_df[GlossaryCore.N2OConcentration],
                                                                gwp=self.gwp_100)

        self.gwp100_obj = np.asarray([gwp100_all_years.mean() / self.pred_indus_gwp100])

        gwp20_all_years: pd.Series = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df[GlossaryCore.CO2Concentration],
                                                               ch4_conc=self.ghg_cycle_df[GlossaryCore.CH4Concentration],
                                                               n2o_conc=self.ghg_cycle_df[GlossaryCore.N2OConcentration],
                                                               gwp=self.gwp_20)

        self.gwp20_obj = np.asarray([gwp20_all_years.mean() / self.pred_indus_gwp20])

    def compute_rockstrom_limit_constraint(self):
        """
        Compute Rockstrom limit constraint
        """
        self.rockstrom_limit_constraint = - (self.ghg_cycle_df[GlossaryCore.CO2Concentration].values -
                                             1.1 * self.rockstrom_limit) / self.rockstrom_constraint_ref

    def compute_minimum_ppm_limit_constraint(self):
        """
        Compute minimum ppm limit constraint
        """
        self.minimum_ppm_constraint = - \
            (self.minimum_ppm_limit -
             self.ghg_cycle_df[GlossaryCore.CO2Concentration].values) / self.minimum_ppm_constraint_ref

    def compute(self, inputs_models):
        """
        Compute results of the pyworld3
        """
        self.create_dataframe()
        self.ghg_emissions_df = inputs_models[GlossaryCore.GHGEmissionsDfValue]

        self.compute_concentration_co2()
        self.compute_concentration_ch4()
        self.compute_concentration_n2o()

        self.compute_objectives()
        self.compute_rockstrom_limit_constraint()
        self.compute_global_warming_potentials()
        self.compute_minimum_ppm_limit_constraint()
        self.compute_extra_CO2_eq_Gt()

    def total_co2_equivalent(self,
                             co2_conc: Union[float, pd.Series],
                             ch4_conc: Union[float, pd.Series],
                             n2o_conc: Union[float, pd.Series], gwp: dict) -> Union[float, pd.Series]:
        """
        Compute the global warming potential (CO2 equivalent) (over 100 years) given the concentrations of
        CO2 (in ppm), CH4 (in ppm), and N20 (in ppm)

        Outputs CO2GtEquivalent
        """
        ch4_total_mass = ch4_conc * self.pp_to_gt[GlossaryCore.CH4]
        n2o_total_mass = n2o_conc * self.pp_to_gt[GlossaryCore.N2O]
        co2_total_mass = co2_conc * self.pp_to_gt[GlossaryCore.CO2]

        total_mass_co2_eq = ch4_total_mass * gwp[GlossaryCore.CH4] + n2o_total_mass * gwp[GlossaryCore.N2O] + co2_total_mass * gwp[GlossaryCore.CO2]
        return total_mass_co2_eq

    def d_total_co2_equivalent_d_conc(self, d_conc: pd.Series, specie: str, gwp: dict):
        """
        Computes the derivative of the total_co2_equivalent w.r.t to the concentration of a species, which can be
        CO2, CH4 or N2O
        """
        return d_conc * self.pp_to_gt[specie] * gwp[specie]

    def compute_extra_CO2_eq_Gt(self):
        """Computes extra Gt of CO2Equivalent in atmosphere since pre-industrial levels"""
        CO2_eq_20y = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df[GlossaryCore.CO2Concentration],
                                               ch4_conc=self.ghg_cycle_df[GlossaryCore.CH4Concentration],
                                               n2o_conc=self.ghg_cycle_df[GlossaryCore.N2OConcentration],
                                               gwp=self.gwp_20)

        CO2_eq_pre_industrial_20y = self.pred_indus_gwp20
        Extra_CO2_eq_Gt_since_pre_industrial_20y = CO2_eq_20y - CO2_eq_pre_industrial_20y

        CO2_eq_100y = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df[GlossaryCore.CO2Concentration],
                                                ch4_conc=self.ghg_cycle_df[GlossaryCore.CH4Concentration],
                                                n2o_conc=self.ghg_cycle_df[GlossaryCore.N2OConcentration],
                                                gwp=self.gwp_100)

        CO2_eq_pre_industrial_100y = self.pred_indus_gwp100
        Extra_CO2_eq_Gt_since_pre_industrial_100y = CO2_eq_100y - CO2_eq_pre_industrial_100y

        self.extra_co2_eq_detailed = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.ExtraCO2EqSincePreIndustrial2OYbasisValue: Extra_CO2_eq_Gt_since_pre_industrial_20y,
            GlossaryCore.ExtraCO2EqSincePreIndustrial10OYbasisValue: Extra_CO2_eq_Gt_since_pre_industrial_100y
        })

        self.extra_co2_eq = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.ExtraCO2EqSincePreIndustrialValue: Extra_CO2_eq_Gt_since_pre_industrial_20y,
        })

    def compute_global_warming_potentials(self):
        global_warming_potential_df = pd.DataFrame({
            GlossaryCore.Years: self.years_range
        })
        for year_basis, gwp in zip([GlossaryCore.YearBasis20, GlossaryCore.YearBasis100],
                                   [self.gwp_20, self.gwp_100]):
            to_sum = []
            for ghg, ghg_conc_name in zip(GlossaryCore.GreenHouseGases,
                                          [GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]):
                ghg_conc = self.ghg_cycle_df[ghg_conc_name].values
                ghg_gt = ghg_conc * self.pp_to_gt[ghg]
                ghg_gwp = gwp[ghg] * ghg_gt
                global_warming_potential_df[f'{ghg} {year_basis}'] = ghg_gwp
                to_sum.append(ghg_gwp)

            global_warming_potential_df[f'Total {year_basis}'] = np.sum(to_sum, axis=0)

        self.global_warming_potential_df = global_warming_potential_df

    def compute_concentration_co2(self):
        conc_boxes = self.boxes_conc
        boxes_list = [conc_boxes]
        for emission_year_Gt in self.ghg_emissions_df[GlossaryCore.TotalCO2Emissions].values[1:]:
            emission_year_Mt = emission_year_Gt * 1e3
            conc_boxes = [decay * box_conc + 0.000471 * em_ratio * emission_year_Mt for (decay, box_conc, em_ratio) in zip(self.decays, conc_boxes, self.em_ratios)]
            boxes_list.append(conc_boxes)

        boxes_array = np.array(boxes_list)

        for i in [1, 2, 3, 4, 5]:
            self.ghg_cycle_df[f'co2_ppm_b{i}'] = boxes_array[:, i - 1]

        # clip value to 0 if negative
        self.ppm_co2_negative_indexes = self.ghg_cycle_df.index[self.ghg_cycle_df['co2_ppm_b1'] < 0].tolist()
        self.ghg_cycle_df.loc[self.ppm_co2_negative_indexes, 'co2_ppm_b1'] = 1e-10
        self.ghg_cycle_df[GlossaryCore.CO2Concentration] = self.ghg_cycle_df['co2_ppm_b1'].values

    def compute_concentration_ch4(self):
        ch4_concentrations = self._forecast_concentration(conc_init=self.init_conc_ch4,
                                                          decay_rate=self.decay_ch4,
                                                          conc_pre_indus=self.pre_indus_conc_ch4,
                                                          emissions_to_pp=self.gt_to_pp[GlossaryCore.CH4],
                                                          emissions=self.ghg_emissions_df[
                                                              GlossaryCore.TotalCH4Emissions].values)

        self.ghg_cycle_df[GlossaryCore.CH4Concentration] = ch4_concentrations

    def compute_concentration_n2o(self):
        n2o_concentrations = self._forecast_concentration(conc_init=self.init_conc_n2o,
                                                          decay_rate=self.decay_n2o,
                                                          conc_pre_indus=self.pre_indus_conc_n2o,
                                                          emissions_to_pp=self.gt_to_pp[GlossaryCore.N2O],
                                                          emissions=self.ghg_emissions_df[
                                                              GlossaryCore.TotalN2OEmissions].values)

        self.ghg_cycle_df[GlossaryCore.N2OConcentration] = n2o_concentrations

    def _forecast_concentration(self, conc_init: float, decay_rate: float, conc_pre_indus: float,
                                emissions_to_pp: float, emissions: np.ndarray):

        # C(t+1) = C(t) + E(t) * E_to_ppm - decay_rate * (C(t) - Cpreindus)
        conc = conc_init
        concentrations = [conc]
        for year, emission_year in zip(self.years_range, emissions):
            conc += emission_year * emissions_to_pp - decay_rate * (conc - conc_pre_indus)
            concentrations.append(conc)

        return np.array(concentrations[:-1])

    def d_conc_d_emission(self, decay_rate: float, emissions_to_pp: float):
        """
        C(t+1) = C(t) + E(t) * E_to_ppm - decay_rate * (C(t) - Cpreindus)
        So for derivative :
        d C[j] / d E[i] = d (C[j-1] + E[j-1] * E_to_ppm - decay_rate * C[j-1]) / d E[i]
                    = (1 - decay_rate) * (d C[j-1] / d E[i]) + E_to_ppm * (j-1 == i)
        """
        n_years = len(self.years_range)
        d_conc_d_emissions = np.zeros((n_years, n_years))
        for i in range(1, n_years):
            d_conc_d_emissions[i] = (1 - decay_rate) * d_conc_d_emissions[i - 1]
            d_conc_d_emissions[i, i - 1] += emissions_to_pp

        return d_conc_d_emissions

    def d_conc_ch4_d_emissions(self):
        return self.d_conc_d_emission(decay_rate=self.decay_ch4,
                                      emissions_to_pp=self.gt_to_pp[GlossaryCore.CH4])

    def d_conc_n2o_d_emissions(self):
        return self.d_conc_d_emission(decay_rate=self.decay_n2o,
                                      emissions_to_pp=self.gt_to_pp[GlossaryCore.N2O])
