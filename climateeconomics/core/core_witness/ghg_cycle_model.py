"""
Copyright 2022 Airbus SAS

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
        self.param = param
        self.set_data()
        self.create_dataframe()

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

        self.em_to_conc_ch4 = self.param['ch4_emis_to_conc']
        self.decay_ch4 = self.param['ch4_decay_rate']
        self.pre_indus_conc_co2 = self.param['co2_pre_indus_conc']
        self.pre_indus_conc_ch4 = self.param['ch4_pre_indus_conc']
        self.conc_ch4 = self.param['ch4_init_conc']
        self.em_to_conc_n2o = self.param['n2o_emis_to_conc']
        self.decay_n2o = self.param['n2o_decay_rate']
        self.pre_indus_conc_n2o = self.param['n2o_pre_indus_conc']
        self.conc_n2o = self.param['n2o_init_conc']

        self.ghg_list = ['CO2', 'CH4', 'N2O']
        self.gwp_20 = self.param['GHG_global_warming_potential20']
        self.gwp_100 = self.param['GHG_global_warming_potential100']

        atmosphere_total_mass_kg = 5.1480 * 10 ** 18
        molar_mass_atmosphere = 0.02897  # kg/mol
        n_moles_in_atmosphere = atmosphere_total_mass_kg / molar_mass_atmosphere
        kg_to_gt = 10 ** (-12)
        molar_mass_co2, molar_mass_ch4, molar_mass_n2o = 0.04401, 0.016_04, 0.044_013  # kg/mol
        self.ppm_to_gt = {
            "CO2": n_moles_in_atmosphere * molar_mass_co2 * kg_to_gt * 10 ** -6,
            "CH4": n_moles_in_atmosphere * molar_mass_ch4 * kg_to_gt * 10 ** -6,
            "N2O": n_moles_in_atmosphere * molar_mass_n2o * kg_to_gt * 10 ** -6,
        }

        self.pred_indus_gwp20 = self.total_co2_equivalent(co2_conc=self.pre_indus_conc_co2,
                                                          ch4_conc=self.pre_indus_conc_ch4,
                                                          n2o_conc=self.pre_indus_conc_n2o,
                                                          gwp=self.gwp_100)
        self.pred_indus_gwp100 = self.total_co2_equivalent(co2_conc=self.pre_indus_conc_co2,
                                                           ch4_conc=self.pre_indus_conc_ch4,
                                                           n2o_conc=self.pre_indus_conc_n2o,
                                                           gwp=self.gwp_20)

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years_range = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range

        self.ghg_cycle_df = pd.DataFrame({GlossaryCore.Years: self.years_range})

        for i in [1, 2, 3, 4, 5]:
            self.ghg_cycle_df[f'co2_ppm_b{i}'] = self.boxes_conc[i-1]
        self.ghg_cycle_df[f'co2_ppm'] = self.boxes_conc[0]
        self.ghg_cycle_df[f'ch4_ppm'] = self.conc_ch4
        self.ghg_cycle_df[f'n2o_ppm'] = self.conc_n2o

    def compute_co2_atm_conc(self, year, boxes):
        """
        computes CO2 concentrations in atmosphere in ppm at t following FUND pyworld3
        """
        emissions = self.GHG_emissions_df.loc[self.GHG_emissions_df[GlossaryCore.Years] == year, GlossaryCore.TotalCO2Emissions].values[0] * 1e3     # in MtCO2
        boxes_tmp = [decay*box_conc + 0.000471*em_ratio*emissions for (decay, box_conc, em_ratio) in zip(self.decays, boxes, self.em_ratios)]
        boxes = boxes_tmp
        for i in [1, 2, 3, 4, 5]:
            self.ghg_cycle_df.loc[self.ghg_cycle_df[GlossaryCore.Years] == year, f'co2_ppm_b{i}'] = boxes[i-1]

        return boxes

    def compute_ch4_atm_conc(self, year, conc_ch4):
        """
        computes CH4 concentrations in atmosphere in ppm at t following FUND pyworld3
        """

        conc_ch4 += self.GHG_emissions_df.loc[self.GHG_emissions_df[GlossaryCore.Years] == year, 'Total CH4 emissions'].values[0] * 1e3 * self.em_to_conc_ch4 - \
                    self.decay_ch4 * (conc_ch4 - self.pre_indus_conc_ch4)

        self.ghg_cycle_df.loc[self.ghg_cycle_df[GlossaryCore.Years] == year, f'ch4_ppm'] = conc_ch4

        return conc_ch4

    def compute_n2o_atm_conc(self, year, conc_n2o):
        """
        computes N2O concentrations in atmosphere in ppm at t following FUND pyworld3
        """

        conc_n2o += self.GHG_emissions_df.loc[self.GHG_emissions_df[GlossaryCore.Years] == year, 'Total N2O emissions'].values[0] * 1e3 * self.em_to_conc_n2o - \
                    self.decay_n2o * (conc_n2o - self.pre_indus_conc_n2o)

        self.ghg_cycle_df.loc[self.ghg_cycle_df[GlossaryCore.Years] == year, f'n2o_ppm'] = conc_n2o

        return conc_n2o

    def compute_dco2_ppm_d_emissions(self):
        """
        computes derivative of co2_ppm with respect to CO2 emissions
        """

        coeff = 0.000471*self.em_ratios[0] * 1e3
        decay = self.decays[0]
        mat = np.diag(coeff*np.ones(len(self.years_range)))

        for i in np.arange(1, len(self.years_range-1)):
            coeff = coeff*decay
            mat += np.diag(coeff*np.ones(len(self.years_range)-i), -i)

        # first year is from initial data and is fixed ==> grad is zero
        mat[:, 0] = 0.0

        return mat

    def d_ppm_d_ghg(self):
        """
        computes derivative of ghg_ppm with respect to GHG emissions for other GHG.
        """
        # CH4
        coeff_ch4 = self.em_to_conc_ch4 * 1e3
        decay_ch4 = 1 - self.decay_ch4
        mat_ch4 = np.diag(coeff_ch4*np.ones(len(self.years_range)))

        # N2O
        coeff_n2o = self.em_to_conc_n2o * 1e3
        decay_n2o = 1 - self.decay_n2o
        mat_n2o = np.diag(coeff_n2o * np.ones(len(self.years_range)))

        for i in np.arange(1, len(self.years_range-1)):
            coeff_ch4 = coeff_ch4*decay_ch4
            coeff_n2o = coeff_n2o*decay_n2o

            mat_ch4 += np.diag(coeff_ch4 * np.ones(len(self.years_range)-i), -i)
            mat_n2o += np.diag(coeff_n2o * np.ones(len(self.years_range) - i), -i)

        # first year is from initial data and is fixed ==> grad is zero
        mat_ch4[:, 0] = 0.0
        mat_n2o[:, 0] = 0.0

        return {'CH4': mat_ch4,
                'N2O': mat_n2o,
                'CO2': self.compute_dco2_ppm_d_emissions()}

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
        gwp100_all_years: pd.Series = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df['co2_ppm'],
                                                                ch4_conc=self.ghg_cycle_df['ch4_ppm'],
                                                                n2o_conc=self.ghg_cycle_df['n2o_ppm'],
                                                                gwp=self.gwp_100)

        self.gwp100_obj = np.asarray([gwp100_all_years.mean() / self.pred_indus_gwp100])

        gwp20_all_years: pd.Series = self.total_co2_equivalent(co2_conc=self.ghg_cycle_df['co2_ppm'],
                                                               ch4_conc=self.ghg_cycle_df['ch4_ppm'],
                                                               n2o_conc=self.ghg_cycle_df['n2o_ppm'],
                                                               gwp=self.gwp_20)

        self.gwp20_obj = np.asarray([gwp20_all_years.mean() / self.pred_indus_gwp20])

    def compute_rockstrom_limit_constraint(self):
        """
        Compute Rockstrom limit constraint
        """
        self.rockstrom_limit_constraint = - (self.ghg_cycle_df['co2_ppm'].values -
                                             1.1 * self.rockstrom_limit) / self.rockstrom_constraint_ref

    def compute_minimum_ppm_limit_constraint(self):
        """
        Compute minimum ppm limit constraint
        """
        self.minimum_ppm_constraint = - \
            (self.minimum_ppm_limit -
             self.ghg_cycle_df['co2_ppm'].values) / self.minimum_ppm_constraint_ref

    def compute(self, inputs_models):
        """
        Compute results of the pyworld3
        """
        self.create_dataframe()
        self.inputs_models = inputs_models
        self.GHG_emissions_df = self.inputs_models['GHG_emissions_df']

        conc_boxes = self.boxes_conc
        conc_ch4 = self.conc_ch4
        conc_n2o = self.conc_n2o

        for year in self.years_range[1:]:
            conc_boxes = self.compute_co2_atm_conc(year, conc_boxes)
            conc_ch4 = self.compute_ch4_atm_conc(year, conc_ch4)
            conc_n2o = self.compute_n2o_atm_conc(year, conc_n2o)

        self.ghg_cycle_df[f'co2_ppm'] = self.ghg_cycle_df[f'co2_ppm_b1']

        self.compute_objectives()
        self.compute_rockstrom_limit_constraint()
        self.compute_minimum_ppm_limit_constraint()

    def total_co2_equivalent(self,
                             co2_conc: Union[float, pd.Series],
                             ch4_conc: Union[float, pd.Series],
                             n2o_conc: Union[float, pd.Series], gwp: dict) -> Union[float, pd.Series]:
        """
        Compute the global warming potential (CO2 equivalent) (over 100 years) given the concentrations of
        CO2 (in ppm), CH4 (in ppm), and N20 (in ppm)
        """
        ch4_total_mass = ch4_conc * self.ppm_to_gt["CH4"]
        n2o_total_mass = n2o_conc * self.ppm_to_gt["N2O"]
        co2_total_mass = co2_conc * self.ppm_to_gt["CO2"]

        total_mass_co2_eq = ch4_total_mass * gwp["CH4"] + n2o_total_mass * gwp["N2O"] + co2_total_mass * gwp["CO2"]
        return total_mass_co2_eq

    def d_total_co2_equivalent_d_conc(self, d_conc: pd.Series, specie: str, gwp: dict):
        """
        Computes the derivative of the total_co2_equivalent w.r.t to the concentration of a species, which can be
        CO2, CH4 or N2O
        """
        return d_conc * self.ppm_to_gt[specie] * gwp[specie]
