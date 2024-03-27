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
import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from energy_models.core.stream_type.carbon_models.nitrous_oxide import N2O


class GHGEmissions:
    """
    Used to compute ghg emissions from different sectors
    """
    GHG_TYPE_LIST = [N2O.name, GlossaryCore.CO2, GlossaryCore.CH4]

    def __init__(self, param):
        """
        Constructor
        """
        self.carbon_intensity_of_energy_mix = None
        self.affine_co2_objective: bool = False
        self.co2_emissions_objective = None
        self.total_energy_co2eq_emissions = None
        self.param = param
        self.configure_parameters()
        self.create_dataframe()
        self.sector_list = ['energy', 'land', 'industry']
        self.total_energy_production = None
        self.epsilon = 1.e-5 # for the CO2 objective function

    def configure_parameters(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]

        self.CO2_land_emissions = self.param['CO2_land_emissions']
        self.CH4_land_emissions = self.param['CH4_land_emissions']
        self.N2O_land_emissions = self.param['N2O_land_emissions']
        self.CO2_indus_emissions_df = self.param['CO2_indus_emissions_df']
        self.GHG_total_energy_emissions = self.param['GHG_total_energy_emissions']
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12

        self.gwp_20 = self.param['GHG_global_warming_potential20']
        self.gwp_100 = self.param['GHG_global_warming_potential100']

        self.CO2EmissionsRef = self.param[GlossaryCore.CO2EmissionsRef['var_name']]
        self.total_energy_production = self.param[GlossaryCore.EnergyProductionValue]


    def configure_parameters_update(self, inputs_dict):

        self.CO2_land_emissions = inputs_dict['CO2_land_emissions']
        self.CH4_land_emissions = inputs_dict['CH4_land_emissions']
        self.N2O_land_emissions = inputs_dict['N2O_land_emissions']
        self.CO2_indus_emissions_df = inputs_dict['CO2_indus_emissions_df']
        self.GHG_total_energy_emissions = inputs_dict['GHG_total_energy_emissions']
        self.affine_co2_objective = inputs_dict['affine_co2_objective']
        self.total_energy_production = inputs_dict[GlossaryCore.EnergyProductionValue]
        self.create_dataframe()

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        # declare class variable as local variable
        year_start = self.year_start
        year_end = self.year_end

        self.years_range = np.arange(
            year_start, year_end + 1, self.time_step)

        self.ghg_emissions_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        self.gwp_emissions = pd.DataFrame({GlossaryCore.Years: self.years_range})

    def compute_land_emissions(self):
        """
        Compute emissions from land
        """

        self.ghg_emissions_df['CO2 land_emissions'] = self.CO2_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values
        self.ghg_emissions_df['CH4 land_emissions'] = self.CH4_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values
        self.ghg_emissions_df['N2O land_emissions'] = self.N2O_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values

    def compute_total_emissions(self):
        """
        Total emissions taking energy emissions as inputs
        """
        self.ghg_emissions_df['CO2 industry_emissions'] = self.CO2_indus_emissions_df['indus_emissions'].values
        self.ghg_emissions_df['CH4 industry_emissions'] = 0.
        self.ghg_emissions_df['N2O industry_emissions'] = 0.

        for ghg in self.GHG_TYPE_LIST:
            self.ghg_emissions_df[f'{ghg} energy_emissions'] = self.GHG_total_energy_emissions[f'Total {ghg} emissions'].values

            self.ghg_emissions_df[f'Total {ghg} emissions'] = self.ghg_emissions_df[f'{ghg} land_emissions'].values + \
                self.ghg_emissions_df[f'{ghg} industry_emissions'].values + \
                self.ghg_emissions_df[f'{ghg} energy_emissions'].values

    def compute_gwp(self):

        for ghg in self.GHG_TYPE_LIST:

            self.gwp_emissions[f'{ghg}_20'] = self.ghg_emissions_df[f'Total {ghg} emissions'] * self.gwp_20[ghg]
            self.gwp_emissions[f'{ghg}_100'] = self.ghg_emissions_df[f'Total {ghg} emissions'] * self.gwp_100[ghg]

    def compute_co2_emissions_for_carbon_cycle(self):
        co2_emissions_df = self.ghg_emissions_df[[GlossaryCore.Years, GlossaryCore.TotalCO2Emissions]].rename(
            {GlossaryCore.TotalCO2Emissions: 'total_emissions'}, axis=1)
        return co2_emissions_df

    def compute_CO2_emissions_objective(self):
        '''
        CO2emissionsObjective = (sqrt((CO2Ref + CO2)^2 + epsilon^2) - epsilon)/ (2 * CO2 ref)


        CO2Ref = CO2emissionsRef corresponds to mean CO2 emissions during the industrial era until 2022 from the energy sector = 6.49 Gt
        CO2 = mean of the CO2 emissions between 2023 and 2100. Can be < 0 thanks to CCUS.
        When CO2 emissions reaches - CO2emissionsRef, then the energy sector is net zero emission and objective function should be 0
        When CO2 emissions are max, in full fossil, mean emissions between 2023 and 2100 are around 102.9 Gt
        For the full fossil case,  CO2emissionsRef + mean(CO2_emissions between 2023 and 2100 =  6.49 + 102.9 = 109.39
        to keep the objective function between 0 and 10, it is sufficient to normalize the sum above by 2 * CO2emiisionsRef

        The objective must be >= 0, hence the square value. If there is too much CCUS invest and CO2 < CO2Ref, there is
        also a climate change (icing) that is unwanted and the objective should depart from 0.
        Epsilon value allows to have the objective function infinitely derivable in -CO2Ref
        '''
        annual_co2_emissions_ref = self.CO2EmissionsRef
        mean_annual_co2_emissions = self.GHG_total_energy_emissions[GlossaryCore.TotalCO2Emissions].mean()
        epsilon = self.epsilon

        if self.affine_co2_objective:
            self.co2_emissions_objective = np.array(
                [(10 * self.CO2EmissionsRef + self.GHG_total_energy_emissions[GlossaryCore.TotalCO2Emissions].mean()) / \
                 (20. * self.CO2EmissionsRef)])
        else:
            self.co2_emissions_objective = np.array([(np.sqrt((annual_co2_emissions_ref + mean_annual_co2_emissions)**2 + epsilon**2) - epsilon) / (2. * annual_co2_emissions_ref)])


    def d_CO2_emissions_objective_d_total_co2_emissions(self):
        '''
        Compute gradient of CO2 emissions objective wrt ToTalCO2Emissions
        f' = CO2' * (CO2Ref + CO2)/sqrt(CO2^2 + epsilon^2)/ CO2Ref
        '''
        annual_co2_emissions_ref = self.CO2EmissionsRef
        mean_annual_co2_emissions = self.GHG_total_energy_emissions[GlossaryCore.TotalCO2Emissions].mean()
        dCO2 = np.ones(len(self.years_range)) / len(self.years_range)

        if self.affine_co2_objective:
            d_CO2_emissions_objective_d_total_co2_emissions = np.ones(len(self.years_range)) / len(self.years_range) / (20. * self.CO2EmissionsRef)
        else:
            epsilon = self.epsilon
            d_CO2_emissions_objective_d_total_co2_emissions = np.array([dCO2 * (annual_co2_emissions_ref + mean_annual_co2_emissions) / np.sqrt((annual_co2_emissions_ref + mean_annual_co2_emissions)**2 + epsilon**2) / (2. * annual_co2_emissions_ref)])

        return d_CO2_emissions_objective_d_total_co2_emissions


    def compute_total_co2_eq_energy_emissions(self):
        columns_to_sum = [f"Total {ghg} emissions" for ghg in self.gwp_100.keys()]
        self.total_energy_co2eq_emissions = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.TotalEnergyEmissions: self.GHG_total_energy_emissions[columns_to_sum].multiply(self.gwp_100.values()).sum(axis=1)
        })

    def d_total_co2_eq_energy_emissions(self, d_ghg_total_emissions, ghg: str):
        return d_ghg_total_emissions * self.gwp_100[ghg]


    def compute(self):
        """
        Compute outputs of the pyworld3
        """

        self.compute_land_emissions()
        self.compute_total_emissions()
        self.compute_gwp()
        self.compute_CO2_emissions_objective()
        self.compute_total_co2_eq_energy_emissions()
        self.compute_carbon_intensity_of_energy_mix()

    def compute_carbon_intensity_of_energy_mix(self):
        """
        Compute the carbon intensity of energy mix defined as
        Total Energy emissions / Total Energy production
        """
        total_energy_emissions = self.total_energy_co2eq_emissions[GlossaryCore.TotalEnergyEmissions].values  # GtCO2Eq : 1e12 kgCO2Eq
        total_energy_production = self.total_energy_production[GlossaryCore.TotalProductionValue].values  # PWh : 1e12 kWh
        carbon_intensity = total_energy_emissions / total_energy_production  # kgCO2Eq / kWh
        self.carbon_intensity_of_energy_mix = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.EnergyCarbonIntensityDfValue: carbon_intensity
        })

    def d_carbon_intensity_of_energy_mix_d_energy_production(self):
        total_energy_emissions = self.total_energy_co2eq_emissions[GlossaryCore.TotalEnergyEmissions].values
        total_energy_production = self.total_energy_production[GlossaryCore.TotalProductionValue].values
        return np.diag(- total_energy_emissions / total_energy_production ** 2)

    def d_carbon_intensity_of_energy_mix_d_ghg_energy_emissions(self, ghg: str):
        total_energy_production = self.total_energy_production[GlossaryCore.TotalProductionValue].values
        return np.diag(self.gwp_100[ghg] / total_energy_production)
