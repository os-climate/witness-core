'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/07-2024/06/24 Copyright 2023 Capgemini

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
from energy_models.core.stream_type.carbon_models.nitrous_oxide import N2O

from climateeconomics.glossarycore import GlossaryCore


class GHGEmissions:
    """
    Used to compute ghg emissions from different sectors
    """
    GHG_TYPE_LIST = [GlossaryCore.CO2, GlossaryCore.CH4, N2O.name]

    def __init__(self, param):
        """
        Constructor
        """
        self.constraint_nze_2050_ref = None
        self.emissions_after_2050_df = None
        self.energy_emission_households_df = None
        self.residential_energy_consumption = None
        self.dict_sector_sections_energy_emissions = {}
        self.dict_sector_sections_non_energy_emissions = {}
        self.dict_sector_sections_emissions = {}
        self.dict_sector_emissions = {}
        self.total_economics_emisssions = None
        self.new_sector_list = []
        self.carbon_intensity_of_energy_mix = None
        self.get_sosdisc_inputs = None
        self.affine_co2_objective: bool = False
        self.co2_emissions_objective = None
        self.total_energy_co2eq_emissions = None
        self.param = param
        self.configure_parameters()
        self.create_dataframe()
        self.total_energy_production = None
        self.epsilon = 1.e-5 # for the CO2 objective function
        self.all_sections_emissions_df = None

    def configure_parameters(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.new_sector_list = self.param[GlossaryCore.SectorListValue]
        self.economic_sectors_except_agriculture = [sector for sector in self.new_sector_list if sector != GlossaryCore.SectorAgriculture]
        self.CO2_land_emissions = self.param[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)]
        self.CH4_land_emissions = self.param[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)]
        self.N2O_land_emissions = self.param[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)]
        self.GHG_total_energy_emissions = self.param['GHG_total_energy_emissions']
        self.constraint_nze_2050_ref = self.param['constraint_nze_2050_ref']
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12

        self.gwp_20 = self.param['GHG_global_warming_potential20']
        self.gwp_100 = self.param['GHG_global_warming_potential100']

        self.CO2EmissionsRef = self.param[GlossaryCore.CO2EmissionsRef['var_name']]
        self.total_energy_production = self.param[GlossaryCore.EnergyProductionValue]


    def configure_parameters_update(self, inputs_dict):

        self.CO2_land_emissions = inputs_dict[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)]
        self.CH4_land_emissions = inputs_dict[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)]
        self.N2O_land_emissions = inputs_dict[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)]
        self.residential_energy_consumption = inputs_dict[GlossaryCore.ResidentialEnergyConsumptionDfValue]
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
            year_start, year_end + 1)

        self.ghg_emissions_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        self.gwp_emissions = pd.DataFrame({GlossaryCore.Years: self.years_range})

    def compute_land_emissions(self):
        """
        Compute emissions from land
        """

        self.ghg_emissions_df[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)] = self.CO2_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values
        self.ghg_emissions_df[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)] = self.CH4_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values
        self.ghg_emissions_df[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)] = self.N2O_land_emissions.drop(
            GlossaryCore.Years, axis=1).sum(axis=1).values

    def compute_total_emissions(self):
        """
        Total emissions is defined as : land use emissions + energy mix emissions + non energy emissions from economy
        
        Note : Non energy emissions from economy are only in CO2 Equivalent
        """
        for ghg in self.GHG_TYPE_LIST:
            self.ghg_emissions_df[GlossaryCore.insertGHGNonEnergyEmissions.format(ghg)] = 0.
        self.ghg_emissions_df[GlossaryCore.insertGHGNonEnergyEmissions.format(GlossaryCore.CO2)] = self.total_economics_emisssions[GlossaryCore.NonEnergyEmissions].values

        for ghg in self.GHG_TYPE_LIST:
            self.ghg_emissions_df[GlossaryCore.insertGHGTotalEmissions.format(ghg)] = \
                self.ghg_emissions_df[GlossaryCore.insertGHGAgriLandEmissions.format(ghg)].values + \
                self.ghg_emissions_df[GlossaryCore.insertGHGNonEnergyEmissions.format(ghg)].values + \
                self.ghg_emissions_df[GlossaryCore.insertGHGEnergyEmissions.format(ghg)].values

    def compute_total_gwp(self):

        for ghg in self.GHG_TYPE_LIST:

            self.gwp_emissions[f'{ghg}_20'] = self.ghg_emissions_df[GlossaryCore.insertGHGTotalEmissions.format(ghg)] * self.gwp_20[ghg]
            self.gwp_emissions[f'{ghg}_100'] = self.ghg_emissions_df[GlossaryCore.insertGHGTotalEmissions.format(ghg)] * self.gwp_100[ghg]

        self.gwp_emissions['Total GWP (20-year basis)'] = self.gwp_emissions[[f'{ghg}_20' for ghg in self.GHG_TYPE_LIST]].sum(axis=1)
        self.gwp_emissions['Total GWP (100-year basis)'] = self.gwp_emissions[[f'{ghg}_100' for ghg in self.GHG_TYPE_LIST]].sum(axis=1)

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
        columns_to_sum = [GlossaryCore.insertGHGTotalEmissions.format(ghg) for ghg in self.gwp_100.keys()]
        self.total_energy_co2eq_emissions = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.TotalEnergyEmissions: self.GHG_total_energy_emissions[columns_to_sum].multiply(self.gwp_100.values()).sum(axis=1)
        })

    def d_total_co2_eq_energy_emissions(self, d_ghg_total_emissions, ghg: str):
        return d_ghg_total_emissions * self.gwp_100[ghg]

    def compute_energy_emission_per_section(self):
        """
        Computing the energy emission for each section of the sector

        section_energy_emission (GtCO2eq) = section_energy_consumption (PWh) x carbon_intensity (kgCO2eq/kWh)
        """
        carbon_intensity = self.carbon_intensity_of_energy_mix[GlossaryCore.EnergyCarbonIntensityDfValue].values
        for sector in self.new_sector_list:
            section_energy_emissions = {
                GlossaryCore.Years: self.years_range
            }
            energy_sections_consumptions_consumptions_of_sector = self.param[f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}"]
            for section in GlossaryCore.SectionDictSectors[sector]:
                section_energy_emissions[section] = energy_sections_consumptions_consumptions_of_sector[section].values * carbon_intensity

            self.dict_sector_sections_energy_emissions[sector] = pd.DataFrame(section_energy_emissions)

    def compute_non_energy_emission_per_section(self):
        """
        Computing the energy emission for each section of the sector

        section_non_energy_emission (GtCO2eq) = section_non_energy_emission_wrt_gdp (tCO2eq/M$) x section_gdp (T$) / 1000.
        """

        for sector in self.new_sector_list:
            section_non_energy_emissions = {
                GlossaryCore.Years: self.years_range
            }
            energy_sections_non_energy_emissions_gdp_of_sector = self.param[f"{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"]
            sector_sections_gdp = self.param[f"{sector}.{GlossaryCore.SectionGdpDfValue}"]
            for section in GlossaryCore.SectionDictSectors[sector]:
                section_non_energy_emissions[section] = energy_sections_non_energy_emissions_gdp_of_sector[section].values * \
                                                        sector_sections_gdp[section].values / 1000.

            self.dict_sector_sections_non_energy_emissions[sector] = pd.DataFrame(section_non_energy_emissions)

    def compute_total_emission_per_section(self):
        """
        Computing the total emission for each section of the sector

        section_emission (GtCO2eq) = section_energy_emission (GtCO2eq) + section_non_energy_emission (GtCO2eq)
        """
        for sector in self.new_sector_list:
            sections_emissions = {
                GlossaryCore.Years: self.years_range
            }
            for section in GlossaryCore.SectionDictSectors[sector]:
                sections_emissions[section] = self.dict_sector_sections_energy_emissions[sector][section].values + \
                                              self.dict_sector_sections_non_energy_emissions[sector][section].values

            self.dict_sector_sections_emissions[sector] = pd.DataFrame(sections_emissions)

    def compute_total_emissions_for_section_agriculture(self):
        """
        Agriculture is not computed with the other sectors as there is no energy and non energy emissions
        Calculate the total Global Warming Potential (GWP) over a 100-year time horizon for CO2, CH4, and N2O emissions
        for agriculture sector (and the associated section)

        This method combines the greenhouse gas emissions data from self.ghg_emissions_df with the GWP100 conversion
        factors stored . It calculates the GWP100 for each gas and then sums them up to get the
        total GWP100 for each year.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing two columns:
            - GlossaryCore.Years: The year of the emissions
            - 'Total_GWP100': The total GWP100 value for all three gases combined for each year
        """
        # List of greenhouse gases
        gases = [GlossaryCore.CO2, GlossaryCore.CH4, GlossaryCore.N2O]

        # Create a dictionary with emission columns for each gas
        emissions_columns = {
            gas: GlossaryCore.insertGHGAgriLandEmissions.format(gas)
            for gas in gases
        }

        # Calculate GWP100 for each gas and year
        gwp_100_by_gas = {
            gas: self.ghg_emissions_df[emissions_columns[gas]] * self.gwp_100[gas]
            for gas in gases
        }

        # Calculate total GWP100 per year
        total_gwp_100 = pd.DataFrame({
            GlossaryCore.Years: self.ghg_emissions_df[GlossaryCore.Years],
            GlossaryCore.SectionA: sum(gwp_100_by_gas.values())  #store values in the only section of agriculture sector
        })

        # add it to dictionary
        self.dict_sector_sections_emissions[GlossaryCore.SectorAgriculture] = total_gwp_100

    def aggregate_emissions_per_section(self):
        """
        Aggregates emissions data from all sectors and converts units from Gt to Mt.

        This method processes the emissions data stored in self.dict_sector_sections_emissions,
        which contains emissions data for different sectors and their subsections.
        """
        # Get all DataFrames in a list
        all_dfs = [sector_data for sector_data in self.dict_sector_sections_emissions.values()
                   ]

        # Extract the 'years' column from the first DataFrame
        years = all_dfs[0][GlossaryCore.Years]

        # Aggregate all DataFrames without the 'years' column and convert Gt to Mt
        aggregated_df = pd.concat([df.drop(columns='years').mul(1000) for df in all_dfs], axis=1)

        # Add the 'years' column at the beginning
        aggregated_df.insert(0, GlossaryCore.Years, years)
        self.all_sections_emissions_df = aggregated_df

    def compute_total_emission_sectors(self):
        """
        Computing the total emissions for each sector
        """
        # sector_emission = sum of section_emission
        for sector in self.new_sector_list:
            sector_emissions = {GlossaryCore.Years: self.years_range}
            sector_sections = GlossaryCore.SectionDictSectors[sector]
            sector_emissions[GlossaryCore.EnergyEmissions] = self.dict_sector_sections_energy_emissions[sector][sector_sections].values.sum(axis=1)
            sector_emissions[GlossaryCore.NonEnergyEmissions] = self.dict_sector_sections_non_energy_emissions[sector][sector_sections].values.sum(axis=1)
            sector_emissions[GlossaryCore.TotalEmissions] = sector_emissions[GlossaryCore.EnergyEmissions] + sector_emissions[GlossaryCore.NonEnergyEmissions]
            self.dict_sector_emissions[sector] = pd.DataFrame(sector_emissions)

    def compute_total_economics_emission(self):
        """Compute economics emissions : sum of emissions for sectors Services and Industry"""
        total_economics_emisssions = {GlossaryCore.Years: self.years_range}
        total_economics_emisssions[GlossaryCore.EnergyEmissions] = np.sum([self.dict_sector_emissions[sector][GlossaryCore.EnergyEmissions].values for sector in self.economic_sectors_except_agriculture], axis=0)
        total_economics_emisssions[GlossaryCore.NonEnergyEmissions] = np.sum([self.dict_sector_emissions[sector][GlossaryCore.NonEnergyEmissions].values for sector in self.economic_sectors_except_agriculture], axis=0)
        total_economics_emisssions[GlossaryCore.TotalEmissions] = np.sum([self.dict_sector_emissions[sector][GlossaryCore.TotalEmissions].values for sector in self.economic_sectors_except_agriculture], axis=0)
        self.total_economics_emisssions = pd.DataFrame(total_economics_emisssions)

    def compute(self, inputs_dict):
        """
        Compute outputs of the pyworld3
        """
        self.param = inputs_dict

        # compute land emissions
        self.compute_land_emissions()

        # compute energy emissions
        self.compute_total_co2_eq_energy_emissions()
        self.compute_energy_mix_total_emissions()

        # compute emissions from economy
        self.compute_carbon_intensity_of_energy_mix()
        self.compute_energy_emission_per_section()
        self.compute_non_energy_emission_per_section()
        self.compute_total_emission_per_section()
        self.compute_total_emissions_for_section_agriculture()
        self.compute_total_emission_sectors()
        self.compute_total_economics_emission()

        # compute total emissions
        self.compute_total_emissions()
        self.aggregate_emissions_per_section()

        # compute other indicators
        self.compute_total_gwp()
        self.compute_gwp_per_sector()
        self.compute_CO2_emissions_objective()
        self.compute_net_zero_2050_constraint_df()


        # compute emission households
        self.compute_energy_emission_households()


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

    def d_section_energy_emissions_d_user_input(self, d_carbon_intensity_d_user_input, sector_name:str, section_name: str):
        return np.diag(self.param[f"{sector_name}.{GlossaryCore.SectionEnergyConsumptionDfValue}"][section_name].values) * d_carbon_intensity_d_user_input

    def d_section_non_energy_emissions_d_gdp_section(self, sector: str, section: str):
        """
        Derivative of section non energy emissions for section S wrt any input (named X)
        User should provide the derivative of output net of damage wrt input variable X in order to
        compute the chain rule
        """

        return np.diag(self.param[f"{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"][section] / 1000.)

    def d_section_energy_emissions_d_section_energy_consumption(self):
        return np.diag(self.carbon_intensity_of_energy_mix[GlossaryCore.EnergyCarbonIntensityDfValue].values)

    def compute_energy_mix_total_emissions(self):
        for ghg in self.GHG_TYPE_LIST:
            self.ghg_emissions_df[GlossaryCore.insertGHGEnergyEmissions.format(ghg)] = \
                self.GHG_total_energy_emissions[GlossaryCore.insertGHGTotalEmissions.format(ghg)].values

    def compute_gwp_per_sector(self):
        """computes global warming potential per sector"""
        emission_types = {GlossaryCore.AgricultureAndLandUse: GlossaryCore.insertGHGAgriLandEmissions,
                          GlossaryCore.Energy: GlossaryCore.insertGHGEnergyEmissions,
                          GlossaryCore.NonEnergy: GlossaryCore.insertGHGNonEnergyEmissions}

        for emission_type, column_name in emission_types.items():
            emissions_type_gwp_20_values = []
            emissions_type_gwp_100_values = []
            for ghg in self.GHG_TYPE_LIST:
                gwp_20_emission_type_ghg = self.ghg_emissions_df[column_name.format(ghg)].values * self.gwp_20[ghg]
                gwp_100_emission_type_ghg = self.ghg_emissions_df[column_name.format(ghg)].values * self.gwp_100[ghg]

                emissions_type_gwp_20_values.append(gwp_20_emission_type_ghg)
                emissions_type_gwp_100_values.append(gwp_100_emission_type_ghg)

            self.gwp_emissions[f'{emission_type}_20'] = np.sum(emissions_type_gwp_20_values, axis=0)
            self.gwp_emissions[f'{emission_type}_100'] = np.sum(emissions_type_gwp_100_values, axis=0)

    def compute_energy_emission_households(self):
        """ Emissions (Gt CO2 Eq) : Households energy consumption (PWh) X Carbon intensity (kgCO2Eq/KWh)"""

        energy_emission_households = (self.residential_energy_consumption[GlossaryCore.TotalProductionValue].values *
                                      self.carbon_intensity_of_energy_mix[
                                          GlossaryCore.EnergyCarbonIntensityDfValue].values)

        self.energy_emission_households_df = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.TotalEmissions: energy_emission_households
        })

    def compute_net_zero_2050_constraint_df(self):

        self.emissions_after_2050_df = self.ghg_emissions_df.loc[self.years_range >= 2050][[GlossaryCore.Years, GlossaryCore.insertGHGTotalEmissions.format(GlossaryCore.CO2)]]
        self.emissions_after_2050_df[GlossaryCore.insertGHGTotalEmissions.format(GlossaryCore.CO2)] = self.emissions_after_2050_df[GlossaryCore.insertGHGTotalEmissions.format(GlossaryCore.CO2)].values / self.constraint_nze_2050_ref

    def d_2050_carbon_negative_constraint(self, d_total_co2_emissions):
        return d_total_co2_emissions[self.years_range >= 2050] / self.constraint_nze_2050_ref
