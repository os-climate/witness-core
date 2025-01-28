'''
Copyright 2024 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline import (
    GHGemissionsDiscipline,
)


def update_dspace_with(dspace_dict, name, value, lower, upper):
    ''' type(value) has to be ndarray
    '''
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)
    dspace_dict['variable'].append(name)
    dspace_dict['value'].append(value.tolist())
    dspace_dict['lower_bnd'].append(lower)
    dspace_dict['upper_bnd'].append(upper)
    dspace_dict['dspace_size'] += len(value)


def update_dspace_dict_with(dspace_dict, name, value, lower, upper, activated_elem=None, enable_variable=True):
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)

    if activated_elem is None:
        activated_elem = [True] * len(value)
    dspace_dict[name] = {'value': value,
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable,
                         'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, name='', execution_engine=None,
                 main_study: bool = True):
        super().__init__(__file__, execution_engine=execution_engine)
        self.main_study = main_study
        self.study_name = 'usecase'
        self.macro_name = 'Macroeconomics'
        self.labormarket_name = 'LaborMarket'
        self.redistrib_energy_name = 'SectorsEnergyDistribution'
        self.year_start = year_start
        self.year_end = year_end
        self.nb_poles = 8
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        self.nb_per = round(self.year_end - self.year_start + 1)

        # Damage
        damage_fraction_df = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.DamageFractionOutput: np.zeros(self.nb_per),
        })

        damage_df = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.Damages: np.zeros(self.nb_per),
             GlossaryCore.EstimatedDamages: np.zeros(self.nb_per)}
        )

        # economisc df to init mda
        gdp = [130.187] * len(years)
        economics_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp})

        # Investment
        invest_indus_start = DatabaseWitnessCore.InvestInduspercofgdp2020.value
        invest_agri_start = DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value
        invest_services_start = DatabaseWitnessCore.InvestServicespercofgdpYearStart.value
        invest_energy_start = 1.077
        total_invest_start = invest_indus_start + invest_agri_start + invest_services_start + invest_energy_start

        total_invests = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: total_invest_start})

        # Energy
        energy_investment_wo_tax = pd.DataFrame({GlossaryCore.Years: years,
                                                 GlossaryCore.EnergyInvestmentsWoTaxValue: 1000.
                                                 })

        carbon_intensity_of_energy_mix = pd.DataFrame({
            GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
            GlossaryCore.EnergyCarbonIntensityDfValue: 100.0
        })

        cons_input = {
            f"{self.study_name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.study_name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.study_name}.{self.macro_name}.{GlossaryCore.InvestmentDfValue}": total_invests,
            f"{self.study_name}.{GlossaryCore.DamageFractionDfValue}": damage_fraction_df,
            f"{self.study_name}.{GlossaryCore.EconomicsDfValue}": economics_df,
            f"{self.study_name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}": energy_investment_wo_tax,
             f'{self.study_name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': carbon_intensity_of_energy_mix,
        }
        for sector in GlossaryCore.SectorsPossibleValues:
            cons_input[f'{self.study_name}.GHGemissions.{GlossaryCore.EconomicSectors}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}'] = DatabaseWitnessCore.SectionsNonEnergyEmissionsDict.value[sector]

        if self.main_study:

            invest_indus = pd.DataFrame(
                {GlossaryCore.Years: years,
                 GlossaryCore.ShareInvestment: invest_indus_start})

            invest_services = pd.DataFrame(
                {GlossaryCore.Years: years,
                 GlossaryCore.ShareInvestment: invest_services_start})

            invest_agriculture = pd.DataFrame(
                {GlossaryCore.Years: years,
                 GlossaryCore.ShareInvestment: invest_agri_start})

            # Energy
            share_energy_resi_2020 = DatabaseWitnessCore.EnergyshareResidentialYearStart.value
            share_energy_other_2020 = DatabaseWitnessCore.EnergyshareOtherYearStart.value
            share_energy_agri_2020 = DatabaseWitnessCore.EnergyshareAgricultureYearStart.value
            share_energy_services_2020 = DatabaseWitnessCore.EnergyshareServicesYearStart.value
            share_energy_agriculture = pd.DataFrame({GlossaryCore.Years: years,
                                                     GlossaryCore.ShareSectorEnergy: share_energy_agri_2020})
            share_energy_services = pd.DataFrame({GlossaryCore.Years: years,
                                                  GlossaryCore.ShareSectorEnergy: share_energy_services_2020})
            share_energy_resi = pd.DataFrame({GlossaryCore.Years: years,
                                              GlossaryCore.ShareSectorEnergy: share_energy_resi_2020})
            share_energy_other = pd.DataFrame({GlossaryCore.Years: years,
                                               GlossaryCore.ShareSectorEnergy: share_energy_other_2020})

            brut_net = 1 / 1.45
            energy_outlook = pd.DataFrame({
                GlossaryCore.Years: [2000, 2005, 2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
                'energy': [118.112, 134.122, 149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084,
                           197.8418842,
                           206.1201182, 220.000, 250.0, 300.0]})
            f2 = interp1d(energy_outlook[GlossaryCore.Years], energy_outlook['energy'])
            # Find values for 2020, 2050 and concat dfs
            energy_supply = f2(np.arange(self.year_start, self.year_end + 1))
            energy_supply_values = energy_supply * brut_net

            energy_production = pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.7})

            # data for consumption
            temperature = np.linspace(1, 3, len(years))
            temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
            energy_price = np.arange(110, 110 + len(years))
            energy_mean_price = pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.EnergyPriceValue: energy_price})

            # workforce share
            agrishare = 27.4
            indusshare = 21.7
            serviceshare = 50.9
            workforce_share = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.SectorAgriculture: agrishare,
                                            GlossaryCore.SectorIndustry: indusshare,
                                            GlossaryCore.SectorServices: serviceshare})

            CO2_emitted_land = pd.DataFrame()
            # GtCO2
            emission_forest = np.linspace(0.04, 0.04, len(years))
            cum_emission = np.cumsum(emission_forest)
            CO2_emitted_land[GlossaryCore.Years] = years
            CO2_emitted_land['Crop'] = np.zeros(len(years))
            CO2_emitted_land['Forest'] = cum_emission
            GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                       GlossaryCore.TotalCO2Emissions: np.linspace(37., 10.,
                                                                                                   len(years)),
                                                       GlossaryCore.TotalN2OEmissions: np.linspace(1.7e-3, 5.e-4,
                                                                                                   len(years)),
                                                       GlossaryCore.TotalCH4Emissions: np.linspace(0.17, 0.01,
                                                                                                   len(years))})
            CO2_indus_emissions_df = pd.DataFrame({
                GlossaryCore.Years: years,
                "indus_emissions": 0.
            })

            for sector in GlossaryCore.SectorsPossibleValues:
                global_data_dir = join(dirname(dirname(dirname(dirname(dirname(dirname(__file__)))))), 'data')
                section_non_energy_emission_gdp_df = pd.read_csv(
                    join(global_data_dir, f'non_energy_emission_gdp_{sector.lower()}_sections.csv'))
                subsector_share_dict = {
                    **{GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1), },
                    **dict(zip(section_non_energy_emission_gdp_df.columns[1:],
                               section_non_energy_emission_gdp_df.values[0, 1:]))
                }
                section_non_energy_emission_gdp_df = pd.DataFrame(subsector_share_dict)
                cons_input[f"{self.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"] = section_non_energy_emission_gdp_df

            cons_input.update({
                f"{self.study_name}.{self.labormarket_name}.{'workforce_share_per_sector'}": workforce_share,
                f"{self.study_name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
                f"{self.study_name}.{GlossaryCore.EnergyMeanPriceValue}": energy_mean_price,
                f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorInvestmentDfValue}": invest_agriculture,
                f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorInvestmentDfValue}": invest_services,
                f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ShareSectorInvestmentDfValue}": invest_indus,
                f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorEnergyDfValue}": share_energy_services,
                f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorEnergyDfValue}": share_energy_agriculture,
                f"{self.study_name}.{GlossaryCore.ShareResidentialEnergyDfValue}": share_energy_resi,
                f"{self.study_name}.{self.redistrib_energy_name}.{GlossaryCore.ShareOtherEnergyDfValue}": share_energy_other,
                f"{self.study_name}.{GlossaryCore.StreamProductionValue}": energy_production,
                f"{self.study_name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}": CO2_emitted_land,
                f"{self.study_name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}": CO2_emitted_land,
                f"{self.study_name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}": CO2_emitted_land,
                f"{self.study_name}.CO2_indus_emissions_df": CO2_indus_emissions_df,
                f"{self.study_name}.GHG_total_energy_emissions": GHG_total_energy_emissions,
                f'{self.study_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}': damage_df,
                f'{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.DamageDfValue}': damage_df,
                f'{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}': damage_df,
                f'{self.study_name}.Utility.{GlossaryCore.SectorAgriculture}_strech_scurve': 3.7,
                f'{self.study_name}.Utility.{GlossaryCore.SectorAgriculture}_shift_scurve': -0.4,
                f'{self.study_name}.Utility.{GlossaryCore.SectorServices}_strech_scurve': 1.8,
                f'{self.study_name}.Utility.{GlossaryCore.SectorServices}_shift_scurve': -0.2,
                f'{self.study_name}.Utility.{GlossaryCore.SectorIndustry}_strech_scurve': 1.7,
                f'{self.study_name}.Utility.{GlossaryCore.SectorIndustry}_shift_scurve': -0.25,
            })

        setup_data_list.append(cons_input)

        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 70,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.inner_mda_name': 'MDAGaussSeidel'}

        setup_data_list.append(numerical_values_dict)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
