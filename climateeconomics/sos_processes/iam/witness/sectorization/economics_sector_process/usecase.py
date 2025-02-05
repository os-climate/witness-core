'''
Copyright 2023 Capgemini

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

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


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

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, name='', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.macro_name = 'Macroeconomics'
        self.year_start = year_start
        self.year_end = year_end
        self.nb_poles = 8
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):


        setup_data_list = []

        data_agri = {}
        setup_data_list.append(data_agri)

        years = np.arange(self.year_start, self.year_end + 1, 1)
        self.nb_per = round(self.year_end - self.year_start + 1)

        # data dir 
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')

        # Energy
        brut_net = 1 / 1.45
        energy_outlook = pd.DataFrame({
            GlossaryCore.Years: [2000, 2005, 2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [118.112, 134.122, 149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084,
                       197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook[GlossaryCore.Years], energy_outlook['energy'])
        # Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(self.year_start, self.year_end + 1))
        energy_supply_values = energy_supply * brut_net
        if self.year_start == 2000 and self.year_end == 2020:
            data_dir = join(
                dirname(dirname(dirname(dirname(dirname(dirname(__file__)))))), 'tests', 'data/sectorization_fitting')
            # Energy
            hist_energy = pd.read_csv(join(data_dir, 'hist_energy_sect.csv'))
            services_energy = pd.DataFrame({GlossaryCore.Years: hist_energy[GlossaryCore.Years], GlossaryCore.TotalProductionValue: hist_energy[GlossaryCore.SectorServices]})
            indus_energy = pd.DataFrame({GlossaryCore.Years: hist_energy[GlossaryCore.Years], GlossaryCore.TotalProductionValue: hist_energy[GlossaryCore.SectorIndustry]})
            # Workforce
            hist_workforce = pd.read_csv(join(data_dir, 'hist_workforce_sect.csv'))
            workforce_df = hist_workforce
            #Tshare sectors invest

        else:
            indus_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.2894})
            services_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.37})

            workforce = np.ones_like(years) * 3389
            workforce[0] = 3389.556200
            workforce[1] = 3450.067707
            workforce_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.SectorAgriculture: workforce * 0.274,
                                         GlossaryCore.SectorServices: workforce * 0.509, GlossaryCore.SectorIndustry: workforce * 0.217})

        agri_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.02136})
        # Damage
        damage_df = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.DamageFractionOutput: np.zeros(self.nb_per),})

        invest_indus = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40,65, len(years))*1/3})

        invest_services = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, len(years)) * 1/6})


        invest_energy_wo_tax = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: np.linspace(40, 65, len(years))})


        energy_emission_df = pd.DataFrame({
            GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
            GlossaryCore.EnergyCarbonIntensityDfValue: 100.0
        })


        sect_input = {}
        sect_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        sect_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        sect_input[f"{self.study_name}.{GlossaryCore.WorkforceDfValue}"] = workforce_df
        sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}"] = invest_indus
        sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}"] = invest_services
        sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}"] = indus_energy
        sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}"] = agri_energy
        sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.EnergyProductionValue}"] = services_energy
        sect_input[f"{self.study_name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}"] = invest_energy_wo_tax
        sect_input[f"{self.study_name}.{GlossaryCore.DamageFractionDfValue}"] = damage_df
        sect_input[f"{self.study_name}.{GlossaryCore.EnergyCarbonIntensityDfValue}"] = energy_emission_df

        for sector in GlossaryCore.SectorsPossibleValues:
            global_data_dir = join(dirname(dirname(dirname(dirname(dirname(dirname(__file__)))))), 'data')
            weighted_average_percentage_per_sector_df = pd.read_csv(
                join(global_data_dir, f'weighted_average_percentage_{sector.lower()}_sections.csv'))
            subsector_share_dict = {
                **{GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1), },
                **dict(zip(weighted_average_percentage_per_sector_df.columns[1:],
                           weighted_average_percentage_per_sector_df.values[0, 1:]))
            }
            section_gdp_df = pd.DataFrame(subsector_share_dict)
            sect_input[f"{self.study_name}.{self.macro_name}.{sector}.{GlossaryCore.SectionGdpPercentageDfValue}"] = section_gdp_df

            section_non_energy_emission_gdp_df = pd.read_csv(
                join(global_data_dir, f'non_energy_emission_gdp_{sector.lower()}_sections.csv'))
            subsector_share_dict = {
                **{GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1), },
                **dict(zip(section_non_energy_emission_gdp_df.columns[1:],
                           section_non_energy_emission_gdp_df.values[0, 1:]))
            }
            section_non_energy_emission_gdp_df = pd.DataFrame(subsector_share_dict)
            sect_input[f"{self.study_name}.{self.macro_name}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"] = section_non_energy_emission_gdp_df

            section_energy_consumption_percentage_df = pd.read_csv(
                join(global_data_dir, f'energy_consumption_percentage_{sector.lower()}_sections.csv'))
            subsector_share_dict = {
                **{GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1), },
                **dict(zip(section_energy_consumption_percentage_df.columns[1:],
                           section_energy_consumption_percentage_df.values[0, 1:]))
            }
            section_energy_consumption_percentage_df = pd.DataFrame(subsector_share_dict)
            sect_input[
                f"{self.study_name}.{self.macro_name}.{sector}.{GlossaryCore.SectionEnergyConsumptionPercentageDfValue}"] = section_energy_consumption_percentage_df

            # section non-energy emissions per dollar of pib
            section_non_energy_emission_gdp_df = pd.read_csv(
                join(global_data_dir, f'non_energy_emission_gdp_{sector.lower()}_sections.csv'))
            section_non_energy_emission_gdp_dict = {
                **{GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1), },
                **dict(zip(section_non_energy_emission_gdp_df.columns[1:],
                           section_non_energy_emission_gdp_df.values[0, 1:]))
            }
            section_non_energy_emission_gdp_df = pd.DataFrame(section_non_energy_emission_gdp_dict)
            sect_input[
                f"{self.study_name}.{self.macro_name}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"] = section_non_energy_emission_gdp_df

        if self.year_start == 2000:
            sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{'capital_start'}"] = 31.763
            sect_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{'capital_start'}"] = 139.1369
            sect_input[f"{self.study_name}.{GlossaryCore.DamageToProductivity}"] = True


        transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": 7.6})
        margin = pd.DataFrame({GlossaryCore.Years: years, 'margin': 110.})

        population_2021 = 7_954_448_391
        population_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.PopulationValue: np.linspace(population_2021 / 1e6, 7870 * 1.2, self.nb_per),
        })
        crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: np.linspace(0., 4.5, len(years)) * 0,  # fake
        })
        energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.EnergyPriceValue: np.linspace(70, 120, self.nb_per)
        })

        share_investments_between_agri_subsectors = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.Crop: 90.,
            GlossaryCore.Forestry: 10.,
        })

        share_investments_inside_forestry = pd.DataFrame({
            GlossaryCore.Years: years,
            "Managed wood": 33.,
            "Deforestation": 33.,
            "Reforestation": 33.,
        })

        share_investments_inside_crop = pd.DataFrame({
            GlossaryCore.Years: years,
            **GlossaryCore.crop_calibration_data["invest_food_type_share_start"]
        })

        economics_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.GrossOutput: 0.,
            GlossaryCore.OutputNetOfDamage: 1.015 ** np.arange(0,
                                                               len(years)) * DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(
                self.year_start) * 0.98,
        })
        sect_input.update({
            f'{self.study_name}.transport_cost': transport_df,
            f'{self.study_name}.margin': margin,
            f'{self.study_name}.mdo_sectors_invest_level': 0,
            f'{self.study_name}.{GlossaryCore.PopulationDfValue}': population_df,
            f'{self.study_name}.{GlossaryCore.CropProductivityReductionName}': crop_productivity_reduction,
            f'{self.study_name}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price,
            f'{self.study_name}.{GlossaryCore.EconomicsDfValue}': economics_df,
            f'{self.study_name}.Macroeconomics.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}': share_investments_between_agri_subsectors,
            f'{self.study_name}.Macroeconomics.Agriculture.Crop.{GlossaryCore.SubShareSectorInvestDfValue}': share_investments_inside_crop,
            f'{self.study_name}.Macroeconomics.Agriculture.Forestry.{GlossaryCore.SubShareSectorInvestDfValue}': share_investments_inside_forestry,
        })

        setup_data_list.append(sect_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
