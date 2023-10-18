'''
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
'''
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline import SectorDiscipline
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sostrades_core.study_manager.study_manager import StudyManager

from os.path import join, dirname
from pandas import read_csv
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.macro_name = 'Macroeconomics'
        self.labormarket_name = 'LaborMarket'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):
        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        self.nb_per = round(self.year_end - self.year_start + 1)

        # Energy
        brut_net = 1 / 1.45
        energy_outlook = pd.DataFrame({
            'year': [2000, 2005, 2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [118.112, 134.122, 149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842,
                       206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        # Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(self.year_start, self.year_end + 1))
        energy_supply_values = energy_supply * brut_net

        energy_production = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values*0.7})


        # workforce share
        agrishare = 27.4
        indusshare = 21.7
        serviceshare = 50.9
        workforce_share = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.SectorAgriculture: agrishare,
                                        GlossaryCore.SectorIndustry: indusshare, GlossaryCore.SectorServices: serviceshare})

        # Damage
        damage_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.Damages: np.zeros(self.nb_per), GlossaryCore.DamageFractionOutput: np.zeros(self.nb_per),
             GlossaryCore.BaseCarbonPrice: np.zeros(self.nb_per)})


        # data for consumption
        temperature = np.linspace(1, 3, len(years))
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature, GlossaryCore.TempOcean: temperature / 100})
        temperature_df.index = years
        residential_energy = np.linspace(21, 58, len(years))
        residential_energy_df = pd.DataFrame(
            {GlossaryCore.Years: years, 'residential_energy': residential_energy})
        energy_price = np.arange(110, 110 + len(years))
        energy_mean_price = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.EnergyPriceValue: energy_price})


        # economisc df to init mda
        # Test With a GDP that grows at 2%
        gdp = [130.187] * len(years)
        economics_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp})
        economics_df.index = years

        total_invests = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, len(years))})

        invest_indus = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareInvestment: np.linspace(40, 65, len(years)) * 1 / 3})

        invest_services = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareInvestment: np.linspace(40, 65, len(years)) * 1 / 6})

        invest_agriculture = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareInvestment: np.linspace(40, 65, len(years)) * 1 / 2})

        share_energy_agriculture = pd.DataFrame({GlossaryCore.Years: years,
                                                      GlossaryCore.ShareSectorEnergy: np.linspace(12, 20, len(years))})

        share_energy_services = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.ShareSectorEnergy: np.linspace(39, 59, len(years))})


        cons_input = {}
        cons_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        cons_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        cons_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.InvestmentDfValue}"] = total_invests
        cons_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}"] = damage_df
        cons_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}"] = damage_df
        cons_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.DamageDfValue}"] = damage_df
        cons_input[f"{self.study_name}.{GlossaryCore.TemperatureDfValue}"] = temperature_df
        cons_input[f"{self.study_name}.EnergyMix.residential_energy"] = residential_energy_df
        # TODO clean this quick workaround
        cons_input[f"{self.study_name}.residential_energy"] = residential_energy_df

        cons_input[f"{self.study_name}.{GlossaryCore.EnergyMeanPriceValue}"] = energy_mean_price
        #cons_input[f"{self.study_name}.{self.macro_name}.{GlossaryCore.InvestmentDfValue}"] = base_dummy_data
        cons_input[f"{self.study_name}.{self.labormarket_name}.{'workforce_share_per_sector'}"] = workforce_share
        cons_input[f"{self.study_name}.{GlossaryCore.EconomicsDfValue}"] = economics_df

        cons_input[f"{self.study_name}.{self.macro_name}.Services.{GlossaryCore.ShareSectorInvestmentDfValue}"] = invest_services
        cons_input[f"{self.study_name}.{self.macro_name}.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}"] = invest_agriculture
        cons_input[f"{self.study_name}.{self.macro_name}.Industry.{GlossaryCore.ShareSectorInvestmentDfValue}"] = invest_indus

        cons_input[f"{self.study_name}.{self.macro_name}.Services.{GlossaryCore.ShareSectorEnergyDfValue}"] = share_energy_services
        cons_input[f"{self.study_name}.{self.macro_name}.Agriculture.{GlossaryCore.ShareSectorEnergyDfValue}"] = share_energy_agriculture

        cons_input[f"{self.study_name}.{GlossaryCore.EnergyProductionValue}"] = energy_production

        setup_data_list.append(cons_input)

        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 70,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.sub_mda_class': 'MDAGaussSeidel'}

        setup_data_list.append(numerical_values_dict)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()

