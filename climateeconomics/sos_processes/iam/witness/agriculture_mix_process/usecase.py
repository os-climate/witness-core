'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2024/06/24 Copyright 2023 Capgemini

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
import scipy.interpolate as sc
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore

AGRI_MIX_MODEL_LIST = ['Crop', 'Forest']
AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT = []
COARSE_AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT = []


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
    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, execution_engine=None,
                 agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 model_list=AGRI_MIX_MODEL_LIST):
        super().__init__(__file__, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.techno_list = agri_techno_list
        self.model_list = model_list
        self.energy_name = None
        self.nb_poles = 8
        self.additional_ns = ''
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):

        agriculture_mix = 'Agriculture'
        energy_name = f'{agriculture_mix}'
        years = np.arange(self.year_start, self.year_end + 1)
        self.stream_prices = pd.DataFrame({GlossaryCore.Years: years,
                                           GlossaryEnergy.electricity: 16.0})
        year_range = self.year_end - self.year_start + 1

        temperature = np.array(np.linspace(1.05, 5.0, year_range))
        temperature_df = pd.DataFrame({
            GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature,
        })
        temperature_df.index = years

        population = np.array(np.linspace(7800.0, 9000.0, year_range))
        population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        population_df.index = years
        diet_df_default = pd.DataFrame({"red meat": [11.02],
                                        "white meat": [31.11],
                                        "milk": [79.27],
                                        "eggs": [9.68],
                                        "rice and maize": [98.08],
                                        "cereals": [78],
                                        "fruits and vegetables": [293],
                                        GlossaryCore.Fish: [23.38],
                                        GlossaryCore.OtherFood: [77.24]
                                        })
        default_kg_to_kcal = {GlossaryCore.RedMeat: 1551.05,
                              GlossaryCore.WhiteMeat: 2131.99,
                              GlossaryCore.Milk: 921.76,
                              GlossaryCore.Eggs: 1425.07,
                              GlossaryCore.RiceAndMaize: 2572.46,
                              GlossaryCore.Cereals: 2964.99,
                              GlossaryCore.FruitsAndVegetables: 559.65,
                              GlossaryCore.Fish: 609.17,
                              GlossaryCore.OtherFood: 3061.06,
                              }

        crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: np.linspace(0., 4.5, len(years)) * 0,  # fake
        })

        red_meat_average_ca_daily_intake = default_kg_to_kcal[GlossaryCore.RedMeat] * diet_df_default[GlossaryCore.RedMeat].values[0] / 365
        milk_eggs_average_ca_daily_intake = default_kg_to_kcal[GlossaryCore.Eggs] * diet_df_default[GlossaryCore.Eggs].values[0] / 365 + \
                                            default_kg_to_kcal[GlossaryCore.Milk] * diet_df_default[GlossaryCore.Milk].values[0] / 365
        white_meat_average_ca_daily_intake = default_kg_to_kcal[
                                                 GlossaryCore.WhiteMeat] * diet_df_default[GlossaryCore.WhiteMeat].values[0] / 365
        # kcal per kg 'vegetables': 200 https://www.fatsecret.co.in/calories-nutrition/generic/raw-vegetable?portionid=54903&portionamount=100.000&frc=True#:~:text=Nutritional%20Summary%3A&text=There%20are%2020%20calories%20in,%25%20carbs%2C%2016%25%20prot.
        vegetables_and_carbs_average_ca_daily_intake = diet_df_default[GlossaryCore.FruitsAndVegetables].values[0] / 365 * \
                                                       default_kg_to_kcal[GlossaryCore.FruitsAndVegetables] + \
                                                       diet_df_default[GlossaryCore.Cereals].values[0] / 365 * default_kg_to_kcal[
                                                           GlossaryCore.Cereals] + \
                                                       diet_df_default[GlossaryCore.RiceAndMaize].values[0] / 365 * \
                                                       default_kg_to_kcal[GlossaryCore.RiceAndMaize]
        fish_average_ca_daily_intake = default_kg_to_kcal[
                                           GlossaryCore.Fish] * diet_df_default[GlossaryCore.Fish].values[0] / 365
        other_average_ca_daily_intake = default_kg_to_kcal[
                                            GlossaryCore.OtherFood] * diet_df_default[GlossaryCore.OtherFood].values[
                                            0] / 365
        self.red_meat_ca_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'red_meat_calories_per_day': [red_meat_average_ca_daily_intake] * year_range})
        self.white_meat_ca_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'white_meat_calories_per_day': [white_meat_average_ca_daily_intake] * year_range})
        self.fish_ca_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.FishDailyCal: [fish_average_ca_daily_intake] * year_range})
        self.other_ca_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.OtherDailyCal: [other_average_ca_daily_intake] * year_range})
        self.vegetables_and_carbs_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'vegetables_and_carbs_calories_per_day': [vegetables_and_carbs_average_ca_daily_intake] * year_range})
        self.milk_and_eggs_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'milk_and_eggs_calories_per_day': [milk_eggs_average_ca_daily_intake] * year_range})

        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {GlossaryCore.Years: years, 'transport': np.ones(len(years)) * 7.6})

        self.stream_co2_emissions = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryEnergy.biomass_dry: - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, GlossaryEnergy.electricity: 0.0,
             GlossaryEnergy.methane: 0.123 / 15.4, 'syngas': 0.0, f"{GlossaryEnergy.hydrogen}.{GlossaryEnergy.gaseous_hydrogen}": 0.0, 'crude oil': 0.02533})



        self.reforestation_investment_df = pd.DataFrame(
            {GlossaryCore.Years: years, "reforestation_investment": forest_invest})

        deforest_invest_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.InvestmentsValue: np.linspace(114, 39, len(self.years))})

        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01, 34.05, 39.08, 44.69, 50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        co2_taxes = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2Tax: func(years)})

        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years),
            GlossaryCore.NonUseCapital: 0.,
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

        share_sectors_invest = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.SectorAgriculture: DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value,
        })

        values_dict = {
            f'{self.study_name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.study_name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.study_name}.{energy_name}.{GlossaryCore.techno_list}': self.model_list,
            f'{self.study_name}.margin': self.margin,
            f'{self.study_name}.transport_cost': self.transport,
            f'{self.study_name}.transport_margin': self.margin,
            f'{self.study_name}.{GlossaryCore.CO2TaxesValue}': co2_taxes,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.diet_df': diet_df_default,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.red_meat_calories_per_day': self.red_meat_ca_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.white_meat_calories_per_day': self.white_meat_ca_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.vegetables_and_carbs_calories_per_day': self.vegetables_and_carbs_calories_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.{GlossaryCore.FishDailyCal}': self.fish_ca_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.{GlossaryCore.OtherDailyCal}': self.other_ca_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.milk_and_eggs_calories_per_day': self.milk_and_eggs_calories_per_day,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.crop_investment': crop_investment,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Forestry}.techno_capital': techno_capital,
            f'{self.study_name}.{energy_name}.{GlossaryEnergy.Crop}.techno_capital': techno_capital,
            f'{self.study_name}.{GlossaryCore.PopulationDfValue}': population_df,
            f'{self.study_name}.{GlossaryCore.TemperatureDfValue}': temperature_df,
            f'{self.study_name}.{GlossaryCore.CropProductivityReductionName}': crop_productivity_reduction,
            f'{self.study_name}.{GlossaryCore.EconomicsDfValue}': economics_df,
            f'{self.study_name}.{GlossaryCore.ShareSectorInvestmentDfValue}': share_sectors_invest,
            f'{self.study_name}.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}': share_investments_between_agri_subsectors,
            f'{self.study_name}.Agriculture.Crop.{GlossaryCore.SubShareSectorInvestDfValue}': share_investments_inside_crop,
            f'{self.study_name}.Agriculture.Forestry.{GlossaryCore.SubShareSectorInvestDfValue}': share_investments_inside_forestry,
        }

        red_meat_percentage_ctrl = np.linspace(600, 900, self.nb_poles)
        white_meat_percentage_ctrl = np.linspace(700, 900, self.nb_poles)
        vegetables_and_carbs_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        milk_and_eggs_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        fish_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        other_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)

        deforestation_investment_ctrl = np.linspace(10.0, 5.0, self.nb_poles)
        reforestation_investment_array_mix = np.linspace(5.0, 8.0, self.nb_poles)
        crop_investment_array_mix = np.linspace(1.0, 1.5, self.nb_poles)
        managed_wood_investment_array_mix = np.linspace(
            2.0, 3.0, self.nb_poles)

        design_space_ctrl_dict = {}
        design_space_ctrl_dict['red_meat_calories_per_day_ctrl'] = red_meat_percentage_ctrl
        design_space_ctrl_dict['white_meat_calories_per_day_ctrl'] = white_meat_percentage_ctrl
        design_space_ctrl_dict[GlossaryCore.FishDailyCal + '_ctrl'] = fish_calories_per_day_ctrl
        design_space_ctrl_dict[GlossaryCore.OtherDailyCal + '_ctrl'] = other_calories_per_day_ctrl
        design_space_ctrl_dict[
            'vegetables_and_carbs_calories_per_day_ctrl'] = vegetables_and_carbs_calories_per_day_ctrl
        design_space_ctrl_dict['milk_and_eggs_calories_per_day_ctrl'] = milk_and_eggs_calories_per_day_ctrl
        design_space_ctrl_dict['deforestation_investment_ctrl'] = deforestation_investment_ctrl
        design_space_ctrl_dict['reforestation_investment_array_mix'] = reforestation_investment_array_mix

        design_space_ctrl = pd.DataFrame(design_space_ctrl_dict)
        self.design_space_ctrl = design_space_ctrl
        self.dspace = self.setup_design_space_ctrl_new()

        return ([values_dict])

    def setup_design_space_ctrl_new(self):
        # Design Space
        # header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables

        update_dspace_dict_with(ddict, 'deforestation_investment_ctrl',
                                self.design_space_ctrl['deforestation_investment_ctrl'].values,
                                [0.0] * self.nb_poles, [100.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        # -----------------------------------------
        # Invests
        update_dspace_dict_with(ddict, 'reforestation_investment_array_mix',
                                self.design_space_ctrl['reforestation_investment_array_mix'].values,
                                [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles,
                                activated_elem=[True] * self.nb_poles)

        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()

    """
    uc_cls.load_data()
    uc_cls.run()
    """
