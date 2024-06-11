'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.study_manager.study_manager import StudyManager

AGRI_MIX_MODEL_LIST = ['Crop', 'Forest']
AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT = [
    'ManagedWood', 'CropEnergy']
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
    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1, execution_engine=None,
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

    def setup_usecase(self, study_folder_path=None):

        agriculture_mix = 'AgricultureMix'
        energy_name = f'{agriculture_mix}'
        years = np.arange(self.year_start, self.year_end + 1)
        self.energy_prices = pd.DataFrame({GlossaryCore.Years: years,
                                           'electricity': 16.0})
        year_range = self.year_end - self.year_start + 1

        temperature = np.array(np.linspace(1.05, 5.0, year_range))
        temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
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
        default_kg_to_kcal = {'red meat': 1551.05,
                              'white meat': 2131.99,
                              'milk': 921.76,
                              'eggs': 1425.07,
                              'rice and maize': 2572.46,
                              'cereals': 2964.99,
                              'fruits and vegetables': 559.65,
                              GlossaryCore.Fish: 609.17,
                              GlossaryCore.OtherFood: 3061.06,
                              }
        red_meat_average_ca_daily_intake = default_kg_to_kcal['red meat'] * diet_df_default['red meat'].values[0] / 365
        milk_eggs_average_ca_daily_intake = default_kg_to_kcal['eggs'] * diet_df_default['eggs'].values[0] / 365 + \
                                            default_kg_to_kcal['milk'] * diet_df_default['milk'].values[0] / 365
        white_meat_average_ca_daily_intake = default_kg_to_kcal[
                                                 'white meat'] * diet_df_default['white meat'].values[0] / 365
        # kcal per kg 'vegetables': 200 https://www.fatsecret.co.in/calories-nutrition/generic/raw-vegetable?portionid=54903&portionamount=100.000&frc=True#:~:text=Nutritional%20Summary%3A&text=There%20are%2020%20calories%20in,%25%20carbs%2C%2016%25%20prot.
        vegetables_and_carbs_average_ca_daily_intake = diet_df_default['fruits and vegetables'].values[0] / 365 * \
                                                       default_kg_to_kcal['fruits and vegetables'] + \
                                                       diet_df_default['cereals'].values[0] / 365 * default_kg_to_kcal[
                                                           'cereals'] + \
                                                       diet_df_default['rice and maize'].values[0] / 365 * \
                                                       default_kg_to_kcal['rice and maize']
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

        self.energy_carbon_emissions = pd.DataFrame(
            {GlossaryCore.Years: years, 'biomass_dry': - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, 'electricity': 0.0,
             'methane': 0.123 / 15.4, 'syngas': 0.0, 'hydrogen.gaseous_hydrogen': 0.0, 'crude oil': 0.02533})

        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface})

        forest_invest = np.linspace(5, 8, year_range)

        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        if 'CropEnergy' in self.techno_list:
            crop_invest = np.linspace(0.5, 0.25, year_range)
        else:
            crop_invest = [0] * year_range
        if 'ManagedWood' in self.techno_list:
            mw_invest = np.linspace(1, 4, year_range)
        else:
            mw_invest = [0] * year_range

        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.crop_investment = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: crop_invest})
        # deforest_invest = np.linspace(10, 1, year_range)
        deforest_invest = [114.548759121872, 117.184116493791, 119.627727466938, 121.881049634694, 123.945540590445,
                           125.82265792757, 127.513859239454, 129.02060211948, 130.34434416103, 131.486542957486,
                           132.448656102233, 133.232141188651, 133.838455810126, 134.269057560038, 134.525404031772,
                           134.60895281871, 134.521161514233, 134.264348221602, 133.844273083575, 133.267556752788,
                           132.540819881875, 131.670683123473, 130.663767130216, 129.52669255474, 128.266080049678,
                           126.888550267669, 125.400723861344, 123.809221483342, 122.120663786294, 120.341671422839,
                           118.47886504561, 116.538865307244, 114.528292860374, 112.453964415913, 110.323480917878,
                           108.144639368564, 105.925236770266, 103.673070125275, 101.395936435889, 99.1016327044,
                           96.7979559331032, 94.492703124292, 92.1936712802624, 89.9086574033064, 87.6454584957192,
                           85.411871559796, 83.2156935978296, 81.0647216121152, 78.9667526049464, 76.9282359492362,
                           74.9502305003717, 73.0324474843579, 71.1745981272006, 69.3763936549051, 67.6375452934767,
                           65.957764268921, 64.3367618072431, 62.7742491344488, 61.2699374765432, 59.8235380595319,
                           58.4347621094202, 57.1033208522136, 55.8289255139174, 54.6112873205371, 53.4501174980782,
                           52.3443211421342, 51.2895788266526, 50.2807649951686, 49.3127540912181, 48.3804205583364,
                           47.4786388400591, 46.6022833799217, 45.7462286214598, 44.905349008209, 44.0745189837049,
                           43.2486129914827, 42.4225054750783, 41.591070878027, 40.7491836438646, 39.8917182161264,
                           39.013549038348]

        deforest_invest_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.InvestmentsValue: np.linspace(114, 39, len(self.years))})

        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01, 34.05, 39.08, 44.69, 50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        self.co2_taxes = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2Tax: func(years)})

        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years)
        })

        values_dict = {
            f'{self.study_name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.study_name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.study_name}.{energy_name}.{GlossaryCore.techno_list}': self.model_list,
            f'{self.study_name}.margin': self.margin,
            f'{self.study_name}.transport_cost': self.transport,
            f'{self.study_name}.transport_margin': self.margin,
            f'{self.study_name}.{GlossaryCore.CO2TaxesValue}': self.co2_taxes,
            f'{self.study_name}.{energy_name}.Crop.diet_df': diet_df_default,
            f'{self.study_name}.{energy_name}.Crop.red_meat_calories_per_day': self.red_meat_ca_per_day,
            f'{self.study_name}.{energy_name}.Crop.white_meat_calories_per_day': self.white_meat_ca_per_day,
            f'{self.study_name}.{energy_name}.Crop.vegetables_and_carbs_calories_per_day': self.vegetables_and_carbs_calories_per_day,
            f'{self.study_name}.{energy_name}.Crop.{GlossaryCore.FishDailyCal}': self.fish_ca_per_day,
            f'{self.study_name}.{energy_name}.Crop.{GlossaryCore.OtherDailyCal}': self.other_ca_per_day,
            f'{self.study_name}.{energy_name}.Crop.milk_and_eggs_calories_per_day': self.milk_and_eggs_calories_per_day,
            f'{self.study_name}.{energy_name}.Crop.crop_investment': self.crop_investment,
            f'{self.study_name + self.additional_ns}.forest_investment': self.forest_invest_df,
            f'{self.study_name}.{energy_name}.Forest.managed_wood_investment': self.mw_invest_df,
            f'{self.study_name}.{energy_name}.Forest.deforestation_investment': deforest_invest_df,
            f'{self.study_name}.{energy_name}.Forest.techno_capital': techno_capital,
            f'{self.study_name}.{energy_name}.Crop.techno_capital': techno_capital,
            f'{self.study_name}.{GlossaryCore.PopulationDfValue}': population_df,
            f'{self.study_name}.{GlossaryCore.TemperatureDfValue}': temperature_df
        }

        red_meat_percentage_ctrl = np.linspace(600, 900, self.nb_poles)
        white_meat_percentage_ctrl = np.linspace(700, 900, self.nb_poles)
        vegetables_and_carbs_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        milk_and_eggs_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        fish_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)
        other_calories_per_day_ctrl = np.linspace(900, 900, self.nb_poles)

        deforestation_investment_ctrl = np.linspace(10.0, 5.0, self.nb_poles)
        forest_investment_array_mix = np.linspace(5.0, 8.0, self.nb_poles)
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
        design_space_ctrl_dict['forest_investment_array_mix'] = forest_investment_array_mix

        if 'CropEnergy' in self.techno_list:
            design_space_ctrl_dict['crop_investment_array_mix'] = crop_investment_array_mix
        if 'ManagedWood' in self.techno_list:
            design_space_ctrl_dict['managed_wood_investment_array_mix'] = managed_wood_investment_array_mix

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
        # -----------------------------------------
        # Crop related
        update_dspace_dict_with(ddict, 'red_meat_calories_per_day_ctrl',
                                np.asarray(self.design_space_ctrl['red_meat_calories_per_day_ctrl']),
                                [1.0] * self.nb_poles, [1000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        update_dspace_dict_with(ddict, 'white_meat_calories_per_day_ctrl',
                                np.asarray(self.design_space_ctrl['white_meat_calories_per_day_ctrl']),
                                [5.0] * self.nb_poles, [2000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        update_dspace_dict_with(ddict, 'vegetables_and_carbs_calories_per_day_ctrl',
                                np.asarray(self.design_space_ctrl['vegetables_and_carbs_calories_per_day_ctrl']),
                                [5.0] * self.nb_poles, [2000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        update_dspace_dict_with(ddict, 'milk_and_eggs_calories_per_day_ctrl',
                                np.asarray(self.design_space_ctrl['milk_and_eggs_calories_per_day_ctrl']),
                                [5.0] * self.nb_poles, [2000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)

        update_dspace_dict_with(ddict, 'deforestation_investment_ctrl',
                                self.design_space_ctrl['deforestation_investment_ctrl'].values,
                                [0.0] * self.nb_poles, [100.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        # -----------------------------------------
        # Invests
        update_dspace_dict_with(ddict, 'forest_investment_array_mix',
                                self.design_space_ctrl['forest_investment_array_mix'].values,
                                [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles,
                                activated_elem=[True] * self.nb_poles)
        if 'CropEnergy' in self.techno_list:
            update_dspace_dict_with(ddict, 'crop_investment_array_mix',
                                    self.design_space_ctrl['crop_investment_array_mix'].values,
                                    [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles,
                                    activated_elem=[True] * self.nb_poles, enable_variable=False, )
        if 'ManagedWood' in self.techno_list:
            update_dspace_dict_with(ddict, 'managed_wood_investment_array_mix',
                                    self.design_space_ctrl['managed_wood_investment_array_mix'].values,
                                    [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles,
                                    activated_elem=[True] * self.nb_poles, enable_variable=False)

        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
    """
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        for graph in graph_list:
            graph.to_plotly().show()

"""
