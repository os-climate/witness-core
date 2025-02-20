'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2024/06/24 Copyright 2023 Capgemini

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
from copy import deepcopy

import numpy as np
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min

from climateeconomics.glossarycore import GlossaryCore


class OrderOfMagnitude():
    KILO = 'k'
    MEGA = 'M'
    GIGA = 'G'
    TERA = 'T'

    magnitude_factor = {
        KILO: 10 ** 3,
        MEGA: 10 ** 6,
        GIGA: 10 ** 9,
        TERA: 10 ** 12
    }


class Crop():
    """
    Crop model class 
    """

    KM_2_unit = 'km2'
    HECTARE = 'ha'
    GHa_unit = 'Gha'

    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    POPULATION_DF = GlossaryCore.PopulationDfValue
    DIET_DF = 'diet_df'
    KG_TO_KCAL_DICT = 'kg_to_kcal_dict'
    KG_TO_M2_DICT = 'kg_to_m2_dict'
    FOOD_LAND_SURFACE_DF = 'food_land_surface_df'
    CROP_INVESTMENT = 'crop_investment'

    min_value_invest = 1.e-12

    def __init__(self, param):
        '''
        Constructor
        '''
        self.produced_kcal = None
        self.organic_waste_df = None
        self.consumed_calories_pc_breakdown_per_day_df = None
        self.techno_production = None
        self.year_start = None
        self.year_end = None
        self.years = None
        self.diet_df = None
        self.kcal_diet_df = None
        self.kg_to_kcal_dict = None
        self.kg_to_m2_dict = None
        self.param_a = None
        self.param_b = None
        self.crop_investment = None
        self.transport_cost = None
        self.transport_margin = None
        self.data_fuel_dict = None
        self.techno_infos_dict = None
        self.scaling_factor_techno_consumption = None
        self.scaling_factor_techno_production = None
        self.initial_age_distrib = None
        self.initial_production = None
        self.nb_years_amort_capex = None
        self.margin = None
        self.construction_delay = None
        self.food_land_surface_df = None
        self.total_food_land_surface = None
        self.mix_detailed_production = None
        self.mix_detailed_prices = None
        self.production = None
        self.land_use_required = None
        self.cost_details = None
        self.techno_prices = None
        self.column_dict = None
        self.techno_consumption = None
        self.techno_consumption_woratio = None
        self.CO2_emissions = None
        self.CO2_land_emissions = None
        self.CO2_land_emissions_detailed = None
        self.CH4_land_emissions = None
        self.CH4_land_emissions_detailed = None
        self.N2O_land_emissions = None
        self.N2O_land_emissions_detailed = None
        self.updated_diet_df = None
        self.calories_pc_df = None
        self.margin = None
        self.red_meat_calories_per_day = None
        self.white_meat_calories_per_day = None
        self.vegetables_and_carbs_calories_per_day = None
        self.milk_and_eggs_calories_per_day = None
        self.fish_calories_per_day = None
        self.other_calories_per_day = None
        self.constaint_calories_limit = None
        self.constraint_calories_ref = None
        self.crop_investment = None
        self.population_df = None
        self.temperature_df = None
        self.co2_emissions_per_kg = None
        self.ch4_emissions_per_kg = None
        self.n2o_emissions_per_kg = None
        self.updated_diet_df = None
        self.food_surface_df_without_climate_change = None
        self.food_land_surface_df = None
        self.food_land_surface_percentage_df = None
        self.residue_prod_from_food_surface = None
        self.diet_init_kcal = None
        self.prod_reduction = None
        self.productivity_evolution = None
        self.crf = None
        self.nb_years_amort_capex = None
        self.production = None
        self.age_distrib_prod_df = None
        self.calories_per_day_constraint = None
        self.food_waste_percentage_df = None
        self.param = param
        self.world_surface_data = None
        self.ha_to_m2 = 0.0001
        self.m2toha = 10000.
        self.surface_df = None
        self.scaling_factor_crop_investment = None
        self.product_energy_unit = 'TWh'
        self.mass_unit = 'Mt'
        self.set_data()
        self.create_dataframe()
        self.lifetime: int = 20

    def set_data(self):
        self.year_start = self.param[Crop.YEAR_START]
        self.year_end = self.param[Crop.YEAR_END]
        years = np.arange(
            self.year_start,
            self.year_end + 1)
        self.years = years
        self.diet_df = self.param[Crop.DIET_DF]
        self.kcal_diet_df = {}
        self.kg_to_kcal_dict = self.param[Crop.KG_TO_KCAL_DICT]
        self.kg_to_m2_dict = self.param[Crop.KG_TO_M2_DICT]
        self.param_a = self.param['param_a']
        self.param_b = self.param['param_b']
        self.crop_investment = self.param['crop_investment']
        self.transport_cost = self.param['transport_cost']
        self.transport_margin = self.param['transport_margin']
        self.data_fuel_dict = self.param['data_fuel_dict']
        self.techno_infos_dict = self.param['techno_infos_dict']
        self.scaling_factor_crop_investment = self.param['scaling_factor_crop_investment']
        self.scaling_factor_techno_consumption = self.param['scaling_factor_techno_consumption']
        self.scaling_factor_techno_production = self.param['scaling_factor_techno_production']
        self.initial_age_distrib = self.param['initial_age_distrib']
        self.initial_production = self.param['initial_production']
        self.lifetime = self.param[GlossaryCore.LifetimeName]

        self.nb_years_amort_capex = 10
        self.construction_delay = 3  # default value
        self.margin = self.param['margin']
        if GlossaryCore.ConstructionDelay in self.techno_infos_dict:
            self.construction_delay = self.inputs['techno_infos_dict'][GlossaryCore.ConstructionDelay]
        else:
            print(
                'The construction_delay data is not set for Crop : default = 3 years  ')

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''

        self.food_land_surface_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.total_food_land_surface = pd.DataFrame({GlossaryCore.Years: self.years})
        self.mix_detailed_production = pd.DataFrame({GlossaryCore.Years: self.years})
        self.mix_detailed_prices = pd.DataFrame({GlossaryCore.Years: self.years})
        self.production = pd.DataFrame({GlossaryCore.Years: self.years})
        self.land_use_required = pd.DataFrame({GlossaryCore.Years: self.years})
        self.cost_details = pd.DataFrame({GlossaryCore.Years: self.years})
        self.techno_prices = pd.DataFrame({GlossaryCore.Years: self.years})
        self.column_dict = {'red meat (Gha)': GlossaryCore.RedMeat, 'white meat (Gha)': GlossaryCore.WhiteMeat,
                            'milk (Gha)': GlossaryCore.Milk, 'eggs (Gha)': GlossaryCore.Eggs, 'rice and maize (Gha)': GlossaryCore.RiceAndMaize,
                            'cereals (Gha)': GlossaryCore.Cereals, 'fruits and vegetables (Gha)': GlossaryCore.FruitsAndVegetables,
                            'fish (Gha)': GlossaryCore.Fish, 'other (Gha)': GlossaryCore.OtherFood,
                            'total surface (Gha)': 'total surface'}
        self.techno_consumption = pd.DataFrame({GlossaryCore.Years: self.years})
        self.techno_consumption_woratio = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CO2_emissions = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CO2_land_emissions = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CO2_land_emissions_detailed = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CH4_land_emissions = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CH4_land_emissions_detailed = pd.DataFrame({GlossaryCore.Years: self.years})
        self.N2O_land_emissions = pd.DataFrame({GlossaryCore.Years: self.years})
        self.N2O_land_emissions_detailed = pd.DataFrame({GlossaryCore.Years: self.years})
        self.updated_diet_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.calories_pc_df = pd.DataFrame({GlossaryCore.Years: self.years})

    def configure_parameters_update(self, inputs_dict):
        '''
        Configure coupling variables with inputs_dict from the discipline
        '''
        if inputs_dict['margin'] is not None:
            self.margin = inputs_dict['margin'].loc[inputs_dict['margin'][GlossaryCore.Years]
                                                    <= self.year_end]
        # diet design variables

        self.red_meat_calories_per_day = inputs_dict['red_meat_calories_per_day']['red_meat_calories_per_day'].values
        self.white_meat_calories_per_day = inputs_dict['white_meat_calories_per_day'][
            'white_meat_calories_per_day'].values
        self.vegetables_and_carbs_calories_per_day = inputs_dict['vegetables_and_carbs_calories_per_day'][
            'vegetables_and_carbs_calories_per_day'].values
        self.milk_and_eggs_calories_per_day = inputs_dict['milk_and_eggs_calories_per_day'][
            'milk_and_eggs_calories_per_day'].values
        self.fish_calories_per_day = inputs_dict[GlossaryCore.FishDailyCal][GlossaryCore.FishDailyCal].values
        self.other_calories_per_day = inputs_dict[GlossaryCore.OtherDailyCal][GlossaryCore.OtherDailyCal].values
        self.constaint_calories_limit = inputs_dict['constraint_calories_limit']
        self.constraint_calories_ref = inputs_dict['constraint_calories_ref']
        # crop_investment from G$ to M$
        self.crop_investment = inputs_dict[Crop.CROP_INVESTMENT]
        self.scaling_factor_crop_investment = inputs_dict['scaling_factor_crop_investment']
        self.crop_investment = deepcopy(inputs_dict['crop_investment'])
        self.crop_investment[GlossaryCore.InvestmentsValue] = self.crop_investment[GlossaryCore.InvestmentsValue] * \
                                                              self.scaling_factor_crop_investment
        if self.initial_age_distrib['distrib'].sum() > 100.001 or self.initial_age_distrib[
            'distrib'].sum() < 99.999:
            sum_distrib = self.initial_age_distrib['distrib'].sum()
            raise Exception(
                f'The distribution sum is not equal to 100 % : {sum_distrib}')
        self.population_df = inputs_dict[GlossaryCore.PopulationDfValue]
        self.temperature_df = inputs_dict[GlossaryCore.TemperatureDfValue].set_index(GlossaryCore.Years)
        self.co2_emissions_per_kg = inputs_dict['co2_emissions_per_kg']
        self.ch4_emissions_per_kg = inputs_dict['ch4_emissions_per_kg']
        self.n2o_emissions_per_kg = inputs_dict['n2o_emissions_per_kg']
        self.food_waste_percentage_df = inputs_dict[GlossaryCore.FoodWastePercentageValue]
        self.lifetime = inputs_dict[GlossaryEnergy.LifetimeName]

    def compute(self):
        ''' 
        Computation methods
        Compute the different output : updated diet, surface used (Gha), surface used (%)

        @param population_df: population from input
        @type population_df: dataframe
        @unit population_df: million of people

        @param food_land_surface_df:  Gives the surface used by each food type + the total surface used
        @type food_land_surface_df: dataframe
        @unit food_land_surface_df: Gha

        @param land_surface_percentage_df:  Gives the share of surface used by each food type
        @type land_surface_percentage_df: dataframe
        @unit land_surface_percentage_df: %

        @param updated_diet_df:  Gives the updatred diet, ie, the quantity of food for each year and each food type
        @type updated_diet_df: dataframe
        @unit updated_diet_df: kg/person/year

        '''

        # construct the diet over time
        self.compute_updated_diet()
        self.compute_consumed_calories_per_capita_breakdown()
        self.compute_calories_per_capita()
        self.compute_organic_waste()
        self.compute_calories_per_day_constraint()

        self.compute_surface_usage_by_food_type_wo_climate_impact()
        self.compute_surface_usage_by_food_type()
        self.compute_total_food_land_surface()

        self.compute_price()
        # compute prod from invests
        self.compute_primary_energy_production()
        # production of residue is the production from food surface and from
        # crop energy
        self.compute_residue_from_food()
        self.compute_mix_detailed_production()

        # compute crop for energy land use
        self.compute_crop_for_energy_land_use()

        # CO2 emissions
        self.compute_carbon_emissions()
        self.compute_land_emissions()

        # consumption
        self.compute_techno_consumption()
        self.rescale_production_and_consumption()

    def compute_quantity_of_food(self):
        """
        Compute the quantity of each food of the diet eaten each year

        @param population_df: input, give the population of each year
        @type population_df: dataframe
        @unit population_df: millions of people

        @param diet_df: input, contain the amount of food consumed each year, for each food considered
        @type diet_df: dataframe
        @unit diet_df: kg / person / year

        @param result: amount of food consumed by the global population, eahc year, for each food considered
        @type result: dataframe
        @unit result: kg / year
        """
        result = pd.DataFrame()
        for key in self.updated_diet_df.keys():
            if key == GlossaryCore.Years:
                result[key] = self.updated_diet_df[key]
            else:
                result[key] = self.population_df[GlossaryCore.PopulationValue].values * self.updated_diet_df[
                    key].values * 1e6
        # as population is in million of habitants, *1e6 is needed
        return result

    def compute_surface_usage_by_food_type_wo_climate_impact(self):
        """
        Compute the surface needed to produce a certain amount of food

        @param quantity_of_food_df: amount of food consumed by the global population, each year, for each food considered
        @type quantity_of_food_df: dataframe
        @unit quantity_of_food_df: kg / year

        @param kg_food_to_surface: input, the surface needed to produce 1kg of the considered food
        @type kg_food_to_surface: dict
        @unit kg_food_to_surface: m^2 / kg

        @param population_df: input, give the population of each year
        @type population_df: dataframe
        @unit population_df: millions of people

        @param result: the surface needed to produce the food quantity in input
        @type result: dataframe
        @unit result: m^2
        """
        quantity_of_food_df = self.compute_quantity_of_food()

        result = pd.DataFrame()
        to_sum = []
        for key in quantity_of_food_df.keys():
            if key != GlossaryCore.Years:
                result[key + ' (Gha)'] = self.kg_to_m2_dict[key] * \
                                         quantity_of_food_df[key]
                to_sum.append(result[key + ' (Gha)'])

        total_surface = np.sum(to_sum, axis=0)
        result['total surface (Gha)'] = total_surface

        # put data in [Gha]
        result = result * self.ha_to_m2 / 1e9
        result.insert(0, GlossaryCore.Years, quantity_of_food_df[GlossaryCore.Years])
        self.food_surface_df_without_climate_change = result

    def compute_updated_diet(self):
        '''
            update diet data:
                - compute new kg/person/year from red and white meat
                - update proportionally all vegetable kg/person/year
                - compute new diet_df
        '''
        changed_diet_df = pd.DataFrame({GlossaryCore.Years: self.years})
        changed_diet_df[GlossaryCore.RedMeat] = self.red_meat_calories_per_day * 365
        changed_diet_df[GlossaryCore.WhiteMeat] = self.white_meat_calories_per_day * 365
        changed_diet_df[GlossaryCore.Fish] = self.fish_calories_per_day * 365
        changed_diet_df[GlossaryCore.OtherFood] = self.other_calories_per_day * 365
        # design var is milk + eggs => need to recompute milk and egg separately
        milk_and_eggs_calories = self.milk_and_eggs_calories_per_day * 365
        # same for fruits, cereals and rice that are summed into design var vegetables_and_carbs_calories_per_day
        vegetables_and_carbs_calories = self.vegetables_and_carbs_calories_per_day * 365

        # Compute initial diet in kcal/day/person (diet df in kg/year/person)
        diet_init_kcal = self.diet_df.copy()
        diet_init_kcal = diet_init_kcal.mul(pd.Series(self.kg_to_kcal_dict) / 365, axis=1)
        self.diet_init_kcal = diet_init_kcal

        list_variables = [GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize, GlossaryCore.Eggs, GlossaryCore.Milk]

        for key in list_variables:
            # compute new vegetable diet in kg_food/person/year: add the removed_kcal/3 for each 3 category of vegetable
            if key == GlossaryCore.FruitsAndVegetables or key == GlossaryCore.Cereals or key == GlossaryCore.RiceAndMaize:
                # proportion of current key wrt to other
                proportion = diet_init_kcal[key].values[0] / \
                             (diet_init_kcal[GlossaryCore.FruitsAndVegetables].values[0] + diet_init_kcal[GlossaryCore.Cereals].values[0] +
                              diet_init_kcal[GlossaryCore.RiceAndMaize].values[0])
                changed_diet_df[key] = vegetables_and_carbs_calories * proportion
            # eggs and milk have fixed value for now
            elif key == GlossaryCore.Eggs or key == GlossaryCore.Milk:
                proportion = diet_init_kcal[key].values[0] / \
                             (diet_init_kcal[GlossaryCore.Eggs].values[0] + diet_init_kcal[GlossaryCore.Milk].values[0])

                changed_diet_df[key] = milk_and_eggs_calories * proportion

        for col in changed_diet_df:
            if col != GlossaryCore.Years:
                changed_diet_df[col] = changed_diet_df[col] / self.kg_to_kcal_dict[col]

        self.updated_diet_df = changed_diet_df

    def convert_surface_to_percentage(self, surface_df):
        """
        Express the surface taken by each food type in % and not in Gha

        @param surface_df: dataframe with the surface of each type of food expressed in Gha, and a columun with the total surface taken in Gha.
        @type surface_df: dataframe
        @unit surface_df: Gha

        @param land_surface_percentage_df: output of the method, with the share of surface taken by each type of food, in %
        @type land_surface_percentage_df: dataframe
        @unit land_surface_percentage_df: %
        """
        land_surface_percentage_df = deepcopy(surface_df)
        for key in land_surface_percentage_df.keys():
            if key != GlossaryCore.Years:
                land_surface_percentage_df[key] = land_surface_percentage_df[key] / \
                                                  land_surface_percentage_df['total surface (Gha)'].values * 100

        land_surface_percentage_df.rename(
            columns=self.column_dict, inplace=True)
        return (land_surface_percentage_df)

    def compute_surface_usage_by_food_type(self):
        """ Add productivity reduction due to temperature increase and compute the new required surface
        Inputs: - surface_df_before: dataframe, Gha ,land required for food production without climate change
                - parameters of productivity function
                - temperature_df: dataframe, degree celsius wrt preindustrial level, dataframe of temperature increase
        """

        temperature = self.temperature_df[GlossaryCore.TempAtmo].values
        # Compute the difference in temperature wrt 2020 reference
        temp = temperature - self.temperature_df.at[self.year_start, GlossaryCore.TempAtmo]
        # Compute reduction in productivity due to increase in temperature 
        pdctivity_reduction = self.param_a * temp ** 2 + self.param_b * temp
        self.prod_reduction = pdctivity_reduction
        self.productivity_evolution = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'productivity_evolution': pdctivity_reduction})
        # Apply this reduction to increase land surface needed
        non_years_columns = self.food_surface_df_without_climate_change.columns[1:]
        surface_usage_by_food_type_with_climate_impact = self.food_surface_df_without_climate_change[
            non_years_columns].multiply(other=(1 - pdctivity_reduction), axis=0)
        surface_usage_by_food_type_with_climate_impact.insert(loc=0, column=GlossaryCore.Years, value=self.years)
        self.food_land_surface_df = surface_usage_by_food_type_with_climate_impact

    def compute_price(self):
        """
        Compute the cost details for crop & price
        """
        # Gather invests in crop for energy input
        invest_inputs = self.crop_investment.loc[self.crop_investment[GlossaryCore.Years]
                                                 <= self.cost_details[GlossaryCore.Years].max()][
            GlossaryCore.InvestmentsValue].values

        # Maximize with smooth exponential
        # this investment is not used in the price computation
        self.cost_details[GlossaryCore.InvestmentsValue] = compute_func_with_exp_min(
            invest_inputs, self.min_value_invest)

        # CAPEX 
        capex_init = self.check_capex_unity(self.techno_infos_dict)
        self.cost_details['Capex ($/MWh)'] = capex_init * np.ones(len(self.cost_details[GlossaryCore.InvestmentsValue]))

        # CRF
        self.crf = self.compute_crf(self.techno_infos_dict)

        # Energy costs
        self.cost_details['Energy costs ($/MWh)'] = self.compute_other_primary_energy_costs()

        # Factory cost including CAPEX OPEX
        self.cost_details['Factory ($/MWh)'] = self.cost_details['Capex ($/MWh)'] * (
                self.crf + self.inputs['techno_infos_dict']['Opex_percentage'])

        if 'nb_years_amort_capex' in self.techno_infos_dict:
            self.nb_years_amort_capex = self.inputs['techno_infos_dict']['nb_years_amort_capex']

        # pylint: disable=no-member
        len_y = max(self.cost_details[GlossaryCore.Years]) + \
                1 - min(self.cost_details[GlossaryCore.Years])
        self.cost_details['Crop_factory_amort'] = (
                np.tril(np.triu(np.ones((len_y, len_y)), k=0), k=self.nb_years_amort_capex - 1).transpose() *
                np.array(self.cost_details['Factory ($/MWh)'].values / self.nb_years_amort_capex)).T.sum(axis=0)
        # pylint: enable=no-member

        # Compute and add transport
        self.cost_details['Transport ($/MWh)'] = self.inputs[f'{GlossaryEnergy.TransportCostValue}:transport'] * \
                                                 self.transport_margin['margin'].values / 100.0 / self.data_fuel_dict[
                                                     'calorific_value']

        # Crop amort
        self.cost_details['Crop amort ($/MWh)'] = self.cost_details['Crop_factory_amort'].values + self.cost_details[
            'Transport ($/MWh)'].values + self.cost_details['Energy costs ($/MWh)'].values

        # Total cost (MWh)
        self.cost_details['Total ($/MWh)'] = self.cost_details['Energy costs ($/MWh)'].values + \
                                             self.cost_details['Factory ($/MWh)'].values\
                                             + self.cost_details['Transport ($/MWh)'].values

        # Add margin in %
        self.cost_details['Total ($/MWh)'] *= self.margin.loc[self.margin[GlossaryCore.Years]
                                                              <= self.cost_details[GlossaryCore.Years].max()][
                                                  'margin'].values / 100.0
        self.cost_details['Crop amort ($/MWh)'] *= self.margin.loc[self.margin[GlossaryCore.Years]
                                                                   <= self.cost_details[GlossaryCore.Years].max()][
                                                       'margin'].values / 100.0
        # Total cost (t)
        self.cost_details['Total ($/t)'] = self.cost_details['Total ($/MWh)'].values * self.data_fuel_dict['calorific_value']
        self.techno_prices['Crop'] = self.cost_details['Total ($/MWh)'].values

        if 'CO2_taxes_factory' in self.cost_details:
            self.techno_prices['Crop_wotaxes'] = self.cost_details['Total ($/MWh)'].values - \
                                                 self.cost_details['CO2_taxes_factory'].values
        else:
            self.techno_prices['Crop_wotaxes'] = self.cost_details['Total ($/MWh)'].values

        price_crop = self.cost_details['Total ($/t)'].values / \
                     (1 + self.inputs['techno_infos_dict']['residue_density_percentage'] *
                      (self.inputs['techno_infos_dict']['crop_residue_price_percent_dif'] - 1))

        price_residue = price_crop * \
                        self.inputs['techno_infos_dict']['crop_residue_price_percent_dif']

        # Price_residue = crop_residue_ratio * Price_crop
        # Price_crop = Price_tot / ((1-ratio_prices)*crop_residue_ratio + ratio_prices)
        self.mix_detailed_prices['Price crop ($/t)'] = price_crop
        self.mix_detailed_prices['Price residue ($/t)'] = price_residue

    def check_capex_unity(self, data_tocheck):
        """
        Put all capex in $/MWh
        """
        capex_init = None  # initialize capex to None
        if data_tocheck['Capex_init_unit'] == 'euro/ha':

            density_per_ha = data_tocheck['density_per_ha']

            if data_tocheck['density_per_ha_unit'] == 'm^3/ha':
                capex_init = data_tocheck['Capex_init'] * \
                             data_tocheck['euro_dollar'] / \
                             density_per_ha / \
                             data_tocheck['density'] / \
                             self.data_fuel_dict['calorific_value']

            elif data_tocheck['density_per_ha_unit'] == 'kg/ha':
                capex_init = data_tocheck['Capex_init'] * \
                             data_tocheck['euro_dollar'] / \
                             density_per_ha / \
                             self.data_fuel_dict['calorific_value']
        else:
            capex_unit = data_tocheck['Capex_init_unit']
            raise Exception(
                f'The CAPEX unity {capex_unit} is not handled yet in techno_type')
        # Ensure capex_init was set
        if capex_init is None:
            raise Exception("capex_init was not set properly.")
        # return capex in $/MWh
        return capex_init * 1.0e3

    def compute_other_primary_energy_costs(self):
        """
        Compute primary costs to produce 1 MWh of crop
        """

        return 0.0

    def compute_crf(self, data_config):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        crf = (data_config['WACC'] * (1.0 + data_config['WACC']) ** self.lifetime) / \
              ((1.0 + data_config['WACC']) ** self.lifetime - 1.0)

        return crf

    def compute_primary_energy_production(self):
        '''
        Compute biomass_dry production
        '''
        # Compute the aging distribution over the years of study to determine the total production over the years
        # This function also erase old factories from the distribution
        self.compute_aging_distribution_production()

        # Finally compute the production by summing all aged production for
        # each year
        # Grouping the aging distribution production by year and summing up the production
        age_distrib_prod_sum = self.age_distrib_prod_df.groupby([GlossaryCore.Years], as_index=False).agg(
            {'distrib_prod (TWh)': 'sum'}
        )
        # Delete the 'biomass_dry (TWh)' column if it already exists in the self.production dataframe
        if 'biomass_dry (TWh)' in self.production:
            del self.production['biomass_dry (TWh)']
        # Merging production data with aged distribution data
        self.production = pd.merge(self.production, age_distrib_prod_sum, how='left', on=GlossaryCore.Years).rename(
            columns={'distrib_prod (TWh)': 'biomass_dry (TWh)'}).fillna(0.0)

    def compute_aging_distribution_production(self):
        '''
        Compute the aging distribution production of primary energy for years of study
        Start with the initial distribution and add a year on the age each year 
        Add also the yearly production regarding the investment
        All productions older than the lifetime are removed from the dataframe  
        '''
        # To break the object link with initial distrib
        # Creating a DataFrame to separate the initial distribution of production from the rest

        aging_distrib_year_df = pd.DataFrame(
            {'age': self.initial_age_distrib['age'].values})
        aging_distrib_year_df['distrib_prod (TWh)'] = self.initial_age_distrib['distrib'] * \
                                                       self.initial_production / 100.0
        # Calculating yearly production based on investment
        production_from_invest = self.compute_prod_from_invest(
            construction_delay=self.construction_delay)

        # get the whole dataframe for new production with one line for each
        # year at each age
        # concatenate tuple to get correct df to mimic the old year loop
        # Creating DataFrame for new production with one line for each year at each age
        len_years = len(self.years)
        range_years = np.arange(
            self.year_start, self.year_end + len_years)

        year_array = np.concatenate(
            tuple(range_years[i:i + len_years] for i in range(len_years)))
        age_array = np.concatenate(tuple(np.ones(
            len_years) * (len_years - i) for i in range(len_years, 0, -1)))
        prod_array = production_from_invest['prod_from_invest'].values.tolist(
        ) * len_years

        new_prod_aged = pd.DataFrame(
            {GlossaryCore.Years: year_array, 'age': age_array, 'distrib_prod (TWh)': prod_array})

        # Creating DataFrame for old production with one line for each year at each age
        year_array = np.array([[year] * len(aging_distrib_year_df)
                               for year in self.years]).flatten()
        age_values = aging_distrib_year_df['age'].values
        age_array = np.concatenate(tuple(
            age_values + i for i in range(len_years)))
        prod_array = aging_distrib_year_df['distrib_prod (TWh)'].values.tolist(
        ) * len_years

        old_prod_aged = pd.DataFrame({GlossaryCore.Years: year_array, 'age': age_array,
                                      'distrib_prod (TWh)': prod_array})

        # Concatenating the two created DataFrames
        self.age_distrib_prod_df = pd.concat(
            [new_prod_aged, old_prod_aged], ignore_index=True)
        # Removing all lines where age is higher than lifetime and all-zero productions
        self.age_distrib_prod_df = self.age_distrib_prod_df.loc[
            # Suppress all lines where age is higher than lifetime
            (self.age_distrib_prod_df['age'] <
             self.lifetime)
            # delete years after year end
            & (self.age_distrib_prod_df[GlossaryCore.Years] < self.year_end + 1)
            # Fill Nan with zeros and suppress all zeros
            & (self.age_distrib_prod_df['distrib_prod (TWh)'] != 0.0)
            ]
        # Fill Nan with zeros
        self.age_distrib_prod_df.fillna(0.0, inplace=True)

    def compute_prod_from_invest(self, construction_delay):
        '''
        Compute the crop production from investment in TWh
        '''
        # Creating a DataFrame for production before the year of study start
        prod_before_ystart = pd.DataFrame(
            {GlossaryCore.Years: np.arange(self.year_start - construction_delay, self.year_start),
             GlossaryCore.InvestmentsValue: [0.0] * (construction_delay),
             'Capex ($/MWh)': self.cost_details.loc[self.cost_details[
                                                        GlossaryCore.Years] == self.year_start, 'Capex ($/MWh)'].values[
                 0]})

        # Concatenating production data with production before year of study start
        production_from_invest = pd.concat(
            [self.cost_details[[GlossaryCore.Years, GlossaryCore.InvestmentsValue, 'Capex ($/MWh)']],
             prod_before_ystart], ignore_index=True)
        # Sorting the production data by years
        production_from_invest.sort_values(by=[GlossaryCore.Years], inplace=True)
        # Calculate production from investment based on Capex and InvestmentsValue
        # # Invest in M$ | Capex in $/MWh | Prod in TWh
        production_from_invest['prod_from_invest'] = production_from_invest[GlossaryCore.InvestmentsValue].values / \
                                                     production_from_invest['Capex ($/MWh)'].values
        # Incrementing years by construction_delay (to start production at year + construction_delay)
        production_from_invest[GlossaryCore.Years] += construction_delay
        # Removing data for years beyond the study period
        production_from_invest = production_from_invest[production_from_invest[GlossaryCore.Years]
                                                        <= self.year_end]
        return production_from_invest

    def compute_residue_from_food(self):
        '''
        Compute residue part from the land surface for food.
        '''
        # compute residue part from food land surface for energy sector in TWh
        residue_production = self.total_food_land_surface['total surface (Gha)'] * \
                             self.inputs['techno_infos_dict']['residue_density_percentage'] * \
                             self.inputs['techno_infos_dict']['density_per_ha'] * \
                             self.data_fuel_dict['calorific_value'] * \
                             self.inputs['techno_infos_dict']['residue_percentage_for_energy']

        self.residue_prod_from_food_surface = residue_production

    def compute_crop_for_energy_land_use(self):
        """
        Compute land use required for crop for energy
        """
        self.land_use_required['Crop (Gha)'] = self.mix_detailed_production['Crop for Energy (TWh)'] / \
                                               self.inputs['techno_infos_dict']['density_per_ha']

    def compute_carbon_emissions(self):
        '''
        Compute the carbon emissions from the technology taking into account 
        CO2 from production + CO2 from primary resources 
        '''
        if 'CO2_from_production' not in self.techno_infos_dict:
            self.CO2_emissions['production'] = self.get_theoretical_co2_prod(
                unit='kg/kWh')
        elif self.inputs['techno_infos_dict']['CO2_from_production'] == 0.0:
            self.CO2_emissions['production'] = 0.0
        else:
            if self.inputs['techno_infos_dict']['CO2_from_production_unit'] == 'kg/kg':
                self.CO2_emissions['production'] = self.inputs['techno_infos_dict']['CO2_from_production'] / \
                                                   self.data_fuel_dict['high_calorific_value']
            elif self.inputs['techno_infos_dict']['CO2_from_production_unit'] == 'kg/kWh':
                self.CO2_emissions['production'] = self.inputs['techno_infos_dict']['CO2_from_production']

        # Add carbon emission from input energies (resources or other
        # energies)

        co2_emissions_frominput_energies = self.compute_CO2_emissions_from_input_resources(
        )

        # Add CO2 from production + C02 from input energies
        self.CO2_emissions['Crop'] = self.CO2_emissions['production'] + \
                                     co2_emissions_frominput_energies

    def compute_land_emissions(self):
        '''
        Emissions are computed for each food from land surface used by the food:
            emission_per_kg * kg_to_m2 * surface * m2toha * 10^9 * 10^-12
        For fish, this does not make sense as no land is used. However, there are some emissions from fish farming.
            emission_per_kg * kg_food_per_capita_per_year * population
        '''

        self.CO2_land_emissions['emitted_CO2_evol_cumulative'] = 0.0
        self.CH4_land_emissions['emitted_CH4_evol_cumulative'] = 0.0
        self.N2O_land_emissions['emitted_N2O_evol_cumulative'] = 0.0
        for food in self.co2_emissions_per_kg:

            # add crop energy surface for rice and maize
            if food == GlossaryCore.RiceAndMaize:
                surface = self.food_land_surface_df[f'{food} (Gha)'] + self.land_use_required['Crop (Gha)']

            else:
                surface = self.food_land_surface_df[f'{food} (Gha)']
            if self.kg_to_m2_dict[food] > 0.:
                # CO2
                self.CO2_land_emissions_detailed[f'{food} (Gt)'] = self.co2_emissions_per_kg[food] / \
                                                                   self.kg_to_m2_dict[food] * surface * \
                                                                   self.m2toha * 1e9 * 1e-12  # to m^2 and then to GtCo2
                # CH4
                self.CH4_land_emissions_detailed[f'{food} (Gt)'] = self.ch4_emissions_per_kg[food] / \
                                                                   self.kg_to_m2_dict[food] * surface * \
                                                                   self.m2toha * 1e9 * 1e-12  # to m^2 and then to GtCo2
                # N20
                self.N2O_land_emissions_detailed[f'{food} (Gt)'] = self.n2o_emissions_per_kg[food] / \
                                                                   self.kg_to_m2_dict[food] * surface * \
                                                                   self.m2toha * 1e9 * 1e-12  # to m^2 and then to GtCo2
            elif food == GlossaryCore.Fish:
                # CO2
                self.CO2_land_emissions_detailed[f'{food} (Gt)'] = self.co2_emissions_per_kg[food] * \
                                                                   self.updated_diet_df[food] * \
                                                                   self.population_df[
                                                                       GlossaryCore.PopulationValue].values * \
                                                                   1e6 / 1e12  # pop in millions, emissions in Gt
                # CH4
                self.CH4_land_emissions_detailed[f'{food} (Gt)'] = self.ch4_emissions_per_kg[food] * \
                                                                   self.updated_diet_df[food] * \
                                                                   self.population_df[
                                                                       GlossaryCore.PopulationValue].values * \
                                                                   1e6 / 1e12  # pop in millions, emissions in Gt
                # N20
                self.N2O_land_emissions_detailed[f'{food} (Gt)'] = self.n2o_emissions_per_kg[food] * \
                                                                   self.updated_diet_df[food] * \
                                                                   self.population_df[
                                                                       GlossaryCore.PopulationValue].values * \
                                                                   1e6 / 1e12  # pop in millions, emissions in Gt
            else:
                self.CO2_land_emissions_detailed[f'{food} (Gt)'] = 0.
                self.CH4_land_emissions_detailed[f'{food} (Gt)'] = 0.
                self.N2O_land_emissions_detailed[f'{food} (Gt)'] = 0.

            self.CO2_land_emissions['emitted_CO2_evol_cumulative'] += self.CO2_land_emissions_detailed[f'{food} (Gt)']
            self.CH4_land_emissions['emitted_CH4_evol_cumulative'] += self.CH4_land_emissions_detailed[f'{food} (Gt)']
            self.N2O_land_emissions['emitted_N2O_evol_cumulative'] += self.N2O_land_emissions_detailed[f'{food} (Gt)']

    def get_theoretical_co2_prod(self, unit='kg/kWh'):
        ''' 
        Get the theoretical CO2 production for a given technology,
        '''
        return 0.0

    def compute_CO2_emissions_from_input_resources(self):
        '''
        Need to take into account  CO2 from electricity/fuel production
        '''
        return 0.0

    ####### Gradient #########

    def d_land_surface_d_population(self, column_name_Gha):
        """
        Compute the derivate of food_land_surface_df wrt population, for a specific column.
        derivate_step1 = diet_expression(kg/person/year) * kg_to_m2 / 1e7
        derivative = derivative_step1 * (1- productivity reduction due to temperature increase)
        diet_expression is computed in a separated method

        need self.column_dict because input column get '(Gha)' at the end
        / 1e7 comes from the unit : *1e6 (population in million) /1e4 (m2 to ha) /1e9 (ha to Gha)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        column_name = self.column_dict[column_name_Gha]
        d_land_surface_d_pop_before = np.identity(
            number_of_values) * self.updated_diet_df[column_name].values * self.kg_to_m2_dict[column_name] / 1e7
        # Add climate change impact 
        d_land_surface_d_pop = d_land_surface_d_pop_before * (1 - self.prod_reduction)

        return (d_land_surface_d_pop)

    def d_ghg_fish_emission_d_population(self):
        '''
        computing the derivative of the ghg emission (in Gt) wrt population (in million people) for the
        particular case of fish that does not depend on land surface but on population
        ghg(fish) = ghg_per_kg(fish)/ 10^12 * kg(fish) * population * 10^6

        dghg(fish)/dpop = ghg_per_kg(fish)/ 10^12 * kg(fish) * 10^6
        '''
        dghg_dpop_co2 = np.diag(self.co2_emissions_per_kg[GlossaryCore.Fish] * \
                                self.updated_diet_df[GlossaryCore.Fish] * 1e6 / 1e12)
        dghg_dpop_ch4 = np.diag(self.ch4_emissions_per_kg[GlossaryCore.Fish] * \
                                self.updated_diet_df[GlossaryCore.Fish] * 1e6 / 1e12)
        dghg_dpop_n2o = np.diag(self.n2o_emissions_per_kg[GlossaryCore.Fish] * \
                                self.updated_diet_df[GlossaryCore.Fish] * 1e6 / 1e12)

        return {GlossaryCore.CO2: dghg_dpop_co2,
                GlossaryCore.CH4: dghg_dpop_ch4,
                GlossaryCore.N2O: dghg_dpop_n2o, }

    def d_ghg_fish_emission_d_fish_kcal(self):
        '''
        computing the derivative of the ghg emission (in Gt) wrt fish kcal for the
        particular case of fish that does not depend on land surface but on kg of food
        ghg(fish) = ghg_per_kg(fish)/ 10^12 * kg(fish) * population * 10^6
        d_kcal(fish) = kcal_per_kg(fish) * d_kg(Fish)
        dghg(fish)/dkcal(fish) = 1/kcal_per_kg(fish) * dghg(fish)/dkg(fish) = ghg_per_kg(fish)/ 10^12/kcal_per_kg(fish) * population * 10^6
        '''
        dco2_dkcal_fish = np.diag(
            self.co2_emissions_per_kg[GlossaryCore.Fish] * 365 / self.kg_to_kcal_dict[GlossaryCore.Fish] * \
            self.population_df[GlossaryCore.PopulationValue].values * 1e6 / 1e12)
        dch4_dkcal_fish = np.diag(
            self.ch4_emissions_per_kg[GlossaryCore.Fish] * 365 / self.kg_to_kcal_dict[GlossaryCore.Fish] * \
            self.population_df[GlossaryCore.PopulationValue].values * 1e6 / 1e12)
        dn2o_dkcal_fish = np.diag(
            self.n2o_emissions_per_kg[GlossaryCore.Fish] * 365 / self.kg_to_kcal_dict[GlossaryCore.Fish] * \
            self.population_df[GlossaryCore.PopulationValue].values * 1e6 / 1e12)

        return {GlossaryCore.CO2: dco2_dkcal_fish,
                GlossaryCore.CH4: dch4_dkcal_fish,
                GlossaryCore.N2O: dn2o_dkcal_fish, }

    def d_food_land_surface_d_temperature(self, temperature_df, column_name):
        """
        Compute the derivative of land surface wrt temperature
        Final land surface = food_land_surface_step_one * (1 - productivity)
        productivity = f(temperature)
        d_food_land_surface_d_temperature =  d_land_d_productivity * d_productivity_d_temperature
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        temp_zero = temperature_df[GlossaryCore.TempAtmo].values[0]
        temp = temperature_df[GlossaryCore.TempAtmo].values
        a = self.param_a
        b = self.param_b
        land_before = self.food_surface_df_without_climate_change[column_name].values
        # Step 1: Productivity reduction
        # temp = temperature - temperature_df.at[self.year_start, GlossaryCore.TempAtmo]
        # pdctivity_reduction = self.param_a * temp**2 + self.param_b * temp
        # =at**2 + at0**2 - 2att0 + bt - bt0
        # Derivative wrt t each year:  2at-2at0 +b
        d_productivity_d_temperature = idty * (2 * a * temp - 2 * a * temp_zero + b)
        # Add derivative wrt t0: 2at0 -2at -b
        d_productivity_d_temperature[:, 0] += 2 * a * temp_zero - 2 * a * temp - b
        # Step 2:d_climate_d_productivity for each t: land = land_before * (1 - productivity) 
        d_land_d_productivity = -idty * land_before
        d_food_land_surface_d_temperature = d_land_d_productivity.dot(d_productivity_d_temperature)

        return d_food_land_surface_d_temperature

    def d_surface_d_calories_per_day(self, population_df, grad_value):
        """
        Generic method to compute derivative of total food land surface wrt to variables of
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)

        total_surface_grad = grad_value * population_df[GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad * idty

    def d_surface_d_vegetables_carbs_calories_per_day(self, population_df):
        """
        Generic method to compute derivative of total food land surface wrt to variables of vegetables and carbs
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        diet_df = self.diet_init_kcal

        kg_food_to_surface = self.kg_to_m2_dict

        vegetables_column_names = [GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize]
        l_years = len(self.years)
        grad_res = np.zeros((l_years, l_years))
        for veg in vegetables_column_names:
            proportion = diet_df[veg].values[0] / (diet_df[GlossaryCore.FruitsAndVegetables].values[0] +
                                                   diet_df[GlossaryCore.Cereals].values[0] +
                                                   diet_df[GlossaryCore.RiceAndMaize].values[0])

            grad_value = 365 * kg_food_to_surface[veg] * proportion / self.kg_to_kcal_dict[veg]

            total_surface_grad = grad_value * population_df[
                GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
            total_surface_climate_grad = total_surface_grad * (
                    1 - self.productivity_evolution['productivity_evolution'])
            grad_res = grad_res + total_surface_climate_grad.values * idty

        return grad_res

    def d_surface_d_eggs_milk_calories_per_day(self, population_df):
        """
        Generic method to compute derivative of total food land surface wrt to variables of eggs and milk
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        diet_df = self.diet_init_kcal

        kg_food_to_surface = self.kg_to_m2_dict

        eggs_milk_column_names = [GlossaryCore.Eggs, GlossaryCore.Milk]
        l_years = len(self.years)
        grad_res = np.zeros((l_years, l_years))
        for veg in eggs_milk_column_names:
            proportion = diet_df[veg].values[0] / (diet_df[GlossaryCore.Eggs].values[0] +
                                                   diet_df[GlossaryCore.Milk].values[0])

            grad_value = 365 * kg_food_to_surface[veg] * proportion / self.kg_to_kcal_dict[veg]

            total_surface_grad = grad_value * population_df[
                GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
            total_surface_climate_grad = total_surface_grad * (
                    1 - self.productivity_evolution['productivity_evolution'])
            grad_res = grad_res + total_surface_climate_grad.values * idty

        return grad_res

    def d_surface_d_calories(self, population_df, key_name):
        """
        Compute the derivative of total food land surface wrt red meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        # red to white meat value influences red meat, white meat, and vegetable surface
        red_meat_diet_grad = 365 * kg_food_to_surface[key_name] / self.kg_to_kcal_dict[key_name]

        total_surface_grad = red_meat_diet_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_red_meat(self, population_df):
        """
        Compute the derivative of total food land surface wrt red meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        # red to white meat value influences red meat, white meat, and vegetable surface
        red_meat_diet_grad = 365 * kg_food_to_surface[GlossaryCore.RedMeat]

        total_surface_grad = red_meat_diet_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_white_meat(self, population_df):
        """
        Compute the derivative of total food land surface wrt white meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        # red to white meat value influences red meat, white meat, and vegetable surface
        white_meat_diet_grad = 365 * kg_food_to_surface[GlossaryCore.WhiteMeat]

        sub_total_surface_grad = white_meat_diet_grad

        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_fish(self, population_df):
        """
        Compute the derivative of total food land surface wrt fish percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        fish_diet_grad = 365 * kg_food_to_surface[GlossaryCore.Fish]

        sub_total_surface_grad = fish_diet_grad

        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_other_food(self, population_df):
        """
        Compute the derivative of total food land surface wrt other food percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        other_food_diet_grad = 365 * kg_food_to_surface[GlossaryCore.OtherFood]

        sub_total_surface_grad = other_food_diet_grad

        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_other_calories_percentage(self, population_df, veg):
        """
        Compute the derivative of total food land surface wrt white meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        diet_df = self.diet_init_kcal
        # red to white meat value influences red meat, white meat, and vegetable surface

        proportion = diet_df[veg].values[0] / (diet_df[GlossaryCore.FruitsAndVegetables].values[0] +
                                               diet_df[GlossaryCore.Cereals].values[0] +
                                               diet_df[GlossaryCore.RiceAndMaize].values[0])

        white_meat_diet_grad = 365 * kg_food_to_surface[veg] * proportion / self.kg_to_kcal_dict[veg]

        # sub total gradient is the sum of all gradients of food category
        sub_total_surface_grad = white_meat_diet_grad
        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_milk_eggs_calories_percentage(self, population_df, veg):
        """
        Compute the derivative of total food land surface wrt white meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        diet_df = self.diet_init_kcal
        # red to white meat value influences red meat, white meat, and vegetable surface

        proportion = diet_df[veg].values[0] / (diet_df[GlossaryCore.Milk].values[0] +
                                               diet_df[GlossaryCore.Eggs].values[0])

        white_meat_diet_grad = 365 * kg_food_to_surface[veg] * proportion / self.kg_to_kcal_dict[veg]

        # sub total gradient is the sum of all gradients of food category
        sub_total_surface_grad = white_meat_diet_grad
        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_dprod_from_dinvest(self):
        # return dproduction_from_dinvest

        '''
               Compute the partial derivative of prod vs investment  and the partial derivative of prod vs capex
               To compute after the total derivative of prod vs investment = dpprod_dpinvest + dpprod_dpcapex*dcapexdinvest
               with dcapexdinvest already computed for detailed prices
               '''

        nb_years = (self.year_end - self.year_start + 1)
        arr_type = 'float64'
        dprod_list_dinvest_list = np.zeros(
            (nb_years, nb_years), dtype=arr_type)

        # We fill this jacobian column by column because it is the same element
        # in the entire column
        for i in range(nb_years):
            len_non_zeros = min(max(0, nb_years - self.construction_delay - i), self.lifetime)
            first_len_zeros = min(i + self.construction_delay, nb_years)
            last_len_zeros = max(0, nb_years - len_non_zeros - first_len_zeros)
            # For prod in each column there is lifetime times the same value which is dpprod_dpinvest
            # This value is delayed in time (means delayed in lines for
            # jacobian by construction _delay)
            # Each column is then composed of [0,0,0... (dp/dx,dp/dx)*lifetime,
            # 0,0,0]
            dpprod_dpinvest = 1 / self.cost_details['Capex ($/MWh)'].values[i] / \
                              self.data_fuel_dict['calorific_value']
            is_invest_negative = max(
                np.sign(self.cost_details[GlossaryCore.InvestmentsValue].values[i] + np.finfo(float).eps), 0.0)
            dprod_list_dinvest_list[:, i] = np.hstack((np.zeros(first_len_zeros),
                                                       np.ones(
                                                           len_non_zeros) * dpprod_dpinvest * is_invest_negative,
                                                       np.zeros(last_len_zeros)))

        # Mt to GWh
        return dprod_list_dinvest_list

    def compute_d_prod_dland_for_food(self, dland_for_food):
        '''
        Compute gradient of production from land surface from food
        an identity matrice with the same scalar on diagonal

        '''
        # production = residue production from food + crop energy production
        # residue production from food = compute_residue_from_food_investment
        d_prod_dland_for_food = dland_for_food * \
                                self.inputs['techno_infos_dict']['residue_density_percentage'] * \
                                self.inputs['techno_infos_dict']['density_per_ha'] * \
                                self.inputs['techno_infos_dict']['residue_percentage_for_energy'] * \
                                self.data_fuel_dict['calorific_value'] / \
                                self.scaling_factor_techno_production
        return d_prod_dland_for_food

    def compute_dland_emissions_dfood_land_surface_df(self, food):
        '''
        Compute gradient of co2 land emissions from land surface from food
        '''
        if self.kg_to_m2_dict[food] > 0.:
            co2 = self.co2_emissions_per_kg[food] / self.kg_to_m2_dict[food] * self.m2toha * 1e9 * 1e-12
            ch4 = self.ch4_emissions_per_kg[food] / self.kg_to_m2_dict[food] * self.m2toha * 1e9 * 1e-12
            n2O = self.n2o_emissions_per_kg[food] / self.kg_to_m2_dict[food] * self.m2toha * 1e9 * 1e-12
        else:
            co2 = self.co2_emissions_per_kg[food] * 0.
            ch4 = self.ch4_emissions_per_kg[food] * 0.
            n2O = self.n2o_emissions_per_kg[food] * 0.
        return {GlossaryCore.CO2: co2,
                GlossaryCore.CH4: ch4,
                GlossaryCore.N2O: n2O, }

    def compute_d_food_surface_d_red_meat_percentage(self, population_df, food):
        """
        Compute the derivative of food land surface wrt red meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        sub_total_surface_grad = 0.0
        kg_food_to_surface = self.kg_to_m2_dict

        red_meat_diet_grad = 365 * kg_food_to_surface[GlossaryCore.RedMeat]
        # removed_red_kcal = -self.kcal_diet_df['total'] / 100

        # red to white meat value influences red meat, white meat, and vegetable surface
        if food in [GlossaryCore.RedMeat, GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize]:

            if food == GlossaryCore.RedMeat:
                sub_total_surface_grad = red_meat_diet_grad / self.kg_to_kcal_dict[food]

            else:
                pass
                """proportion = self.kcal_diet_df[food] / (self.kcal_diet_df[GlossaryCore.FruitsAndVegetables] + self.kcal_diet_df[GlossaryCore.Cereals] + self.kcal_diet_df[GlossaryCore.RiceAndMaize])
                sub_total_surface_grad = removed_red_kcal * proportion / self.kg_to_kcal_dict[food] * kg_food_to_surface[food]
                """
        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_d_food_surface_d_white_meat_percentage(self, population_df, food):
        """
        Compute the derivative of food land surface wrt white meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        sub_total_surface_grad = 0.0
        kg_food_to_surface = self.kg_to_m2_dict

        white_meat_diet_grad = 365 * kg_food_to_surface[GlossaryCore.WhiteMeat]
        # removed_white_kcal = -self.kcal_diet_df['total'] / 100

        # red to white meat value influences red meat, white meat, and vegetable surface
        if food in [GlossaryCore.WhiteMeat, GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize]:

            if food == GlossaryCore.WhiteMeat:
                sub_total_surface_grad = white_meat_diet_grad / self.kg_to_kcal_dict[food]
        """
            else:
                proportion = self.kcal_diet_df[food] / (self.kcal_diet_df[GlossaryCore.FruitsAndVegetables] + self.kcal_diet_df[GlossaryCore.Cereals] + self.kcal_diet_df[GlossaryCore.RiceAndMaize])
                sub_total_surface_grad = removed_white_kcal * proportion / self.kg_to_kcal_dict[food] * kg_food_to_surface[food]
        """
        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_d_food_surface_d_fish_percentage(self, population_df, food):
        """
        Compute the derivative of food land surface wrt fish percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        sub_total_surface_grad = 0.0
        kg_food_to_surface = self.kg_to_m2_dict

        fish_diet_grad = 365 * kg_food_to_surface[GlossaryCore.Fish]

        # red to white meat value influences red meat, white meat, and vegetable surface
        if food in [GlossaryCore.WhiteMeat, GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize, GlossaryCore.OtherFood,
                    GlossaryCore.Fish]:

            if food == GlossaryCore.Fish:
                sub_total_surface_grad = fish_diet_grad / self.kg_to_kcal_dict[food]

        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_d_food_surface_d_other_food_percentage(self, population_df, food):
        """
        Compute the derivative of food land surface wrt other food percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        sub_total_surface_grad = 0.0
        kg_food_to_surface = self.kg_to_m2_dict

        other_food_diet_grad = 365 * kg_food_to_surface[GlossaryCore.OtherFood]

        # red to white meat value influences red meat, white meat, and vegetable surface
        if food in [GlossaryCore.WhiteMeat, GlossaryCore.FruitsAndVegetables, GlossaryCore.Cereals, GlossaryCore.RiceAndMaize, GlossaryCore.OtherFood]:

            if food == GlossaryCore.OtherFood:
                sub_total_surface_grad = other_food_diet_grad / self.kg_to_kcal_dict[food]

        total_surface_grad = sub_total_surface_grad * population_df[
            GlossaryCore.PopulationValue].values * 1e6 * self.ha_to_m2 / 1e9
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_calories_per_day_constraint(self):
        self.calories_per_day_constraint = (self.calories_pc_df[
                                                'kcal_pc'].values - self.constaint_calories_limit) / self.constraint_calories_ref

    def compute_calories_per_capita(self):
        non_wasted_share = 1 - self.food_waste_percentage_df[GlossaryCore.FoodWastePercentageValue].values / 100.
        self.produced_kcal = self.red_meat_calories_per_day \
                             + self.white_meat_calories_per_day \
                             + self.vegetables_and_carbs_calories_per_day \
                             + self.milk_and_eggs_calories_per_day \
                             + self.fish_calories_per_day \
                             + self.other_calories_per_day

        self.calories_pc_df['kcal_pc'] = self.produced_kcal * non_wasted_share

    def compute_organic_waste(self):
        waste_share = self.food_waste_percentage_df[GlossaryCore.FoodWastePercentageValue] / 100.
        self.organic_waste_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            "kcal": self.produced_kcal * waste_share
        })

    def compute_total_food_land_surface(self):
        self.total_food_land_surface[GlossaryCore.Years] = self.food_land_surface_df[GlossaryCore.Years]
        self.total_food_land_surface['total surface (Gha)'] = self.food_land_surface_df['total surface (Gha)']

        self.food_land_surface_percentage_df = self.convert_surface_to_percentage(
            self.food_land_surface_df)

    def compute_mix_detailed_production(self):
        self.mix_detailed_production['Crop residues (TWh)'] = self.residue_prod_from_food_surface.values + \
                                                              self.inputs['techno_infos_dict'][
                                                                  'residue_density_percentage'] * self.production[
                                                                  'biomass_dry (TWh)']

        self.mix_detailed_production['Crop for Energy (TWh)'] = self.production['biomass_dry (TWh)'] * (
                1 - self.inputs['techno_infos_dict']['residue_density_percentage'])

        self.mix_detailed_production['Total (TWh)'] = self.mix_detailed_production['Crop for Energy (TWh)'] + \
                                                      self.mix_detailed_production['Crop residues (TWh)']

    def compute_techno_consumption(self):
        self.techno_consumption[f'{GlossaryEnergy.carbon_captured} ({self.mass_unit})'] = -self.inputs['techno_infos_dict']['CO2_from_production'] / \
                                                                                          self.data_fuel_dict['high_calorific_value'] * \
                                                                                          self.mix_detailed_production['Total (TWh)']
        self.techno_consumption_woratio[f'{GlossaryEnergy.carbon_captured} ({self.mass_unit})'] = -self.inputs['techno_infos_dict'][
            'CO2_from_production'] / \
                                                                                                  self.data_fuel_dict[
                                                                                'high_calorific_value'] * \
                                                                                                  self.mix_detailed_production['Total (TWh)']

    def rescale_production_and_consumption(self):
        # Scale production TWh -> PWh
        self.techno_production = self.mix_detailed_production[[GlossaryCore.Years, 'Total (TWh)']]
        self.techno_production = self.techno_production.rename(
            columns={'Total (TWh)': "biomass_dry (TWh)"})
        for column in self.techno_production.columns:
            if column != GlossaryCore.Years:
                self.techno_production[column] /= self.scaling_factor_techno_production

        # Scale production Mt -> Gt
        for column in self.techno_consumption.columns:
            if column != GlossaryCore.Years:
                self.techno_consumption[column] /= self.scaling_factor_techno_consumption
                self.techno_consumption_woratio[column] /= self.scaling_factor_techno_consumption

    def compute_consumed_calories_per_capita_breakdown(self):
        non_years_columns = self.updated_diet_df.columns[1:]
        non_wasted_share = 1 - self.food_waste_percentage_df[GlossaryCore.FoodWastePercentageValue].values / 100.
        self.consumed_calories_pc_breakdown_per_day_df = pd.DataFrame({
            key: self.updated_diet_df[key] / 365 * self.kg_to_kcal_dict[key] * non_wasted_share for key in
            non_years_columns
        })

        self.consumed_calories_pc_breakdown_per_day_df.insert(0, GlossaryCore.Years,
                                                              self.updated_diet_df[GlossaryCore.Years])
