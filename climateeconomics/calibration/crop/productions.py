from copy import copy
from climateeconomics.calibration.crop.shares_of_waste import food_waste_share_supply_chain
from climateeconomics.calibration.crop.tools import CalibrationData
from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy

population_2021 = 7_954_448_391

# FAO data food avalaible for consumption kg per pers per year
per_capita_prod_other_red_meat = 6_209_141 * 1000 / population_2021 + (10.037 + 16.311 + 0.757) * 1e9 / population_2021  # duck + lamb and mutton + goat and sheep + horse
dict_of_available_food_type_per_capita_2021 = {
    GlossaryCore.RedMeat: 9.35 + per_capita_prod_other_red_meat, # beef + others
    GlossaryCore.WhiteMeat: 16.96 + 13.89 + (5.371 + 0.870) * 1e9 / population_2021, # poultry + pork + turkey + rabbit
    GlossaryCore.Milk: 87.03,
    GlossaryCore.Eggs: 10.34,
    GlossaryCore.Fish: 20.03,
    GlossaryCore.Maize: 17.79,
    GlossaryCore.SugarCane: 5.23,
    GlossaryCore.Rice: 80.67,
    GlossaryCore.FruitsAndVegetables: 86.40 + 147.04,
}

# cereals = total - (rice + maize)
# only 45% of cereals are used for human food
dict_of_available_food_type_per_capita_2021[GlossaryCore.Cereals] = 3_071_264_000 * 1000 / population_2021 * 0.45 - dict_of_available_food_type_per_capita_2021[GlossaryCore.Rice] - dict_of_available_food_type_per_capita_2021[GlossaryCore.Maize]


# total available calories for consumption
actual_calories_per_capita_per_day_2021 = CalibrationData(
        varname=GlossaryCore.CaloriesPerCapitaValue,
        column_name='kcal_pc',
        year=2021,
        value=2959,
        unit='kcal/person/day',
        source='Food and Agriculture Organization of the United Nations (2023) and other sources',
        link='https://ourworldindata.org/grapher/daily-per-capita-caloric-supply?country=~OWID_WRL')


zeros_dict = {ft: 0 for ft in GlossaryCore.DefaultFoodTypesV2}
share_usage_for_biomass_dry = copy(zeros_dict)
share_usage_for_biomass_dry.update({
    GlossaryCore.Maize: 15.6,  # https://www.ifpri.org/blog/food-versus-fuel-v20-biofuel-policies-and-current-food-crisis/#:~:text=Global%20biofuel%20production%20consumes%20a,is%20used%20for%20ethanol%20production.
    GlossaryCore.SugarCane: 21.6, # https://www.ifpri.org/blog/food-versus-fuel-v20-biofuel-policies-and-current-food-crisis/#:~:text=Global%20biofuel%20production%20consumes%20a,is%20used%20for%20ethanol%20production.
})

# deduce production from what is available for consumption by person by year according to FAO.
# deduced production for human food (Mt) = available for consumption per person per year (kg) * population / (1 - share waste supply chain) / 1e9
dict_of_production_in_megatons_2021 = {food_type: dict_of_available_food_type_per_capita_2021[food_type] * population_2021 / 1e9 /
                                       (1 - food_waste_share_supply_chain[food_type] / 100.) /
                                       (1 - share_usage_for_biomass_dry[food_type] / 100.) for food_type in dict_of_available_food_type_per_capita_2021.keys()}

# from capgemini sharepoint
kcal_by_kg_produced = GlossaryCore.FoodTypeKcalByProdUnitVar["default"]

# deducing production for category "Other" by deducing the missing calories from the other categories
kg_per_pers_per_year_2021_per_food_type_available_for_consumption = {food_type: dict_of_production_in_megatons_2021[food_type] * 1e9 / population_2021 * (1 - food_waste_share_supply_chain[food_type] / 100) for food_type in dict_of_production_in_megatons_2021.keys()}
total_kcal_pc_available_wo_other_2021 = sum(kg_per_pers_per_year_2021_per_food_type_available_for_consumption[food_type] * kcal_by_kg_produced[food_type] for food_type in dict_of_production_in_megatons_2021.keys())
calories_per_capita_per_day_wo_other_2021 = total_kcal_pc_available_wo_other_2021 / 365  # kcal/person/day * days in a year
missing_kcals_per_day_per_person = actual_calories_per_capita_per_day_2021.value - calories_per_capita_per_day_wo_other_2021
if missing_kcals_per_day_per_person < 0 :
    raise ValueError("There is more food available for consumption than needed.")
missing_prod_other = missing_kcals_per_day_per_person / kcal_by_kg_produced[GlossaryCore.OtherFood] * 365 * population_2021 / 1e9  # Megatons
dict_of_production_in_megatons_2021[GlossaryCore.OtherFood] = missing_prod_other

to_export = {
    GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.biomass_dry): share_usage_for_biomass_dry,
    GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.wet_biomass): zeros_dict,

    GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): zeros_dict,
    GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): zeros_dict,

    GlossaryCore.FoodTypeShareWasteSupplyChainUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): zeros_dict,
    GlossaryCore.FoodTypeShareWasteSupplyChainUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): zeros_dict,
}

if __name__ == '__main__':
    for food_type, food_prod in dict_of_production_in_megatons_2021.items():
        per_capita_available = food_prod * 1e9 / population_2021 * (1 - food_waste_share_supply_chain[food_type] / 100)
        print(f"Per capita available for {food_type} : {round(per_capita_available, 2)} kg (modeled)")

    print("\nPer capita available for consumption for 2021 relative error:")
    for ft, value_fao in dict_of_available_food_type_per_capita_2021.items():
        relative_diff = (kg_per_pers_per_year_2021_per_food_type_available_for_consumption[ft] / value_fao - 1) * 100
        print(f"{ft}: {round(relative_diff, 2)}%")

