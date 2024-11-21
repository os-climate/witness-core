from climateeconomics.calibration.crop.productions import dict_of_production_in_megatons_2021, \
    per_capita_prod_other_red_meat
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore

# existing data from capgemini sharepoint.
# Need to update with rice and maize and sugar cane
# kg {ghg} / kg produced
co2_emissions = {
            GlossaryCore.RedMeat: 0.0,
                GlossaryCore.WhiteMeat: 3.95,
                GlossaryCore.Milk: 0.0,
                GlossaryCore.Eggs: 1.88,
                GlossaryCore.Cereals: 0.12,
                GlossaryCore.FruitsAndVegetables: 0.44,
                GlossaryCore.Fish: 2.37,
                GlossaryCore.OtherFood: 0.48
            }
ch4_emissions = {
    GlossaryCore.RedMeat: 6.823e-1,
                GlossaryCore.WhiteMeat: 1.25e-2,
                GlossaryCore.Milk: 3.58e-2,
                GlossaryCore.Eggs: 0.0,
                # negligible methane in this category
                GlossaryCore.Cereals: 0.0,
                GlossaryCore.FruitsAndVegetables: 0.0,
                # consider fish farm only
                GlossaryCore.Fish: 3.39e-2,
            }
n2o_emissions = {
                GlossaryCore.RedMeat: 9.268e-3,
                GlossaryCore.WhiteMeat: 3.90e-4,
                GlossaryCore.Milk: 2.40e-4,
                GlossaryCore.Eggs: 1.68e-4,
                GlossaryCore.Cereals: 1.477e-3,
                GlossaryCore.FruitsAndVegetables: 2.63e-4,
                GlossaryCore.Fish: 0.,  # no crop or livestock related
                GlossaryCore.OtherFood: 1.535e-3,
            }

def solve_breakdown_ghg(total_ghg_eq, methane_share, n2o_share):
    # total_ghg_eq = total_ghg_emissions_per_kg[food_type]
    # methane_share = methane_share_of_emissions_co2eq[food_type]
    # n2o_share = n2o_share_of_emissions_co2_eq[food_type]
    # total = sum val_i * gwp_i

    co2_share = 1 - methane_share - n2o_share

    return {
        GlossaryCore.CO2: total_ghg_eq / ClimateEcoDiscipline.GWP_100_default[GlossaryCore.CO2] * co2_share,
        GlossaryCore.CH4: total_ghg_eq / ClimateEcoDiscipline.GWP_100_default[GlossaryCore.CH4] * methane_share,
        GlossaryCore.N2O: total_ghg_eq / ClimateEcoDiscipline.GWP_100_default[GlossaryCore.N2O] * n2o_share
    }

# https://ourworldindata.org/carbon-footprint-food-methane
# https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions
# 100 year time scale
total_ghg_emissions_per_kg = {
    GlossaryCore.Rice: 2.5,
    GlossaryCore.Maize: 2,
    GlossaryCore.SugarCane: 3.2, # https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions
}
agriculture_n2o_share_of_emissions_co2_eq = 2.33 / (3.54 + 2.33 + 0.) # 2020 value https://ourworldindata.org/emissions-by-sector

methane_share_of_emissions_co2eq = {
    GlossaryCore.Rice: 2.5 / 4.5, # major source of methane # https://www.worldwildlife.org/blogs/sustainability-works/posts/innovation-in-reducing-methane-emissions-from-the-food-sector-side-of-rice-hold-the-methane
    GlossaryCore.Maize: 0,  # no methane
    GlossaryCore.SugarCane: 0, # not a major source of methane emissions https://ourworldindata.org/environmental-impacts-of-food#methane
}
# assumed .. just to have some n2o emisssions for fertilizer
n2o_share_of_emissions_co2_eq = {
    GlossaryCore.Rice: agriculture_n2o_share_of_emissions_co2_eq,
    GlossaryCore.Maize: agriculture_n2o_share_of_emissions_co2_eq,
    GlossaryCore.SugarCane: agriculture_n2o_share_of_emissions_co2_eq,
}
for ft in total_ghg_emissions_per_kg:
    total_ghg_eq = total_ghg_emissions_per_kg[ft]
    methane_share = methane_share_of_emissions_co2eq[ft]
    n2o_share = n2o_share_of_emissions_co2_eq[ft]
    breakdown_ghg = solve_breakdown_ghg(total_ghg_eq, methane_share, n2o_share)
    n2o_emissions[ft] = breakdown_ghg[GlossaryCore.N2O]
    ch4_emissions[ft] = breakdown_ghg[GlossaryCore.CH4]
    co2_emissions[ft] = breakdown_ghg[GlossaryCore.CO2]
    modeled_emi = sum(list(breakdown_ghg[ghg] * ClimateEcoDiscipline.GWP_100_default[ghg] for ghg in breakdown_ghg.keys()))
    if not abs(modeled_emi - total_ghg_eq) < 1e-6:
        raise Exception("Error in solving breakdown of GHG emissions")




# FOCUS METHANE EMISSIONS : tuning other so that total methane emissions are OK

# 43 % of methane emissions come from agriculture https://searchinger.princeton.edu/sites/g/files/toruqf4701/files/methane_discussion_paper_nov_2021.pdf
# https://www.iea.org/reports/global-methane-tracker-2023/overview
biofuel_emissions = 9.2
energy_emissions_w_biofuel = 44.4 + 37.2 + 40.1 + biofuel_emissions
ch4_emissions_global_2021 = (energy_emissions_w_biofuel) / 0.40
agri_emissions_for_food = ch4_emissions_global_2021 * 0.43
agri_emissions_including_biofuel = agri_emissions_for_food + biofuel_emissions

emissions_breakdown = {ft : dict_of_production_in_megatons_2021[ft] * ch4_emissions[ft] for ft in ch4_emissions.keys()}
modeled_agriculture_ch4_emissions_2021_wo_other = sum(emissions_breakdown.values())
missing_ch4_emissions = agri_emissions_including_biofuel - modeled_agriculture_ch4_emissions_2021_wo_other
if missing_ch4_emissions < 0:
    raise Exception("Error in tuning CH4 emissions")

ch4_emissions[GlossaryCore.OtherFood] = missing_ch4_emissions / dict_of_production_in_megatons_2021[GlossaryCore.OtherFood]
modeled_emissions_ch4 = sum(dict_of_production_in_megatons_2021[ft] * ch4_emissions[ft] for ft in ch4_emissions.keys())
print("Relative error on modeled agriculture CH4 emissions in 2021:")
print((modeled_emissions_ch4 - agri_emissions_including_biofuel) / agri_emissions_including_biofuel)

# Focus N2O emissions ... #https://www.statista.com/statistics/1351550/agriculture-nitrous-oxide-emissions-worldwide/#:~:text=Global%20emissions%20of%20nitrous%20oxides,than%2025%20percent%20since%201990.
agriculture_2021_n2o_emissions = 2.31 * 1e3 / ClimateEcoDiscipline.GWP_100_default[GlossaryCore.N2O]  # Mt

ft_wo_other = [ft for ft in n2o_emissions.keys() if ft != GlossaryCore.OtherFood]
modeled_emissions_breakdown_n2o = {ft : dict_of_production_in_megatons_2021[ft] * n2o_emissions[ft] for ft in ft_wo_other}
modeled_agriculture_n2o_emissions_2021_wo_other = sum(modeled_emissions_breakdown_n2o.values())
missing_n2o_emissions = agriculture_2021_n2o_emissions - modeled_agriculture_n2o_emissions_2021_wo_other
if missing_n2o_emissions < 0:
    raise Exception("Error in tuning N2O emissions")
print("Relative error on modeled agriculture N2O emissions in 2021:")
n2o_emissions[GlossaryCore.OtherFood] = missing_n2o_emissions / dict_of_production_in_megatons_2021[GlossaryCore.OtherFood]
modeled_emissions_breakdown_n2o[GlossaryCore.OtherFood] = dict_of_production_in_megatons_2021[GlossaryCore.OtherFood] * n2o_emissions[GlossaryCore.OtherFood]
modeled_emissions_n2o = sum(modeled_emissions_breakdown_n2o.values())
print((modeled_emissions_n2o - agriculture_2021_n2o_emissions) / agriculture_2021_n2o_emissions)


modeled_emissions_co2 = sum(dict_of_production_in_megatons_2021[ft] * co2_emissions[ft] for ft in co2_emissions.keys())
total_emissions_of_agri_excluding_land_use_change = 5.3 # GtCO2eq https://openknowledge.fao.org/bitstreams/487c7f4e-91ff-4d23-b1e4-f72dd867e939/download#:~:text=Of%20the%20three%20components%20that,change%20(3.1%20Gt%20CO2eq).

total_ghg_eq_modeled = (modeled_emissions_co2 * ClimateEcoDiscipline.GWP_100_default[GlossaryCore.CO2] +
                        modeled_emissions_ch4 * ClimateEcoDiscipline.GWP_100_default[GlossaryCore.CH4] +
                        modeled_emissions_n2o * ClimateEcoDiscipline.GWP_100_default[GlossaryCore.N2O])\
                       / 1e3

print("Relative error on total agriculture emissions in 2021:")
print((round((total_ghg_eq_modeled - total_emissions_of_agri_excluding_land_use_change) / total_emissions_of_agri_excluding_land_use_change, 2)))

# Our crop discipline should not include emissions associated to land use change. However it is hard to know in every data found if it is included or not. It appears that most of the time it is.
# Without any rescaling, the total of CO2Eq emissions is closer to total emissions (food + land use change (about 9-10 Gt https://openknowledge.fao.org/server/api/core/bitstreams/cc09fbbc-eb1d-436b-a88a-bed42a1f12f3/content) than only food (5.3 for 2021).
# therefore we decided to rescale all unitary emissions prod by a factor so that the total emissions match the total emissions of agriculture excluding land use change.
# By doing this we conserve the relative order of emissions between food types, which seemed very coherent.

rescaling_factor = 0.5226  # found by hand
new_total_emissions_co2eq = 0
for ghg, dict_emissions in zip(
        [GlossaryCore.CO2, GlossaryCore.CH4, GlossaryCore.N2O,],
        [co2_emissions, ch4_emissions, n2o_emissions]):
    for ft in dict_emissions.keys():
        dict_emissions[ft] *= rescaling_factor
        new_total_emissions_co2eq += dict_of_production_in_megatons_2021[ft] * dict_emissions[ft] * ClimateEcoDiscipline.GWP_100_default[ghg] / 1e3

print("Relative error on total agriculture emissions in 2021 after rescaling:")
print((new_total_emissions_co2eq - total_emissions_of_agri_excluding_land_use_change) / total_emissions_of_agri_excluding_land_use_change)


to_export = {
    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CO2): co2_emissions,
    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CH4): ch4_emissions,
    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.N2O): n2o_emissions,
}


