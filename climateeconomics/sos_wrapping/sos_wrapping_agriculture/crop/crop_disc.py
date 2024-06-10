'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
import logging
from copy import deepcopy
from os.path import dirname, join

import numpy as np
import pandas as pd
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.pie_charts.instanciated_pie_chart import (
    InstanciatedPieChart,
)

from climateeconomics.core.core_agriculture.crop import Crop
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class CropDiscipline(ClimateEcoDiscipline):
    ''' Crop discipline transforms crops and crops residues
        into biomass_dry resource

        Some Foods are detailed, most of them are grouped within the category Glossary.OtherFood which includes
        wheat, cereals, sugar and 'other' as defined in https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=OgMTTe
        tab "food supply kg" with a 1 in column "other". Only includes non-animal food
        other =
        Beverages, Alcoholic
        Beverages, Fermented
        Cocoa Beans and products
        Coconut Oil
        Coffee and products
        Cottonseed
        Cottonseed Oil
        Groundnut Oil
        Groundnuts
        Honey
        Maize Germ Oil
        Nuts and products
        Oilcrops Oil, Other
        Oilcrops, Other
        Olive Oil
        Palm Oil
        Palmkernel Oil
        Rape and Mustard Oil
        Ricebran Oil
        Sesameseed Oil
        Soyabean Oil
        Soyabeans
        Sugar
        Sunflowerseed Oil
        Tea (including mate)
        Wine
    '''
    energy_name = "biomass_dry"
    # ontology information
    _ontology_data = {
        'label': 'Crop Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-seedling fa-fw',
        'version': '',
    }
    techno_name = 'CropEnergy'
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = GlossaryCore.YearEndDefault
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    '''
    Sources:
    [1]: https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=OgMTTe
    [2] : https://capgemini.sharepoint.com/:p:/r/sites/SoSTradesCapgemini/_layouts/15/Doc.aspx?sourcedoc=%7B24B3F100-A5AD-4CCA-8021-3A273C1E4D9E%7D&file=diet%20problem.pptx&action=edit&mobileredirect=true

    does not consider farm fish for land use at this stage as it would require to split farm fish from wild fish in the computations
    The land use for category 'other' is adjusted to calibrate the land surface
    Method 1:  
    It used to be computed from the variable self.other_use_crop = 0.01719 [ha/person] that has been removed. 
    relationship between self.other_use_crop and other: [m2/kg] = [m2/ha*ha/person*person*year/kg/year]
    other[m2/kg] = 10000.other_use_crop/other[kg/person/year]/year = 10000*0.01719/77.24/year = 2.25 m2/kg/year WARNING on units
    Method 2:
    It could be computed from the average of all land use per kg of all components of others (12.41 m2/kg) 
    but assuming the same ponderation for each food of others does not provide reliable results
    '''
    default_kg_to_m2 = {'red meat': 345.,
                        'white meat': 14.5,
                        'milk': 8.95,
                        'eggs': 6.27,
                        'rice and maize': 2.89,
                        'cereals': 4.5,
                        'fruits and vegetables': 0.8,
                        GlossaryCore.Fish: 0.,
                        GlossaryCore.OtherFood: 5.1041,
                        }
    # unit kcal/kg
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

    #diet default for veg = 260+32.93 of cereals
    # unit: kg/person/year
    '''
    Default diet baseline comes from [1] modified manually following [2] in order to:
    - have rougly 2900kcal
    - have a correct landuse
    '''
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

    # Our World in Data (emissions per kg of food product)
    # https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions
    # FAO Stats (total production)
    # https://www.fao.org/faostat/en/#data/FBS

    co2_gwp_100 = 1.0
    ch4_gwp_100 = 28.0
    n2o_gwp_100 = 265.0

    ghg_emissions_unit = 'kg/kg'  # in kgCo2eq per kg of food
    # Consider farm fish emissions, assuming that 53% of fish consumed comes from farm fish
    # values computed in https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=uqOX8c
    default_ghg_emissions = {'red meat': 21.56,
                             'white meat': 4.41,
                             'milk': 1.07,
                             'eggs': 1.93,
                             'rice and maize': 1.98,
                             'cereals': 0.52,
                             'fruits and vegetables': 0.51,
                             GlossaryCore.Fish: 3.32,
                             GlossaryCore.OtherFood: 0.93,
                             }

    # Our World in Data
    # https://ourworldindata.org/carbon-footprint-food-methane
    ch4_emissions_unit = 'kg/kg'  # in kgCH4 per kg food
    # values computed in https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=uqOX8c
    default_ch4_emissions = {'red meat': 6.823e-1,
                            'white meat': 1.25e-2,
                            'milk': 3.58e-2,
                            'eggs': 0.0,
                            'rice and maize': 3.17e-2,
                            # negligible methane in this category
                            'cereals': 0.0,
                            'fruits and vegetables': 0.0,
                            # consider fish farm only
                            GlossaryCore.Fish: 3.39e-2,
                            GlossaryCore.OtherFood: 0.,
                            }

    # FAO Stats
    # https://www.fao.org/faostat/en/#data/GT
    # data preparation detailed in https://capgemini.sharepoint.com/:x:/r/sites/SoSTradesCapgemini/Shared%20Documents/General/Development/WITNESS/Agriculture/Faostatfoodsupplykgandkcalpercapita.xlsx?d=w2b79154f7109433c86a28a585d9f6276&csf=1&web=1&e=U4Tbjl&nav=MTVfe0VBRUJDOTAyLUIzNjEtNEI0Ni05RjhCLUE5QTZDQzc5NTY1OX0
    # n2o_emission_per_kg_food = n2o_emission / (kg_food/capita/year * population)
    # where n2o_emission = land_use_food/total_land_use * land_n2O_emissions
    # where total_land_use is split between crop and pastures since land_n2O_emissions is split between land and pastures
    n2o_emissions_unit = 'kg/kg'  # in kgN2O per kg food$

    default_n2o_emissions = {'red meat': 9.268e-3,
                             'white meat': 3.90e-4,
                             'milk': 2.40e-4,
                             'eggs': 1.68e-4,
                             'rice and maize': 9.486e-4,
                             'cereals': 1.477e-3,
                             'fruits and vegetables': 2.63e-4,
                             GlossaryCore.Fish: 0., #no crop or livestock related
                             GlossaryCore.OtherFood: 1.68e-3,
                             }

    co2_emissions_unit = 'kg/kg'  # in kgCO2 per kg food$
    # co2 emissions = (total_emissions - ch4_emissions * ch4_gwp_100 -
    # n2o_emissions * n2o_gwp_100)/co2_gwp_100
    # Difference method
    default_co2_emissions = {}
    for food in default_ghg_emissions:
        default_co2_emissions[food] = max(0., (default_ghg_emissions[food] - default_ch4_emissions[food]*ch4_gwp_100 - default_n2o_emissions[food]*n2o_gwp_100) / co2_gwp_100)


    year_range = default_year_end - default_year_start + 1
    total_kcal = 1067961. #kcal/person/year
    # Take diet per year in kg / 365 * kg to cal to get consumption per day in kcal
    red_meat_average_ca_daily_intake = default_kg_to_kcal['red meat'] * diet_df_default['red meat'].values[0]/365
    milk_eggs_average_ca_daily_intake = default_kg_to_kcal['eggs'] * diet_df_default['eggs'].values[0]/365 + \
                                        default_kg_to_kcal['milk'] * diet_df_default['milk'].values[0]/365
    white_meat_average_ca_daily_intake = default_kg_to_kcal[
                                             'white meat'] * diet_df_default['white meat'].values[0]/365
    #kcal per kg 'vegetables': 200 https://www.fatsecret.co.in/calories-nutrition/generic/raw-vegetable?portionid=54903&portionamount=100.000&frc=True#:~:text=Nutritional%20Summary%3A&text=There%20are%2020%20calories%20in,%25%20carbs%2C%2016%25%20prot.
    vegetables_and_carbs_average_ca_daily_intake =  diet_df_default['fruits and vegetables'].values[0]/365 * default_kg_to_kcal['fruits and vegetables'] + \
                                                    diet_df_default['cereals'].values[0]/365 * default_kg_to_kcal['cereals'] + \
                                                    diet_df_default['rice and maize'].values[0] / 365 * default_kg_to_kcal['rice and maize']
    fish_average_ca_daily_intake = default_kg_to_kcal[GlossaryCore.Fish] * \
                                    diet_df_default[GlossaryCore.Fish].values[0] / 365
    other_average_ca_daily_intake = default_kg_to_kcal[GlossaryCore.OtherFood] * diet_df_default[GlossaryCore.OtherFood].values[0] / 365
    default_red_meat_ca_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'red_meat_calories_per_day': [red_meat_average_ca_daily_intake] * year_range})
    default_white_meat_ca_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'white_meat_calories_per_day': [white_meat_average_ca_daily_intake] * year_range})
    default_vegetables_and_carbs_calories_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'vegetables_and_carbs_calories_per_day': [vegetables_and_carbs_average_ca_daily_intake] * year_range})
    default_milk_and_eggs_calories_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'milk_and_eggs_calories_per_day': [milk_eggs_average_ca_daily_intake] * year_range})
    default_fish_ca_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        GlossaryCore.FishDailyCal: [fish_average_ca_daily_intake] * year_range})
    default_other_ca_per_day = pd.DataFrame({
        GlossaryCore.Years: default_years,
        GlossaryCore.OtherDailyCal: [other_average_ca_daily_intake] * year_range})

    # mdpi: according to the NASU recommendations,
    # a fixed value of 0.25 is applied to all crops
    # 50% of crops are left on the field,
    # 50% of the left on the field can be used as crop residue =>
    # 25% of the crops is residue
    residue_percentage = 0.25 # TODO delete residue percentage from here as it'll be computed in organic waste stream
    # 23$/t for residue, 60$/t for crop
    crop_residue_price_percent_dif = 23 / 60
    # bioenergyeurope.org : Dedicated energy crops
    # represent 0.1% of the total biomass production in 2018
    energy_crop_percentage = 0.005
    # ourworldindata, average cereal yield: 4070kg/ha +
    # average yield of switchgrass on grazing lands: 2565,67kg/ha
    # residue is 0.25 more than that
    density_per_ha = 2903 * 1.25
    # reference of the value in database
    initial_production = DatabaseWitnessCore.InitialProductionCropForEnergy.value
    construction_delay = 1  # years
    # defined lifetime here is the supposed lifetime of crop farm
    lifetime = 50

    techno_infos_dict_default = {
        'maturity': 5,
        # computed 87.7euro/ha, counting harvest,
        # fertilizing, drying...from gov.mb.ca
        # plus removing residue price:
        # FACT_Sheet_Harvesting_Crop_Residues_-_revised_2016-2
        # 22$/t for harvest residue + 23$/t for
        # fertilizing => 37.5euro/ha for residues
        'Opex_percentage': 0.52,
        'Opex_percentage_for_residue_only': 0.15,
        # CO2 from production from tractor is taken
        # into account into the energy net factor
        # land CO2 absorption is static and set in carbonemission model
        'CO2_from_production': -0.425 * 44.01 / 12.0,
        'CO2_from_production_unit': 'kg/kg',
        'elec_demand': 0,
        'elec_demand_unit': 'kWh/kWh',
        'WACC': 0.07,  # ?
        'lifetime': lifetime,
        'lifetime_unit': GlossaryCore.Years,
        # capex from
        # gov.mb.ca/agriculture/farm-management/production-economics/pubs/cop-crop-production.pdf
        # 237.95 euro/ha (717 $/acre)
        # 1USD = 0,82 euro in 2021
        'Capex_init': 237.95,
        'Capex_init_unit': 'euro/ha',
        'full_load_hours': 8760.0,
        'euro_dollar': 1.2195,  # in 2021, date of the paper
        'density_per_ha': density_per_ha,  # average, worldbioenergy.org
        'density_per_ha_unit': 'kg/ha',
        'residue_density_percentage': residue_percentage,
        'crop_percentage_for_energy': energy_crop_percentage,
        'residue_percentage_for_energy': 0.,  # TODO delete all use of residues in crop : residue will be computed in organic waste
        'efficiency': 1.0,
        'techno_evo_eff': 'no',
        'crop_residue_price_percent_dif': crop_residue_price_percent_dif,
        GlossaryCore.ConstructionDelay: construction_delay,  # years
    }

    # Age distribution of forests in 2008 (
    # this is the initial distribution considered of crop farms
    initial_age_distribution = pd.DataFrame({'age': np.arange(1, lifetime),
                                             'distrib': [0.16, 0.24, 0.31, 0.39, 0.47, 0.55, 0.63, 0.71, 0.78, 0.86,
                                                         0.94, 1.02, 1.1, 1.18, 1.26, 1.33, 1.41, 1.49, 1.57, 1.65,
                                                         1.73, 1.81, 1.88, 1.96, 2.04, 2.12, 2.2, 2.28, 2.35, 2.43,
                                                         2.51, 2.59, 2.67, 2.75, 2.83, 2.9, 2.98, 3.06, 3.14, 3.22,
                                                         3.3, 3.38, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.92]})



    crop_investment_default = pd.read_csv(join(dirname(__file__), 'data/crop_investment.csv'), index_col=0)

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        'diet_df': {'type': 'dataframe', 'unit': 'kg_food/person/year', 'default': diet_df_default,
                    'dataframe_descriptor': {  # GlossaryCore.Years: ('float', None, False),
                        'red meat': ('float', [0, 1e9], True),
                        'white meat': ('float', [0, 1e9], True),
                        'milk': ('float', [0, 1e9], True),
                        'eggs': ('float', [0, 1e9], True),
                        'rice and maize': ('float', [0, 1e9], True),
                        'cereals': ('float', [0, 1e9], True),
                        'fruits and vegetables': ('float', [0, 1e9], True),
                        GlossaryCore.Fish: ('float', [0, 1e9], True),
                        GlossaryCore.OtherFood: ('float', [0, 1e9], True)
                    },
                    'dataframe_edition_locked': False, 'namespace': 'ns_crop',
                    'description': 'average annual consumption in kg/person of food '
                                   'by category. Category other includes sugar, oil '
                                   'wheat, alcohol, cocoa etc (see full list in crop_disc) '
                                   },
        'kg_to_kcal_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_kcal,
                            'unit': 'kcal/kg', 'namespace': 'ns_crop',
                            'description': 'Used to convert kg to kcal. kcal per kg of for each category of food '
                            'Category other includes sugar, oil wheat, alcohol, '
                                           'cocoa etc (see full list in crop_disc)'
                            },
        'kg_to_m2_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_m2,
                          'unit': 'm^2/kg', 'namespace': 'ns_crop',
                          'description': 'Used to convert kg to m2. m2 per kg of for each category of food '
                                         'Category other includes sugar, oil wheat, alcohol, '
                                         'cocoa etc (see full list in crop_disc)'
                          },
        # design variables of changing diet
        'red_meat_calories_per_day': {'type': 'dataframe', 'default': default_red_meat_ca_per_day,
                                      'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                               'red_meat_calories_per_day': ('float', None, True)},
                                      'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                      'namespace': 'ns_crop',
                                      'description': 'number of calories coming from red meat consumed per day per person'},
        'white_meat_calories_per_day': {'type': 'dataframe', 'default': default_white_meat_ca_per_day,
                                        'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                 'white_meat_calories_per_day': ('float', None, True)},
                                        'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                        'namespace': 'ns_crop',
                                        'description': 'number of calories coming from white meat consumed per day per person'},
        'vegetables_and_carbs_calories_per_day': {'type': 'dataframe',
                                                  'default': default_vegetables_and_carbs_calories_per_day,
                                                  'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                           'vegetables_and_carbs_calories_per_day': (
                                                                               'float', None, True)},
                                                  'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                                  'namespace': 'ns_crop',
                                                  'description': 'number of calories coming from vegetables and fruits '
                                                                 'consumed per day per person'},
        'milk_and_eggs_calories_per_day': {'type': 'dataframe', 'default': default_milk_and_eggs_calories_per_day,
                                           'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                    'milk_and_eggs_calories_per_day': (
                                                                        'float', None, True)},
                                           'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                           'namespace': 'ns_crop',
                                           'description': 'number of calories coming from egg and milk consumed per day per person'},
        GlossaryCore.FishDailyCal: {'type': 'dataframe', 'default': default_fish_ca_per_day,
                                      'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                               GlossaryCore.FishDailyCal: ('float', None, True)},
                                      'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                      'namespace': 'ns_crop',
                                    'description': 'number of calories coming from fish consumed per day per person'},
        GlossaryCore.OtherDailyCal: {'type': 'dataframe', 'default': default_other_ca_per_day,
                                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                             GlossaryCore.OtherDailyCal: ('float', None, True)},
                                    'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_crop',
                                     'description': 'number of calories coming from food other than '
                                                    'red and white meat, fish, vegetables, fruits, egg and milk '
                                                    'consumed per day per person. This category typically includes '
                                                    'sugar, oil wheat, alcohol, cocoa etc (see full list in crop_disc)'
                                     },

        GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,

        'param_a': {'type': 'float', 'default': -0.00833, 'unit': '-', 'user_level': 3},
        'param_b': {'type': 'float', 'default': -0.04167, 'unit': '-', 'user_level': 3},
        'crop_investment': {'type': 'dataframe', 'unit': 'G$',
                            'dataframe_descriptor': {GlossaryCore.Years: ('int', [1900, GlossaryCore.YearEndDefault], False),
                                                     GlossaryCore.InvestmentsValue: ('float', None, True)},
                            'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_crop',
                            'default': crop_investment_default},
        'scaling_factor_crop_investment': {'type': 'float', 'default': 1e3, 'unit': '-', 'user_level': 2},
        'scaling_factor_techno_consumption': {'type': 'float', 'default': 1e3, 'unit': '-',
                                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                              'namespace': 'ns_public', 'user_level': 2},
        'scaling_factor_techno_production': {'type': 'float', 'default': 1e3, 'unit': '-',
                                             'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                             'namespace': 'ns_public', 'user_level': 2},
        'margin': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'unit': '%',
                   'namespace': GlossaryCore.NS_WITNESS,
                   'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                            'margin': ('float', None, True)}},
        'transport_cost': {'type': 'dataframe', 'unit': '$/t', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                           'namespace': GlossaryCore.NS_WITNESS,
                           'dataframe_descriptor': {GlossaryCore.Years: ('int', [1900, GlossaryCore.YearEndDefault], False),
                                                    'transport': ('float', None, True)},
                           'dataframe_edition_locked': False},
        'transport_margin': {'type': 'dataframe', 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                             'namespace': GlossaryCore.NS_WITNESS,
                             'dataframe_descriptor': {GlossaryCore.Years: ('int', [1900, GlossaryCore.YearEndDefault], False),
                                                      'margin': ('float', None, True)},
                             'dataframe_edition_locked': False},
        'data_fuel_dict': {'type': 'dict', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                           'namespace': 'ns_biomass_dry', 'default': BiomassDry.data_energy_dict,
                           'unit': 'defined in dict'},
        'techno_infos_dict': {'type': 'dict', 'unit': 'defined in dict',
                              'default': techno_infos_dict_default},
        'initial_production': {'type': 'float', 'unit': 'TWh', 'default': initial_production},
        'initial_age_distrib': {'type': 'dataframe', 'unit': '%', 'default': initial_age_distribution,
                                'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                                         'age': ('float', None, True),
                                                         'distrib': ('float', None, True)}},
        'co2_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg',
                                 'default': default_co2_emissions},
        'ch4_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg',
                                 'default': default_ch4_emissions},
        'n2o_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg',
                                 'default': default_n2o_emissions},
        'constraint_calories_ref': {'type': 'float', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': GlossaryCore.NS_REFERENCE, 'default': 4000.},
        'constraint_calories_limit': {'type': 'float', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                      'namespace': GlossaryCore.NS_REFERENCE, 'default': 2000.},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.FoodWastePercentageValue: GlossaryCore.FoodWastePercentage,
    }

    DESC_OUT = {
        'total_food_land_surface': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': GlossaryCore.NS_WITNESS},
        'food_land_surface_df': {
            'type': 'dataframe', 'unit': 'Gha'},
        'food_land_surface_percentage_df': {'type': 'dataframe', 'unit': '%'},
        'updated_diet_df': {'type': 'dataframe', 'unit': 'kg/person/year'},
        'crop_productivity_evolution': {'type': 'dataframe', 'unit': '%'},
        'mix_detailed_prices': {'type': 'dataframe', 'unit': '$/MWh'},
        'mix_detailed_production': {'type': 'dataframe', 'unit': 'TWh'},
        'cost_details': {'type': 'dataframe', 'unit': '$/MWh'},
        'techno_production': {
            'type': 'dataframe', 'unit': 'TWh or Mt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_crop'},
        'techno_prices': {
            'type': 'dataframe', 'unit': '$/MWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_crop'},
        'techno_consumption': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop',
            'unit': 'TWh or Mt'},
        'techno_consumption_woratio': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop',
            'unit': 'TWh or Mt'},
        'land_use_required': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_crop'},
        # emissions from production that goes into energy mix
        'CO2_emissions': {
            'type': 'dataframe', 'unit': 'kg/kWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_crop'},
        # crop land emissions
        'CO2_land_emission_df': {'type': 'dataframe', 'unit': 'GtCO2',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CO2_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtCO2',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CH4_land_emission_df': {'type': 'dataframe', 'unit': 'GtCH4',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CH4_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtCH4',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'N2O_land_emission_df': {'type': 'dataframe', 'unit': 'GtN2O',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'N2O_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtN2O',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'calories_per_day_constraint': {'type': 'array', 'unit': 'kcal',
                                        'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                        'namespace': GlossaryCore.NS_FUNCTIONS},
        GlossaryCore.CaloriesPerCapitaValue: GlossaryCore.CaloriesPerCapita,
        GlossaryCore.CaloriesPerCapitaBreakdownValue: GlossaryCore.CaloriesPerCapitaBreakdown,
        GlossaryCore.OrganicWasteValue: GlossaryCore.OrganicWaste,
    }

    CROP_CHARTS = 'crop and diet charts'

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.crop_model = None

    def setup_sos_disciplines(self):  # type: (...) -> None

        if "red_meat_calories_per_day" in self.get_data_in():
            red_meat_calories_per_day = self.get_sosdisc_inputs("red_meat_calories_per_day")
            default_food_waste_percentage_df = pd.DataFrame({
                GlossaryCore.Years: red_meat_calories_per_day[GlossaryCore.Years].values,
                GlossaryCore.FoodWastePercentageValue: DatabaseWitnessCore.FoodWastePercentage.value
            })
            self.set_dynamic_default_values({GlossaryCore.FoodWastePercentageValue: default_food_waste_percentage_df})


    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.crop_model = Crop(param)

    def run(self):
        # -- get inputs
        input_dict = self.get_sosdisc_inputs()

        # -- configure class with inputs
        self.crop_model.configure_parameters_update(input_dict)
        # -- compute
        self.crop_model.compute()



        outputs_dict = {
            'food_land_surface_df': self.crop_model.food_land_surface_df,
            'total_food_land_surface': self.crop_model.total_food_land_surface,
            'food_land_surface_percentage_df': self.crop_model.food_land_surface_percentage_df,
            'updated_diet_df': self.crop_model.updated_diet_df,
            'crop_productivity_evolution': self.crop_model.productivity_evolution,
            'mix_detailed_prices': self.crop_model.mix_detailed_prices,
            'cost_details': self.crop_model.cost_details,
            'mix_detailed_production': self.crop_model.mix_detailed_production,
            'techno_production': self.crop_model.techno_production,
            'techno_prices': self.crop_model.techno_prices,
            'land_use_required': self.crop_model.land_use_required,
            'techno_consumption': self.crop_model.techno_consumption,
            'techno_consumption_woratio': self.crop_model.techno_consumption_woratio,
            'CO2_emissions': self.crop_model.CO2_emissions,
            'CO2_land_emission_df': self.crop_model.CO2_land_emissions,
            'CO2_land_emission_detailed': self.crop_model.CO2_land_emissions_detailed,
            'CH4_land_emission_df': self.crop_model.CH4_land_emissions,
            'CH4_land_emission_detailed': self.crop_model.CH4_land_emissions_detailed,
            'N2O_land_emission_df': self.crop_model.N2O_land_emissions,
            'N2O_land_emission_detailed': self.crop_model.N2O_land_emissions_detailed,
            'calories_per_day_constraint': self.crop_model.calories_per_day_constraint,
            GlossaryCore.CaloriesPerCapitaValue: self.crop_model.calories_pc_df,
            GlossaryCore.CaloriesPerCapitaBreakdownValue: self.crop_model.consumed_calories_pc_breakdown_per_day_df,
            GlossaryCore.OrganicWasteValue: self.crop_model.organic_waste_df
        }
        
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        inputs_dict = deepcopy(self.get_sosdisc_inputs())
        population_df = inputs_dict[GlossaryCore.PopulationDfValue]
        temperature_df = inputs_dict[GlossaryCore.TemperatureDfValue]
        scaling_factor_crop_investment = inputs_dict['scaling_factor_crop_investment']
        scaling_factor_techno_production = inputs_dict['scaling_factor_techno_production']
        density_per_ha = inputs_dict['techno_infos_dict']['density_per_ha']
        residue_density_percentage = inputs_dict['techno_infos_dict']['residue_density_percentage']
        calorific_value = inputs_dict['data_fuel_dict']['calorific_value']
        CO2_from_production = inputs_dict['techno_infos_dict']['CO2_from_production']
        high_calorific_value = inputs_dict['data_fuel_dict']['high_calorific_value']
        food_waste_percentage = inputs_dict[GlossaryCore.FoodWastePercentageValue][GlossaryCore.FoodWastePercentageValue].values
        model = self.crop_model

        # get variable
        food_land_surface_df = model.food_land_surface_df

        # get column of interest
        food_land_surface_df_columns = list(food_land_surface_df)
        if GlossaryCore.Years in food_land_surface_df_columns:
            food_land_surface_df_columns.remove(GlossaryCore.Years)
        food_land_surface_df_columns.remove('total surface (Gha)')

        # sum is needed to have d_total_surface_d_population
        summ = np.identity(len(food_land_surface_df.index)) * 0
        for column_name in food_land_surface_df_columns:
            result = model.d_land_surface_d_population(column_name)
            summ += result

        d_total_d_temperature = model.d_food_land_surface_d_temperature(
            temperature_df, 'total surface (Gha)')


        d_surface_d_red_meat_percentage = model.d_surface_d_calories(population_df, 'red meat')
        d_surface_d_white_meat_percentage = model.d_surface_d_calories(population_df, 'white meat')
        d_surface_d_fish_percentage = model.d_surface_d_calories(population_df, GlossaryCore.Fish)
        d_surface_d_other_food_percentage = model.d_surface_d_calories(population_df, GlossaryCore.OtherFood)

        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            summ)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            d_total_d_temperature)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('red_meat_calories_per_day', 'red_meat_calories_per_day'), d_surface_d_red_meat_percentage)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('white_meat_calories_per_day', 'white_meat_calories_per_day'), d_surface_d_white_meat_percentage)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            (GlossaryCore.FishDailyCal, GlossaryCore.FishDailyCal), d_surface_d_fish_percentage)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            (GlossaryCore.OtherDailyCal, GlossaryCore.OtherDailyCal), d_surface_d_other_food_percentage)

        l_years = len(model.years)

        d_surface_d_vegetables_carbs = model.d_surface_d_vegetables_carbs_calories_per_day(population_df)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
            d_surface_d_vegetables_carbs)

        d_surface_d_eggs_milk = model.d_surface_d_eggs_milk_calories_per_day(population_df)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'), d_surface_d_eggs_milk)

        food_types = [
            'vegetables_and_carbs_calories_per_day',
            'red_meat_calories_per_day',
            'white_meat_calories_per_day',
            GlossaryCore.FishDailyCal,
            GlossaryCore.OtherDailyCal,
            'milk_and_eggs_calories_per_day',
        ]

        for food_type in food_types:
            self.set_partial_derivative_for_other_types(
                ('calories_per_day_constraint',),
                (food_type, food_type),
                np.identity(l_years) * (1 -  food_waste_percentage /100.) / self.crop_model.constraint_calories_ref)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CaloriesPerCapitaValue, 'kcal_pc'),
                (food_type, food_type),
                np.identity(l_years) * (1 -  food_waste_percentage /100.))


        # gradients for techno_production from total food land surface
        d_prod_dpopulation = model.compute_d_prod_dland_for_food(summ)
        d_prod_dtemperature = model.compute_d_prod_dland_for_food(d_total_d_temperature)
        d_prod_dred_to_white = model.compute_d_prod_dland_for_food(d_surface_d_red_meat_percentage)
        d_prod_dmeat_to_vegetable = model.compute_d_prod_dland_for_food(d_surface_d_white_meat_percentage)
        d_prod_dfish_cal = model.compute_d_prod_dland_for_food(d_surface_d_fish_percentage)
        d_prod_dother_cal = model.compute_d_prod_dland_for_food(d_surface_d_other_food_percentage)
        d_prod_dveg_carbs_cal = model.compute_d_prod_dland_for_food(d_surface_d_vegetables_carbs)
        dprod_deggs_milk_cal = model.compute_d_prod_dland_for_food(d_surface_d_eggs_milk)
        # --------------------------------------------------------------
        # Techno production gradients
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
                                                    d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            ('red_meat_calories_per_day', 'red_meat_calories_per_day'),
            d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            ('white_meat_calories_per_day', 'white_meat_calories_per_day'),
            d_prod_dmeat_to_vegetable)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
            d_prod_dveg_carbs_cal)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
            dprod_deggs_milk_cal)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            (GlossaryCore.FishDailyCal, GlossaryCore.FishDailyCal),
            d_prod_dfish_cal)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            (GlossaryCore.OtherDailyCal, GlossaryCore.OtherDailyCal),
            d_prod_dother_cal)

        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'),
            ('crop_investment', GlossaryCore.InvestmentsValue),
            dprod_dinvest * scaling_factor_crop_investment * calorific_value / scaling_factor_techno_production)
        # --------------------------------------------------------------
        # Techno consumption gradients
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            -CO2_from_production / high_calorific_value * d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), -CO2_from_production / high_calorific_value * d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            ('red_meat_calories_per_day', 'red_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            ('white_meat_calories_per_day', 'white_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dmeat_to_vegetable)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dveg_carbs_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
            -CO2_from_production / high_calorific_value * dprod_deggs_milk_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            (GlossaryCore.FishDailyCal, GlossaryCore.FishDailyCal),
            -CO2_from_production / high_calorific_value * d_prod_dfish_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            (GlossaryCore.OtherDailyCal, GlossaryCore.OtherDailyCal),
            -CO2_from_production / high_calorific_value * d_prod_dother_cal)

        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'),
            ('crop_investment', GlossaryCore.InvestmentsValue),
            -CO2_from_production / high_calorific_value *dprod_dinvest * scaling_factor_crop_investment * calorific_value / scaling_factor_techno_production)

        # --------------------------------------------------------------
        # Techno consumption wo ratio gradients
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            -CO2_from_production / high_calorific_value * d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            -CO2_from_production / high_calorific_value * d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            ('red_meat_calories_per_day', 'red_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            ('white_meat_calories_per_day', 'white_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dmeat_to_vegetable)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dveg_carbs_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
            -CO2_from_production / high_calorific_value * dprod_deggs_milk_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            (GlossaryCore.FishDailyCal, GlossaryCore.FishDailyCal),
            -CO2_from_production / high_calorific_value * d_prod_dfish_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            (GlossaryCore.OtherDailyCal, GlossaryCore.OtherDailyCal),
            -CO2_from_production / high_calorific_value * d_prod_dother_cal)

        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio', 'CO2_resource (Mt)'),
            ('crop_investment', GlossaryCore.InvestmentsValue),
            -CO2_from_production / high_calorific_value * dprod_dinvest * scaling_factor_crop_investment * calorific_value / scaling_factor_techno_production)

        # gradient for land demand
        self.set_partial_derivative_for_other_types(
            ('land_use_required', 'Crop (Gha)'),
            ('crop_investment', GlossaryCore.InvestmentsValue),
            dprod_dinvest * scaling_factor_crop_investment * (
                        1 - residue_density_percentage) / density_per_ha * calorific_value)

        vegetables_column_names = ['fruits and vegetables', 'cereals', 'rice and maize']

        # gradient for emissions. Specific case of Fish. Needs be computed only once for all ghg
        dghg_fish_dpop = model.d_ghg_fish_emission_d_population()
        dghg_fish_dkcal = model.d_ghg_fish_emission_d_fish_kcal()

        # gradient for land emissions
        for ghg in [GlossaryCore.CO2, GlossaryCore.CH4, GlossaryCore.N2O]:
            dco2_dpop = 0.0
            dco2_dtemp = 0.0
            dco2_dred_meat = 0.0
            dco2_dwhite_meat = 0.0
            dco2_dfish = dghg_fish_dkcal[ghg]
            dco2_dinvest = 0.0
            dco2_dother_cal = 0.0
            dco2_dveg_cal = 0.0
            dco2_deggs_milk = 0.0

            for food in self.get_sosdisc_inputs('co2_emissions_per_kg'):
                food = food.replace(' (Gt)', '')

                # particular processing of Fish since no land surface is associated but ghg are emitted by this sector
                if food == GlossaryCore.Fish:
                    dco2_dpop += dghg_fish_dpop[ghg]

                elif food != GlossaryCore.Years:

                    dland_emissions_dfood_land_surface = model.compute_dland_emissions_dfood_land_surface_df(
                        food)

                    dland_dpop = model.d_land_surface_d_population(
                            f'{food} (Gha)')

                    dland_dtemp = model.d_food_land_surface_d_temperature(
                        temperature_df, f'{food} (Gha)')
                    d_food_surface_d_red_meat_percentage = \
                        model.compute_d_food_surface_d_red_meat_percentage(
                            population_df, food)
                    d_food_surface_d_white_meat_percentage = \
                        model.compute_d_food_surface_d_white_meat_percentage(
                            population_df, food)
                    d_food_surface_d_fish_percentage = \
                        model.compute_d_food_surface_d_fish_percentage(
                            population_df, food)
                    d_food_surface_d_other_food_percentage = \
                        model.compute_d_food_surface_d_other_food_percentage(
                            population_df, food)

                    dco2_dpop += dland_emissions_dfood_land_surface[ghg] * dland_dpop
                    dco2_dfish += dland_emissions_dfood_land_surface[ghg] * \
                                        d_food_surface_d_fish_percentage
                    dco2_dtemp += dland_emissions_dfood_land_surface[ghg] * dland_dtemp
                    dco2_dred_meat += dland_emissions_dfood_land_surface[ghg] * \
                                      d_food_surface_d_red_meat_percentage
                    dco2_dwhite_meat += dland_emissions_dfood_land_surface[ghg] * \
                                        d_food_surface_d_white_meat_percentage
                    dco2_dother_cal += dland_emissions_dfood_land_surface[ghg] * \
                                  d_food_surface_d_other_food_percentage

                    if food == 'rice and maize':
                        dco2_dinvest += dland_emissions_dfood_land_surface[ghg] * dprod_dinvest * \
                                        scaling_factor_crop_investment * (1 - residue_density_percentage) / \
                                        density_per_ha * calorific_value
            for food in vegetables_column_names:
                dland_emissions_dfood_land_surface_oth = model.compute_dland_emissions_dfood_land_surface_df(
                    food)[ghg]
                dco2_dveg_cal += dland_emissions_dfood_land_surface_oth * model.d_surface_d_other_calories_percentage(
                    population_df, food)

            for food in ['eggs', 'milk']:
                dland_emissions_dfood_land_surface_oth = model.compute_dland_emissions_dfood_land_surface_df(
                    food)[ghg]
                dco2_deggs_milk += dland_emissions_dfood_land_surface_oth * model.d_surface_d_milk_eggs_calories_percentage(
                    population_df, food)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
                dco2_dpop)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
                dco2_dtemp)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('red_meat_calories_per_day', 'red_meat_calories_per_day'),
                dco2_dred_meat)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('white_meat_calories_per_day', 'white_meat_calories_per_day'),
                dco2_dwhite_meat)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                (GlossaryCore.FishDailyCal, GlossaryCore.FishDailyCal),
                dco2_dfish)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                (GlossaryCore.OtherDailyCal, GlossaryCore.OtherDailyCal),
                dco2_dother_cal)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
                dco2_dveg_cal)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
                dco2_deggs_milk)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('crop_investment', GlossaryCore.InvestmentsValue),
                dco2_dinvest)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []
        chart_list = [
            CropDiscipline.CROP_CHARTS, 'Crop Productivity Evolution', 'Crop Energy', 'Crop Emissions']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        '''
        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then
        '''
        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values
        year_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
        year_end = self.get_sosdisc_inputs(GlossaryCore.YearEnd)
        if CropDiscipline.CROP_CHARTS in chart_list:

            surface_df = self.get_sosdisc_outputs('food_land_surface_df')
            years = surface_df[GlossaryCore.Years].values.tolist()

            crop_surfaces = surface_df['total surface (Gha)'].values
            crop_surface_series = InstanciatedSeries(
                years, crop_surfaces.tolist(), 'Total crop surface', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []

            for key in surface_df.keys():

                if key == GlossaryCore.Years:
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, (surface_df[key]).values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'surface [Gha]',
                                                 chart_name='Surface taken to produce food over time', stacked_bar=True)
            new_chart.add_series(crop_surface_series)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # chart of land surface in %
            surface_percentage_df = self.get_sosdisc_outputs(
                'food_land_surface_percentage_df')

            series_to_add = []
            for key in surface_percentage_df.keys():

                if key == GlossaryCore.Years:
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, surface_percentage_df[key].values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'surface [%]',
                                                 chart_name='Share of the surface used to produce food over time',
                                                 stacked_bar=True)
            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(
                years, surface_percentage_df[key].values.tolist() * 0, '', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)
            # -------------------------- chart of the updated diet
            updated_diet_df = self.get_sosdisc_outputs('updated_diet_df')
            starting_diet = self.get_sosdisc_inputs('diet_df')
            kg_to_kcal_dict = self.get_sosdisc_inputs('kg_to_kcal_dict')
            total_kcal = 0
            # compute total kcal
            for key in starting_diet:
                total_kcal += starting_diet[key].values[0] * \
                              kg_to_kcal_dict[key]

            series_to_add = []
            for key in updated_diet_df.keys():

                if key == GlossaryCore.Years:
                    pass
                elif key.startswith('total'):
                    pass
                else:
                    l_values = updated_diet_df[key].values / 365 * kg_to_kcal_dict[key]
                    new_series = InstanciatedSeries(
                        years, l_values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'food calories [kcal / person / day]',
                                                 chart_name='Food calories produced per person (before waste)',
                                                 stacked_bar=True)

            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(years, surface_percentage_df[key].values.tolist() * 0, '',
                                            InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)
            # -------------------------- chart of the consumed calories
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'kcal / person / day',
                                                 chart_name='Food calories consumed per person (production - waste)',
                                                 stacked_bar=True)

            consumed_breakdown_pc_per_day = self.get_sosdisc_outputs(GlossaryCore.CaloriesPerCapitaBreakdownValue)
            years = list(consumed_breakdown_pc_per_day[GlossaryCore.Years].values)
            total_consumed_pc_per_day = self.get_sosdisc_outputs(GlossaryCore.CaloriesPerCapitaValue)[
                "kcal_pc"].values.tolist()
            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(years, surface_percentage_df[key].values.tolist() * 0, '',
                                            InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(fake_serie)
            for key in consumed_breakdown_pc_per_day.columns[1:]:
                l_values = consumed_breakdown_pc_per_day[key]
                new_series = InstanciatedSeries(
                    years, l_values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, total_consumed_pc_per_day, "Total", InstanciatedSeries.LINES_DISPLAY)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # ------------------------------------------
            # -------------------------- chart of constraint
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'kcal / person / day',
                                                 chart_name='Minimum daily kcalories constraint',
                                                 stacked_bar=True)

            total_consumed_pc_per_day_df = self.get_sosdisc_outputs(GlossaryCore.CaloriesPerCapitaValue)
            total_consumed_pc_per_day = total_consumed_pc_per_day_df["kcal_pc"].values
            food_waste_percentage = self.get_sosdisc_inputs(GlossaryCore.FoodWastePercentageValue)[
                                        GlossaryCore.FoodWastePercentageValue] / 100.
            produced_food = total_consumed_pc_per_day / (1 - food_waste_percentage)
            wasted_food = produced_food - total_consumed_pc_per_day
            years = list(total_consumed_pc_per_day_df[GlossaryCore.Years].values)
            minimum_kcal_daily_apport = self.get_sosdisc_inputs('constraint_calories_limit')

            new_series = InstanciatedSeries(
                years, total_consumed_pc_per_day.tolist(), "Total consumed", InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                years, wasted_food.tolist(), "Wasted food", InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                years, [minimum_kcal_daily_apport] * len(years), "Minimum daily kcalories",
                InstanciatedSeries.DASH_LINES_DISPLAY)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        ##################### Kcal per kg per category #################

        list_categories = ['red meat', 'white meat', 'eggs_milk', 'vegetables_and_carbs', GlossaryCore.Fish, GlossaryCore.OtherFood]

        co2_gwp_100 = 1.0
        ch4_gwp_100 = 28.0
        n2o_gwp_100 = 265.0

        kg_to_m2_dict, co2_emissions_per_kg, ch4_emissions_per_kg, n2o_emissions_per_kg = self.get_sosdisc_inputs(
            ['kg_to_m2_dict',
             'co2_emissions_per_kg', 'ch4_emissions_per_kg', 'n2o_emissions_per_kg'])
        kg_to_kcal_mean_dict = {'vegetables_and_carbs': 0, 'eggs_milk': 0}
        m2_per_kcal_dict = {'vegetables_and_carbs': 0, 'eggs_milk': 0}
        co2_emissions_per_kcal_dict_mean = {'vegetables_and_carbs': 0, 'eggs_milk': 0}
        ch4_emissions_per_kcal_dict_mean = {'vegetables_and_carbs': 0, 'eggs_milk': 0}
        n2o_emissions_per_kcal_dict_mean = {'vegetables_and_carbs': 0, 'eggs_milk': 0}

        ghg_emissions_per_kcal = {}
        starting_diet = self.get_sosdisc_inputs('diet_df')
        kg_to_kcal_dict = self.get_sosdisc_inputs('kg_to_kcal_dict')
        for key in starting_diet:
            if key == 'fruits and vegetables' or key == 'cereals' or key == 'rice and maize':

                proportion = starting_diet[key].values[0] / \
                             (starting_diet['fruits and vegetables'].values[0] + starting_diet['cereals'].values[0] +
                              starting_diet['rice and maize'].values[0])

                kg_to_kcal_mean_dict['vegetables_and_carbs'] += proportion * kg_to_kcal_dict[key]
                m2_per_kcal_dict['vegetables_and_carbs'] += proportion * kg_to_m2_dict[key] / kg_to_kcal_dict[key]
                co2_emissions_per_kcal_dict_mean['vegetables_and_carbs'] += proportion * co2_emissions_per_kg[key] / \
                                                                            kg_to_kcal_dict[key]
                ch4_emissions_per_kcal_dict_mean['vegetables_and_carbs'] += ch4_gwp_100 * proportion * \
                                                                            ch4_emissions_per_kg[key] / kg_to_kcal_dict[
                                                                                key]
                n2o_emissions_per_kcal_dict_mean['vegetables_and_carbs'] += n2o_gwp_100 * proportion * \
                                                                            n2o_emissions_per_kg[key] / kg_to_kcal_dict[
                                                                                key]

            elif key == 'eggs' or key == 'milk':
                proportion = starting_diet[key].values[0] / (
                            starting_diet['eggs'].values[0] + starting_diet['milk'].values[0])
                kg_to_kcal_mean_dict['eggs_milk'] += proportion * kg_to_kcal_dict[key]
                m2_per_kcal_dict['eggs_milk'] += proportion * kg_to_m2_dict[key] / kg_to_kcal_dict[key]
                co2_emissions_per_kcal_dict_mean['eggs_milk'] += proportion * co2_emissions_per_kg[key] / \
                                                                 kg_to_kcal_dict[key]
                ch4_emissions_per_kcal_dict_mean['vegetables_and_carbs'] += ch4_gwp_100 * proportion * \
                                                                            ch4_emissions_per_kg[key] / kg_to_kcal_dict[
                                                                                key]
                n2o_emissions_per_kcal_dict_mean['vegetables_and_carbs'] += n2o_gwp_100 * proportion * \
                                                                            n2o_emissions_per_kg[key] / kg_to_kcal_dict[
                                                                                key]

        for food in ['red meat', 'white meat', GlossaryCore.Fish, GlossaryCore.OtherFood]:
            kg_to_kcal_mean_dict[food] = kg_to_kcal_dict[food]
            m2_per_kcal_dict[food] = kg_to_m2_dict[food] / kg_to_kcal_dict[food]
            co2_emissions_per_kcal_dict_mean[food] = co2_emissions_per_kg[food] / kg_to_kcal_dict[food]
            ch4_emissions_per_kcal_dict_mean[food] = ch4_gwp_100 * ch4_emissions_per_kg[food] / kg_to_kcal_dict[
                food]
            n2o_emissions_per_kcal_dict_mean[food] = n2o_gwp_100 * n2o_emissions_per_kg[food] / kg_to_kcal_dict[
                food]


        for category in list_categories:
            ghg_emissions_per_kcal[category] = co2_emissions_per_kcal_dict_mean[category] + \
                                               ch4_emissions_per_kcal_dict_mean[category] + \
                                               n2o_emissions_per_kcal_dict_mean[category]

        ##################### m2/kcal ############

        chart_name = "m2 per kcal per category"

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'm2 per category per kcal [m2/kcal]',
                                             chart_name=chart_name)

        for key in list_categories:
            new_series = InstanciatedSeries(
                [key], [m2_per_kcal_dict[key]], f'm2 per kcal of category {key}', 'bar')
            new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

        ################## ghg total ###############
        chart_name = "ghg emissions per kcal per category"

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'ghg per category per kcal [co2eq/kcal]',
                                             chart_name=chart_name)

        for key in list_categories:
            new_series = InstanciatedSeries(
                [key], [ghg_emissions_per_kcal[key]], f'co2eq per kcal of category {key}', 'bar')
            new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

        if 'Crop Productivity Evolution' in chart_list:
            prod_df = self.get_sosdisc_outputs(
                'crop_productivity_evolution')
            years = list(prod_df[GlossaryCore.Years])

            chart_name = 'Crop productivity evolution'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' productivity evolution [%]',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(prod_df['productivity_evolution'] * 100)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'productivity_evolution', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Crop Energy' in chart_list:

            mix_detailed_production = deepcopy(self.get_sosdisc_outputs(
                'mix_detailed_production'))
            land_use_required = deepcopy(self.get_sosdisc_outputs(
                'land_use_required'))
            mix_detailed_prices = deepcopy(
                self.get_sosdisc_outputs('mix_detailed_prices'))
            data_fuel_dict = deepcopy(
                self.get_sosdisc_inputs('data_fuel_dict'))
            cost_details = deepcopy(self.get_sosdisc_outputs('cost_details'))
            crop_investment = deepcopy(self.get_sosdisc_inputs(
                'crop_investment') * self.get_sosdisc_inputs('scaling_factor_crop_investment'))
            years = list(prod_df[GlossaryCore.Years])

            # ------------------------------------------
            # INVEST (M$)
            chart_name = 'Input investments over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Investments [M$]',
                                                 chart_name=chart_name)

            visible_line = True

            for investment in crop_investment:
                if investment != GlossaryCore.Years:
                    ordonate_data = list(crop_investment[investment])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, investment, 'bar', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (Mt)
            chart_name = 'Crop for Energy production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop mass for energy production [Mt]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != GlossaryCore.Years:
                    ordonate_data = list(
                        mix_detailed_production[crop] * data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(TWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (TWh)
            chart_name = 'Crop for Energy production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop for Energy production [TWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != GlossaryCore.Years:
                    ordonate_data = list(mix_detailed_production[crop])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(TWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # LAND USE (Gha)
            chart_name = 'Land demand for Crop for Energy'

            land_demand_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Land demand [Gha]',
                                                         chart_name=chart_name)
            ordonate_data = list(land_use_required['Crop (Gha)'])
            land_demand_serie = InstanciatedSeries(
                years, ordonate_data, 'Crop', 'lines', visible_line)
            land_demand_chart.series.append(land_demand_serie)

            instanciated_charts.append(land_demand_chart)
            # ------------------------------------------
            # PRICE ($/MWh)
            chart_name = 'Crop energy prices by type'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop prices [$/MWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_prices:
                if crop != GlossaryCore.Years:
                    ordonate_data = list(
                        mix_detailed_prices[crop] / data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("($/t)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # PRICE DETAILED($/MWh)
            chart_name = 'Detailed prices for crop energy production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop prices [$/MWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for price in cost_details:
                if 'Transport' in price or 'Factory' in price or 'Total ($/MWh)' in price:
                    ordonate_data = list(cost_details[price])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, price.replace("($/MWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRICE ($/t)
            chart_name = 'Crop mass prices'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop prices [$/t]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_prices:
                if crop != GlossaryCore.Years:
                    ordonate_data = list(mix_detailed_prices[crop])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("($/t)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # PRICE DETAILED($/t)
            chart_name = 'Detailed prices for crop energy production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Crop prices [$/t]',
                                                 chart_name=chart_name)

            visible_line = True

            for price in cost_details:
                if 'Transport' in price or 'Factory' in price or 'Total ($/MWh)' in price:
                    ordonate_data = list(
                        cost_details[price] * data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, price.replace("($/MWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Crop Emissions' in chart_list:

            # total emissions + by food
            co2_emissions_df = self.get_sosdisc_outputs('CO2_land_emission_df')
            co2_emissions_detailed_df = self.get_sosdisc_outputs(
                'CO2_land_emission_detailed')
            ch4_emissions_df = self.get_sosdisc_outputs('CH4_land_emission_df')
            ch4_emissions_detailed_df = self.get_sosdisc_outputs(
                'CH4_land_emission_detailed')
            n2o_emissions_df = self.get_sosdisc_outputs('N2O_land_emission_df')
            n2o_emissions_detailed_df = self.get_sosdisc_outputs(
                'N2O_land_emission_detailed')
            years = co2_emissions_df[GlossaryCore.Years].values.tolist()

            co2_crop_emissions = co2_emissions_df['emitted_CO2_evol_cumulative'].values
            co2_crop_emissions_series = InstanciatedSeries(
                years, co2_crop_emissions.tolist(), 'Total Crop CO2 Emissions', InstanciatedSeries.BAR_DISPLAY)

            ch4_crop_emissions = ch4_emissions_df['emitted_CH4_evol_cumulative'].values
            ch4_crop_emissions_series = InstanciatedSeries(
                years, ch4_crop_emissions.tolist(), 'Total Crop CH4 Emissions', InstanciatedSeries.BAR_DISPLAY)

            n2o_crop_emissions = n2o_emissions_df['emitted_N2O_evol_cumulative'].values
            n2o_crop_emissions_series = InstanciatedSeries(
                years, n2o_crop_emissions.tolist(), 'Total Crop N2O Emissions', InstanciatedSeries.BAR_DISPLAY)

            # total emissions
            series_to_add = [co2_crop_emissions_series,
                             ch4_crop_emissions_series, n2o_crop_emissions_series]

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Greenhouse Gas Emissions [Gt]',
                                                 chart_name='Greenhouse Gas Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # total emissions GWP

            co2_crop_emissions = co2_emissions_df['emitted_CO2_evol_cumulative'].values * self.co2_gwp_100
            co2_crop_emissions_series = InstanciatedSeries(
                years, co2_crop_emissions.tolist(), 'Total Crop CO2 Emissions [GtCO2eq]',
                InstanciatedSeries.BAR_DISPLAY)

            ch4_crop_emissions = ch4_emissions_df['emitted_CH4_evol_cumulative'].values * self.ch4_gwp_100
            ch4_crop_emissions_series = InstanciatedSeries(
                years, ch4_crop_emissions.tolist(), 'Total Crop CH4 Emissions [GtCO2eq]',
                InstanciatedSeries.BAR_DISPLAY)

            n2o_crop_emissions = n2o_emissions_df['emitted_N2O_evol_cumulative'].values * self.n2o_gwp_100
            n2o_crop_emissions_series = InstanciatedSeries(
                years, n2o_crop_emissions.tolist(), 'Total Crop N2O Emissions [GtCO2eq]',
                InstanciatedSeries.BAR_DISPLAY)

            series_to_add = [co2_crop_emissions_series,
                             ch4_crop_emissions_series, n2o_crop_emissions_series]

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Greenhouse Gas Emissions [GtCO2eq]',
                                                 chart_name='Greenhouse Gas CO2 Eq. Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # CO2 per food
            series_to_add = []
            for key in co2_emissions_detailed_df.columns:

                if key != GlossaryCore.Years:
                    new_series = InstanciatedSeries(
                        years, (co2_emissions_detailed_df[key]).values.tolist(), key.replace(' (Gt)', ''),
                        InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 Emissions [Gt]',
                                                 chart_name='CO2 Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # CH4 per food
            series_to_add = []
            for key in ch4_emissions_detailed_df.columns:

                if key != GlossaryCore.Years:
                    new_series = InstanciatedSeries(
                        years, (ch4_emissions_detailed_df[key]).values.tolist(), key.replace(' (Gt)', ''),
                        InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CH4 Emissions [Gt]',
                                                 chart_name='CH4 Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # N2O per food
            series_to_add = []
            for key in n2o_emissions_detailed_df.columns:

                if key != GlossaryCore.Years:
                    new_series = InstanciatedSeries(
                        years, (n2o_emissions_detailed_df[key]).values.tolist(
                        ), key.replace(' (Gt)', ''),
                        InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'N2O Emissions [Gt]',
                                                 chart_name='N2O Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # CO2 emissions by food (pie chart)
            years_list = [year_start, year_end]
            food_names = co2_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [co2_emissions_detailed_df.loc[co2_emissions_detailed_df[GlossaryCore.Years]
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'CO2 Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

            # CH4 emissions by food (pie chart)
            years_list = [year_start, year_end]
            food_names = ch4_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [ch4_emissions_detailed_df.loc[ch4_emissions_detailed_df[GlossaryCore.Years]
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'CH4 Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

            # N2O emissions by food (pie chart)
            years_list = [year_start, year_end]
            food_names = n2o_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [n2o_emissions_detailed_df.loc[n2o_emissions_detailed_df[GlossaryCore.Years]
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'N2O Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

        return instanciated_charts
