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
import matplotlib.pyplot as plt

from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from energy_models.core.stream_type.carbon_models.carbon_dioxyde import CO2
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from climateeconomics.core.core_agriculture.crop import Crop, \
    OrderOfMagnitude
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.pie_charts.instanciated_pie_chart import InstanciatedPieChart
from plotly import graph_objects as go
import numpy as np
import pandas as pd
from copy import deepcopy

from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


class CropDiscipline(ClimateEcoDiscipline):
    ''' Crop discipline transforms crops and crops residues
        into biomass_dry resource
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
    default_year_start = 2020
    default_year_end = 2050
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    default_kg_to_m2 = {'red meat': 348,
                        'white meat': 14.5,
                        'milk': 8.9,
                        'eggs': 6.3,
                        'rice and maize': 2.9,
                        'potatoes': 0.9,
                        'fruits and vegetables': 0.8,
                        'other': 21.4,
                        }
    default_kg_to_kcal = {'red meat': 2566,
                          'white meat': 1860,
                          'milk': 550,
                          'eggs': 1500,
                          'rice and maize': 1150,
                          'potatoes': 670,
                          'fruits and vegetables': 624,
                          }

    # Our World in Data (emissions per kg of food product)
    # https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions
    # FAO Stats (total production)
    # https://www.fao.org/faostat/en/#data/FBS

    co2_gwp_100 = 1.0
    ch4_gwp_100 = 28.0
    n2o_gwp_100 = 265.0

    ghg_emissions_unit = 'kg/kg'  # in kgCo2eq per kg of food
    default_ghg_emissions = {'red meat': 32.7,
                             'white meat': 4.09,
                             'milk': 1.16,
                             'eggs': 1.72,
                             'rice and maize': 1.45,
                             'potatoes': 0.170,
                             'fruits and vegetables': 0.372,
                             'other': 3.44,
                             }

    # Our World in Data
    # https://ourworldindata.org/carbon-footprint-food-methane
    ch4_emissions_unit = 'kg/kg'  # in kgCH4 per kg food
    calibration = 0.134635 / 0.108958
    # set up as a ratio of total ghg emissions
    ch4_emissions_ratios = {'red meat': 49 / 100 * calibration,
                            'white meat': 2 / 20 * calibration,
                            'milk': 17.0 / 33 * calibration,
                            'eggs': 0.0 * calibration,
                            'rice and maize': 4 / 6.5 * calibration,
                            'potatoes': 0.0 * calibration,
                            # negligible methane in this category
                            'fruits and vegetables': 0.0 * calibration,
                            'other': (0.0 + 0.0 + 11.0 + 4.0 + 5.0 + 17.0) / (14 + 24 + 33 + 27 + 29 + 34) * calibration,
                            }

    default_ch4_emissions = {}
    for food in default_ghg_emissions:
        default_ch4_emissions[food] = (
            default_ghg_emissions[food] * ch4_emissions_ratios[food] / ch4_gwp_100)

    # FAO Stats
    # https://www.fao.org/faostat/en/#data/GT
    n2o_emissions_unit = 'kg/kg'  # in kgN2O per kg food$
    calibration = 7.332 / 6.0199
    pastures_emissions = 3.039e-3 * calibration
    crops_emissions = 1.504e-3 * calibration

    # with land use ratio on n2o emissions
    default_n2o_emissions = {'red meat': pastures_emissions * 3.0239241372696104 / 0.9959932034220041,
                             'white meat': pastures_emissions * 0.3555438662130599 / 0.9959932034220041,
                             'milk': pastures_emissions * 0.5564085980770741 / 0.9959932034220041,
                             'eggs': pastures_emissions * 0.048096212128271996 / 0.9959932034220041,
                             'rice and maize': crops_emissions * 0.2236252183903196 / 0.29719264680276536,
                             'potatoes': crops_emissions * 0.023377379498821543 / 0.29719264680276536,
                             'fruits and vegetables': crops_emissions * 0.13732524416192043 / 0.29719264680276536,
                             'other': crops_emissions * 0.8044427451599999 / 0.29719264680276536,
                             }

    # co2 emissions = (total_emissions - ch4_emissions * ch4_gwp_100 -
    # n2o_emissions * n2o_gwp_100)/co2_gwp_100
    co2_emissions_unit = 'kg/kg'  # in kgCo2 per kg food
    calibration = 0.722 / 3.417569
    default_co2_emissions = {'red meat': 0.0 * calibration,
                             'white meat': 0.0 * calibration,
                             'milk': 0.0 * calibration,
                             'eggs': 0.0 * calibration,
                             'rice and maize': 1.45 * calibration,
                             'potatoes': 0.170 * calibration,
                             'fruits and vegetables': 0.372 * calibration,
                             'other': 3.44 * calibration,
                             }

    # Difference method
    # default_co2_emissions = {}
    # for food in default_ghg_emissions:
    #     default_co2_emissions[food] = (default_ghg_emissions[food] - default_ch4_emissions[food]*ch4_gwp_100 - default_n2o_emissions[food]*n2o_gwp_100) / co2_gwp_100
    #     # default_co2_emissions[food] = 0.0

    year_range = default_year_end - default_year_start + 1
    total_kcal = 414542.4
    red_meat_percentage = default_kg_to_kcal['red meat'] / total_kcal * 100
    white_meat_percentage = default_kg_to_kcal['white meat'] / total_kcal * 100
    default_red_meat_percentage = pd.DataFrame({
        'years': default_years,
        'red_meat_percentage': np.linspace(red_meat_percentage, 0.3 * red_meat_percentage, year_range)})
    default_white_meat_percentage = pd.DataFrame({
        'years': default_years,
        'white_meat_percentage': np.linspace(white_meat_percentage, 0.3 * white_meat_percentage, year_range)})

    # mdpi: according to the NASU recommendations,
    # a fixed value of 0.25 is applied to all crops
    # 50% of crops are left on the field,
    # 50% of the left on the field can be used as crop residue =>
    # 25% of the crops is residue
    residue_percentage = 0.25
    # 23$/t for residue, 60$/t for crop
    crop_residue_price_percent_dif = 23 / 60
    # bioenergyeurope.org : Dedicated energy crops
    # represent 0.1% of the total biomass production in 2018
    energy_crop_percentage = 0.005
    # ourworldindata, average cereal yield: 4070kg/ha +
    # average yield of switchgrass on grazing lands: 2565,67kg/ha
    # residue is 0.25 more than that
    density_per_ha = 2903 * 1.25
    # available ha of crop: 4.9Gha, initial prod = crop energy + residue for
    # energy of all surfaces
    initial_production = 4.8 * density_per_ha * \
        3.6 * energy_crop_percentage  # in Twh
    construction_delay = 1  # years
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
        'CO2_from_production':-0.425 * 44.01 / 12.0,
        'CO2_from_production_unit': 'kg/kg',
        'elec_demand': 0,
        'elec_demand_unit': 'kWh/kWh',
        'WACC': 0.07,  # ?
        'lifetime': lifetime,
        'lifetime_unit': 'years',
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
        'residue_percentage_for_energy': 0.05,  # hypothesis
        'efficiency': 1.0,
        'techno_evo_eff': 'no',
        'crop_residue_price_percent_dif': crop_residue_price_percent_dif,
        'construction_delay': construction_delay,  # years
    }

    # Age distribution of forests in 2008 (
    initial_age_distribution = pd.DataFrame({'age': np.arange(1, lifetime),
                                             'distrib': [0.16, 0.24, 0.31, 0.39, 0.47, 0.55, 0.63, 0.71, 0.78, 0.86,
                                                         0.94, 1.02, 1.1, 1.18, 1.26, 1.33, 1.41, 1.49, 1.57, 1.65,
                                                         1.73, 1.81, 1.88, 1.96, 2.04, 2.12, 2.2, 2.28, 2.35, 2.43,
                                                         2.51, 2.59, 2.67, 2.75, 2.83, 2.9, 2.98, 3.06, 3.14, 3.22,
                                                         3.3, 3.38, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.92]})
    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'population_df': {'type': 'dataframe', 'unit': 'millions of people',
                          'dataframe_descriptor': {'years': ('float', None, False),
                                                   'population': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                          'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'diet_df': {'type': 'dataframe', 'unit': 'kg_food/person/year',
                    'dataframe_descriptor': {'years': ('float', None, False),
                                             'red meat': ('float', [0, 1e9], True), 'white meat': ('float', [0, 1e9], True), 'milk': ('float', [0, 1e9], True),
                                             'eggs': ('float', [0, 1e9], True), 'rice and maize': ('float', [0, 1e9], True), 'potatoes': ('float', [0, 1e9], True),
                                             'fruits and vegetables': ('float', [0, 1e9], True)},
                    'dataframe_edition_locked': False, 'namespace': 'ns_crop'},
        'kg_to_kcal_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_kcal, 'unit': 'kcal/kg', 'namespace': 'ns_crop'},
        'kg_to_m2_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_m2, 'unit': 'm^2/kg', 'namespace': 'ns_crop'},
        # design variables of changing diet
        'red_meat_calories_per_day': {'type': 'dataframe', 'default': default_red_meat_percentage,
                                'dataframe_descriptor': {'years': ('float', None, False),
                                                         'red_meat_calories_per_day': ('float', [0, 100], True)},
                                'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'white_meat_calories_per_day': {'type': 'dataframe', 'default': default_white_meat_percentage,
                                  'dataframe_descriptor': {'years': ('float', None, False),
                                                           'white_meat_calories_per_day': ('float', [0, 100], True)},
                                  'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'vegetables_and_carbs_calories_per_day': {'type': 'dataframe', 'default': default_white_meat_percentage,
                                        'dataframe_descriptor': {'years': ('float', None, False),
                                                                 'vegetables_and_carbs_calories_per_day': (
                                                                 'float', [0, 100], True)},
                                        'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                        'namespace': 'ns_crop'},
        'milk_and_eggs_calories_per_day': {'type': 'dataframe', 'default': default_white_meat_percentage,
                                                  'dataframe_descriptor': {'years': ('float', None, False),
                                                                           'milk_and_eggs_calories_per_day': (
                                                                               'float', [0, 100], True)},
                                                  'unit': 'kcal', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                                  'namespace': 'ns_crop'},
        'other_use_crop': {'type': 'array', 'unit': 'ha/person', 'namespace': 'ns_crop'},
        'temperature_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'degree Celsius'},
        'param_a': {'type': 'float', 'default':-0.00833, 'unit': '-', 'user_level': 3},
        'param_b': {'type': 'float', 'default':-0.04167, 'unit': '-', 'user_level': 3},
        'crop_investment': {'type': 'dataframe', 'unit': 'G$',
                            'dataframe_descriptor': {'years': ('int', [1900, 2100], False),
                                                     'investment': ('float', None, True)},
                            'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_crop'},
        'scaling_factor_crop_investment': {'type': 'float', 'default': 1e3, 'unit': '-', 'user_level': 2},
        'scaling_factor_techno_consumption': {'type': 'float', 'default': 1e3, 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public', 'user_level': 2},
        'scaling_factor_techno_production': {'type': 'float', 'default': 1e3, 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public', 'user_level': 2},
        'margin': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'unit': '%', 'namespace': 'ns_witness'},
        'transport_cost': {'type': 'dataframe', 'unit': '$/t', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness',
                           'dataframe_descriptor': {'years': ('int', [1900, 2100], False),
                                                    'transport': ('float', None, True)},
                           'dataframe_edition_locked': False},
        'transport_margin': {'type': 'dataframe', 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness',
                             'dataframe_descriptor': {'years': ('int', [1900, 2100], False),
                                                      'margin': ('float', None, True)},
                             'dataframe_edition_locked': False},
        'data_fuel_dict': {'type': 'dict', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                           'namespace': 'ns_biomass_dry', 'default': BiomassDry.data_energy_dict,
                           'unit': 'defined in dict'},
        'techno_infos_dict': {'type': 'dict', 'unit': 'defined in dict',
                              'default': techno_infos_dict_default},
        'initial_production': {'type': 'float', 'unit': 'TWh', 'default': initial_production},
        'initial_age_distrib': {'type': 'dataframe', 'unit': '%', 'default': initial_age_distribution},
        'co2_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg', 'default': default_co2_emissions},
        'ch4_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg', 'default': default_ch4_emissions},
        'n2o_emissions_per_kg': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': 'kg/kg', 'default': default_n2o_emissions},
        'constraint_calories_ref': {'type': 'float','visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref', 'default': 3400. },
        'constraint_calories_limit': {'type': 'float', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_ref', 'default': 1700.},
    }

    DESC_OUT = {
        'total_food_land_surface': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'food_land_surface_df': {
            'type': 'dataframe', 'unit': 'Gha'},
        'food_land_surface_percentage_df': {'type': 'dataframe', 'unit': '%'},
        'updated_diet_df': {'type': 'dataframe', 'unit': 'kg/person/year'},
        'crop_productivity_evolution': {'type': 'dataframe', 'unit': '%'},
        'mix_detailed_prices': {'type': 'dataframe', 'unit': '$/MWh'},
        'mix_detailed_production': {'type': 'dataframe', 'unit': 'TWh'},
        'cost_details': {'type': 'dataframe', 'unit': '$/MWh'},
        'techno_production': {
            'type': 'dataframe', 'unit': 'TWh or Mt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'techno_prices': {
            'type': 'dataframe', 'unit': '$/MWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'techno_consumption': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop', 'unit': 'TWh or Mt'},
        'techno_consumption_woratio': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop', 'unit': 'TWh or Mt'},
        'land_use_required': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        # emissions from production that goes into energy mix
        'CO2_emissions': {
            'type': 'dataframe', 'unit': 'kg/kWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        # crop land emissions
        'CO2_land_emission_df': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CO2_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CH4_land_emission_df': {'type': 'dataframe', 'unit': 'GtCH4',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'CH4_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtCH4',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'N2O_land_emission_df': {'type': 'dataframe', 'unit': 'GtN2O',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'N2O_land_emission_detailed': {'type': 'dataframe', 'unit': 'GtN2O',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_crop'},
        'calories_per_day_constraint': {'type': 'array', 'unit': 'kcal',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        'calories_pc_df': {'type': 'dataframe', 'unit': 'kcal',
                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }

    CROP_CHARTS = 'crop and diet charts'

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

        # Scale production TWh -> PWh
        techno_production = self.crop_model.mix_detailed_production[[
            'years', 'Total (TWh)']]
        techno_production = techno_production.rename(
            columns={'Total (TWh)': "biomass_dry (TWh)"})
        for column in techno_production.columns:
            if column == 'years':
                continue
            techno_production[column] = techno_production[column].values / \
                input_dict['scaling_factor_techno_production']
        # Scale production Mt -> Gt
        techno_consumption = deepcopy(self.crop_model.techno_consumption)
        techno_consumption_woratio = deepcopy(
            self.crop_model.techno_consumption_woratio)
        for column in techno_consumption.columns:
            if column == 'years':
                continue
            techno_consumption[column] = techno_consumption[column].values / \
                input_dict['scaling_factor_techno_consumption']
            techno_consumption_woratio[column] = techno_consumption_woratio[column].values / \
                input_dict['scaling_factor_techno_consumption']

        outputs_dict = {
            'food_land_surface_df': self.crop_model.food_land_surface_df,
            'total_food_land_surface': self.crop_model.total_food_land_surface,
            'food_land_surface_percentage_df': self.crop_model.food_land_surface_percentage_df,
            'updated_diet_df': self.crop_model.updated_diet_df,
            'crop_productivity_evolution': self.crop_model.productivity_evolution,
            'mix_detailed_prices': self.crop_model.mix_detailed_prices,
            'cost_details': self.crop_model.cost_details,
            'mix_detailed_production': self.crop_model.mix_detailed_production,
            'techno_production': techno_production,
            'techno_prices': self.crop_model.techno_prices,
            'land_use_required': self.crop_model.land_use_required,
            'techno_consumption': techno_consumption,
            'techno_consumption_woratio': techno_consumption_woratio,
            'CO2_emissions': self.crop_model.CO2_emissions,
            'CO2_land_emission_df': self.crop_model.CO2_land_emissions,
            'CO2_land_emission_detailed': self.crop_model.CO2_land_emissions_detailed,
            'CH4_land_emission_df': self.crop_model.CH4_land_emissions,
            'CH4_land_emission_detailed': self.crop_model.CH4_land_emissions_detailed,
            'N2O_land_emission_df': self.crop_model.N2O_land_emissions,
            'N2O_land_emission_detailed': self.crop_model.N2O_land_emissions_detailed,
            'calories_per_day_constraint': self.crop_model.calories_per_day_constraint,
            'calories_pc_df': self.crop_model.calories_pc_df
        }

        # -- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        inputs_dict = deepcopy(self.get_sosdisc_inputs())
        population_df = inputs_dict['population_df']
        temperature_df = inputs_dict['temperature_df']
        scaling_factor_crop_investment = inputs_dict['scaling_factor_crop_investment']
        scaling_factor_techno_production = inputs_dict['scaling_factor_techno_production']
        density_per_ha = inputs_dict['techno_infos_dict']['density_per_ha']
        residue_density_percentage = inputs_dict['techno_infos_dict']['residue_density_percentage']
        calorific_value = inputs_dict['data_fuel_dict']['calorific_value']
        CO2_from_production = inputs_dict['techno_infos_dict']['CO2_from_production']
        high_calorific_value = inputs_dict['data_fuel_dict']['high_calorific_value']
        model = self.crop_model
        model.configure_parameters_update(inputs_dict)
        model.compute()

        # get variable
        food_land_surface_df = model.food_land_surface_df

        # get column of interest
        food_land_surface_df_columns = list(food_land_surface_df)
        if 'years' in food_land_surface_df_columns:
            food_land_surface_df_columns.remove('years')
        food_land_surface_df_columns.remove('total surface (Gha)')

        # sum is needed to have d_total_surface_d_population
        summ = np.identity(len(food_land_surface_df.index)) * 0
        for column_name in food_land_surface_df_columns:
            if column_name == 'other (Gha)':
                result = model.d_other_surface_d_population()
            else:
                result = model.d_land_surface_d_population(column_name)
            summ += result

        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('population_df', 'population'), summ)
        d_total_d_temperature = model.d_food_land_surface_d_temperature(
            temperature_df, 'total surface (Gha)')
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('temperature_df', 'temp_atmo'), d_total_d_temperature)

        d_surface_d_red_meat_percentage = model.d_surface_d_calories(
            population_df, 'red meat')
        d_surface_d_white_meat_percentage = model.d_surface_d_calories(
            population_df, 'white meat')

        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('red_meat_calories_per_day', 'red_meat_calories_per_day'), d_surface_d_red_meat_percentage)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('white_meat_calories_per_day', 'white_meat_calories_per_day'), d_surface_d_white_meat_percentage)
        """
        vegetables_column_names = ['fruits and vegetables', 'potatoes', 'rice and maize']
        d_surface_d_other_cal = np.zeros((l_years , l_years))
        for veg in vegetables_column_names:
            grad_res = model.d_surface_d_other_calories_percentage(population_df, veg)
            d_surface_d_other_cal = d_surface_d_other_cal + grad_res
        """
        l_years = len(model.years)

        d_surface_d_vegetables_carbs = model.d_surface_d_vegetables_carbs_calories_per_day(population_df)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'), d_surface_d_vegetables_carbs)

        d_surface_d_eggs_milk = model.d_surface_d_eggs_milk_calories_per_day(population_df)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'),
            ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'), d_surface_d_eggs_milk)


        grad_constraint = np.identity(l_years)/self.crop_model.constraint_calories_ref
        self.set_partial_derivative_for_other_types(('calories_per_day_constraint',),('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),grad_constraint )
        self.set_partial_derivative_for_other_types(('calories_per_day_constraint',),('red_meat_calories_per_day', 'red_meat_calories_per_day'),grad_constraint )
        self.set_partial_derivative_for_other_types(('calories_per_day_constraint',),('white_meat_calories_per_day', 'white_meat_calories_per_day'),grad_constraint )
        self.set_partial_derivative_for_other_types(('calories_per_day_constraint',),('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),grad_constraint )

        self.set_partial_derivative_for_other_types(('calories_pc_df', 'kcal_pc'),('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'), np.identity(l_years))
        self.set_partial_derivative_for_other_types(('calories_pc_df', 'kcal_pc'),('red_meat_calories_per_day', 'red_meat_calories_per_day'), np.identity(l_years))
        self.set_partial_derivative_for_other_types(('calories_pc_df', 'kcal_pc'),('white_meat_calories_per_day', 'white_meat_calories_per_day'), np.identity(l_years))
        self.set_partial_derivative_for_other_types(('calories_pc_df', 'kcal_pc'),('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'), np.identity(l_years))

        # gradients for techno_production from total food land surface
        d_prod_dpopulation = model.compute_d_prod_dland_for_food(summ)
        d_prod_dtemperature = model.compute_d_prod_dland_for_food(
            d_total_d_temperature)
        d_prod_dred_to_white = model.compute_d_prod_dland_for_food(
            d_surface_d_red_meat_percentage)
        d_prod_dmeat_to_vegetable = model.compute_d_prod_dland_for_food(
            d_surface_d_white_meat_percentage)
        d_prod_dveg_carbs_cal = model.compute_d_prod_dland_for_food(d_surface_d_vegetables_carbs)
        dprod_deggs_milk_cal = model.compute_d_prod_dland_for_food(d_surface_d_eggs_milk)
        # --------------------------------------------------------------
        # Techno production gradients
        self.set_partial_derivative_for_other_types(('techno_production', 'biomass_dry (TWh)'), ('population_df', 'population'),
                                                    d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'), ('temperature_df', 'temp_atmo'),
            d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'), ('red_meat_calories_per_day',
                                                         'red_meat_calories_per_day'),
            d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'), ('white_meat_calories_per_day',
                                                         'white_meat_calories_per_day'),
            d_prod_dmeat_to_vegetable)
        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'), ('vegetables_and_carbs_calories_per_day',
                                                         'vegetables_and_carbs_calories_per_day'),
            d_prod_dveg_carbs_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_production', 'biomass_dry (TWh)'), ('milk_and_eggs_calories_per_day',
                                                         'milk_and_eggs_calories_per_day'),
            dprod_deggs_milk_cal)

        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(('techno_production', 'biomass_dry (TWh)'), ('crop_investment', 'investment'),
                                                    dprod_dinvest * scaling_factor_crop_investment * calorific_value / scaling_factor_techno_production)
        # --------------------------------------------------------------
        # Techno consumption gradients
        self.set_partial_derivative_for_other_types(('techno_consumption', 'CO2_resource (Mt)'), ('population_df', 'population'),
                                                    -CO2_from_production / high_calorific_value * d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'), ('temperature_df', 'temp_atmo'),
            -CO2_from_production / high_calorific_value * d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'), ('red_meat_calories_per_day',
                                                          'red_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'), ('white_meat_calories_per_day',
                                                          'white_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dmeat_to_vegetable)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'), ('vegetables_and_carbs_calories_per_day',
                                                          'vegetables_and_carbs_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dveg_carbs_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption', 'CO2_resource (Mt)'), ('milk_and_eggs_calories_per_day',
                                                          'milk_and_eggs_calories_per_day'),
            -CO2_from_production / high_calorific_value * dprod_deggs_milk_cal)


        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(('techno_consumption', 'CO2_resource (Mt)'), ('crop_investment', 'investment'),
                                                    -CO2_from_production / high_calorific_value * 
                                                    dprod_dinvest * scaling_factor_crop_investment
                                                    * calorific_value / scaling_factor_techno_production)
        # --------------------------------------------------------------
        # Techno consumption wo ratio gradients
        self.set_partial_derivative_for_other_types(('techno_consumption_woratio', 'CO2_resource (Mt)'), ('population_df', 'population'),
                                                    -CO2_from_production / high_calorific_value * d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio',
             'CO2_resource (Mt)'), ('temperature_df', 'temp_atmo'),
            -CO2_from_production / high_calorific_value * d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio',
             'CO2_resource (Mt)'), ('red_meat_calories_per_day', 'red_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio',
             'CO2_resource (Mt)'), ('white_meat_calories_per_day', 'white_meat_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dmeat_to_vegetable)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio',
             'CO2_resource (Mt)'), ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
            -CO2_from_production / high_calorific_value * d_prod_dveg_carbs_cal)

        self.set_partial_derivative_for_other_types(
            ('techno_consumption_woratio',
             'CO2_resource (Mt)'), ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
            -CO2_from_production / high_calorific_value * dprod_deggs_milk_cal)


        # gradients for techno_production from investment
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(('techno_consumption_woratio', 'CO2_resource (Mt)'), ('crop_investment', 'investment'),
                                                    -CO2_from_production / high_calorific_value * 
                                                    dprod_dinvest * scaling_factor_crop_investment
                                                    * calorific_value / scaling_factor_techno_production)

        # gradient for land demand
        self.set_partial_derivative_for_other_types(
            ('land_use_required', 'Crop (Gha)'),
            ('crop_investment', 'investment'),
            dprod_dinvest * scaling_factor_crop_investment * (1 - residue_density_percentage) / density_per_ha * calorific_value)

        vegetables_column_names = ['fruits and vegetables', 'potatoes', 'rice and maize']


        # gradient for land emissions
        for ghg in ['CO2', 'CH4', 'N2O']:
            dco2_dpop = 0.0
            dco2_dtemp = 0.0
            dco2_dred_meat = 0.0
            dco2_dwhite_meat = 0.0
            dco2_dinvest = 0.0
            dco2_dother_cal = 0.0
            dco2_deggs_milk = 0.0

            for food in self.get_sosdisc_inputs('co2_emissions_per_kg'):
                food = food.replace(' (Gt)', '')
                if food != 'years':

                    dland_emissions_dfood_land_surface = model.compute_dland_emissions_dfood_land_surface_df(
                        food)

                    if food == 'other':
                        dland_dpop = model.d_other_surface_d_population()
                    else:
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

                    dco2_dpop += dland_emissions_dfood_land_surface[ghg] * dland_dpop
                    dco2_dtemp += dland_emissions_dfood_land_surface[ghg] * dland_dtemp
                    dco2_dred_meat += dland_emissions_dfood_land_surface[ghg] * \
                        d_food_surface_d_red_meat_percentage
                    dco2_dwhite_meat += dland_emissions_dfood_land_surface[ghg] * \
                        d_food_surface_d_white_meat_percentage
                    if food == 'rice and maize':
                        dco2_dinvest += dland_emissions_dfood_land_surface[ghg] * dprod_dinvest * \
                            scaling_factor_crop_investment * (1 - residue_density_percentage) / \
                            density_per_ha * calorific_value
            for food in vegetables_column_names:
                dland_emissions_dfood_land_surface_oth = model.compute_dland_emissions_dfood_land_surface_df(
                        food)[ghg]
                dco2_dother_cal += dland_emissions_dfood_land_surface_oth * model.d_surface_d_other_calories_percentage(population_df, food)

            for food in ['eggs', 'milk']:
                dland_emissions_dfood_land_surface_oth = model.compute_dland_emissions_dfood_land_surface_df(
                    food)[ghg]
                dco2_deggs_milk += dland_emissions_dfood_land_surface_oth * model.d_surface_d_milk_eggs_calories_percentage(
                    population_df, food)


            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('population_df', 'population'),
                dco2_dpop)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('temperature_df', 'temp_atmo'),
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
                ('vegetables_and_carbs_calories_per_day', 'vegetables_and_carbs_calories_per_day'),
                dco2_dother_cal)

            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('milk_and_eggs_calories_per_day', 'milk_and_eggs_calories_per_day'),
                dco2_deggs_milk)



            self.set_partial_derivative_for_other_types(
                (f'{ghg}_land_emission_df', f'emitted_{ghg}_evol_cumulative'),
                ('crop_investment', 'investment'),
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

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if CropDiscipline.CROP_CHARTS in chart_list:

            surface_df = self.get_sosdisc_outputs('food_land_surface_df')
            years = surface_df['years'].values.tolist()

            crop_surfaces = surface_df['total surface (Gha)'].values
            crop_surface_series = InstanciatedSeries(
                years, crop_surfaces.tolist(), 'Total crop surface', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []

            for key in surface_df.keys():

                if key == 'years':
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, (surface_df[key]).values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'surface [Gha]',
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

                if key == 'years':
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, surface_percentage_df[key].values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'surface [%]',
                                                 chart_name='Share of the surface used to produce food over time', stacked_bar=True)
            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(
                years, surface_percentage_df[key].values.tolist() * 0, '', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # chart of the updated diet
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

                if key == 'years':
                    pass
                elif key.startswith('total'):
                    pass
                else:
                    l_values = updated_diet_df[key].values / 365
                    new_series = InstanciatedSeries(
                        years, l_values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'food calories [kcal / person / day]',
                                                 chart_name='Food calories per person', stacked_bar=True)

            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(
                years, surface_percentage_df[key].values.tolist() * 0, '', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)


            # ------------------------------------------
            # DIET EVOLUTION VARIABLES
            chart_name = "Calories per day"

            red_meat_calories = self.get_sosdisc_inputs('red_meat_calories_per_day')
            white_meat_calories = self.get_sosdisc_inputs(
                'white_meat_calories_per_day')
            veg_carbs = self.get_sosdisc_inputs(
                'vegetables_and_carbs_calories_per_day')
            eggs_milk = self.get_sosdisc_inputs(
                'milk_and_eggs_calories_per_day')

            new_chart = TwoAxesInstanciatedChart('years', 'calories per day [kcal]',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(
                red_meat_calories['red_meat_calories_per_day'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Calories of red meat per day per person', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data = list(
                white_meat_calories['white_meat_calories_per_day'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Calories of white meat per day per person', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data = list(
                veg_carbs['vegetables_and_carbs_calories_per_day'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Calories of vegetables and carbs per day per person', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data = list(
                eggs_milk['milk_and_eggs_calories_per_day'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Calories of eggs and milk per day per person', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data = list(
                eggs_milk['milk_and_eggs_calories_per_day'].values +
                veg_carbs['vegetables_and_carbs_calories_per_day'].values +
                red_meat_calories['red_meat_calories_per_day'].values +
                white_meat_calories['white_meat_calories_per_day'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Total calories per day per person', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Crop Productivity Evolution' in chart_list:

            prod_df = self.get_sosdisc_outputs(
                'crop_productivity_evolution')
            years = list(prod_df['years'])

            chart_name = 'Crop productivity evolution'

            new_chart = TwoAxesInstanciatedChart('years', ' productivity evolution [%]',
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
            years = list(prod_df['years'])

            # ------------------------------------------
            # INVEST (M$)
            chart_name = 'Input investments over the years'

            new_chart = TwoAxesInstanciatedChart('years', 'Investments [M$]',
                                                 chart_name=chart_name)

            visible_line = True

            for investment in crop_investment:
                if investment != 'years':
                    ordonate_data = list(crop_investment[investment])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, investment, 'bar', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (Mt)
            chart_name = 'Crop for Energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop mass for energy production [Mt]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != 'years':
                    ordonate_data = list(
                        mix_detailed_production[crop] * data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(TWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (TWh)
            chart_name = 'Crop for Energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop for Energy production [TWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != 'years':
                    ordonate_data = list(mix_detailed_production[crop])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(TWh)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # LAND USE (Gha)
            chart_name = 'Land demand for Crop for Energy'

            land_demand_chart = TwoAxesInstanciatedChart('years', 'Land demand [Gha]',
                                                         chart_name=chart_name)
            ordonate_data = list(land_use_required['Crop (Gha)'])
            land_demand_serie = InstanciatedSeries(
                years, ordonate_data, 'Crop', 'lines', visible_line)
            land_demand_chart.series.append(land_demand_serie)

            instanciated_charts.append(land_demand_chart)
            # ------------------------------------------
            # PRICE ($/MWh)
            chart_name = 'Crop energy prices by type'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop prices [$/MWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_prices:
                if crop != 'years':
                    ordonate_data = list(
                        mix_detailed_prices[crop] / data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("($/t)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # PRICE DETAILED($/MWh)
            chart_name = 'Detailed prices for crop energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop prices [$/MWh]',
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

            new_chart = TwoAxesInstanciatedChart('years', 'Crop prices [$/t]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_prices:
                if crop != 'years':
                    ordonate_data = list(mix_detailed_prices[crop])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("($/t)", ""), 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # PRICE DETAILED($/t)
            chart_name = 'Detailed prices for crop energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop prices [$/t]',
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
            years = co2_emissions_df['years'].values.tolist()

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

            new_chart = TwoAxesInstanciatedChart('years', 'Greenhouse Gas Emissions [Gt]',
                                                 chart_name='Greenhouse Gas Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # total emissions GWP

            co2_crop_emissions = co2_emissions_df['emitted_CO2_evol_cumulative'].values * self.co2_gwp_100
            co2_crop_emissions_series = InstanciatedSeries(
                years, co2_crop_emissions.tolist(), 'Total Crop CO2 Emissions [GtCO2eq]', InstanciatedSeries.BAR_DISPLAY)

            ch4_crop_emissions = ch4_emissions_df['emitted_CH4_evol_cumulative'].values * self.ch4_gwp_100
            ch4_crop_emissions_series = InstanciatedSeries(
                years, ch4_crop_emissions.tolist(), 'Total Crop CH4 Emissions [GtCO2eq]', InstanciatedSeries.BAR_DISPLAY)

            n2o_crop_emissions = n2o_emissions_df['emitted_N2O_evol_cumulative'].values * self.n2o_gwp_100
            n2o_crop_emissions_series = InstanciatedSeries(
                years, n2o_crop_emissions.tolist(), 'Total Crop N2O Emissions [GtCO2eq]', InstanciatedSeries.BAR_DISPLAY)

            series_to_add = [co2_crop_emissions_series,
                             ch4_crop_emissions_series, n2o_crop_emissions_series]

            new_chart = TwoAxesInstanciatedChart('years', 'Greenhouse Gas Emissions [GtCO2eq]',
                                                 chart_name='Greenhouse Gas CO2 Eq. Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # CO2 per food
            series_to_add = []
            for key in co2_emissions_detailed_df.columns:

                if key != 'years':
                    new_series = InstanciatedSeries(
                        years, (co2_emissions_detailed_df[key]).values.tolist(), key.replace(' (Gt)', ''), InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'CO2 Emissions [Gt]',
                                                 chart_name='CO2 Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)
            instanciated_charts.append(new_chart)

            # CH4 per food
            series_to_add = []
            for key in ch4_emissions_detailed_df.columns:

                if key != 'years':
                    new_series = InstanciatedSeries(
                        years, (ch4_emissions_detailed_df[key]).values.tolist(), key.replace(' (Gt)', ''), InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'CH4 Emissions [Gt]',
                                                 chart_name='CH4 Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # N2O per food
            series_to_add = []
            for key in n2o_emissions_detailed_df.columns:

                if key != 'years':
                    new_series = InstanciatedSeries(
                        years, (n2o_emissions_detailed_df[key]).values.tolist(
                        ), key.replace(' (Gt)', ''),
                        InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'N2O Emissions [Gt]',
                                                 chart_name='N2O Emissions of food and energy production over time',
                                                 stacked_bar=True)
            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            # CO2 emissions by food (pie chart)
            years_list = [self.crop_model.year_start, self.crop_model.year_end]
            food_names = co2_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [co2_emissions_detailed_df.loc[co2_emissions_detailed_df['years']
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'CO2 Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

            # CH4 emissions by food (pie chart)
            years_list = [self.crop_model.year_start, self.crop_model.year_end]
            food_names = ch4_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [ch4_emissions_detailed_df.loc[ch4_emissions_detailed_df['years']
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'CH4 Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

            # N2O emissions by food (pie chart)
            years_list = [self.crop_model.year_start, self.crop_model.year_end]
            food_names = n2o_emissions_detailed_df.columns[1:].to_list()
            for year in years_list:
                values = [n2o_emissions_detailed_df.loc[n2o_emissions_detailed_df['years']
                                                        == year][food].values[0] for food in food_names]

                if sum(values) != 0.0:
                    pie_chart = InstanciatedPieChart(
                        f'N2O Emissions of food and energy production in {year}', food_names, values)
                    instanciated_charts.append(pie_chart)

        
        return instanciated_charts
