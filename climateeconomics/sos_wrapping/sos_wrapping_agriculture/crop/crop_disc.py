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
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from climateeconomics.core.core_agriculture.crop import Crop,\
    OrderOfMagnitude
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


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
                        }
    default_kg_to_kcal = {'red meat': 2566,
                          'white meat': 1860,
                          'milk': 550,
                          'eggs': 1500,
                          'rice and maize': 1150,
                          'potatoes': 670,
                          'fruits and vegetables': 624,
                          }
    year_range = default_year_end - default_year_start + 1
    default_red_to_white_meat = np.linspace(0, 50, year_range)
    default_meat_to_vegetables = np.linspace(0, 25, year_range)

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
        3.6 * energy_crop_percentage   # in Twh
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
        'CO2_from_production': - 0.425 * 44.01 / 12.0,  # same as biomass_dry
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
        'construction_delay' : construction_delay, # years
        }
    
    # Age distribution of forests in 2008 (
    initial_age_distribution = pd.DataFrame({'age': np.arange(1, lifetime),
                                             'distrib': [0.16, 0.24, 0.31, 0.39, 0.47, 0.55, 0.63, 0.71, 0.78, 0.86,
                                                         0.94, 1.02, 1.1, 1.18, 1.26, 1.33, 1.41, 1.49, 1.57, 1.65,
                                                         1.73, 1.81, 1.88, 1.96, 2.04, 2.12, 2.2, 2.28, 2.35, 2.43,
                                                         2.51, 2.59, 2.67, 2.75, 2.83, 2.9, 2.98, 3.06, 3.14, 3.22,
                                                         3.3, 3.38, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.92]})
    DESC_IN = {
        'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
        'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'population_df': {'type': 'dataframe', 'unit': 'millions of people',
                            'dataframe_descriptor': {'years': ('float', None, False),
                                                    'population': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                            'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'diet_df': {'type': 'dataframe', 'unit': 'kg_food/person/year',
                    'dataframe_descriptor': {'years': ('float', None, False),
                                            'red meat': ('float', [0, 1e9], True), 'white meat': ('float', [0, 1e9], True), 'milkt': ('float', [0, 1e9], True),
                                            'eggs': ('float', [0, 1e9], True), 'rice and maize': ('float', [0, 1e9], True), 'potatoes': ('float', [0, 1e9], True),
                                            'fruits and vegetables': ('float', [0, 1e9], True)},
                    'dataframe_edition_locked': False, 'namespace': 'ns_agriculture'},
        'kg_to_kcal_dict': {'type': 'dict', 'default': default_kg_to_kcal, 'unit': 'kcal/kg', 'namespace': 'ns_agriculture'},
        'kg_to_m2_dict': {'type': 'dict', 'default': default_kg_to_m2, 'unit': 'm^2/kg',  'namespace': 'ns_agriculture'},
        'red_to_white_meat': {'type': 'array', 'default': default_red_to_white_meat, 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'},
        'meat_to_vegetables': {'type': 'array', 'default': default_meat_to_vegetables, 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'},
        'other_use_crop': {'type': 'array', 'unit': 'ha/person', 'namespace': 'ns_agriculture'},
        'temperature_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'param_a': {'type': 'float', 'default': - 0.00833, 'user_level': 3},
        'param_b': {'type': 'float', 'default': - 0.04167, 'user_level': 3},
        'invest_level': {'type': 'dataframe', 'unit': 'G$',
                    'dataframe_descriptor': {'years': ('int',  [1900, 2100], False),
                                            'invest': ('float',  None, True)},
                    'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'scaling_factor_invest_level': {'type': 'float', 'default': 1e3, 'user_level': 2},
        'margin': {'type': 'dataframe', 'unit': '%'},
        'transport_cost': {'type': 'dataframe', 'unit': '$/t', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture',
                            'dataframe_descriptor': {'years': ('int',  [1900, 2100], False),
                                                    'transport': ('float',  None, True)},
                            'dataframe_edition_locked': False},
        'transport_margin': {'type': 'dataframe', 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture',
                            'dataframe_descriptor': {'years': ('int',  [1900, 2100], False),
                                                        'margin': ('float',  None, True)},
                            'dataframe_edition_locked': False},
        'data_fuel_dict': {'type': 'dict', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                            'namespace': 'ns_biomass_dry', 'default': BiomassDry.data_energy_dict},
        'techno_infos_dict': {'type': 'dict',
                                     'default': techno_infos_dict_default},
        'initial_production': {'type': 'float', 'unit': 'TWh', 'default': initial_production},
        'initial_age_distrib': {'type': 'dataframe', 'unit': '%', 'default': initial_age_distribution},
        }

    DESC_OUT = {
        'total_food_land_surface': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'food_land_surface_df': {
            'type': 'dataframe', 'unit': 'Gha'},
        'food_land_surface_percentage_df': {'type': 'dataframe', 'unit': '%'},
        'updated_diet_df': {'type': 'dataframe', 'unit': 'kg/person/year'},
        'crop_productivity_evolution': {'type': 'dataframe'},
        'mix_detailed_prices': {'type': 'dataframe', 'unit': '$/t'},
        'mix_detailed_production': {'type': 'dataframe', 'unit': 'Mt'},
        'cost_details': {'type': 'dataframe'},
        'biomass_production_df': {'type': 'dataframe', 'unit': 'Mt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'biomass_price_df': {'type': 'dataframe', 'unit': '$/t', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'land_demand_df': {'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_land_use'}
        }

    CROP_CHARTS = 'crop and diet charts'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.crop_model = Crop(param)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        input_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        # -- configure class with inputs
        self.crop_model.red_to_white_meat = input_dict[Crop.RED_TO_WHITE_MEAT]
        self.crop_model.meat_to_vegetables = input_dict[Crop.MEAT_TO_VEGETABLES]
        self.crop_model.invest_level = input_dict[Crop.INVEST_LEVEL]
        self.crop_model.configure_parameters(input_dict)
        #-- compute
        population_df = input_dict.pop('population_df')
        temperature_df = input_dict.pop('temperature_df')
        self.crop_model.compute(population_df, temperature_df)

        biomass_production_df = self.crop_model.mix_detailed_production[['years', 'Total (Mt)']]
        biomass_production_df = biomass_production_df.rename(columns={'Total (Mt)': "biomass_dry (Mt)"})

        outputs_dict = {
            'food_land_surface_df': self.crop_model.food_land_surface_df,
            'total_food_land_surface': self.crop_model.total_food_land_surface,
            'food_land_surface_percentage_df': self.crop_model.food_land_surface_percentage_df,
            'updated_diet_df': self.crop_model.updated_diet_df,
            'crop_productivity_evolution': self.crop_model.productivity_evolution,
            'mix_detailed_prices': self.crop_model.mix_detailed_prices,
            'cost_details': self.crop_model.cost_details,
            'mix_detailed_production': self.crop_model.mix_detailed_production,
            'biomass_production_df': biomass_production_df,
            'biomass_price_df': self.crop_model.biomass_price_df,
            'land_demand_df': self.crop_model.land_demand_df,
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        inputs_dict = self.get_sosdisc_inputs()
        population_df = inputs_dict.pop('population_df')
        temperature_df = inputs_dict['temperature_df']
        scaling_factor_invest_level = inputs_dict['scaling_factor_invest_level']
        density_per_ha = inputs_dict['techno_infos_dict']['density_per_ha']
        residue_density_percentage = inputs_dict['techno_infos_dict']['residue_density_percentage']
        model = self.crop_model
        model.configure_parameters(inputs_dict)
        model.compute(population_df, temperature_df)

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

        d_surface_d_red_to_white = model.d_surface_d_red_to_white(population_df)
        d_surface_d_meat_to_vegetable = model.d_surface_d_meat_to_vegetable(population_df)

        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('red_to_white_meat', 'red_to_white_meat'), d_surface_d_red_to_white)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('meat_to_vegetables', 'meat_to_vegetables'), d_surface_d_meat_to_vegetable)

        # gradients for biomass_production_df from total food land surface
        d_prod_dpopulation = model.compute_d_prod_dland_for_food(summ)
        d_prod_dtemperature = model.compute_d_prod_dland_for_food(d_total_d_temperature)
        d_prod_dred_to_white = model.compute_d_prod_dland_for_food(d_surface_d_red_to_white)
        d_prod_dmeat_to_vegetable = model.compute_d_prod_dland_for_food(d_surface_d_meat_to_vegetable)

        self.set_partial_derivative_for_other_types(
            ('biomass_production_df', f'biomass_dry (Mt)'),
            ('population_df', 'population'),
            d_prod_dpopulation)
        self.set_partial_derivative_for_other_types(
            ('biomass_production_df', f'biomass_dry (Mt)'), ('temperature_df', 'temp_atmo'),
            d_prod_dtemperature)
        self.set_partial_derivative_for_other_types(
            ('biomass_production_df', f'biomass_dry (Mt)'), ('red_to_white_meat', 'red_to_white_meat'),
            d_prod_dred_to_white)
        self.set_partial_derivative_for_other_types(
            ('biomass_production_df', f'biomass_dry (Mt)'), ('meat_to_vegetables', 'meat_to_vegetables'),
            d_prod_dmeat_to_vegetable)

        # gradients for biomass_production_df from invest
        dprod_dinvest = model.compute_dprod_from_dinvest()
        self.set_partial_derivative_for_other_types(
            ('biomass_production_df', f'biomass_dry (Mt)'),
            ('invest_level', 'invest'),
            dprod_dinvest * scaling_factor_invest_level)

        # gradient for land demand
        self.set_partial_derivative_for_other_types(
            ('land_demand_df', 'Crop for energy (Gha)'),
            ('invest_level', 'invest'),
            dprod_dinvest * scaling_factor_invest_level * (1 - residue_density_percentage) / density_per_ha)

        self.set_partial_derivative_for_other_types(
            ('land_demand_df', 'Crop for food (Gha)'), ('population_df', 'population'), summ)
        self.set_partial_derivative_for_other_types(
            ('land_demand_df', 'Crop for food (Gha)'), ('temperature_df', 'temp_atmo'), d_total_d_temperature)
        self.set_partial_derivative_for_other_types(
            ('land_demand_df', 'Crop for food (Gha)'), ('red_to_white_meat', 'red_to_white_meat'),
            d_surface_d_red_to_white)
        self.set_partial_derivative_for_other_types(
            ('land_demand_df', 'Crop for food (Gha)'), ('meat_to_vegetables', 'meat_to_vegetables'),
            d_surface_d_meat_to_vegetable)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            CropDiscipline.CROP_CHARTS, 'Crop Productivity Evolution', 'Crop Energy']

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

            new_chart = TwoAxesInstanciatedChart('years', 'surface (Gha)',
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

            new_chart = TwoAxesInstanciatedChart('years', 'surface (%)',
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
            updated_diet_df = self.get_sosdisc_outputs(
                'updated_diet_df')

            series_to_add = []
            for key in updated_diet_df.keys():

                if key == 'years':
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, updated_diet_df[key].values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('years', 'food quantity (kg / person / year)',
                                                 chart_name='Evolution of the diet over time', stacked_bar=True)

            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(
                years, surface_percentage_df[key].values.tolist() * 0, '', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

        if 'Crop Productivity Evolution' in chart_list:

            prod_df = self.get_sosdisc_outputs(
                'crop_productivity_evolution')
            years = list(prod_df.index)

            chart_name = 'Crop productivity evolution'

            new_chart = TwoAxesInstanciatedChart('years', ' productivity evolution (%)',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(prod_df['productivity_evolution'] * 100)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'productivity_evolution', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Crop Energy' in chart_list:

            mix_detailed_production = self.get_sosdisc_outputs(
                'mix_detailed_production')
            land_demand_df = self.get_sosdisc_outputs(
                'land_demand_df')
            mix_detailed_prices = self.get_sosdisc_outputs('mix_detailed_prices')
            data_fuel_dict = self.get_sosdisc_inputs('data_fuel_dict')
            cost_details = self.get_sosdisc_outputs('cost_details')
            invest_level = self.get_sosdisc_inputs('invest_level') * self.get_sosdisc_inputs('scaling_factor_invest_level')
            years = list(prod_df.index)

            # ------------------------------------------
            # INVEST (M$)
            chart_name = 'Input investments over the years'

            new_chart = TwoAxesInstanciatedChart('years', 'Investments [M$]',
                                                 chart_name=chart_name)

            visible_line = True

            for invest in invest_level:
                if invest != 'years':
                    ordonate_data = list(invest_level[invest])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, invest , 'bar', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (Mt)
            chart_name = 'Crop for energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop mass for energy production [Mt]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != 'years':
                    ordonate_data = list(mix_detailed_production[crop])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(Mt)", "") , 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # PRODUCTION (TWh)
            chart_name = 'Crop for energy production'

            new_chart = TwoAxesInstanciatedChart('years', 'Crop for energy production [TWh]',
                                                 chart_name=chart_name)

            visible_line = True

            for crop in mix_detailed_production:
                if crop != 'years':
                    ordonate_data = list(mix_detailed_production[crop] * data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("(Mt)", "") , 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            # ------------------------------------------
            # LAND USE (Gha)
            chart_name = 'Land demand for Crop for energy'

            land_demand_chart = TwoAxesInstanciatedChart('years', 'Land demand [Gha]',
                                                 chart_name=chart_name)
            ordonate_data = list(land_demand_df['Crop for energy (Gha)'])
            land_demand_serie = InstanciatedSeries(
                        years, ordonate_data, 'Crop' , 'lines', visible_line)
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
                    ordonate_data = list(mix_detailed_prices[crop] / data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, crop.replace("($/t)", "") , 'lines', visible_line)
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
                        years, ordonate_data, price.replace("($/MWh)", "") , 'lines', visible_line)
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
                    ordonate_data = list(cost_details[price] * data_fuel_dict['calorific_value'])
                    new_series = InstanciatedSeries(
                        years, ordonate_data, price.replace("($/MWh)", "") , 'lines', visible_line)
                    new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        return instanciated_charts
