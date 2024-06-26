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
from copy import deepcopy

import numpy as np
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_agriculture.agriculture import Agriculture
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class AgricultureDiscipline(ClimateEcoDiscipline):
    '''Disscipline intended to host agricluture pyworld3'''

    # ontology information
    _ontology_data = {
        'label': 'Agriculture Model',
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
    default_year_start = GlossaryCore.YearStartDefault
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

    total_kcal = 414542.4
    red_meat_percentage = default_kg_to_kcal['red meat'] / total_kcal * 100
    white_meat_percentage = default_kg_to_kcal['white meat'] / total_kcal * 100
    default_red_meat_percentage = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'red_meat_percentage': np.linspace(red_meat_percentage, 0.3 * red_meat_percentage, year_range)})
    default_white_meat_percentage = pd.DataFrame({
        GlossaryCore.Years: default_years,
        'white_meat_percentage': np.linspace(white_meat_percentage, 0.3 * white_meat_percentage, year_range)})

    default_other_use = np.linspace(0.102, 0.102, year_range)
    default_diet_df = pd.DataFrame({'red meat': [11.02],
                                    'white meat': [31.11],
                                    'milk': [79.27],
                                    'eggs': [9.68],
                                    'rice and maize': [97.76],
                                    'potatoes': [32.93],
                                    'fruits and vegetables': [217.62],
                                    })
    DESC_IN = {GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
               GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
               'diet_df': {'type': 'dataframe', 'unit': 'kg_food/person/year', 'default': default_diet_df,
                                   'dataframe_descriptor': {'red meat': ('float', [0, 1e9], True),
                                                            'cereals': ('float', [0, 1e9], True),
                                                            'white meat': ('float', [0, 1e9], True), 'milk': ('float', [0, 1e9], True),
                                                            'eggs': ('float', [0, 1e9], True), 'rice and maize': ('float', [0, 1e9], True), 'potatoes': ('float', [0, 1e9], True),
                                                            'fruits and vegetables': ('float', [0, 1e9], True)},
                                   'dataframe_edition_locked': False, 'namespace': 'ns_agriculture'},
               'kg_to_kcal_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_kcal, 'unit': 'kcal/kg', 'namespace': 'ns_agriculture'},
               'kg_to_m2_dict': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'default': default_kg_to_m2, 'unit': 'm^2/kg',  'namespace': 'ns_agriculture'},
               # design variables of changing diet
               'red_meat_percentage': {'type': 'dataframe', 'default': default_red_meat_percentage,
                                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                        'red_meat_percentage': ('float', [0, 100], True)},
                                               'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'},
               'white_meat_percentage': {'type': 'dataframe', 'default': default_white_meat_percentage,
                                                 'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                          'white_meat_percentage': ('float', [0, 100], True)},
                                                 'unit': '%', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'},

               'other_use_agriculture': {'type': 'array', 'unit': 'ha/person', 'default': default_other_use, 'namespace': 'ns_agriculture'},
               GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
               'param_a': {'type': 'float', 'unit': '-', 'default': - 0.00833, 'user_level': 3},
               'param_b': {'type': 'float', 'unit': '-', 'default': - 0.04167, 'user_level': 3},
               GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool
               }

    DESC_OUT = {
        'total_food_land_surface': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS},
        'food_land_surface_df': {
            'type': 'dataframe', 'unit': 'Gha'},
        'food_land_surface_percentage_df': {'type': 'dataframe', 'unit': '%'},
        'updated_diet_df': {'type': 'dataframe', 'unit': 'kg/person/year'},
        'agriculture_productivity_evolution': {'type': 'dataframe', 'unit': '%', }
    }

    AGRICULTURE_CHARTS = 'agriculture and diet charts'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.agriculture_model = Agriculture(param)

    def run(self):

        #-- get inputs

        inp_dict = self.get_sosdisc_inputs()
        

        self.agriculture_model.apply_percentage(inp_dict)
        #-- compute
        population_df = deepcopy(inp_dict[GlossaryCore.PopulationDfValue])
        temperature_df = deepcopy(inp_dict[GlossaryCore.TemperatureDfValue])
        self.agriculture_model.compute(population_df, temperature_df)

        outputs_dict = {
            'food_land_surface_df': self.agriculture_model.food_land_surface_df,
            'total_food_land_surface': self.agriculture_model.total_food_land_surface,
            'food_land_surface_percentage_df': self.agriculture_model.food_land_surface_percentage_df,
            'updated_diet_df': self.agriculture_model.updated_diet_df,
            'agriculture_productivity_evolution': self.agriculture_model.productivity_evolution,
        }
        
        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        inputs_dict = self.get_sosdisc_inputs()
        population_df = inputs_dict.pop(GlossaryCore.PopulationDfValue)
        temperature_df = inputs_dict[GlossaryCore.TemperatureDfValue]
        model = self.agriculture_model
        model.compute(population_df, temperature_df)

        # get variable
        food_land_surface_df = model.food_land_surface_df

        # get column of interest
        food_land_surface_df_columns = list(food_land_surface_df)
        food_land_surface_df_columns.remove(GlossaryCore.Years)
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
            ('total_food_land_surface', 'total surface (Gha)'), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue), summ)
        d_total_d_temperature = model.d_food_land_surface_d_temperature(
            temperature_df, 'total surface (Gha)')
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), d_total_d_temperature)

        d_surface_d_red_meat_percentage = model.d_surface_d_red_meat_percentage(
            population_df)
        d_surface_d_white_meat_percentage = model.d_surface_d_white_meat_percentage(
            population_df)

        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('red_meat_percentage', 'red_meat_percentage'), d_surface_d_red_meat_percentage)
        self.set_partial_derivative_for_other_types(
            ('total_food_land_surface', 'total surface (Gha)'), ('white_meat_percentage', 'white_meat_percentage'), d_surface_d_white_meat_percentage)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            AgricultureDiscipline.AGRICULTURE_CHARTS, 'Agriculture Productivity Evolution']

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

        if AgricultureDiscipline.AGRICULTURE_CHARTS in chart_list:

            surface_df = self.get_sosdisc_outputs('food_land_surface_df')
            years = surface_df[GlossaryCore.Years].values.tolist()

            agriculture_surfaces = surface_df['total surface (Gha)'].values
            agriculture_surface_series = InstanciatedSeries(
                years, agriculture_surfaces.tolist(), 'Total agriculture surface', InstanciatedSeries.LINES_DISPLAY)

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

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'surface (Gha)',
                                                 chart_name='Surface taken to produce food over time', stacked_bar=True)
            new_chart.add_series(agriculture_surface_series)

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

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'surface (%)',
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

                if key == GlossaryCore.Years:
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, (updated_diet_df[key].values * kg_to_kcal_dict[key]).tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'food calories [kcal / person / year]',
                                                 chart_name='Evolution of the diet over time', stacked_bar=True)

            # add a fake serie of value before the other serie to keep the same color than in the first graph,
            # where the line plot of total surface take the first color
            fake_serie = InstanciatedSeries(
                years, surface_percentage_df[key].values.tolist() * 0, '', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(fake_serie)

            for serie in series_to_add:
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

            series_to_add = []
            for key in updated_diet_df.keys():

                if key == GlossaryCore.Years:
                    pass
                elif key.startswith('total'):
                    pass
                else:

                    new_series = InstanciatedSeries(
                        years, (updated_diet_df[key].values * kg_to_kcal_dict[key] * 100 / total_kcal).tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'food calories proportion [% / person / year]',
                                                 chart_name='Evolution of the diet proportion over time', stacked_bar=True)
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
            chart_name = "Diet evolution, percentage of red and white meat in a person's diet"

            red_meat_evolution = self.get_sosdisc_inputs('red_meat_percentage')
            white_meat_evolution = self.get_sosdisc_inputs(
                'white_meat_percentage')

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Diet evolution [%]',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(
                red_meat_evolution['red_meat_percentage'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'percentage of red meat calories in diet', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data = list(
                white_meat_evolution['white_meat_percentage'].values)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'percentage of white meat calories in diet', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Agriculture Productivity Evolution' in chart_list:

            prod_df = self.get_sosdisc_outputs(
                'agriculture_productivity_evolution')
            years = list(prod_df.index)

            chart_name = 'Agriculture productivity evolution'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' productivity evolution (%)',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(prod_df['productivity_evolution'] * 100)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'productivity_evolution', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        return instanciated_charts
