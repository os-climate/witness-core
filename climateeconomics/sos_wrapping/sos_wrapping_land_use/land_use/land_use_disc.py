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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from climateeconomics.core.core_land_use.land_use import LandUse
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


class LandUseDiscipline(SoSDiscipline):
    ''' Disscipline intended to host land use model
    '''
    default_year_start = 2020
    default_year_end = 2050
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    default_percentage = np.linspace(
        100, 50, default_year_end - default_year_start + 1)
    default_meat_food_df = pd.DataFrame(
        {'years': default_years, 'percentage': default_percentage}, index=default_years)

    # source of crop_land_use_per_capita and livestock_land_use_per_capita
    # http://www.fao.org/sustainability/news/detail/en/c/1274219/
    DESC_IN = {LandUse.LAND_DEMAND_DF: {'type': 'dataframe', 'unit': 'Gha',
                                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_land_use'},
               'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'crop_land_use_per_capita': {'type': 'float', 'default': 0.21, 'unit': 'ha/capita'},
               'livestock_land_use_per_capita': {'type': 'float', 'default': 0.42, 'unit': 'ha/capita'},
               'population_df': {'type': 'dataframe', 'unit': 'millions of people',
                                 'dataframe_descriptor': {'years': ('float', None, False),
                                                          'population': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               'livestock_usage_factor_df': {'type': 'dataframe', 'unit': '%',
                                             'dataframe_descriptor': {'years': ('float', None, False),
                                                                      'percentage': ('float', [0, 100], True)}, 'dataframe_edition_locked': False,
                                             'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               LandUse.LAND_USE_CONSTRAINT_REF: {
                   'type': 'float', 'default': 0.01,  'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'}
               }
    # add a variable design livestock_usage_factor_df with % of usage of meat food surface
    # as quick fix for optim

    DESC_OUT = {
        LandUse.LAND_DEMAND_CONSTRAINT_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        LandUse.LAND_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        LandUse.LAND_SURFACE_DETAIL_DF: {'type': 'dataframe', 'unit': 'Gha'},
        LandUse.LAND_SURFACE_FOR_FOOD_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'}
    }

    AGRICULTURE_CHARTS = 'agriculture usage (Giga ha)'
    FOREST_CHARTS = 'Forest usage (Giga ha)'
    AGRICULTURE_FOOD_CHARTS = 'Agriculture usage for food (Giga ha)'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.land_use_model = LandUse(param)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        population_df = inp_dict.pop('population_df')
        # rescaling from billion to million
        cols = [col for col in population_df.columns if col != 'years']
        population_df[cols] = population_df[cols] * 1e3
        land_demand_df = inp_dict.pop('land_demand_df')
        livestock_usage_factor_df = inp_dict.pop('livestock_usage_factor_df')
        self.land_use_model.compute(
            population_df, land_demand_df, livestock_usage_factor_df)

        outputs_dict = {
            LandUse.LAND_DEMAND_CONSTRAINT_DF: self.land_use_model.land_demand_constraint_df,
            LandUse.LAND_SURFACE_DETAIL_DF: self.land_use_model.land_surface_df,
            LandUse.LAND_SURFACE_DF: self.land_use_model.land_surface_df[[
                'Agriculture (Gha)', 'Forest (Gha)']],
            LandUse.LAND_SURFACE_FOR_FOOD_DF: self.land_use_model.land_surface_for_food_df
        }
        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        land_demand_objective_df wrt land_demand_df
        """
        inputs_dict = self.get_sosdisc_inputs()
        population = inputs_dict.pop('population_df')
        # rescaling from billion to million
        cols = [col for col in population.columns if col != 'years']
        population[cols] = population[cols] * 1e3
        land_demand_df = inputs_dict.pop('land_demand_df')
        livestock_usage_factor_df = inputs_dict.pop(
            'livestock_usage_factor_df')

        model = self.land_use_model
        model.compute(population, land_demand_df, livestock_usage_factor_df)
        land_demand_ref = inputs_dict.pop(model.LAND_USE_CONSTRAINT_REF)
        # Retrieve variables
        land_demand_df = model.land_demand_df
        land_demand_constraint_df = model.land_demand_constraint_df
        land_surface_df = model.land_surface_df

        # build columns
        land_demand_df_columns = list(land_demand_df)
        land_demand_df_columns.remove('years')

        land_demand_constraint_df_columns = list(land_demand_constraint_df)
        land_demand_constraint_df_columns.remove('years')

        land_surface_df_columns = list(land_surface_df)
        land_surface_df_columns.remove('Agriculture total (Gha)')
        land_surface_df_columns.remove('Crop Usage (Gha)')
        land_surface_df_columns.remove('Livestock Usage (Gha)')

        for objective_column in land_demand_constraint_df_columns:
            for demand_column in land_demand_df_columns:
                self.set_partial_derivative_for_other_types(
                    (LandUse.LAND_DEMAND_CONSTRAINT_DF, objective_column),  (LandUse.LAND_DEMAND_DF, demand_column), model.get_derivative(objective_column, demand_column),)

        self.set_partial_derivative_for_other_types(
            (LandUse.LAND_SURFACE_FOR_FOOD_DF, 'Agriculture total (Gha)'),  ('livestock_usage_factor_df', 'percentage'), model.d_land_surface_for_food_d_livestock_usage_factor(population),)

        self.set_partial_derivative_for_other_types(
            (LandUse.LAND_SURFACE_DF, 'Agriculture (Gha)'),  ('livestock_usage_factor_df', 'percentage'), -model.d_land_surface_for_food_d_livestock_usage_factor(population),)

        self.set_partial_derivative_for_other_types(
            (LandUse.LAND_DEMAND_CONSTRAINT_DF, 'Agriculture demand constraint (Gha)'),  ('livestock_usage_factor_df', 'percentage'), -model.d_land_surface_for_food_d_livestock_usage_factor(population) / land_demand_ref,)

        # remove land use gradient for population but keep it in case of
        # population model later
#             self.set_partial_derivative_for_other_types(
#                 (LandUse.LAND_DEMAND_CONSTRAINT_DF, objective_column),  (LandUse.POPULATION_DF, 'population'), model.d_land_demand_constraint_d_population(objective_column),)
#
#         d_surface_d_population = model.d_land_surface_for_food_d_population()
#         self.set_partial_derivative_for_other_types(
#             (LandUse.LAND_SURFACE_FOR_FOOD_DF, 'Agriculture total (Gha)'),  (LandUse.POPULATION_DF, 'population'), d_surface_d_population)
#         for objective_column in land_surface_df_columns:
#             self.set_partial_derivative_for_other_types(
#                 (LandUse.LAND_SURFACE_DF, objective_column),  (LandUse.POPULATION_DF, 'population'), model.d_agriculture_surface_d_population(objective_column),)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            LandUseDiscipline.AGRICULTURE_CHARTS, LandUseDiscipline.FOREST_CHARTS]

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

        if LandUseDiscipline.FOREST_CHARTS in chart_list:

            demand_df = self.get_sosdisc_inputs(LandUse.LAND_DEMAND_DF)
            if demand_df is not None:
                surface_df = self.get_sosdisc_outputs(
                    LandUse.LAND_SURFACE_DETAIL_DF)
                years = demand_df['years'].values.tolist()

                forest_surfaces = np.ones(
                    len(years)) * (surface_df['Forest (Gha)'].values[0])
                forest_surface_series = InstanciatedSeries(
                    years, forest_surfaces.tolist(), 'Available surface', InstanciatedSeries.LINES_DISPLAY)

                series_to_add = []
                for column in list(demand_df):

                    if column in LandUse.FOREST_TECHNO:

                        new_series = InstanciatedSeries(
                            years, (demand_df[column]).values.tolist(), column.replace('(Gha)', ''), InstanciatedSeries.BAR_DISPLAY)

                        series_to_add.append(new_series)

                new_chart = TwoAxesInstanciatedChart('years', LandUseDiscipline.FOREST_CHARTS,
                                                     chart_name=LandUseDiscipline.FOREST_CHARTS, stacked_bar=True)
                new_chart.add_series(forest_surface_series)

                for serie in series_to_add:
                    new_chart.add_series(serie)

                instanciated_charts.append(new_chart)

        if LandUseDiscipline.AGRICULTURE_CHARTS in chart_list:

            demand_df = self.get_sosdisc_inputs(LandUse.LAND_DEMAND_DF)
            if demand_df is not None:
                surface_df = self.get_sosdisc_outputs(
                    LandUse.LAND_SURFACE_DETAIL_DF)
                years = demand_df['years'].values.tolist()

                agriculture_surfaces = surface_df['Agriculture total (Gha)'].values
                agriculture_surface_series = InstanciatedSeries(
                    years, agriculture_surfaces.tolist(), 'Total agriculture surface', InstanciatedSeries.LINES_DISPLAY)

                series_to_add = []
                to_plot = ['Crop Usage (Gha)', 'Livestock Usage (Gha)']
                legend = {'Crop Usage (Gha)': 'Food usage : Crop',
                          'Livestock Usage (Gha)': 'Food usage: Lifestock'}
                for key in to_plot:

                    new_series = InstanciatedSeries(
                        years, (surface_df[key]).values.tolist(), legend[key], InstanciatedSeries.BAR_DISPLAY)

                    series_to_add.append(new_series)

                for column in list(demand_df):

                    if column in LandUse.AGRICULTURE_TECHNO:

                        new_series = InstanciatedSeries(
                            years, (demand_df[column]).values.tolist(), column.replace('(Gha)', ''), InstanciatedSeries.BAR_DISPLAY)

                        series_to_add.append(new_series)

                new_chart = TwoAxesInstanciatedChart('years', LandUseDiscipline.AGRICULTURE_FOOD_CHARTS,
                                                     chart_name=LandUseDiscipline.AGRICULTURE_FOOD_CHARTS, stacked_bar=True)
                new_chart.add_series(agriculture_surface_series)

                for serie in series_to_add:
                    new_chart.add_series(serie)

                instanciated_charts.append(new_chart)

        return instanciated_charts
