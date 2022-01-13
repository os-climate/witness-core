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
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_land_use.land_use_v1 import LandUseV1,\
    OrderOfMagnitude
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


class LandUseV1Discipline(SoSDiscipline):
    ''' Discipline intended to host land use model with land use for food input from agriculture model
    '''
    default_year_start = 2020
    default_year_end = 2050

    DESC_IN = {'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               LandUseV1.LAND_DEMAND_DF: {'type': 'dataframe', 'unit': 'Gha',
                                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_land_use'},
               LandUseV1.TOTAL_FOOD_LAND_SURFACE: {'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               }

    DESC_OUT = {
        LandUseV1.LAND_DEMAND_CONSTRAINT_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        LandUseV1.LAND_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        LandUseV1.LAND_SURFACE_DETAIL_DF: {'type': 'dataframe', 'unit': 'Gha'},
        LandUseV1.LAND_SURFACE_FOR_FOOD_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'}
    }

    AGRICULTURE_CHARTS = 'Agriculture usage (Giga ha)'
    FOREST_CHARTS = 'Forest usage (Giga ha)'
    AVAILABLE_FOREST_CHARTS = 'Forests surface evolution (Giga ha)'
    AVAILABLE_AGRICULTURE_CHARTS = 'Agriculture surface evolution (Giga ha)'
    AGRICULTURE_FOOD_CHARTS = 'Agriculture usage for food (Giga ha)'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.land_use_model = LandUseV1(param)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        land_demand_df = inp_dict.pop('land_demand_df')
        total_food_land_surface = inp_dict.pop('total_food_land_surface')
        self.land_use_model.compute(total_food_land_surface, land_demand_df)

        outputs_dict = {
            LandUseV1.LAND_DEMAND_CONSTRAINT_DF: self.land_use_model.land_demand_constraint_df,
            LandUseV1.LAND_SURFACE_DETAIL_DF: self.land_use_model.land_surface_df,
            LandUseV1.LAND_SURFACE_DF: self.land_use_model.land_surface_df[[
                'Agriculture (Gha)', 'Forest (Gha)']],
            LandUseV1.LAND_SURFACE_FOR_FOOD_DF: self.land_use_model.land_surface_for_food_df
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
        land_demand_df = inputs_dict.pop('land_demand_df')
        total_food_land_surface = inputs_dict.pop('total_food_land_surface')
        model = self.land_use_model
        model.compute(total_food_land_surface, land_demand_df)

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
        land_surface_df_columns.remove('Food Usage (Gha)')
        land_surface_df_columns.remove('Added Forest (Gha)')
        land_surface_df_columns.remove('Added Agriculture (Gha)')

        for objective_column in land_demand_constraint_df_columns:
            for demand_column in land_demand_df_columns:
                self.set_partial_derivative_for_other_types(
                    (LandUseV1.LAND_DEMAND_CONSTRAINT_DF, objective_column),  (LandUseV1.LAND_DEMAND_DF, demand_column), model.get_derivative(objective_column, demand_column),)
               
            self.set_partial_derivative_for_other_types(
                (LandUseV1.LAND_DEMAND_CONSTRAINT_DF, objective_column),  (LandUseV1.TOTAL_FOOD_LAND_SURFACE, 'total surface (Gha)'), model.d_land_demand_constraint_d_food_land_surface(objective_column),)

        d_surface_d_population = model.d_land_surface_for_food_d_food_land_surface()
        self.set_partial_derivative_for_other_types(
            (LandUseV1.LAND_SURFACE_FOR_FOOD_DF, 'Agriculture total (Gha)'),  (LandUseV1.TOTAL_FOOD_LAND_SURFACE, 'total surface (Gha)'), d_surface_d_population)
        for objective_column in land_surface_df_columns:
            self.set_partial_derivative_for_other_types(
                (LandUseV1.LAND_SURFACE_DF, objective_column),  (LandUseV1.TOTAL_FOOD_LAND_SURFACE, 'total surface (Gha)'), model.d_agriculture_surface_d_food_land_surface(objective_column),)

            for demand_column in land_demand_df_columns:
                self.set_partial_derivative_for_other_types(
                    (LandUseV1.LAND_SURFACE_DF, objective_column),  (LandUseV1.LAND_DEMAND_DF, demand_column), model.d_constraint_d_surface(objective_column, demand_column),)


    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            LandUseV1Discipline.AGRICULTURE_CHARTS, LandUseV1Discipline.FOREST_CHARTS]

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

        if LandUseV1Discipline.FOREST_CHARTS in chart_list:

            demand_df = self.get_sosdisc_inputs(LandUseV1.LAND_DEMAND_DF)
            if demand_df is not None:
                surface_df = self.get_sosdisc_outputs(
                    LandUseV1.LAND_SURFACE_DETAIL_DF)
                years = demand_df['years'].values.tolist()
    
                forest_surfaces = surface_df['Forest (Gha)'].values
                forest_surface_series = InstanciatedSeries(
                    years, forest_surfaces.tolist(), 'Available surface', InstanciatedSeries.LINES_DISPLAY)
    
                series_to_add = []
                for column in list(demand_df):
    
                    if column in LandUseV1.FOREST_TECHNO:
    
                        new_series = InstanciatedSeries(
                            years, (demand_df[column]).values.tolist(), column.replace('(Gha)', ''), InstanciatedSeries.BAR_DISPLAY)
    
                        series_to_add.append(new_series)
    
                new_chart = TwoAxesInstanciatedChart('years', LandUseV1Discipline.FOREST_CHARTS,
                                                     chart_name=LandUseV1Discipline.FOREST_CHARTS, stacked_bar=True)
                new_chart.add_series(forest_surface_series)
    
                for serie in series_to_add:
                    new_chart.add_series(serie)
    
                instanciated_charts.append(new_chart)

                # create chart of added forest constraint
                new_chart2 = TwoAxesInstanciatedChart('years', LandUseV1Discipline.AVAILABLE_FOREST_CHARTS,
                                                     chart_name=LandUseV1Discipline.AVAILABLE_FOREST_CHARTS, stacked_bar=True)

                added_forest = surface_df['Added Forest (Gha)'].values
                added_forest_series = InstanciatedSeries(
                    years, (added_forest).tolist(), 'Global evolution of forest surface',
                    InstanciatedSeries.LINES_DISPLAY)
                new_chart2.add_series(added_forest_series)
                series_to_add = []
                # for each techno that change constraint, show its impact
                for column in LandUseV1.FOREST_CONSTRAINT_IMPACT.keys():
                    if column in list(demand_df):
                        # get multiplication coeff for technologies that change forest constraint
                        coeff = [LandUseV1.FOREST_CONSTRAINT_IMPACT[column]] * len((demand_df[column]).values)
                        new_series = InstanciatedSeries(
                            years, ((demand_df[column]).values * coeff).cumsum().tolist(), column.replace('(Gha)', 'cumulated surface'),
                            InstanciatedSeries.BAR_DISPLAY)
                        series_to_add.append(new_series)
                for serie in series_to_add:
                    new_chart2.add_series(serie)

                instanciated_charts.append(new_chart2)

        if LandUseV1Discipline.AGRICULTURE_CHARTS in chart_list:

            demand_df = self.get_sosdisc_inputs(LandUseV1.LAND_DEMAND_DF)
            if demand_df is not None:
                surface_df = self.get_sosdisc_outputs(
                    LandUseV1.LAND_SURFACE_DETAIL_DF)
                years = demand_df['years'].values.tolist()
    
                agriculture_surfaces = surface_df['Agriculture total (Gha)'].values
                agriculture_surface_series = InstanciatedSeries(
                    years, agriculture_surfaces.tolist(), 'Total agriculture surface', InstanciatedSeries.LINES_DISPLAY)
    
                series_to_add = []
                key = 'Food Usage (Gha)'
                legend = 'Food usage : Crop & lifestock'
                new_series = InstanciatedSeries(
                        years, (surface_df[key]).values.tolist(), legend, InstanciatedSeries.BAR_DISPLAY)
    
                series_to_add.append(new_series)
    
                for column in list(demand_df):
    
                    if column in LandUseV1.AGRICULTURE_TECHNO:
    
                        new_series = InstanciatedSeries(
                            years, (demand_df[column]).values.tolist(), column.replace('(Gha)', ''), InstanciatedSeries.BAR_DISPLAY)
    
                        series_to_add.append(new_series)
    
                new_chart = TwoAxesInstanciatedChart('years', LandUseV1Discipline.AGRICULTURE_CHARTS,
                                                     chart_name=LandUseV1Discipline.AGRICULTURE_CHARTS, stacked_bar=True)
                new_chart.add_series(agriculture_surface_series)
    
                for serie in series_to_add:
                    new_chart.add_series(serie)
    
                instanciated_charts.append(new_chart)

                # create chart of added agriculture constraint
                new_chart2 = TwoAxesInstanciatedChart('years', LandUseV1Discipline.AVAILABLE_AGRICULTURE_CHARTS,
                                                      chart_name=LandUseV1Discipline.AVAILABLE_AGRICULTURE_CHARTS,
                                                      stacked_bar=True)

                added_agriculture = surface_df['Added Agriculture (Gha)'].values
                added_agriculture_series = InstanciatedSeries(
                    years, (added_agriculture).tolist(), 'Global evolution of agriculture surface',
                    InstanciatedSeries.LINES_DISPLAY)
                new_chart2.add_series(added_agriculture_series)
                series_to_add = []
                # for each techno that change constraint, show its impact
                for column in LandUseV1.AGRICULTURE_CONSTRAINT_IMPACT.keys():
                    if column in list(demand_df):
                        # get multiplication coeff for technologies that change agriculture constraint
                        coeff = [LandUseV1.AGRICULTURE_CONSTRAINT_IMPACT[column]] * len((demand_df[column]).values)
                        new_series = InstanciatedSeries(
                            years, ((demand_df[column]).values * coeff).cumsum().tolist(), column.replace('(Gha)', 'cumulated surface'),
                            InstanciatedSeries.BAR_DISPLAY)
                        series_to_add.append(new_series)
                for serie in series_to_add:
                    new_chart2.add_series(serie)
                instanciated_charts.append(new_chart2)

        return instanciated_charts
