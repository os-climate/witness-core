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
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
from climateeconomics.core.core_deforest.deforest import Deforest
import numpy as np
import pandas as pd


class DeforestDiscipline(ClimateEcoDiscipline):
    ''' Disscipline intended to host deforestation model
    '''
    default_year_start = 2020
    default_year_end = 2050
    years = np.arange(default_year_start, default_year_end + 1, 1)
    year_range = default_year_end - default_year_start + 1
    forest_surface = np.array(np.linspace(3.4, 2.8, year_range))
    forest_df = pd.DataFrame(
        {"years": years, "forest_surface": forest_surface})
    deforestation_rate = np.array(np.linspace(3, 3, year_range))
    deforestation_df = pd.DataFrame(
        {"years": years, "forest_evolution": deforestation_rate})

    DESC_IN = {Deforest.YEAR_START: {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Deforest.YEAR_END: {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Deforest.TIME_STEP: {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
               Deforest.FOREST_DF: {'type': 'dataframe', 'unit': 'Gha',  'default': forest_df,
                                    'dataframe_descriptor': {'years': ('float', None, False),
                                                             'forest_surface': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               Deforest.DEFORESTATION_RATE_DF: {'type': 'dataframe', 'unit': '%', 'default': deforestation_df,
                                                'dataframe_descriptor': {'years': ('float', None, False),
                                                                         'forest_evolution': ('float', [0, 100], True)}, 'dataframe_edition_locked': False,
                                                'namespace': 'ns_witness',
                                                'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY},
               Deforest.CO2_PER_HA: {'type': 'float', 'default': 4000, 'unit': 'kgCO2/ha/year', 'namespace': 'ns_deforestation'},
               }

    DESC_OUT = {
        Deforest.DEFORESTED_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        Deforest.NON_CAPTURED_CO2_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }

    DEFOREST_CHARTS = 'Deforestation chart'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.deforest_model = Deforest(param)

    def run(self):

        #-- get inputs
        #         inputs = list(self.DESC_IN.keys())
        #         inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        in_dict = self.get_sosdisc_inputs()
        self.deforest_model.compute(in_dict)

        outputs_dict = {
            Deforest.DEFORESTED_SURFACE_DF: self.deforest_model.deforested_surface_df,
            Deforest.NON_CAPTURED_CO2_DF: self.deforest_model.non_captured_CO2_df,
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        in_dict = self.get_sosdisc_inputs()
        self.deforest_model.compute(in_dict)

        # gradient for forests surface
        d_deforestation_surface_d_forest = self.deforest_model.d_deforestation_surface_d_forests()
        d_cum_deforestation_d_forest = self.deforest_model.d_cum(
            d_deforestation_surface_d_forest)

        self.set_partial_derivative_for_other_types(
            (Deforest.DEFORESTED_SURFACE_DF,
             'forest_surface_evol'), (Deforest.FOREST_DF, 'forest_surface'),
            d_deforestation_surface_d_forest)
        self.set_partial_derivative_for_other_types(
            (Deforest.DEFORESTED_SURFACE_DF,
             'forest_surface_evol_cumulative'), (Deforest.FOREST_DF, 'forest_surface'),
            d_cum_deforestation_d_forest)

        d_non_captured_co2_d_forest = self.deforest_model.d_non_captured_CO2(
            d_deforestation_surface_d_forest)
        d_cum_non_captured_co2_d_forest = self.deforest_model.d_cum(
            d_non_captured_co2_d_forest)

        self.set_partial_derivative_for_other_types(
            (Deforest.NON_CAPTURED_CO2_DF,
             'captured_CO2_evol'), (Deforest.FOREST_DF, 'forest_surface'),
            d_non_captured_co2_d_forest)
        self.set_partial_derivative_for_other_types(
            (Deforest.NON_CAPTURED_CO2_DF, 'captured_CO2_evol_cumulative'),
            (Deforest.FOREST_DF, 'forest_surface'),
            d_cum_non_captured_co2_d_forest)

        # gradient for deforestation rate
        d_deforestation_surface_d_deforestation_rate = self.deforest_model.d_deforestation_surface_d_deforestation_rate()
        d_cum_deforestation_d_deforestation_rate = self.deforest_model.d_cum(
            d_deforestation_surface_d_deforestation_rate)

        self.set_partial_derivative_for_other_types(
            (Deforest.DEFORESTED_SURFACE_DF, 'forest_surface_evol'), (
                Deforest.DEFORESTATION_RATE_DF, 'forest_evolution'),
            d_deforestation_surface_d_deforestation_rate)
        self.set_partial_derivative_for_other_types(
            (Deforest.DEFORESTED_SURFACE_DF,
             'forest_surface_evol_cumulative'),
            (Deforest.DEFORESTATION_RATE_DF, 'forest_evolution'),
            d_cum_deforestation_d_deforestation_rate)

        d_non_captured_co2_d_deforestation_rate = self.deforest_model.d_non_captured_CO2(
            d_deforestation_surface_d_deforestation_rate)
        d_cum_non_captured_co2_d_deforestation_rate = self.deforest_model.d_cum(
            d_non_captured_co2_d_deforestation_rate)

        self.set_partial_derivative_for_other_types(
            (Deforest.NON_CAPTURED_CO2_DF, 'captured_CO2_evol'),
            (Deforest.DEFORESTATION_RATE_DF, 'forest_evolution'),
            d_non_captured_co2_d_deforestation_rate)
        self.set_partial_derivative_for_other_types(
            (Deforest.NON_CAPTURED_CO2_DF, 'captured_CO2_evol_cumulative'),
            (Deforest.DEFORESTATION_RATE_DF, 'forest_evolution'),
            d_cum_non_captured_co2_d_deforestation_rate)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [DeforestDiscipline.DEFOREST_CHARTS]

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

        if DeforestDiscipline.DEFOREST_CHARTS in chart_list:

            deforested_surface_df = self.get_sosdisc_outputs(
                Deforest.DEFORESTED_SURFACE_DF)
            years = deforested_surface_df['years'].values.tolist()
            # surface conversion : Gha to Mha
            deforested_surface_by_year = deforested_surface_df['forest_surface_evol'].values * 1000
            deforested_surface_cum = deforested_surface_df['forest_surface_evol_cumulative'].values * 1000

            # deforestation year by year chart
            deforested_series = InstanciatedSeries(
                years, deforested_surface_by_year.tolist(), 'Forest surface', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Forest surface evolution year by year [Mha]')
            new_chart.add_series(deforested_series)

            instanciated_charts.append(new_chart)

            # deforestation cumulative chart
            deforested_series = InstanciatedSeries(
                years, deforested_surface_cum.tolist(), 'Cumulative forest surface', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Cumulative forest surface evolution [Mha]')
            new_chart.add_series(deforested_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            non_captured_CO2_df = self.get_sosdisc_outputs(
                Deforest.NON_CAPTURED_CO2_DF)
            # in Mt
            non_captured_CO2_cum = non_captured_CO2_df['captured_CO2_evol_cumulative'].values

            graph_series = InstanciatedSeries(
                years, non_captured_CO2_cum.tolist(), 'Cumulative non-captured CO2', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Captured CO2 evolution [Mt]',
                                                 chart_name='Cumulative captured CO2 evolution')
            new_chart.add_series(graph_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            # in Mt
            non_captured_CO2 = non_captured_CO2_df['captured_CO2_evol'].values

            graph_series = InstanciatedSeries(
                years, non_captured_CO2.tolist(), 'Non-captured CO2', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Captured CO2 evolution [Mt]',
                                                 chart_name='Captured CO2 evolution')
            new_chart.add_series(graph_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
