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
from climateeconomics.core.core_forest.forest import Forest
import numpy as np
import pandas as pd


class ForestDiscipline(ClimateEcoDiscipline):
    ''' Forest discipline
    '''
    default_year_start = 2020
    default_year_end = 2050
    years = np.arange(default_year_start, default_year_end + 1, 1)
    year_range = default_year_end - default_year_start + 1
    deforestation_surface = np.array(np.linspace(10, 10, year_range))
    deforestation_surface_df = pd.DataFrame(
        {"years": years, "deforested_surface": deforestation_surface})
    deforestation_limit = 1000
    initial_emissions = 3.21

    DESC_IN = {Forest.YEAR_START: {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.YEAR_END: {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.TIME_STEP: {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
               Forest.DEFORESTATION_SURFACE: {'type': 'dataframe', 'unit': 'Mha',  'default': deforestation_surface_df,
                                              'dataframe_descriptor': {'years': ('float', None, False),
                                                                       'deforested_surface': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               Forest.LIMIT_DEFORESTATION_SURFACE: {'type': 'float', 'unit': 'Mha', 'default': deforestation_limit,
                                                    'namespace': 'ns_forest', },
               Forest.INITIAL_CO2_EMISSIONS: {'type': 'float', 'unit': 'GtCO2', 'default': initial_emissions,
                                              'namespace': 'ns_forest', },
               Forest.CO2_PER_HA: {'type': 'float', 'default': 4000, 'unit': 'kgCO2/ha/year', 'namespace': 'ns_forest'},
               }

    DESC_OUT = {
        'CO2_emissions_detail_df': {
            'type': 'dataframe', 'unit': 'Gha', 'namespace': 'ns_forest'},
        Forest.DEFORESTED_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        Forest.CO2_EMITTED_FOREST_DF: {
            'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }

    FOREST_CHARTS = 'Forest chart'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.forest_model = Forest(param)

    def run(self):

        #-- compute
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        outputs_dict = {
            'CO2_emissions_detail_df': self.forest_model.CO2_emitted_df,
            Forest.DEFORESTED_SURFACE_DF: self.forest_model.deforested_surface_df,
            Forest.CO2_EMITTED_FOREST_DF: self.forest_model.CO2_emitted_df[['years', 'emitted_CO2_evol_cumulative']],
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        # gradient for deforestation rate
        d_deforestation_surface_d_deforestation_surface = self.forest_model.d_deforestation_surface_d_deforestation_surface()
        d_cum_deforestation_d_deforestation_surface = self.forest_model.d_cum(
            d_deforestation_surface_d_deforestation_surface)

        self.set_partial_derivative_for_other_types(
            (Forest.DEFORESTED_SURFACE_DF, 'forest_surface_evol'), (
                Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_deforestation_surface_d_deforestation_surface)
        self.set_partial_derivative_for_other_types(
            (Forest.DEFORESTED_SURFACE_DF,
             'forest_surface_evol_cumulative'),
            (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_cum_deforestation_d_deforestation_surface)

        d_CO2_emitted_d_deforestation_surface = self.forest_model.d_CO2_emitted(
            d_deforestation_surface_d_deforestation_surface)
        d_cum_CO2_emitted_d_deforestation_surface = self.forest_model.d_cum(
            d_CO2_emitted_d_deforestation_surface)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, 'emitted_CO2_evol_cumulative'),
            (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_cum_CO2_emitted_d_deforestation_surface)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [ForestDiscipline.FOREST_CHARTS]

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

        if ForestDiscipline.FOREST_CHARTS in chart_list:

            deforested_surface_df = self.get_sosdisc_outputs(
                Forest.DEFORESTED_SURFACE_DF)
            years = deforested_surface_df['years'].values.tolist()
            # values are *1000 to convert from Gha to Mha
            deforested_surface_by_year = deforested_surface_df['forest_surface_evol'].values * 1000
            deforested_surface_cum = deforested_surface_df['forest_surface_evol_cumulative'].values * 1000

            # deforestation year by year chart
            deforested_series = InstanciatedSeries(
                years, deforested_surface_by_year.tolist(), 'Forest surface', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha / year]',
                                                 chart_name='Forest surface evolution')
            new_chart.add_series(deforested_series)

            instanciated_charts.append(new_chart)

            # deforestation cumulative chart
            deforested_series = InstanciatedSeries(
                years, deforested_surface_cum.tolist(), 'Cumulative forest surface', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Cumulative forest surface evolution')
            new_chart.add_series(deforested_series)

            instanciated_charts.append(new_chart)

            # CO2 graph
            CO2_emissions_df = self.get_sosdisc_outputs(
                'CO2_emissions_detail_df')

            # in Gt
            non_captured_CO2 = CO2_emissions_df['emitted_CO2_evol'].values

            graph_series = InstanciatedSeries(
                years, non_captured_CO2.tolist(), 'Forest CO2 outcome', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'CO2 [Gt / year]',
                                                 chart_name='Forest CO2 outcome')
            new_chart.add_series(graph_series)

            instanciated_charts.append(new_chart)
            # Cumulated
            non_captured_CO2_cum = CO2_emissions_df['emitted_CO2_evol_cumulative'].values

            graph_series = InstanciatedSeries(
                years, non_captured_CO2_cum.tolist(), 'Cumulative forest CO2 outcome', InstanciatedSeries.LINES_DISPLAY)
            new_chart = TwoAxesInstanciatedChart('years', 'CO2 [Gt]',
                                                 chart_name='Cumulative forest CO2 outcome')
            new_chart.add_series(graph_series)

            instanciated_charts.append(new_chart)



        return instanciated_charts
