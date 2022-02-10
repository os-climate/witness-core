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
from climateeconomics.core.core_forest.forest_v1 import Forest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


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
    initial_emissions = 3210

    DESC_IN = {Forest.YEAR_START: {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.YEAR_END: {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.TIME_STEP: {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
               Forest.DEFORESTATION_SURFACE: {'type': 'dataframe', 'unit': 'Mha',
                                              'dataframe_descriptor': {'years': ('float', None, False),
                                                                       'deforested_surface': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               Forest.LIMIT_DEFORESTATION_SURFACE: {'type': 'float', 'unit': 'Mha', 'default': deforestation_limit,
                                                    'namespace': 'ns_forest', },
               Forest.INITIAL_CO2_EMISSIONS: {'type': 'float', 'unit': 'MtCO2', 'default': initial_emissions,
                                              'namespace': 'ns_forest', },
               Forest.CO2_PER_HA: {'type': 'float', 'default': 4000, 'unit': 'kgCO2/ha/year', 'namespace': 'ns_forest'},
               Forest.REFORESTATION_COST_PER_HA: {'type': 'float', 'default': 15200, 'unit': '$/ha', 'namespace': 'ns_forest'},
               Forest.REFORESTATION_INVESTMENT: {'type': 'dataframe', 'unit': 'G$',
                                                 'dataframe_descriptor': {'years': ('float', None, False),
                                                                          'forest_investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               }
    # cost per ha 12000 $/ha (buy the land + 2564.128 euro/ha (ground preparation, planting) (www.teagasc.ie)
    # 1USD = 0,7360 euro in 2019

    DESC_OUT = {
        'CO2_emissions_detail_df': {
            'type': 'dataframe', 'unit': 'Gha', 'namespace': 'ns_forest'},
        Forest.FOREST_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        Forest.FOREST_DETAIL_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha'},
        Forest.CO2_EMITTED_FOREST_DF: {
            'type': 'dataframe', 'unit': 'MtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }

    FOREST_CHARTS = 'Forest chart'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.forest_model = Forest(param)

    def run(self):

        #-- get inputs
        #         inputs = list(self.DESC_IN.keys())
        #         inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        outputs_dict = {
            Forest.CO2_EMITTED_DETAIL_DF: self.forest_model.CO2_emitted_df,
            Forest.FOREST_DETAIL_SURFACE_DF: self.forest_model.forest_surface_df,
            Forest.FOREST_SURFACE_DF: self.forest_model.forest_surface_df[['years', 'forest_surface_evol', 'forest_surface_evol_cumulative']],
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
        d_forest_surface_d_invest = self.forest_model.d_forestation_surface_d_invest()
        d_cun_forest_surface_d_invest = self.forest_model.d_cum(
            d_forest_surface_d_invest)

        # forest surface vs deforestation grad
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, 'forest_surface_evol'), (
                Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_deforestation_surface_d_deforestation_surface)
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF,
             'forest_surface_evol_cumulative'),
            (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_cum_deforestation_d_deforestation_surface)

        # forest surface vs forest invest
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, 'forest_surface_evol'), (
                Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
            d_forest_surface_d_invest)
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF,
             'forest_surface_evol_cumulative'),
            (Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
            d_cun_forest_surface_d_invest)

        # d_CO2 d deforestation
        d_CO2_emitted_d_deforestation_surface = self.forest_model.d_CO2_emitted(
            d_deforestation_surface_d_deforestation_surface)
        d_cum_CO2_emitted_d_deforestation_surface = self.forest_model.d_cum(
            d_CO2_emitted_d_deforestation_surface)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, 'emitted_CO2_evol_cumulative'),
            (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_cum_CO2_emitted_d_deforestation_surface)

        # d_CO2 d invest
        d_CO2_emitted_d_invest = self.forest_model.d_CO2_emitted(
            d_forest_surface_d_invest)
        d_cum_CO2_emitted_d_invest = self.forest_model.d_cum(
            d_CO2_emitted_d_invest)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, 'emitted_CO2_evol_cumulative'),
            (Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
            d_cum_CO2_emitted_d_invest)

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

            forest_surface_df = self.get_sosdisc_outputs(
                Forest.FOREST_DETAIL_SURFACE_DF)
            years = forest_surface_df['years'].values.tolist()
            # values are *1000 to convert from Gha to Mha
            surface_evol_by_year = forest_surface_df['forest_surface_evol'].values * 1000
            surface_evol_cum = forest_surface_df['forest_surface_evol_cumulative'].values * 1000
            deforested_surface_by_year = forest_surface_df['deforested_surface'].values * 1000
            deforested_surface_cum = forest_surface_df['deforested_surface_cumulative'].values * 1000
            forested_surface_by_year = forest_surface_df['forested_surface'].values * 1000
            forested_surface_cum = forest_surface_df['forested_surface_cumulative'].values * 1000

            # forest evolution year by year chart
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Forest surface evolution year by year [Mha]', stacked_bar=True)

            deforested_series = InstanciatedSeries(
                years, deforested_surface_by_year.tolist(), 'Deforested surface', 'bar')
            new_chart.add_series(deforested_series)
            forested_series = InstanciatedSeries(
                years, forested_surface_by_year.tolist(), 'Forested surface', 'bar')
            new_chart.add_series(forested_series)
            total_series = InstanciatedSeries(
                years, surface_evol_by_year.tolist(), 'Surface evolution', InstanciatedSeries.LINES_DISPLAY)
            new_chart.add_series(total_series)

            instanciated_charts.append(new_chart)

            # forest cumulative evolution chart
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Cumulative forest surface evolution [Mha]', stacked_bar=True)

            deforested_series = InstanciatedSeries(
                years, deforested_surface_cum.tolist(), 'Deforested surface', 'bar')
            new_chart.add_series(deforested_series)
            forested_series = InstanciatedSeries(
                years, forested_surface_cum.tolist(), 'Forested surface', 'bar')
            new_chart.add_series(forested_series)
            total_series = InstanciatedSeries(
                years, surface_evol_cum.tolist(), 'Surface evolution', InstanciatedSeries.LINES_DISPLAY)
            new_chart.add_series(total_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            CO2_emissions_df = self.get_sosdisc_outputs(
                'CO2_emissions_detail_df')
            CO2_emitted_year_by_year = CO2_emissions_df['emitted_CO2']
            CO2_captured_year_by_year = CO2_emissions_df['captured_CO2']
            CO2_total_year_by_year = CO2_emissions_df['emitted_CO2_evol']
            CO2_emitted_cum = CO2_emissions_df['emitted_CO2_cumulative']
            CO2_captured_cum = CO2_emissions_df['captured_CO2_cumulative']
            CO2_total_cum = CO2_emissions_df['emitted_CO2_evol_cumulative']

            # in Mt

            new_chart = TwoAxesInstanciatedChart('years', 'CO2 emitted [Mt]',
                                                 chart_name='Delta of CO2 emitted due to yearly forest activities', stacked_bar=True)

            CO2_emitted_series = InstanciatedSeries(
                years, CO2_emitted_year_by_year.tolist(), 'Delta of emitted CO2', InstanciatedSeries.BAR_DISPLAY)
            CO2_captured_series = InstanciatedSeries(
                years, CO2_captured_year_by_year.tolist(), 'Delta of captured CO2', InstanciatedSeries.BAR_DISPLAY)
            CO2_total_series = InstanciatedSeries(
                years, CO2_total_year_by_year.tolist(), 'Delta of CO2 quantity', InstanciatedSeries.LINES_DISPLAY)

            new_chart.add_series(CO2_emitted_series)
            new_chart.add_series(CO2_captured_series)
            new_chart.add_series(CO2_total_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            # in Mt
            new_chart = TwoAxesInstanciatedChart('years', 'CO2 emitted evolution [Mt]',
                                                 chart_name='Yearly CO2 emmitted due to forest activities', stacked_bar=True)
            CO2_emitted_series = InstanciatedSeries(
                years, CO2_emitted_cum.tolist(), 'Emitted CO2', InstanciatedSeries.BAR_DISPLAY)
            CO2_captured_series = InstanciatedSeries(
                years, CO2_captured_cum.tolist(), 'Captured CO2', InstanciatedSeries.BAR_DISPLAY)
            CO2_total_series = InstanciatedSeries(
                years, CO2_total_cum.tolist(), 'CO2 quantity', InstanciatedSeries.LINES_DISPLAY, custom_data=['width'])

            new_chart.add_series(CO2_emitted_series)
            new_chart.add_series(CO2_captured_series)
            new_chart.add_series(CO2_total_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
