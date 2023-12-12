'''
Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.tempchange_model_v2 import TempChange
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ['temperature evolution', 'Radiative forcing']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, namespace, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    # execution_engine.dm.get_all_namespaces_from_var_name('temperature_df')[0]

    instanciated_charts = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    if 'temperature evolution' in chart_list:

        model = ProxyDiscipline.get_sosdisc_inputs('temperature_model')
        temperature_df = deepcopy(
            execution_engine.dm.get_sosdisc_outputs('temperature_detail_df'))

        if model == 'DICE':
            to_plot = [GlossaryCore.TempAtmo, GlossaryCore.TempOcean]
            legend = {GlossaryCore.TempAtmo: 'atmosphere temperature',
                      GlossaryCore.TempOcean: 'ocean temperature'}

        elif model == 'FUND':
            to_plot = [GlossaryCore.TempAtmo]
            legend = {GlossaryCore.TempAtmo: 'atmosphere temperature'}

        elif model == 'FAIR':
            raise NotImplementedError('Model not implemented yet')

        years = list(temperature_df.index)

        year_start = years[0]
        year_end = years[len(years) - 1]

        max_values = {}
        min_values = {}
        for key in to_plot:
            min_values[key], max_values[key] = execution_engine.dm.get_greataxisrange(
                temperature_df[to_plot])

        min_value = min(min_values.values())
        max_value = max(max_values.values())

        chart_name = 'Temperature evolution over the years'

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                             'temperature evolution (degrees Celsius above preindustrial)',
                                             [year_start - 5, year_end + 5], [
                                                 min_value, max_value],
                                             chart_name)

        for key in to_plot:
            visible_line = True

            ordonate_data = list(temperature_df[key])

            new_series = InstanciatedSeries(
                years, ordonate_data, legend[key], 'lines', visible_line)

            new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

        # Seal level chart for FUND pyworld3
        if model == 'FUND':
            chart_name = 'Sea level evolution over the years'
            min_value, max_value = execution_engine.dm.get_greataxisrange(temperature_df['sea_level'])
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 'Seal level evolution',
                                                 [year_start - 5, year_end + 5], [min_value, max_value],
                                                 chart_name)
            visible_line = True
            ordonate_data = list(temperature_df['sea_level'])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Seal level evolution [m]', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

    if 'Radiative forcing' in chart_list:

        forcing_df = execution_engine.dm.get_sosdisc_outputs('forcing_detail_df')

        years = forcing_df[GlossaryCore.Years].values.tolist()

        chart_name = 'Gas Radiative forcing evolution over the years'

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Radiative forcing (W.m-2)',
                                             chart_name=chart_name)

        for forcing in forcing_df.columns:
            if forcing != GlossaryCore.Years:
                new_series = InstanciatedSeries(
                    years, forcing_df[forcing].values.tolist(), forcing, 'lines')

                new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)
    return instanciated_charts


