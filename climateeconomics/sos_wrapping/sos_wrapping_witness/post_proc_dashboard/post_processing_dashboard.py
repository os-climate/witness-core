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

import logging

import climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline as TempChange
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

        model = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('temperature_model')[0])
        temperature_df = deepcopy(
            execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('temperature_detail_df')[0]))

        instanciated_charts = TempChange.temperature_evolution(model,temperature_df,instanciated_charts)

    if 'Radiative forcing' in chart_list:

        forcing_df = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('forcing_detail_df')[0])

        instanciated_charts = TempChange.radiative_forcing(forcing_df,instanciated_charts)

    return instanciated_charts


