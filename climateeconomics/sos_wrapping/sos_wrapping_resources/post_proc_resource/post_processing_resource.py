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

from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart

import numpy as np
from matplotlib.pyplot import cm
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
from plotly.express.colors import qualitative


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    filters = []

    chart_list = ['Resource Consumption', ]
    # The filters are set to False by default since the graphs are not yet
    # mature
    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))

    return filters


def post_processings(execution_engine, namespace, filters):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    instanciated_charts = []

    # Overload default value with chart filter
    graphs_list = []
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list.extend(chart_filter.selected_values)

    #---
    if 'Resource Consumption' in graphs_list:
        chart_name = f'Resource Consumption'
        new_chart = get_chart_resource_consumption(
            execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

    return instanciated_charts


def get_chart_resource_consumption(execution_engine, namespace, chart_name='Resource consumption'):
    '''! Function to create the resource consumption chart
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc

    @return new_chart: InstantiatedPlotlyNativeChart Scatter plot
    '''

    # Prepare data
    resource_name = namespace.split('All_resources.')[-1]
    WITNESS_ns = namespace.split('.All_resources')[0]
    EnergyMix = execution_engine.dm.get_disciplines_with_name(
        f'{WITNESS_ns}.EnergyMix')[0]
    years = np.arange(EnergyMix.get_sosdisc_inputs(
        'year_start'), EnergyMix.get_sosdisc_inputs('year_end') + 1)
    # Construct a DataFrame to organize the data
    resource_consumed = pd.DataFrame({'years': years})
    energy_list = EnergyMix.get_sosdisc_inputs('energy_list')
    for energy in energy_list:
        energy_disc = execution_engine.dm.get_disciplines_with_name(
            f'{WITNESS_ns}.EnergyMix.{energy}')[0]
        techno_list = energy_disc.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{WITNESS_ns}.EnergyMix.{energy}.{techno}')[0]
            consumption_techno = techno_disc.get_sosdisc_outputs(
                'techno_consumption')
            if resource_name in consumption_techno.columns:
                resource_consumed[f'{energy} {techno}'] = consumption_techno[resource_name] * techno_disc.get_sosdisc_inputs(
                    'scaling_factor_techno_consumption')
    CCUS = execution_engine.dm.get_disciplines_with_name(
        f'{WITNESS_ns}.ccus')[0]
    ccs_list = CCUS.get_sosdisc_inputs('ccs_list')
    for stream in ccs_list:
        stream_disc = execution_engine.dm.get_disciplines_with_name(
            f'{WITNESS_ns}.ccus.{stream}')[0]
        techno_list = stream_disc.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{WITNESS_ns}.ccus.{stream}.{techno}')[0]
            consumption_techno = techno_disc.get_sosdisc_outputs(
                'techno_consumption')
            if resource_name in consumption_techno.columns:
                resource_consumed[f'{stream} {techno}'] = consumption_techno[resource_name] * techno_disc.get_sosdisc_inputs(
                    'scaling_factor_techno_consumption')

    # Create Figure
    chart_name = f'{resource_name} consumption by technologies'
    new_chart = TwoAxesInstanciatedChart('years', f'{resource_name} consumed by techno',
                                         chart_name=chart_name, stacked_bar=True)
    for col in resource_consumed.columns:
        if 'category' not in col and col != 'years':
            legend_title = f'{col}'
            serie = InstanciatedSeries(
                resource_consumed['years'].values.tolist(),
                resource_consumed[col].values.tolist(), legend_title, 'bar')
            new_chart.series.append(serie)

    return new_chart
