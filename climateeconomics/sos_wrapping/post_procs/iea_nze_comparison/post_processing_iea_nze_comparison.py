'''
Copyright 2024 Capgemini

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
import os.path

import numpy as np
import pandas as pd
from climateeconomics.core.tools.post_proc import get_scenario_value
import logging

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ['Test']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts_grad', chart_list, chart_list, 'Charts_grad')) # name 'Charts' is already used by ssp_comparison post-proc

    return chart_filters

def get_comp_chart_from_df(comp_df, y_axis_name, chart_name):
    """
    Create comparison chart from df with all series to compare.
    """
    years = comp_df[GlossaryCore.Years].values.tolist()
    series = comp_df.loc[:, comp_df.columns != GlossaryCore.Years]
    min_x = min(years)
    max_x = max(years)
    min_y = series.min()
    max_y = series.max()
    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, y_axis_name,
                                         [min_x - 5, max_x + 5], [
                                         min_y - max_y * 0.05, max_y * 1.05],
                                         chart_name)
    for sc in series.columns:
        new_series = InstanciatedSeries(
            years, series[sc].values.tolist(), sc, 'lines')
        new_chart.series.append(new_series)
    return new_chart

def post_processings(execution_engine, namespace, filters):
    """
    Instantiate postprocessing charts.
    """
    logging.info('TEST NODE POST_PROC')

    instanciated_charts = []
    chart_list = []

    # example: 3 methods to recover the dataframe of the variable 'invest_mix'
    # method 1: if invest_mix occurs in different disciplines, first list the full variable name including namespace value
    list_of_variables_with_full_namespace = execution_engine.dm.get_all_namespaces_from_var_name('invest_mix')
    # then recover the values for each occurences of the variable
    invest_mix_dict = {}
    for var in list_of_variables_with_full_namespace:
        invest_mix_dict[var] = execution_engine.dm.get_value(var)
    # method 2: if only one occurrence of the variable, it gets it automatically:
    invest_mix = get_scenario_value(execution_engine, 'invest_mix', namespace)

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts_grad':
                chart_list = chart_filter.selected_values

    if 'Test' not in chart_list:
        # if not in coarse, add primary energy chart
        pass

    return instanciated_charts
