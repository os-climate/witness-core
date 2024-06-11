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

from climateeconomics.core.tools.post_proc import get_scenario_value
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

    chart_list = [
        'Output per sector',
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    total_gdp_macro = get_scenario_value(execution_engine, GlossaryCore.EconomicsDetailDfValue, scenario_name)
    years = list(total_gdp_macro[GlossaryCore.Years].values)

    gdp_all_sectors = get_scenario_value(execution_engine, GlossaryCore.SectorGdpDfValue, scenario_name)

    if 'Output per sector' in chart_list or True:
        chart_name = f"Breakdown of GDP per sector [{GlossaryCore.SectorGdpDf['unit']}]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorGdpDf['unit'], chart_name=chart_name, stacked_bar=True)
        for sector in GlossaryCore.SectorsPossibleValues:
            new_series = InstanciatedSeries(years,
                                            list(gdp_all_sectors[sector].values),
                                            sector, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
        new_series = InstanciatedSeries(years,
                                        list(total_gdp_macro[GlossaryCore.OutputNetOfDamage].values),
                                        "Total", display_type=InstanciatedSeries.LINES_DISPLAY)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    return instanciated_charts
