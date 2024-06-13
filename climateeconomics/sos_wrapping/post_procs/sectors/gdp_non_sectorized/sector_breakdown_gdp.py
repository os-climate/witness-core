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
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = [
        "GDP breakdown on sub-sectors"
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, sector: str, chart_filters=None):
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

    economics_df = get_scenario_value(execution_engine, GlossaryCore.EconomicsDfValue, scenario_name)
    sector_gdp_df = get_scenario_value(execution_engine, GlossaryCore.SectorGdpDfValue, scenario_name)
    years = list(economics_df[GlossaryCore.Years].values)
    if "GDP breakdown on sub-sectors" in chart_list:
        section_gdp_df = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionGdpDfValue}", scenario_name)
        # loop on all sectors to plot a chart per sector

        chart_name = f"Breakdown of GDP betweens sub-sectors for {sector} sector [{GlossaryCore.SectionGdpDf['unit']}]"

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionGdpPart,
                                             chart_name=chart_name, stacked_bar=True)
        # loop on all sections of current sector
        for section in GlossaryCore.SectionDictSectors[sector]:
            section_value = section_gdp_df[section].values
            new_series = InstanciatedSeries(
                years, list(section_value), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

        # plot total gdp for the current sector in line
        new_series = InstanciatedSeries(
            years, list(sector_gdp_df[sector]),
            f"Total GDP for {sector} sector",
            'lines', True)
        new_chart.add_series(new_series)

        # have a full label on chart (for long names)
        fig = new_chart.to_plotly()
        fig.update_traces(hoverlabel=dict(namelength=-1))
        # if dictionaries has big size, do not show legen, otherwise show it
        if len(GlossaryCore.SectionDictSectors[sector]) > 5:
            fig.update_layout(showlegend=False)
        else:
            fig.update_layout(showlegend=True)
        instanciated_charts.append(InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name,
            default_title=True, default_legend=False))

    return instanciated_charts
