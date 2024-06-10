"""
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
"""

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore


def post_processing_filters(execution_engine, namespace):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    chart_filters = []

    chart_list = ["Residential emissions"]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "Charts"))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    """

    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == "Charts":
                chart_list = chart_filter.selected_values

    total_emission_residential = get_scenario_value(
        execution_engine, GlossaryCore.ResidentialEmissionsDfValue, scenario_name
    )
    years = list(total_emission_residential[GlossaryCore.Years].values)

    if "Residential emissions" in chart_list:
        chart_name = (
            f"Emissions of households (only due to energy consumption) [{GlossaryCore.ResidentialEmissionsDf['unit']}]"
        )
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, GlossaryCore.ResidentialEmissionsDf["unit"], chart_name=chart_name, stacked_bar=True
        )
        new_series = InstanciatedSeries(
            years,
            list(total_emission_residential[GlossaryCore.TotalEmissions].values),
            "Total",
            display_type=InstanciatedSeries.BAR_DISPLAY,
        )
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    return instanciated_charts
