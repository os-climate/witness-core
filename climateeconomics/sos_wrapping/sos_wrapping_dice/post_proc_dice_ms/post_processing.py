"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.data_manager import DataManager
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
)
from sostrades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import (
    InstantiatedParetoFrontOptimalChart,
)

from climateeconomics.glossarycore import GlossaryCore


def post_processing_filters(execution_engine, namespace):

    filters = []

    chart_list = ["Temperature vs Welfare"]
    filters.append(ChartFilter("Charts", chart_list, chart_list, "Charts"))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == "Charts":
                graphs_list = chart_filter.selected_values
    else:
        graphs_list = ["Temperature vs Welfare"]

    temperature_df_dict, utility_df_dict, scenario_list, year_end, namespace_w = get_df(execution_engine, namespace)

    """
        -------------
        -------------
        PARETO OPTIMAL CHART
        -------------
        -------------
        """
    if "Temperature vs Welfare" in graphs_list:

        chart_name = f"Temperature in {year_end} vs Welfare"

        temperature = {}
        welfare = {}

        for scenario in scenario_list:
            temperature[scenario] = temperature_df_dict[scenario][GlossaryCore.TempAtmo][year_end]
            welfare[scenario] = utility_df_dict[scenario][GlossaryCore.Welfare][year_end]

        min_temp = min(list(temperature.values()))
        max_temp = max(list(temperature.values()))
        maxs = max(max_temp, abs(min_temp))

        max_value_welfare = max(list(welfare.values()))
        min_value_welfare = min(list(welfare.values()))

        new_pareto_chart = InstantiatedParetoFrontOptimalChart(
            abscissa_axis_name="Temperature increase since industrial revolution in degree Celsius",
            primary_ordinate_axis_name="Welfare",
            abscissa_axis_range=[min_temp - max_temp * 0.05, max_temp * 1.05],
            primary_ordinate_axis_range=[min_value_welfare - max_value_welfare * 0.03, max_value_welfare * 1.03],
            chart_name=chart_name,
        )

        for scenario in scenario_list:
            new_serie = InstanciatedSeries(
                [temperature[scenario]],
                [welfare[scenario]],
                scenario,
                "scatter",
                custom_data=f"{namespace_w}.{scenario}",
            )
            new_pareto_chart.add_serie(new_serie)

            # Calculating and adding pareto front
        sorted_temp = sorted(temperature.values())
        sorted_scenarios = []
        for val in sorted_temp:
            for scen, temp in temperature.items():
                if temp == val:
                    sorted_scenarios.append(scen)

        sorted_list = sorted(
            [[temperature[scenario], welfare[scenario]] for scenario in sorted_scenarios], reverse=True
        )
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

        pareto_front_serie = InstanciatedSeries(
            [pair[0] for pair in pareto_front], [pair[1] for pair in pareto_front], "Pareto front", "lines"
        )
        new_pareto_chart.add_pareto_front_optimal(pareto_front_serie)

        instanciated_charts.append(new_pareto_chart)
    #

    #     if 'Temperature vs Utility' in graphs_list:
    #
    #         chart_name = f'Temperature vs Utility per capita in {year_end}'
    #
    #         temperature = {}
    #         utility = {}
    #
    #         for scenario in scenario_list:
    #             temperature[scenario] = temperature_df_dict[scenario][GlossaryCore.TempAtmo][year_end]
    #             utility[scenario] = utility_df_dict[scenario]['period_utility'][year_end]
    #
    #         min_temp = min(list(temperature.values()))
    #         max_temp = max(list(temperature.values()))
    #         maxs = max(max_temp, abs(min_temp))
    #
    #         max_value_utility = max(list(utility.values()))
    #         min_value_utility = min(list(utility.values()))
    #
    #         new_pareto_chart = InstantiatedParetoFrontOptimalChart(
    #             abscissa_axis_name=f'Temperature increase in degree Celsius at {year_end} since industrial revolution',
    #             primary_ordinate_axis_name=f'Utility per capita in thousand dollars at {year_end}',
    #             abscissa_axis_range=[min_temp - max_temp * 0.05, max_temp * 1.05],
    #             primary_ordinate_axis_range=[
    #                 min_value_utility - max_value_utility * 0.03, max_value_utility * 1.03],
    #             chart_name=chart_name)
    #
    #         for scenario in scenario_list:
    #             new_serie = InstanciatedSeries([temperature[scenario]],
    #                                            [utility[scenario]],
    #                                            scenario, 'scatter',
    #                                            custom_data=f'{namespace_w}.{scenario}')
    #             new_pareto_chart.add_serie(new_serie)
    #
    #             # Calculating and adding pareto front
    #         sorted_temp = sorted(temperature.values())
    #         sorted_scenarios = []
    #         for val in sorted_temp:
    #             for scen, temp in temperature.items():
    #                 if temp == val:
    #                     sorted_scenarios.append(scen)
    #
    #         sorted_list = sorted([[temperature[scenario], utility[scenario]]
    #                               for scenario in sorted_scenarios], reverse=True)
    #         pareto_front = [sorted_list[0]]
    #         for pair in sorted_list[1:]:
    #             if pair[1] <= pareto_front[-1][1]:
    #                 pareto_front.append(pair)
    #
    #         pareto_front_serie = InstanciatedSeries(
    #             [pair[0] for pair in pareto_front], [pair[1] for pair in pareto_front], 'Pareto front', 'lines')
    #         new_pareto_chart.add_pareto_front_optimal(pareto_front_serie)
    #
    #         instanciated_charts.append(new_pareto_chart)
    #         new_pareto_chart.to_plotly().show()

    return instanciated_charts


def get_df(execution_engine, namespace):

    temperature_df_dict = {}
    utility_df_dict = {}
    scatter_scenario = "Control rate scenarios"
    namespace_w = f"{execution_engine.study_name}.{scatter_scenario}"

    scenario_key = execution_engine.dm.get_data_id(f"{namespace_w}.samples_df")
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]["scenario_name"].values.tolist()

    for scenario in scenario_list:
        temperature_df_namespace = f"{namespace_w}.{scenario}.{GlossaryCore.TemperatureDfValue}"
        utility_df_namespace = f"{namespace_w}.{scenario}.{GlossaryCore.UtilityDfValue}"

        x_key = execution_engine.dm.get_data_id(temperature_df_namespace)
        y_key = execution_engine.dm.get_data_id(utility_df_namespace)

        temperature_df_dict[scenario] = execution_engine.dm.data_dict[x_key][DataManager.VALUE]
        utility_df_dict[scenario] = execution_engine.dm.data_dict[y_key][DataManager.VALUE]

    year_end_namespace = f"{namespace_w}.{GlossaryCore.YearEnd}"
    year_key = execution_engine.dm.get_data_id(year_end_namespace)
    year_end = execution_engine.dm.data_dict[year_key][DataManager.VALUE]

    return temperature_df_dict, utility_df_dict, scenario_list, year_end, namespace_w
