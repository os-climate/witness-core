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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import \
    InstantiatedParetoFrontOptimalChart
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.execution_engine.data_manager import DataManager
from copy import deepcopy


def post_processing_filters(execution_engine, namespace):

    filters = []

    chart_list = ['Temperature per scenario', 'Forcing per scenario']

    namespace_w = execution_engine.study_name
    scenario_key = execution_engine.dm.get_data_id(
        f'{namespace_w}.scenario_list')
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))
    filters.append(ChartFilter('Scenarios', scenario_list,
                               scenario_list, 'Scenarios'))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    scatter_scenario = 'Scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    scenario_key = execution_engine.dm.get_data_id(
        f'{execution_engine.study_name}.scenario_list')
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
    graphs_list = ['Temperature per scenario', 'Forcing per scenario']
    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
            if chart_filter.filter_key == 'Scenarios':
                selected_scenarios = chart_filter.selected_values

        selected_scenarios = scenario_list
    """
        -------------
        -------------
        SCENARIO COMPARISON CHART
        -------------
        -------------
    """

    if 'Temperature per scenario' in graphs_list:

        chart_name = 'Atmosphere temperature evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Temperature [degrees Celsius above preindustrial]'

        temperature_dict = {}
        for scenario in scenario_list:
            temperature_detail_df = execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.Temperature.temperature_detail_df')
            temperature_dict[scenario] = temperature_detail_df['temp_atmo'].values.tolist(
            )
            years = temperature_detail_df[GlossaryCore.Years].values.tolist(
            )
        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)
    if 'Forcing per scenario' in graphs_list:

        chart_name = 'Radiative Forcing evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Forcing [W.m-2]'

        temperature_dict = {}
        for scenario in scenario_list:
            temperature_detail_df = execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.Temperature.temperature_detail_df')
            temperature_dict[scenario] = temperature_detail_df['forcing'].values.tolist(
            )
            years = temperature_detail_df[GlossaryCore.Years].values.tolist(
            )
        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

        chart_name = 'CO2 Radiative Forcing evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'CO2 Forcing [W.m-2]'

        forcing_dict = {}
        for scenario in scenario_list:
            forcing_df = execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.Temperature.forcing_detail_df')
            forcing_dict[scenario] = forcing_df['CO2 forcing'].values.tolist(
            )
            years = forcing_df[GlossaryCore.Years].values.tolist(
            )
        new_chart = get_scenario_comparison_chart(years, forcing_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

        chart_name = 'Other Radiative Forcing evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Other Forcing [W.m-2]'

        forcing_dict = {}
        selected_scenarios_other = []
        for scenario in scenario_list:
            forcing_df = deepcopy(execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.Temperature.forcing_detail_df'))
            for col in forcing_df.columns:
                if col not in [GlossaryCore.Years, 'CO2 forcing']:
                    if f'other RF {scenario}' in forcing_dict:
                        forcing_dict[f'other RF {scenario}'] += forcing_df[col].values
                    else:
                        forcing_dict[f'other RF {scenario}'] = forcing_df[col].values
                    if scenario in selected_scenarios:
                        selected_scenarios_other.append(f'other RF {scenario}')
            years = forcing_df[GlossaryCore.Years].values.tolist(
            )
        forcing_dict_in_list = {key: value.tolist()
                                for key, value in forcing_dict.items()}
        new_chart = get_scenario_comparison_chart(years, forcing_dict_in_list,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios_other)

        instanciated_charts.append(new_chart)
    return instanciated_charts


def get_scenario_comparison_chart(x_list, y_dict, chart_name, x_axis_name, y_axis_name, selected_scenarios):

    min_x = min(x_list)
    max_x = max(x_list)
    min_y = min([min(list(y)) for y in y_dict.values()])
    max_y = max([max(list(y)) for y in y_dict.values()])

    new_chart = TwoAxesInstanciatedChart(x_axis_name, y_axis_name,
                                         [min_x - 5, max_x + 5], [
                                             min_y - max_y * 0.05, max_y * 1.05],
                                         chart_name)

    for scenario, y_values in y_dict.items():
        if scenario in selected_scenarios:
            new_series = InstanciatedSeries(
                x_list, y_values, scenario, 'lines', True)

            new_chart.series.append(new_series)

    return new_chart


def get_chart_pareto_front(x_dict, y_dict, scenario_list, namespace_w, chart_name='Pareto Front',
                           x_axis_name='x', y_axis_name='y'):
    '''
    Function that, given two dictionaries and a scenario_list, returns a pareto front

    :params: x_dict, dict containing the data for the x axis of the pareto front per scenario
    :type: dict

    :params: y_dict, dict containing the data for the y axis of the pareto front per scenario
    :type: dict

    :params: scenario_list, list containing the name of the scenarios 
    :type: list

    :params: namespace_w, namespace of scatter scenario
    :type: string

    :params: chart_name, name of the chart used as title 
    :type: string

    :returns: new_pareto_chart, the chart object to be displayed
    :type: InstantiatedParetoFrontOptimalChart
    '''

    min_x = min(list(x_dict.values()))
    max_x = max(list(x_dict.values()))

    max_y = max(list(y_dict.values()))
    min_y = min(list(y_dict.values()))

    new_pareto_chart = InstantiatedParetoFrontOptimalChart(
        abscissa_axis_name=f'{x_axis_name}',
        primary_ordinate_axis_name=f'{y_axis_name}',
        abscissa_axis_range=[min_x - max_x * 0.05, max_x * 1.05],
        primary_ordinate_axis_range=[
            min_y - max_y * 0.03, max_y * 1.03],
        chart_name=chart_name)

    for scenario in scenario_list:
        new_serie = InstanciatedSeries([x_dict[scenario]],
                                       [y_dict[scenario]],
                                       scenario, 'scatter',
                                       custom_data=f'{namespace_w}.{scenario}')
        new_pareto_chart.add_serie(new_serie)

    # Calculating and adding pareto front
    sorted_x = sorted(x_dict.values())
    sorted_scenarios = []
    for val in sorted_x:
        for scen, x_val in x_dict.items():
            if x_val == val:
                sorted_scenarios.append(scen)

    sorted_list = sorted([[x_dict[scenario], y_dict[scenario]]
                          for scenario in sorted_scenarios])
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if pair[1] >= pareto_front[-1][1]:
            pareto_front.append(pair)

    pareto_front_serie = InstanciatedSeries(
        [pair[0] for pair in pareto_front], [pair[1] for pair in pareto_front], 'Pareto front', 'lines')
    new_pareto_chart.add_pareto_front_optimal(pareto_front_serie)

    return new_pareto_chart
