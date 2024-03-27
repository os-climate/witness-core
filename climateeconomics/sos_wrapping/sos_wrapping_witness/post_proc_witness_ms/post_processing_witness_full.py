'''
Copyright 2022 Airbus SAS
Modifications on 2023/07/18-2023/11/03 Copyright 2023 Capgemini

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
import numpy as np

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME, \
    COUPLING_NAME, EXTRA_NAME
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import \
    InstantiatedParetoFrontOptimalChart


def post_processing_filters(execution_engine, namespace):

    filters = []

    chart_list = ['Temperature vs Utility',
                  'CO2 Emissions vs Utility min',
                  'PPM vs Utility',]

    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    scenario_list = execution_engine.dm.get_value(f'{namespace_w}.samples_df')['scenario_name'].tolist()

    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))
    filters.append(ChartFilter('Scenarios', scenario_list,
                               scenario_list, 'Scenarios'))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    scenario_list = execution_engine.dm.get_value(f'{namespace_w}.samples_df')['scenario_name'].tolist()

    # Overload default value with chart filter
    graphs_list = ['Temperature vs Utility',
                   'CO2 Emissions vs Utility min',
                   'PPM vs Utility',]

    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
            if chart_filter.filter_key == 'Scenarios':
                selected_scenarios = chart_filter.selected_values

        selected_scenarios = scenario_list

    df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.YearStart}',
                f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.YearEnd}', ]
    year_start_dict, year_end_dict = get_df_per_scenario_dict(
        execution_engine, df_paths, scenario_list)
    year_start, year_end = year_start_dict[scenario_list[0]
                                           ], year_end_dict[scenario_list[0]]
    years = np.arange(year_start, year_end).tolist()

    """
        -------------
        -------------
        PARETO OPTIMAL CHART
        -------------
        -------------
    """

    if 'Temperature vs Utility' in graphs_list:

        chart_name = f'Temperature in {year_end} vs Utility'
        x_axis_name = f'Temperature anomaly (°C above pre-industrial)'
        y_axis_name = 'Utility'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Temperature_change.temperature_detail_df',
                    f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}'
                    ]
        (temperature_df_dict, utility_df_dict) = get_df_per_scenario_dict(
            execution_engine, df_paths, scenario_list)

        last_temperature_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            last_temperature_dict[scenario] = temperature_df_dict[scenario][GlossaryCore.TempAtmo][year_end]
            welfare_dict[scenario] = utility_df_dict[scenario][GlossaryCore.DiscountedUtility][year_end]
        namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'

        new_pareto_chart = get_chart_pareto_front(last_temperature_dict, welfare_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'CO2 Emissions vs Utility min' in graphs_list:

        chart_name = f'CO2 Emissions vs Minimum of utility'
        x_axis_name = f'Summed CO2 emissions'
        y_axis_name = 'min( Utility )'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.GHGEmissionsDfValue}',
                    f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}',
                    ]
        (co2_emissions_df_dict, utility_df_dict) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        summed_co2_emissions_dict, min_utility_dict = {}, {}
        for scenario in scenario_list:
            summed_co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario][GlossaryCore.TotalCO2Emissions].sum(
            )
            min_utility_dict[scenario] = min(
                utility_df_dict[scenario][GlossaryCore.DiscountedUtility])
        namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'

        new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, min_utility_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'PPM vs Utility' in graphs_list:

        chart_name = f'Mean ppm vs Welfare'
        x_axis_name = f'Mean ppm'
        y_axis_name = f'Utility in {year_end}'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.GHGCycleDfValue}',
                    f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}',
                    ]
        (carboncycle_detail_df_dict, utility_df_dict) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        mean_co2_ppm_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            mean_co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario][GlossaryCore.CO2Concentration].mean(
            )
            welfare_dict[scenario] = utility_df_dict[scenario][GlossaryCore.DiscountedUtility][year_end]
        namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'

        new_pareto_chart = get_chart_pareto_front(mean_co2_ppm_dict, welfare_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    return instanciated_charts


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


def get_df_per_scenario_dict(execution_engine, df_paths, scenario_list=None):
    '''! Function to retrieve dataframes from all the scenarios given a specified path
    @param execution_engine: Execution_engine, object from which the data is gathered
    @param df_paths: list of string, containing the paths to access the df

    @return df_per_scenario_dict: list of dict, with {key = scenario_name: value= requested_dataframe} 
    '''
    df_per_scenario_dicts = [{} for _ in df_paths]
    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    if not scenario_list:
        scenario_list = execution_engine.dm.get_value(f'{namespace_w}.samples_df')['scenario_name'].tolist()

    for scenario in scenario_list:
        for i, df_path in enumerate(df_paths):
            df_per_scenario_dicts[i][scenario] = execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.{df_path}')
    return df_per_scenario_dicts
