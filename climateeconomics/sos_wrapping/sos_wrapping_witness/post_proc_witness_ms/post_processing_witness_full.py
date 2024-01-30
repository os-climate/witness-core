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
                  #'CO2 Emissions vs Welfare',
                  'CO2 Emissions vs min(Utility)',
                  'CO2 tax per scenario', 'Temperature per scenario', #'Welfare per scenario',
                  'Utility per scenario', 'CO2 emissions per scenario', 'ppm(mean) vs Utility',
                  'Total production per scenario', 'ppm per scenario', 'invest per scenario']

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
                   #'CO2 Emissions vs Welfare',
                   'CO2 Emissions vs min(Utility)'
                                               'CO2 tax per scenario', 'Temperature per scenario',
                   #'Welfare per scenario',
                   'Utility per scenario', 'CO2 emissions per scenario', 'ppm(mean) vs Utility',
                   'Total production per scenario', 'ppm per scenario', 'invest per scenario']

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
        x_axis_name = f'Temperature increase since industrial revolution in degree Celsius'
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

    if 'CO2 Emissions vs Welfare' in graphs_list:

        chart_name = f'Sum of CO2 emissions vs Welfare'
        x_axis_name = f'Summed CO2 emissions'
        y_axis_name = f'Welfare in {year_end}'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.GHGEmissionsDfValue}',
                    f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}',
                    ]
        (co2_emissions_df_dict, utility_df_dict) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        summed_co2_emissions_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            summed_co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario][GlossaryCore.TotalCO2Emissions].sum(
            )
            welfare_dict[scenario] = utility_df_dict[scenario][GlossaryCore.Welfare][year_end]
        namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'

        new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, welfare_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'CO2 Emissions vs min(Utility)' in graphs_list:

        chart_name = f'CO2 Emissions vs minimum of Utility'
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

    if 'ppm(mean) vs Utility' in graphs_list:

        chart_name = f'mean ppm vs Welfare'
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

    """
        -------------
        -------------
        SCENARIO COMPARISON CHART
        -------------
        -------------
    """

    if 'CO2 tax per scenario' in graphs_list:

        chart_name = 'CO2 tax per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Price ($/tCO2)'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.CO2TaxesValue}', ]
        (co2_taxes_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        co2_tax_dict = {}
        for scenario in scenario_list:
            co2_tax_dict[scenario] = co2_taxes_df_dict[scenario][GlossaryCore.CO2Tax].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_tax_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'Temperature per scenario' in graphs_list:

        chart_name = 'Atmosphere temperature evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Temperature (degrees Celsius above preindustrial)'

        df_paths = [
            f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Temperature_change.temperature_detail_df', ]
        (temperature_detail_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        temperature_dict = {}
        for scenario in scenario_list:
            temperature_dict[scenario] = temperature_detail_df_dict[scenario][GlossaryCore.TempAtmo].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'Welfare per scenario' in graphs_list:

        chart_name = 'Welfare per scenario'
        y_axis_name = f'Welfare in {year_end}'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}',
                    ]
        (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        welfare_dict = {}
        for scenario in scenario_list:
            welfare_dict[scenario] = utility_df_dict[scenario][GlossaryCore.Welfare][year_end]

        min_y = min(list(welfare_dict.values()))
        max_y = max(list(welfare_dict.values()))

        new_chart = TwoAxesInstanciatedChart('', y_axis_name,
                                             [], [
                                                 min_y * 0.95, max_y * 1.05],
                                             chart_name)

        for scenario, welfare in welfare_dict.items():
            if scenario in selected_scenarios:
                serie = InstanciatedSeries(
                    [''],
                    [welfare], scenario, 'bar')

                new_chart.series.append(serie)

        instanciated_charts.append(new_chart)

    if 'Utility per scenario' in graphs_list:

        chart_name = 'Utility per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Discounted Utility (trill $)'

        df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.UtilityDfValue}', ]
        (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        utility_dict = {}
        for scenario in scenario_list:
            utility_dict[scenario] = utility_df_dict[scenario][GlossaryCore.DiscountedUtility].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, utility_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'CO2 emissions per scenario' in graphs_list:

        chart_name = 'CO2 emissions per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Carbon emissions (Gtc)'

        df_paths = [
            f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.GHGEmissionsDfValue}']
        (co2_emissions_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_emissions_dict = {}
        for scenario in scenario_list:
            co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario][GlossaryCore.TotalCO2Emissions].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_emissions_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'ppm per scenario' in graphs_list:

        chart_name = 'Atmospheric concentrations parts per million per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Atmospheric concentrations parts per million'

        df_paths = [
            f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.GHGCycleDfValue}']
        (carboncycle_detail_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_ppm_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario][GlossaryCore.CO2Concentration].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_ppm_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        # Rockstrom Limit
        ordonate_data = [450] * int(len(years) / 5)
        abscisse_data = np.linspace(
            year_start, year_end, int(len(years) / 5))
        new_series = InstanciatedSeries(
            abscisse_data.tolist(), ordonate_data, 'Rockstrom limit', 'scatter')

        note = {'Rockstrom limit': 'Scientifical limit of the Earth'}
        new_chart.annotation_upper_left = note

        new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

    if 'Total production per scenario' in graphs_list:

        chart_name = 'Total production per scenario'
        x_axis_name = 'Years'
        y_axis_name = GlossaryCore.TotalProductionValue

        df_paths = [
            f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.EnergyMix.energy_production_detailed']
        (energy_production_detailed_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        energy_production_detailed_dict = {}
        for scenario in scenario_list:
            energy_production_detailed_dict[scenario] = energy_production_detailed_df_dict[
                scenario]['Total production (uncut)'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_production_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'invest per scenario' in graphs_list:
        chart_name = f'investments per scenario'
        x_axis_name = 'Years'
        y_axis_name = f'total energy investment'

        # Get the total energy investment

        df_paths = [
            f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.{GlossaryCore.EnergyInvestmentsValue}']
        (energy_investment_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        energy_investment_dict = {}
        for scenario in scenario_list:
            energy_investment_dict[scenario] = energy_investment_df_dict[
                scenario][GlossaryCore.EnergyInvestmentsValue].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_investment_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

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
