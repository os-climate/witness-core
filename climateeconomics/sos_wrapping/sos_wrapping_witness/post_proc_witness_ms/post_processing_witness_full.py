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
from sos_trades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import \
    InstantiatedParetoFrontOptimalChart
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.execution_engine.data_manager import DataManager
import numpy as np
import pandas as pd

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME, COUPLING_NAME, EXTRA_NAME


def post_processing_filters(execution_engine, namespace):

    filters = []

    chart_list = ['Temperature vs Welfare',
                  'CO2 Emissions vs Welfare', 'CO2 Emissions vs min(Utility)',
                  'CO2 tax per scenario', 'Temperature per scenario', 'Welfare per scenario',
                  'Utility per scenario', 'CO2 emissions per scenario', 'ppm(mean) vs Welfare',
                  'Total production per scenario', 'ppm per scenario']

    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    scenario_key = execution_engine.dm.get_data_id(
        f'{namespace_w}.scenario_list')
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
    try:
        # In case of flattened invest distribution, the two graphs in except
        # are not working
        invest_mix = execution_engine.dm.get_value(
            f'{namespace_w}.{scenario_list[0]}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.InvestmentDistribution.all_invest_df')
        list_technos = list(invest_mix.columns)[:-1]
        chart_list += list_technos
    except:
        execution_engine.logger.warning('all invest df not in dm')
    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))
    filters.append(ChartFilter('Scenarios', scenario_list,
                               scenario_list, 'Scenarios'))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    scenario_key = execution_engine.dm.get_data_id(
        f'{namespace_w}.scenario_list')
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
    try:
        # In case of flattened invest distribution, the two graphs in
        # except are not working
        is_flattened_invest_distrib = True
        invest_mix = execution_engine.dm.get_value(
            f'{namespace_w}.{scenario_list[0]}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.InvestmentDistribution.all_invest_df')
        list_technos = list(invest_mix.columns)[:-1]
    except:
        is_flattened_invest_distrib = False

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
            if chart_filter.filter_key == 'Scenarios':
                selected_scenarios = chart_filter.selected_values
    else:
        graphs_list = ['Temperature vs Welfare',
                       'CO2 Emissions vs Welfare', 'CO2 Emissions vs min(Utility)'
                       'CO2 tax per scenario', 'Temperature per scenario', 'Welfare per scenario',
                       'Utility per scenario', 'CO2 emissions per scenario', 'ppm(mean) vs Welfare',
                       'Total production per scenario', 'ppm per scenario']
        if is_flattened_invest_distrib:
            graphs_list += list_technos

        selected_scenarios = scenario_list

    temperature_df_dict, utility_df_dict, co2_emissions_df_dict, co2_ppm_df_dict, CO2_taxes_df_dict, energy_production_detailed_df_dict, energy_investment_df_dict, ccs_investment_df_dict, scenario_list, year_start, year_end, namespace_w, energy_list = get_df(
        execution_engine, is_flattened_invest_distrib)

    # prepare data
    temperature_dict = {}
    last_temperature_dict = {}
    welfare_dict = {}
    co2_emisions_dict = {}
    summed_co2_emissions_dict = {}
    co2_ppm_dict = {}
    mean_co2_ppm_dict = {}
    utility_dict = {}
    min_utility_dict = {}
    co2_tax_dict = {}
    energy_production_detailed_dict = {}
    energy_investment_dict = {}
    ccs_investment_dict = {}
    years = list(np.arange(year_start, year_end + 1))

    for techno in list_technos:
        energy_investment_dict[techno] = {}

    for scenario in scenario_list:
        temperature_dict[scenario] = temperature_df_dict[scenario]['temp_atmo'].values.tolist(
        )
        last_temperature_dict[scenario] = temperature_df_dict[scenario]['temp_atmo'][year_end]
        welfare_dict[scenario] = utility_df_dict[scenario]['welfare'][year_end]
        co2_emisions_dict[scenario] = co2_emissions_df_dict[scenario]['total_emissions'].values.tolist()
        summed_co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario]['total_emissions'].sum(
        )
        co2_ppm_dict[scenario] = co2_ppm_df_dict[scenario]['ppm'].values.tolist()
        mean_co2_ppm_dict[scenario] = co2_ppm_df_dict[scenario]['ppm'].mean()
        utility_dict[scenario] = utility_df_dict[scenario]['discounted_utility'].values.tolist()
        min_utility_dict[scenario] = min(
            utility_df_dict[scenario]['discounted_utility'])

        co2_tax_dict[scenario] = CO2_taxes_df_dict[scenario]['CO2_tax'].values.tolist()
        energy_production_detailed_dict[scenario] = energy_production_detailed_df_dict[
            scenario]['Total production (uncut)'].values.tolist()
        for techno in list_technos:
            energy_investment_dict[techno][scenario] = energy_investment_df_dict[scenario][techno].values.tolist(
            )

    """
        -------------
        -------------
        PARETO OPTIMAL CHART
        -------------
        -------------
    """

    if 'Temperature vs Welfare' in graphs_list:

        chart_name = f'Temperature in {year_end} vs Welfare'
        x_axis_name = f'Temperature increase since industrial revolution in degree Celsius'
        y_axis_name = 'Welfare'

        new_pareto_chart = get_chart_pareto_front(last_temperature_dict, welfare_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'CO2 Emissions vs Welfare' in graphs_list:

        chart_name = f'Sum of CO2 emissions vs Welfare'
        x_axis_name = f'Summed CO2 emissions'
        y_axis_name = f'Welfare in {year_end}'

        new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, welfare_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'CO2 Emissions vs min(Utility)' in graphs_list:

        chart_name = f'CO2 Emissions vs minimum of Utility'
        x_axis_name = f'Summed CO2 emissions'
        y_axis_name = 'min( Utility )'

        new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, min_utility_dict, scenario_list,
                                                  namespace_w, chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

        instanciated_charts.append(new_pareto_chart)

    if 'ppm(mean) vs Welfare' in graphs_list:

        chart_name = f'mean ppm vs Welfare'
        x_axis_name = f'Mean ppm'
        y_axis_name = f'Welfare in {year_end}'

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

        new_chart = get_scenario_comparison_chart(years, co2_tax_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'Temperature per scenario' in graphs_list:

        chart_name = 'Atmosphere temperature evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Temperature (degrees Celsius above preindustrial)'

        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'Welfare per scenario' in graphs_list:

        chart_name = 'Welfare per scenario'
        y_axis_name = f'Welfare in {year_end}'

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

        new_chart = get_scenario_comparison_chart(years, utility_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'CO2 emissions per scenario' in graphs_list:

        chart_name = 'CO2 emissions per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Carbon emissions (Gtc)'

        new_chart = get_scenario_comparison_chart(years, co2_emisions_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    if 'ppm per scenario' in graphs_list:

        chart_name = 'Atmospheric concentrations parts per million per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Atmospheric concentrations parts per million'

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
        y_axis_name = 'Total production'

        new_chart = get_scenario_comparison_chart(years, energy_production_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)

        instanciated_charts.append(new_chart)

    for techno in list_technos:
        if techno in graphs_list:
            chart_name = f'{techno} investment per scenario'
            x_axis_name = 'Years'
            y_axis_name = f'{techno} investment'

            new_chart = get_scenario_comparison_chart(years, energy_investment_dict[techno],
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


def get_df(execution_engine, is_flattened_invest_distrib=False):

    temperature_df_dict = {}
    co2_emissions_df_dict = {}
    utility_df_dict = {}
    co2_ppm_df_dict = {}
    CO2_taxes_df_dict = {}
    energy_production_detailed_df_dict = {}
    energy_investment_df_dict = {}
    ccs_investment_df_dict = {}
    scatter_scenario = 'optimization scenarios'
    namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'

    scenario_key = execution_engine.dm.get_data_id(
        f'{namespace_w}.scenario_list')
    scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]

    for scenario in scenario_list:
        temperature_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Temperature_change.temperature_detail_df'
        utility_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df'
        co2_emissions_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carbon_emissions.emissions_detail_df'
        co2_ppm_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carboncycle.carboncycle_detail_df'
        CO2_taxes_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.CO2_taxes'
        energy_list_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.energy_list'
        energy_production_detailed_df_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.EnergyMix.energy_production_detailed'
        invest_mix_namespace = f'{namespace_w}.{scenario}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.InvestmentDistribution.all_invest_df'
        temperature_df_dict[scenario] = execution_engine.dm.get_value(
            temperature_df_namespace)
        utility_df_dict[scenario] = execution_engine.dm.get_value(
            utility_df_namespace)
        co2_emissions_df_dict[scenario] = execution_engine.dm.get_value(
            co2_emissions_df_namespace)
        co2_ppm_df_dict[scenario] = execution_engine.dm.get_value(
            co2_ppm_df_namespace)
        CO2_taxes_df_dict[scenario] = execution_engine.dm.get_value(
            CO2_taxes_df_namespace)
        energy_production_detailed_df_dict[scenario] = execution_engine.dm.get_value(
            energy_production_detailed_df_namespace)

        energy_list = execution_engine.dm.get_value(energy_list_namespace)
        energy_investment_df_dict[scenario] = execution_engine.dm.get_value(
            invest_mix_namespace)

    year_start_namespace = f'{namespace_w}.{scenario_list[0]}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.year_start'
    year_start = execution_engine.dm.get_value(year_start_namespace)
    year_end_namespace = f'{namespace_w}.{scenario_list[0]}.{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.year_end'
    year_end = execution_engine.dm.get_value(year_end_namespace)

    return temperature_df_dict, utility_df_dict, co2_emissions_df_dict, co2_ppm_df_dict, CO2_taxes_df_dict, energy_production_detailed_df_dict, energy_investment_df_dict, ccs_investment_df_dict, scenario_list, year_start, year_end, namespace_w, energy_list
