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

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import \
    InstantiatedParetoFrontOptimalChart
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.execution_engine.data_manager import DataManager
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME, COUPLING_NAME, EXTRA_NAME
import os.path

WITNESS_SERIES_NAME = 'WITNESS'

REGION = 'Region'
SCENARIO = 'Scenario'
CSV_SEP = ','
CSV_YRS = [str(_yr) for _yr in range(2020, 2101, 10)]

YEARS = 'years'


GDP = 'GDP'

# POSTPROCESSING DICTS KEYS (INTERNAL USAGE)
FILE_NAME = 'file_name'
VAR_NAME = 'var_name'
COLUMN = 'column'
CHART_TITLE = 'chart_title'
Y_AXIS = 'y_axis'
# POSTPROCESSING DICTS WITH ALL THE INFOS TO CONSTRUCT GRAPHS
_gdp = {FILE_NAME: 'gdp_ppp.csv',
        VAR_NAME: 'Macroeconomics.economics_df',
        COLUMN: 'gross_output',
        CHART_TITLE: 'GDP comparison: WITNESS vs. baseline SSP scenarios',
        Y_AXIS: None} #FIXME: manage units
CHARTS_DATA = {GDP: _gdp,
                      }

CHART_LIST = list(CHARTS_DATA.keys())

def get_ssp_data(data_name, region='World'):
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    var_df = pd.read_csv(os.path.join(data_dir, CHARTS_DATA[data_name]['file']), sep=CSV_SEP)
    var_df = var_df[var_df[REGION] == region]
    var_df = var_df[[SCENARIO] + CSV_YRS].set_index(SCENARIO).transpose().reset_index().rename(columns={'index': YEARS})
    var_df.reindex(sorted(var_df.columns), axis=1, copy=False)  # sort the scenarios by name for clarity
    return var_df

def post_processing_filters(execution_engine, namespace):

    filters = []

    # scatter_scenario = 'optimization scenarios'
    # namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
    # scenario_key = execution_engine.dm.get_data_id(
    #     f'{namespace_w}.scenario_list')
    # scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
    filters.append(ChartFilter('Charts', CHART_LIST, CHART_LIST, 'Charts'))
    # filters.append(ChartFilter('Scenarios', scenario_list,
    #                            scenario_list, 'Scenarios'))
    return filters

def _get_comp_chart_from_df(comp_df, y_axis_name, chart_name):
    years = comp_df[YEARS].values.tolist()
    series = comp_df.loc[:, comp_df.columns != YEARS]
    min_x = min(years)
    max_x = max(years)
    min_y = series.min()
    max_y = series.max()
    new_chart = TwoAxesInstanciatedChart(YEARS, y_axis_name,
                                         [min_x - 5, max_x + 5], [
                                         min_y - max_y * 0.05, max_y * 1.05],
                                         chart_name)
    for sc in series.columns:
        new_series = InstanciatedSeries(
            years, series[sc].values.tolist(), sc, 'lines')
        new_chart.series.append(new_series)
    return new_chart

def post_processings(execution_engine, namespace, filters):

    def get_comparison_chart(data_name):
        var_f_name = f"{namespace}.{CHARTS_DATA[data_name][VAR_NAME]}"
        column = CHARTS_DATA[data_name][COLUMN]
        witness_data = execution_engine.dm.get_value(var_f_name)[
            [YEARS, column]].rename(columns={column: WITNESS_SERIES_NAME})
        ssp_data = get_ssp_data(data_name, region='World')
        for scenario in ssp_data.columns:
            f_interp = interp1d(ssp_data[YEARS], ssp_data[scenario])
            scenario_data = f_interp(witness_data[YEARS])
            witness_data[scenario] = scenario_data
        return _get_comp_chart_from_df(witness_data, data_name, CHARTS_DATA[data_name][CHART_TITLE]) # FIXME: manage units

    instanciated_charts = []

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
            # if chart_filter.filter_key == 'Scenarios':
            #     selected_scenarios = chart_filter.selected_values
    else:
        graphs_list = CHART_LIST
        # selected_scenarios = scenario_list

    instanciated_charts.extend(map(get_comparison_chart, graphs_list))
    return instanciated_charts

#     df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.year_start',
#                 f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.year_end', ]
#     year_start_dict, year_end_dict = get_df_per_scenario_dict(
#         execution_engine, df_paths, scenario_list)
#     year_start, year_end = year_start_dict[scenario_list[0]
#                                            ], year_end_dict[scenario_list[0]]
#     years = np.arange(year_start, year_end).tolist()
#
#     """
#         -------------
#         -------------
#         PARETO OPTIMAL CHART
#         -------------
#         -------------
#     """
#
#     if 'Temperature vs Welfare' in graphs_list:
#
#         chart_name = f'Temperature in {year_end} vs Welfare'
#         x_axis_name = f'Temperature increase since industrial revolution in degree Celsius'
#         y_axis_name = 'Welfare'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Temperature_change.temperature_detail_df',
#                     f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df'
#                     ]
#         (temperature_df_dict, utility_df_dict) = get_df_per_scenario_dict(
#             execution_engine, df_paths, scenario_list)
#
#         last_temperature_dict, welfare_dict = {}, {}
#         for scenario in scenario_list:
#             last_temperature_dict[scenario] = temperature_df_dict[scenario]['temp_atmo'][year_end]
#             welfare_dict[scenario] = utility_df_dict[scenario]['welfare'][year_end]
#         namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
#
#         new_pareto_chart = get_chart_pareto_front(last_temperature_dict, welfare_dict, scenario_list,
#                                                   namespace_w, chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name)
#
#         instanciated_charts.append(new_pareto_chart)
#
#     if 'CO2 Emissions vs Welfare' in graphs_list:
#
#         chart_name = f'Sum of CO2 emissions vs Welfare'
#         x_axis_name = f'Summed CO2 emissions'
#         y_axis_name = f'Welfare in {year_end}'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carbon_emissions.CO2_emissions_detail_df',
#                     f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df',
#                     ]
#         (co2_emissions_df_dict, utility_df_dict) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         summed_co2_emissions_dict, welfare_dict = {}, {}
#         for scenario in scenario_list:
#             summed_co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario]['total_emissions'].sum(
#             )
#             welfare_dict[scenario] = utility_df_dict[scenario]['welfare'][year_end]
#         namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
#
#         new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, welfare_dict, scenario_list,
#                                                   namespace_w, chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name)
#
#         instanciated_charts.append(new_pareto_chart)
#
#     if 'CO2 Emissions vs min(Utility)' in graphs_list:
#
#         chart_name = f'CO2 Emissions vs minimum of Utility'
#         x_axis_name = f'Summed CO2 emissions'
#         y_axis_name = 'min( Utility )'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carbon_emissions.CO2_emissions_detail_df',
#                     f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df',
#                     ]
#         (co2_emissions_df_dict, utility_df_dict) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         summed_co2_emissions_dict, min_utility_dict = {}, {}
#         for scenario in scenario_list:
#             summed_co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario]['total_emissions'].sum(
#             )
#             min_utility_dict[scenario] = min(
#                 utility_df_dict[scenario]['discounted_utility'])
#         namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
#
#         new_pareto_chart = get_chart_pareto_front(summed_co2_emissions_dict, min_utility_dict, scenario_list,
#                                                   namespace_w, chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name)
#
#         instanciated_charts.append(new_pareto_chart)
#
#     if 'ppm(mean) vs Welfare' in graphs_list:
#
#         chart_name = f'mean ppm vs Welfare'
#         x_axis_name = f'Mean ppm'
#         y_axis_name = f'Welfare in {year_end}'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carboncycle.carboncycle_detail_df',
#                     f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df',
#                     ]
#         (carboncycle_detail_df_dict, utility_df_dict) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         mean_co2_ppm_dict, welfare_dict = {}, {}
#         for scenario in scenario_list:
#             mean_co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario]['ppm'].mean(
#             )
#             welfare_dict[scenario] = utility_df_dict[scenario]['welfare'][year_end]
#         namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
#
#         new_pareto_chart = get_chart_pareto_front(mean_co2_ppm_dict, welfare_dict, scenario_list,
#                                                   namespace_w, chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name)
#
#         instanciated_charts.append(new_pareto_chart)
#
#     """
#         -------------
#         -------------
#         SCENARIO COMPARISON CHART
#         -------------
#         -------------
#     """
#
#     if 'CO2 tax per scenario' in graphs_list:
#
#         chart_name = 'CO2 tax per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Price ($/tCO2)'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.CO2_taxes', ]
#         (co2_taxes_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#         co2_tax_dict = {}
#         for scenario in scenario_list:
#             co2_tax_dict[scenario] = co2_taxes_df_dict[scenario]['CO2_tax'].values.tolist(
#             )
#
#         new_chart = get_scenario_comparison_chart(years, co2_tax_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     if 'Temperature per scenario' in graphs_list:
#
#         chart_name = 'Atmosphere temperature evolution per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Temperature (degrees Celsius above preindustrial)'
#
#         df_paths = [
#             f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Temperature_change.temperature_detail_df', ]
#         (temperature_detail_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#         temperature_dict = {}
#         for scenario in scenario_list:
#             temperature_dict[scenario] = temperature_detail_df_dict[scenario]['temp_atmo'].values.tolist(
#             )
#
#         new_chart = get_scenario_comparison_chart(years, temperature_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     if 'Welfare per scenario' in graphs_list:
#
#         chart_name = 'Welfare per scenario'
#         y_axis_name = f'Welfare in {year_end}'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df',
#                     ]
#         (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
#
#         welfare_dict = {}
#         for scenario in scenario_list:
#             welfare_dict[scenario] = utility_df_dict[scenario]['welfare'][year_end]
#
#         min_y = min(list(welfare_dict.values()))
#         max_y = max(list(welfare_dict.values()))
#
#         new_chart = TwoAxesInstanciatedChart('', y_axis_name,
#                                              [], [
#                                                  min_y * 0.95, max_y * 1.05],
#                                              chart_name)
#
#         for scenario, welfare in welfare_dict.items():
#             if scenario in selected_scenarios:
#                 serie = InstanciatedSeries(
#                     [''],
#                     [welfare], scenario, 'bar')
#
#                 new_chart.series.append(serie)
#
#         instanciated_charts.append(new_chart)
#
#     if 'Utility per scenario' in graphs_list:
#
#         chart_name = 'Utility per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Discounted Utility (trill $)'
#
#         df_paths = [f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.utility_df', ]
#         (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
#
#         utility_dict = {}
#         for scenario in scenario_list:
#             utility_dict[scenario] = utility_df_dict[scenario]['discounted_utility'].values.tolist(
#             )
#
#         new_chart = get_scenario_comparison_chart(years, utility_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     if 'CO2 emissions per scenario' in graphs_list:
#
#         chart_name = 'CO2 emissions per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Carbon emissions (Gtc)'
#
#         df_paths = [
#             f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carbon_emissions.CO2_emissions_detail_df']
#         (co2_emissions_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         co2_emissions_dict = {}
#         for scenario in scenario_list:
#             co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario]['total_emissions'].values.tolist(
#             )
#
#         new_chart = get_scenario_comparison_chart(years, co2_emissions_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     if 'ppm per scenario' in graphs_list:
#
#         chart_name = 'Atmospheric concentrations parts per million per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Atmospheric concentrations parts per million'
#
#         df_paths = [
#             f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.Carboncycle.carboncycle_detail_df']
#         (carboncycle_detail_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         co2_ppm_dict, welfare_dict = {}, {}
#         for scenario in scenario_list:
#             co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario]['ppm'].values.tolist(
#             )
#
#         new_chart = get_scenario_comparison_chart(years, co2_ppm_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         # Rockstrom Limit
#         ordonate_data = [450] * int(len(years) / 5)
#         abscisse_data = np.linspace(
#             year_start, year_end, int(len(years) / 5))
#         new_series = InstanciatedSeries(
#             abscisse_data.tolist(), ordonate_data, 'Rockstrom limit', 'scatter')
#
#         note = {'Rockstrom limit': 'Scientifical limit of the Earth'}
#         new_chart.annotation_upper_left = note
#
#         new_chart.series.append(new_series)
#
#         instanciated_charts.append(new_chart)
#
#     if 'Total production per scenario' in graphs_list:
#
#         chart_name = 'Total production per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = 'Total production'
#
#         df_paths = [
#             f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.EnergyMix.energy_production_detailed']
#         (energy_production_detailed_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         energy_production_detailed_dict = {}
#         for scenario in scenario_list:
#             energy_production_detailed_dict[scenario] = energy_production_detailed_df_dict[
#                 scenario]['Total production (uncut)'].values.tolist()
#
#         new_chart = get_scenario_comparison_chart(years, energy_production_detailed_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     if 'invest per scenario' in graphs_list:
#         chart_name = f'investments per scenario'
#         x_axis_name = 'Years'
#         y_axis_name = f'total energy investment'
#
#         # Get the total energy investment
#
#         df_paths = [
#             f'{OPTIM_NAME}.{COUPLING_NAME}.{EXTRA_NAME}.energy_investment']
#         (energy_investment_df_dict,) = get_df_per_scenario_dict(
#             execution_engine, df_paths)
#
#         energy_investment_dict = {}
#         for scenario in scenario_list:
#             energy_investment_dict[scenario] = energy_investment_df_dict[
#                 scenario]['energy_investment'].values.tolist()
#
#         new_chart = get_scenario_comparison_chart(years, energy_investment_dict,
#                                                   chart_name=chart_name,
#                                                   x_axis_name=x_axis_name, y_axis_name=y_axis_name, selected_scenarios=selected_scenarios)
#
#         instanciated_charts.append(new_chart)
#
#     return instanciated_charts
#
#
# def get_scenario_comparison_chart(x_list, y_dict, chart_name, x_axis_name, y_axis_name, selected_scenarios):
#
#     min_x = min(x_list)
#     max_x = max(x_list)
#     min_y = min([min(list(y)) for y in y_dict.values()])
#     max_y = max([max(list(y)) for y in y_dict.values()])
#
#     new_chart = TwoAxesInstanciatedChart(x_axis_name, y_axis_name,
#                                          [min_x - 5, max_x + 5], [
#                                              min_y - max_y * 0.05, max_y * 1.05],
#                                          chart_name)
#
#     for scenario, y_values in y_dict.items():
#         if scenario in selected_scenarios:
#             new_series = InstanciatedSeries(
#                 x_list, y_values, scenario, 'lines', True)
#
#             new_chart.series.append(new_series)
#
#     return new_chart
#
#
# def get_chart_pareto_front(x_dict, y_dict, scenario_list, namespace_w, chart_name='Pareto Front',
#                            x_axis_name='x', y_axis_name='y'):
#     '''
#     Function that, given two dictionaries and a scenario_list, returns a pareto front
#
#     :params: x_dict, dict containing the data for the x axis of the pareto front per scenario
#     :type: dict
#
#     :params: y_dict, dict containing the data for the y axis of the pareto front per scenario
#     :type: dict
#
#     :params: scenario_list, list containing the name of the scenarios
#     :type: list
#
#     :params: namespace_w, namespace of scatter scenario
#     :type: string
#
#     :params: chart_name, name of the chart used as title
#     :type: string
#
#     :returns: new_pareto_chart, the chart object to be displayed
#     :type: InstantiatedParetoFrontOptimalChart
#     '''
#
#     min_x = min(list(x_dict.values()))
#     max_x = max(list(x_dict.values()))
#
#     max_y = max(list(y_dict.values()))
#     min_y = min(list(y_dict.values()))
#
#     new_pareto_chart = InstantiatedParetoFrontOptimalChart(
#         abscissa_axis_name=f'{x_axis_name}',
#         primary_ordinate_axis_name=f'{y_axis_name}',
#         abscissa_axis_range=[min_x - max_x * 0.05, max_x * 1.05],
#         primary_ordinate_axis_range=[
#             min_y - max_y * 0.03, max_y * 1.03],
#         chart_name=chart_name)
#
#     for scenario in scenario_list:
#         new_serie = InstanciatedSeries([x_dict[scenario]],
#                                        [y_dict[scenario]],
#                                        scenario, 'scatter',
#                                        custom_data=f'{namespace_w}.{scenario}')
#         new_pareto_chart.add_serie(new_serie)
#
#     # Calculating and adding pareto front
#     sorted_x = sorted(x_dict.values())
#     sorted_scenarios = []
#     for val in sorted_x:
#         for scen, x_val in x_dict.items():
#             if x_val == val:
#                 sorted_scenarios.append(scen)
#
#     sorted_list = sorted([[x_dict[scenario], y_dict[scenario]]
#                           for scenario in sorted_scenarios])
#     pareto_front = [sorted_list[0]]
#     for pair in sorted_list[1:]:
#         if pair[1] >= pareto_front[-1][1]:
#             pareto_front.append(pair)
#
#     pareto_front_serie = InstanciatedSeries(
#         [pair[0] for pair in pareto_front], [pair[1] for pair in pareto_front], 'Pareto front', 'lines')
#     new_pareto_chart.add_pareto_front_optimal(pareto_front_serie)
#
#     return new_pareto_chart
#
#
# def get_df_per_scenario_dict(execution_engine, df_paths, scenario_list=None):
#     '''! Function to retrieve dataframes from all the scenarios given a specified path
#     @param execution_engine: Execution_engine, object from which the data is gathered
#     @param df_paths: list of string, containing the paths to access the df
#
#     @return df_per_scenario_dict: list of dict, with {key = scenario_name: value= requested_dataframe}
#     '''
#     df_per_scenario_dicts = [{} for _ in df_paths]
#     scatter_scenario = 'optimization scenarios'
#     namespace_w = f'{execution_engine.study_name}.{scatter_scenario}'
#     if not scenario_list:
#         scenario_key = execution_engine.dm.get_data_id(
#             f'{namespace_w}.scenario_list')
#         scenario_list = execution_engine.dm.data_dict[scenario_key][DataManager.VALUE]
#     for scenario in scenario_list:
#         for i, df_path in enumerate(df_paths):
#             df_per_scenario_dicts[i][scenario] = execution_engine.dm.get_value(
#                 f'{namespace_w}.{scenario}.{df_path}')
#     return df_per_scenario_dicts
