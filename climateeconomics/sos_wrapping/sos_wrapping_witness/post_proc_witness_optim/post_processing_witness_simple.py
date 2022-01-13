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
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart
from sos_trades_core.execution_engine.data_manager import DataManager

import numpy as np
from plotly import graph_objects as go

import pandas as pd
from copy import deepcopy

#from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    filters = []

    chart_list = ['Aggregated Objectives',
                  'Energy Supply', 'Energy Investments', 'CO2 tax constraints', 'MDA residuals']
    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))

    return filters


def post_processings(execution_engine, namespace, filters):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    instanciated_charts = []

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
    else:
        graphs_list = []

    if 'Aggregated Objectives' in graphs_list:

        chart_name = f'Aggregated objectives'
        disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WitnessModelEval.FunctionManagerDisc')[0]
        new_chart = None
        func_df = disc.get_sosdisc_inputs('function_df')
        optim_output_df = disc.get_sosdisc_outputs(disc.OPTIM_OUTPUT_DF)
        parameters_df, obj_list, ineq_list, eq_list = disc.get_parameters_df(
            func_df)
        new_chart = disc.get_chart_aggregated_iterations(
            optim_output=optim_output_df,
            main_parameters=parameters_df.loc[[
                disc.OBJECTIVE, disc.INEQ_CONSTRAINT, disc.EQ_CONSTRAINT]],
            objectives=parameters_df.loc[obj_list],
            ineq_constraints=parameters_df.loc[ineq_list],
            eq_constraints=parameters_df.loc[eq_list], name=chart_name)
        if new_chart is not None:
            instanciated_charts.append(new_chart)
    if 'Energy Supply' in graphs_list:
        disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WitnessModelEval.Energy')[0]
        new_chart = None
        energy_df = deepcopy(disc.get_sosdisc_outputs(
            'energy_supply_detailed_df'))
        energy_history = pd.DataFrame({
            'years': [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900,
                      1910, 1920, 1930, 1940, 1950, 1960, 1965, 1966, 1967, 1968, 1969,
                      1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980,
                      1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
                      1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                      2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                      2014, 2015, 2016, 2017, 2018, 2019],
            'energy_supply': [5653, 5961, 6264, 6653, 7300, 7791, 8005, 8592, 9519, 10659,
                              12101, 15617, 17963, 19837, 22528, 27972, 40589, 50681, 52945, 54668,
                              57484, 60803, 64105, 66327, 69413, 72939, 73005, 73205, 76762, 79119,
                              81347, 83842, 83167, 82609, 82038, 83018, 86344, 88124, 89819, 92616,
                              95720, 97517, 98551, 99086, 99892, 100322, 101592, 103370, 106244, 107301,
                              107962, 109833, 112381, 113558, 115844, 119927, 124931, 128829, 132161, 136141,
                              137309, 135185, 141057, 144327, 145975, 148182, 148955, 149653, 151254, 153513,
                              157366, 158839]
        })
        energy_outlook = pd.DataFrame({'years': [2010, 2017, 2018, 2019, 2020, 2025, 2030, 2040],
                                       'energy_demand': [141057, 153513, 157366, 158839, 149308, 158392.78, 166743.06376, 180816.73428]})
        lifetime = disc.get_sosdisc_inputs('lifetime_fossil')
        new_chart = disc.get_chart_energy_supply(
            energy_df, energy_history, energy_outlook, lifetime)
        if new_chart is not None:
            instanciated_charts.append(new_chart)
    if 'Energy Investments' in graphs_list:
        disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WitnessModelEval.Energy')[0]
        new_chart = None
        invest_share = disc.get_sosdisc_inputs('invest_mix')
        invest = disc.get_sosdisc_inputs('energy_investment')[
            'energy_investment']
        years = list(disc.get_sosdisc_outputs(
            'energy_supply_detailed_df')['years'].values)
        new_chart = disc.get_chart_investment(invest_share, invest, years)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

    if 'CO2 tax constraints' in graphs_list:
        new_chart = None
        disc_damage = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WitnessModelEval.Damage')[0]
        CO2_damage_price = disc_damage.get_sosdisc_outputs('CO2_damage_price')
        years = list(CO2_damage_price['years'].values)
        damage_constraint_factor = disc_damage.get_sosdisc_inputs(
            'damage_constraint_factor')
        disc_energy = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WitnessModelEval.Energy')[0]
        CO2_taxes = disc_energy.get_sosdisc_inputs('CO2_taxes')
        CCS_price_tCO2 = disc_energy.get_sosdisc_outputs('CCS_price')
        CCS_constraint_factor = disc_energy.get_sosdisc_inputs(
            'CCS_constraint_factor')
        new_chart = get_chart_damage_constraint(CO2_damage_price, CO2_taxes, CCS_price_tCO2, years,
                                                damage_constraint_factor, CCS_constraint_factor)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

    if 'MDA residuals' in graphs_list:
        residuals_history_namespace = f'{namespace}.WitnessModelEval.residuals_history'
        if residuals_history_namespace in execution_engine.dm.data_id_map.keys():
            residuals_history = execution_engine.dm.get_value(
                residuals_history_namespace)
            new_chart = get_chart_mda_residuals_plotly(
                residuals=residuals_history, name='MDA residuals', log=True)
            instanciated_charts.append(new_chart)

    return instanciated_charts


def get_chart_mda_residuals_plotly(residuals, name, log=False):

    chart_name = f'{name} wrt iterations'
    fig = go.Figure()

    for sub_mda in residuals.columns:
        x = [res[1] for res in residuals[sub_mda]]
        y = [res[0] for res in residuals[sub_mda]]
        if 'complex' in str(type(y[0])):
            y = [np.real(res[0]) for res in residuals[sub_mda]]
        fig.add_trace(go.Scatter(x=list(x), y=list(y), name=sub_mda))
    if log:
        fig.update_yaxes(type="log", exponentformat='e')
    fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title='n iterations', yaxis_title=f'value of {name}')
    new_chart = InstantiatedPlotlyNativeChart(
        fig, chart_name=chart_name, default_title=True)
    return new_chart


def get_chart_damage_constraint(CO2_damage_price, CO2_taxes, CCS_price_tCO2, years, damage_constraint_factor, CCS_constraint_factor):

    min_value = min(
        [min((CO2_damage_price['CO2_damage_price'] * damage_constraint_factor).values.tolist()),
         min((CO2_taxes['CO2_tax']).values.tolist()),
         min((CCS_price_tCO2['ccs_price_per_tCO2'] * CCS_constraint_factor).values.tolist())])
    max_value = max(
        [max((CO2_damage_price['CO2_damage_price'] * damage_constraint_factor).values.tolist()),
         max((CO2_taxes['CO2_tax']).values.tolist()),
         max((CCS_price_tCO2['ccs_price_per_tCO2'] * CCS_constraint_factor).values.tolist())])
    min_value += -0.1 * abs(min_value)
    max_value += 0.1 * abs(max_value)
    year_start = years[0]
    year_end = years[-1]
    chart_name = 'CO2 tax Constraints'
    new_chart = TwoAxesInstanciatedChart('years', 'Price ($/tCO2)',
                                         [year_start - 5, year_end + 5],
                                         [min_value, max_value],
                                         chart_name)
    visible_line = True
    # add CO2 damage price serie
    new_series = InstanciatedSeries(
        years, (CO2_damage_price['CO2_damage_price']
                * damage_constraint_factor).values.tolist(),
        'Damage constraint factor * CO2 damage price', 'lines', visible_line)
    new_chart.series.append(new_series)
    # add CO2 tax serie
    new_series = InstanciatedSeries(
        years, CO2_taxes['CO2_tax'].values.tolist(),
        'CO2 tax', 'lines', visible_line)
    new_chart.series.append(new_series)
    # add CCS price serie
    new_series = InstanciatedSeries(
        years, (CCS_price_tCO2['ccs_price_per_tCO2'] *
                CCS_constraint_factor).values.tolist(),
        'CCS constraint factor * CCS price', 'lines', visible_line)
    new_chart.series.append(new_series)
    return new_chart
