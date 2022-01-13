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
from plotly.subplots import make_subplots

import pandas as pd
from copy import deepcopy
from plotly.validators.sankey import domain
from itertools import product

#from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    filters = []

    chart_list = ['Green energies', 'Breakdown price energies']
    # The filters are set to False by default since the graphs are not yet
    # mature
    filters.append(ChartFilter('Charts', chart_list, [], 'Charts'))

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

    if 'Green energies' in graphs_list:

        chart_name = f'Green energies'
        new_chart = get_chart_green_energies(
            execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            new_chart.plotly_fig.add_annotation(xanchor="center", valign="middle", xref="x", yref="y", text="WORK IN PROGRESS",
                                                font={'size': 24},
                                                bordercolor="black", borderwidth=2, borderpad=4,
                                                bgcolor="white", opacity=1.0)
            instanciated_charts.append(new_chart)

    if 'Breakdown price energies' in graphs_list:

        chart_name = f'Breakdown price energies'
        new_chart = get_chart_breakdown_price_energies(
            execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            new_chart.plotly_fig.add_annotation(xanchor="center", valign="middle", xref="x", yref="y", text="WORK IN PROGRESS",
                                                font={'size': 24},
                                                bordercolor="black", borderwidth=2, borderpad=4,
                                                bgcolor="white", opacity=1.0)
            instanciated_charts.append(new_chart)

    return instanciated_charts


def get_chart_green_energies(execution_engine, namespace, chart_name):
    '''! Function to create the green_techno/_energy Sankey diagram with the associated scatter plot
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc

    @return new_chart: InstantiatedPlotlyNativeChart a Sankey Diagram and a Scatter plot side to side  
    '''

    # Prepare data
    sankey_df, years = get_green_energy_sankey_df(
        execution_engine, namespace)
    energy_list = list(set(sankey_df.index.droplevel(1)))
    technologies_list = list(sankey_df.index.droplevel(0))
    green_categories_list = list(
        set([item for sublist in sankey_df['green_category'].values for item in sublist]))
    label_list = energy_list + technologies_list + green_categories_list
    fluxes_name = ['production', 'invest']
    color_name = 'green_category'
    x_scatter = 'price_per_kWh'
    y_scatter = 'CO2_per_kWh'

    # Create Figure
    chart_name = f'{chart_name}'
    fig = go.Figure()

    # Fill figure with data by year
    # i_label_dict associates each label with an integer value
    i_label_dict = dict((key, i) for i, key in enumerate(label_list))
    fig = make_subplots(rows=1, cols=2)
    for i_year in range(len(years)):
        source, target, color = [], [], []
        fluxes = {}
        for flux_name in fluxes_name:
            fluxes[flux_name] = []
        for i, row in sankey_df.iterrows():
            # Add flux from energy to techno
            source += [i_label_dict[i[0]], ]
            target += [i_label_dict[i[1]], ]
            for flux_name in fluxes_name:
                fluxes[flux_name] += [row[flux_name][i_year], ]
            color += [row['color'][i_year], ]
            # Add flux from techno to green category
            source += [i_label_dict[i[1]], ]
            target += [i_label_dict[row[color_name][i_year]], ]
            for flux_name in fluxes_name:
                fluxes[flux_name] += [row[flux_name][i_year], ]
            color += [row['color'][i_year], ]
        link = dict(source=source, target=target,
                    value=fluxes[fluxes_name[0]], color=color)
        customdata = [i_year] + [fluxes[flux_name]
                                 for flux_name in fluxes_name]
        sankey = go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                                     label=list(i_label_dict.keys()), color="black"),
                           link=link,
                           customdata=customdata,
                           domain={'x': [0, 0.45]},
                           visible=False)
        fig.add_trace(sankey)
        price_per_kWh, CO2_per_kWh, color, label = [], [], [], []
        for i, row in sankey_df.iterrows():
            price_per_kWh += [row[x_scatter][i_year], ]
            CO2_per_kWh += [row[y_scatter][i_year], ]
            color += [row[color_name][i_year], ]
            label += [i[1], ]
        scatter = go.Scatter(x=price_per_kWh, y=CO2_per_kWh, hovertext=label,
                             customdata=customdata,
                             marker=dict(size=10, color=color), mode='markers',
                             visible=False)
        fig.add_trace(scatter, row=1, col=2)
        fig.update_xaxes(title_text=x_scatter, row=1, col=2)
        fig.update_yaxes(title_text=y_scatter, row=1, col=2)

    fig.data[0].visible = True
    fig.data[int(len(fig.data) / 2)].visible = True

    # Prepare year slider and layout updates
    steps = []
    for i in range(int(len(fig.data) / 2)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": 'Year: ' + str(2020 + i)}],
            label=str(2020 + i)
        )
        # visible_index = [True if trace['customdata']
        # [0] == i else False for trace in fig.data]
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i + int(len(fig.data) / 2)] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        steps=steps), ]

    layout = {}
    layout.update({'width': 1000})
    layout.update({'height': 450})
    layout.update({'autosize': False})
    layout.update({'showlegend': False})
    fig.update_layout(layout)

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([

                    dict(
                        args=[{'link.value': [trace.customdata[i + 1]
                                              for trace in fig.data]}, ],
                        label=flux_name,
                        method="restyle"
                    ) for i, flux_name in enumerate(fluxes_name)

                ]),
                direction='down',
                type='dropdown',
                pad={"r": 0, "t": 0},
                showactive=True,
                active=0,
                x=0.0,
                y=1.00,
                yanchor='top',
                xanchor='left'
            ),
        ]
    )

    fig.update_layout(
        sliders=sliders
    )

    new_chart = InstantiatedPlotlyNativeChart(
        fig, chart_name=chart_name, default_title=True)
    return new_chart


def get_green_energy_sankey_df(execution_engine, namespace):
    '''! Function to create the sankey dataframe with all the data necessary for the 'green energy' Sankey diagram post_proc
    @param execution engine: Current execution engine object, from which the data is extracted
    @param namespace: Namespace at which the data can be accessed

    @return sankey_df: Dataframe
    '''
    EnergyMix = execution_engine.dm.get_disciplines_with_name(
        f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix')[0]
    # Construct a DataFrame to organize the data on two levels: energy and
    # techno
    idx = pd.MultiIndex.from_tuples([], names=['energy', 'techno'])
    sankey_df = pd.DataFrame(
        index=idx,
        columns=['production', 'invest', 'CO2_per_kWh', 'price_per_kWh'])
    energy_list = EnergyMix.get_sosdisc_inputs('energy_list')
    for energy in energy_list:
        energy_disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix.{energy}')[0]
        techno_list = energy_disc.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{techno}')[0]
            # Data for Sankey diagram
            production_techno = techno_disc.get_sosdisc_outputs(
                'techno_production')[f'{energy} (TWh)'].values *\
                techno_disc.get_sosdisc_inputs(
                    'scaling_factor_techno_production')
            invest_techno = techno_disc.get_sosdisc_inputs('invest_level')[
                f'invest'].values *\
                techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
            CO2_per_kWh_techno = techno_disc.get_sosdisc_outputs('CO2_emissions')[
                f'{techno}'].values
            # Data for scatter plot
            price_per_kWh_techno = techno_disc.get_sosdisc_outputs('techno_prices')[
                f'{techno}'].values
            idx = pd.MultiIndex.from_tuples(
                [(f'{energy}', f'{techno}')], names=['energy', 'techno'])
            columns = ['energy', 'technology',
                       'production', 'invest',
                       'CO2_per_kWh', 'price_per_kWh']
            techno_df = pd.DataFrame([(energy, techno, production_techno, invest_techno, CO2_per_kWh_techno, price_per_kWh_techno)],
                                     index=idx, columns=columns)
            sankey_df = sankey_df.append(techno_df)
    # Add a column with the green categories, based on a threshold on
    # CO2_per_kWh values
    sankey_df['green_category'], sankey_df['color'] = categorize_to_green(
        list(sankey_df['CO2_per_kWh'].values))

    years = np.arange(EnergyMix.get_sosdisc_inputs(
        'year_start'), EnergyMix.get_sosdisc_inputs('year_end') + 1, 1)

    return sankey_df, years


def categorize_to_green(CO2_per_kWh, df_categories=None, dict_category_color=None):
    '''! Function to split CO2_per_kWh list into 'green' categories
    @param CO2_per_kWh: List of list containing the list of techno and their values for each year to be split into categories
    @param df_categories: Dataframe with index == 'category_name' and column == [CO2_per_kWh threshold] 
    @param dict_category_color: Dictionary with the color associated to each category 

    @return categories: List of list containing the name of the category assigned to each value of the input list
    '''
    if df_categories is None:
        df_categories = pd.DataFrame.from_dict({'green': [0.01],
                                                'lightgreen': [0.2],
                                                'grey': [0.5],
                                                'red': [10]}, orient='index',
                                               columns=['threshold'])
    if dict_category_color is None:
        dict_category_color = dict({'green': 'green',
                                    'lightgreen': 'lightgreen',
                                    'grey': 'grey',
                                    'red': 'red'})
    df_categories.sort_values('threshold', inplace=True)
    categories, colors = [], []
    for CO2_per_kWh_techno in CO2_per_kWh:
        category_techno, color_techno = [], []
        for val in CO2_per_kWh_techno:
            for row in df_categories.iterrows():
                if val < row[1]['threshold']:
                    category_techno += [row[0], ]
                    color_techno += [dict_category_color[row[0]], ]
                    break
        categories += [category_techno, ]
        colors += [color_techno, ]
    return categories, colors


def get_chart_breakdown_price_energies(execution_engine, namespace, chart_name):
    '''! Function to create the breakdown_prices_techno/_energy Sankey diagram with the associated scatter plot
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc

    @return new_chart: InstantiatedPlotlyNativeChart a Sankey Diagram and a Scatter plot side to side  
    '''

    # Prepare data
    sankey_df, years = get_price_sankey_df(execution_engine, namespace)

    # Create Figure
    chart_name = f'{chart_name}'
    fig = go.Figure()

    price_sources = ['transport', 'factory',
                     'energy_cost', 'CO2_taxes', 'margin']
    energy_list = list(set(sankey_df.index.droplevel(1)))
    technologies_list = list(sankey_df.index.droplevel(0))
    stream_category_list = list(
        ['Energy', 'CCUS.CarbonCapture', 'CCUS.CarbonStorage'])
    label_list = price_sources + technologies_list + \
        energy_list + stream_category_list

    # i_label_dict associates each label with an integer value
    i_label_dict = dict((key, i) for i, key in enumerate(label_list))
    x_scatter = 'invest'
    y_scatter = 'CO2_per_kWh'

    # Create Figure
    chart_name = f'{chart_name}'
    fig = go.Figure()

    # Fill figure with data by year
    for i_year in range(len(years)):
        source, target, color = [], [], []
        fluxes = []
        for i, row in sankey_df.iterrows():
            # Add flux from price_sources to techno
            for price_source in price_sources:
                source += [i_label_dict[price_source], ]
                target += [i_label_dict[row['technology']], ]
                fluxes += [max(row[f'{price_source}'][i_year], 1e-10), ]
            # Add flux from techno to energy
            source += [i_label_dict[row['technology']], ]
            target += [i_label_dict[row['stream']], ]
            fluxes += [max(row['weighted_techno_price'][i_year], 1e-10), ]
        for key, group_df in sankey_df[['weighted_stream_price', 'stream_category']].groupby(axis=0, level=0):
            # Add flux from energy to stream_category
            source += [i_label_dict[next(group_df.iterrows())[0][0]], ]
            target += [i_label_dict[next(group_df.iterrows())
                                    [1]['stream_category']], ]
            fluxes += [max(next(group_df.iterrows())[1]
                           ['weighted_stream_price'][i_year], 1e-10), ]
        link = dict(source=source, target=target,
                    value=fluxes)
        customdata = [i_year]
        sankey = go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                                     label=list(i_label_dict.keys()), color="black"),
                           link=link,
                           customdata=customdata,
                           visible=False)
        fig.add_trace(sankey)

    fig.data[0].visible = True

    # Prepare year slider and layout updates
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": 'Year: ' + str(2020 + i)}],
            label=str(2020 + i)
        )
        # visible_index = [True if trace['customdata']
        # [0] == i else False for trace in fig.data]
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        steps=steps), ]

    layout = {}
    layout.update({'width': 1000})
    layout.update({'height': 450})
    layout.update({'autosize': False})
    layout.update({'showlegend': False})
    fig.update_layout(layout)

    fig.update_layout(
        sliders=sliders
    )

    new_chart = InstantiatedPlotlyNativeChart(
        fig, chart_name=chart_name, default_title=True)
    return new_chart


def get_price_sankey_df(execution_engine, namespace):
    '''! Function to create the sankey dataframe with all the data necessary for the 'energy price' Sankey diagram post_proc
    @param execution engine: Current execution engine object, from which the data is extracted
    @param namespace: Namespace at which the data can be accessed

    @return sankey_df: Dataframe
    '''
    EnergyMix = execution_engine.dm.get_disciplines_with_name(
        f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix')[0]
    # Construct a DataFrame to organize the data on two levels: energy and
    # techno
    idx = pd.MultiIndex.from_tuples([], names=['stream', 'techno'])
    sankey_df = pd.DataFrame(
        index=idx,
        columns=['stream', 'technology', 'stream_price', 'transport',
                 'factory', 'energy_cost', 'CO2_taxes', 'margin',
                 'production_stream', 'production_techno',
                 'weighted_techno_price', 'weighted_stream_price',
                 'invest', 'CO2_per_kWh', 'stream_category'])
    energy_list = EnergyMix.get_sosdisc_inputs('energy_list')
    for energy in energy_list:
        energy_disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix.{energy}')[0]
        techno_list = energy_disc.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{namespace}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{techno}')[0]
            #---------
            # Data for Sankey diagram
            #---------
            #- Prices
            techno_price = techno_disc.get_sosdisc_outputs('techno_prices')[
                f'{techno}'].values
            stream_price = energy_disc.get_sosdisc_outputs('energy_prices')[
                f'{energy}'].values
            techno_detailed_prices = techno_disc.get_sosdisc_outputs(
                'techno_detailed_prices')
            transport_techno = techno_detailed_prices['transport'].values
            factory_techno = techno_detailed_prices[f'{techno}_factory'].values
            energy_cost_techno = techno_detailed_prices['energy_costs'].values
            CO2_taxes_techno = techno_detailed_prices['CO2_taxes_factory'].values
            margin_techno = (techno_detailed_prices[['transport', f'{techno}_factory', 'energy_costs', 'CO2_taxes_factory']].sum(axis=1) * techno_disc.get_sosdisc_inputs('margin')[
                'margin'] / 100).values
            #- Prod
            production_stream = energy_disc.get_sosdisc_outputs(
                'energy_production')[f'{energy}'].values *\
                energy_disc.get_sosdisc_inputs(
                    'scaling_factor_energy_production')
            production_techno = techno_disc.get_sosdisc_outputs(
                'techno_production')[f'{energy} (TWh)'].values *\
                techno_disc.get_sosdisc_inputs(
                    'scaling_factor_techno_production')
            techno_mix = energy_disc.get_sosdisc_outputs('techno_mix')[
                f'{techno}'].values
            energy_mix = EnergyMix.get_sosdisc_outputs('energy_mix')[
                f'{energy}'].values
            weighted_techno_price = techno_price * techno_mix
            weighted_stream_price = stream_price * energy_mix
            stream_category = 'Energy'

            #---------
            # Data for Scatter plot
            #---------
            invest_techno = techno_disc.get_sosdisc_inputs('invest_level')[
                f'invest'].values *\
                techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
            CO2_per_kWh_techno = techno_disc.get_sosdisc_outputs('CO2_emissions')[
                f'{techno}'].values
            idx = pd.MultiIndex.from_tuples(
                [(f'{energy}', f'{techno}')], names=['stream', 'techno'])
            columns = ['stream', 'technology', 'stream_price', 'transport',
                       'factory', 'energy_cost', 'CO2_taxes', 'margin',
                       'production_stream', 'production_techno',
                       'weighted_techno_price', 'weighted_stream_price',
                       'invest', 'CO2_per_kWh', 'stream_category']
            techno_df = pd.DataFrame([(energy, techno, stream_price, transport_techno, factory_techno,
                                       energy_cost_techno, CO2_taxes_techno, margin_techno,
                                       production_stream, production_techno,
                                       weighted_techno_price, weighted_stream_price,
                                       invest_techno, CO2_per_kWh_techno, stream_category)],
                                     index=idx, columns=columns)
            sankey_df = sankey_df.append(techno_df)

    CCUS = execution_engine.dm.get_disciplines_with_name(
        f'{namespace}.WITNESS_Eval.WITNESS.CCUS')[0]
    ccs_list = CCUS.get_sosdisc_inputs('ccs_list')
    for stream in ccs_list:
        stream_disc = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.WITNESS_Eval.WITNESS.CCUS.{stream}')[0]
        techno_list = stream_disc.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{namespace}.WITNESS_Eval.WITNESS.CCUS.{stream}.{techno}')[0]
            #---------
            # Data for Sankey diagram
            #---------
            #- Prices
            stream_price = stream_disc.get_sosdisc_outputs('energy_prices')[
                f'{stream}'].values
            techno_detailed_prices = techno_disc.get_sosdisc_outputs(
                'techno_detailed_prices')
            transport_techno = techno_detailed_prices['transport'].values
            factory_techno = techno_detailed_prices[f'{techno}_factory'].values
            energy_cost_techno = techno_detailed_prices['energy_costs'].values
            CO2_taxes_techno = techno_detailed_prices['CO2_taxes_factory'].values
            margin_techno = (techno_detailed_prices[['transport', f'{techno}_factory', 'energy_costs', 'CO2_taxes_factory']].sum(axis=1) * techno_disc.get_sosdisc_inputs('margin')[
                'margin'] / 100).values
            #- Prod
            production_stream = stream_disc.get_sosdisc_outputs(
                'energy_production')[f'{stream}'].values *\
                stream_disc.get_sosdisc_inputs(
                    'scaling_factor_energy_production')
            production_techno = techno_disc.get_sosdisc_outputs(
                'techno_production')[f'{stream} (Mt)'].values *\
                techno_disc.get_sosdisc_inputs(
                    'scaling_factor_techno_production')
            techno_mix = stream_disc.get_sosdisc_outputs('techno_mix')[
                f'{techno}'].values
            weighted_techno_price = techno_price * techno_mix
            weighted_stream_price = stream_price
            if stream == 'carbon_capture':
                stream_category = 'CCUS.CarbonCapture'
            if stream == 'carbon_storage':
                stream_category = 'CCUS.CarbonStorage'
            # Data for scatter plot
            invest_techno = techno_disc.get_sosdisc_inputs('invest_level')[
                f'invest'].values *\
                techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
            CO2_per_kWh_techno = techno_disc.get_sosdisc_outputs('CO2_emissions')[
                f'{techno}'].values
            idx = pd.MultiIndex.from_tuples(
                [(f'{stream}', f'{techno}')], names=['stream', 'techno'])
            columns = ['stream', 'technology', 'stream_price', 'transport',
                       'factory', 'energy_cost', 'CO2_taxes', 'margin',
                       'production_stream', 'production_techno',
                       'weighted_techno_price', 'weighted_stream_price',
                       'invest', 'CO2_per_kWh', 'stream_category']
            techno_df = pd.DataFrame([(stream, techno, stream_price, transport_techno, factory_techno,
                                       energy_cost_techno, CO2_taxes_techno, margin_techno,
                                       production_stream, production_techno,
                                       weighted_techno_price, weighted_stream_price,
                                       invest_techno, CO2_per_kWh_techno, stream_category)],
                                     index=idx, columns=columns)
            sankey_df = sankey_df.append(techno_df)

    years = np.arange(EnergyMix.get_sosdisc_inputs(
        'year_start'), EnergyMix.get_sosdisc_inputs('year_end') + 1, 1)

    return sankey_df, years


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
