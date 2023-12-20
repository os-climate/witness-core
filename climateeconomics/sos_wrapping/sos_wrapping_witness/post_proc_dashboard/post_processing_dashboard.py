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

from copy import deepcopy

import numpy as np

import logging

import climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline as TempChange
import climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline as Population
from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import qualitative
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ['temperature and ghg evolution', 'Radiative forcing', 'population and death', 'land use', 'energy mix']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, namespace, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    # execution_engine.dm.get_all_namespaces_from_var_name('temperature_df')[0]

    instanciated_charts = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    if 'temperature and ghg evolution' in chart_list:
        temperature_df = execution_engine.dm.get_value(
                    execution_engine.dm.get_all_namespaces_from_var_name('temperature_detail_df')[0])
        total_ghg_df = execution_engine.dm.get_value(
                    execution_engine.dm.get_all_namespaces_from_var_name('GHG_emissions_df')[0])
        years = temperature_df[GlossaryEnergy.Years].values.tolist()

        chart_name = 'Temperature and GHG evolution over the years'

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=years,
            y=temperature_df[GlossaryCore.TempAtmo].values.tolist(),
            name='Temperature',
            #line=dict(color=qualitative.Set1[0]),
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=years,
            y=total_ghg_df[f'Total CO2 emissions'].to_list(),
            name='Total CO2 emissions',
            line=dict(color=qualitative.Set1[0]),
        ), secondary_y=True)

        fig.update_yaxes(title_text='temperature evolution (degrees Celsius above preindustrial)', rangemode="tozero",
                         secondary_y=False)
        fig.update_yaxes(title_text=f'Total CO2 emissions [Gt]', secondary_y=True, rangemode="tozero",
                         color=qualitative.Set1[0])

        new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if 'temperature evolution' in chart_list:

        model = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('temperature_model')[0])
        temperature_df = deepcopy(
            execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('temperature_detail_df')[0]))

        instanciated_charts = TempChange.temperature_evolution(model,temperature_df,instanciated_charts)

    if 'Radiative forcing' in chart_list:

        forcing_df = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('forcing_detail_df')[0])

        instanciated_charts = TempChange.radiative_forcing(forcing_df,instanciated_charts)

    if 'population and death' in chart_list:

        pop_df = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('population_detail_df')[0])
        death_dict = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('death_dict')[0])
        instanciated_charts = Population.graph_model_world_pop_and_cumulative_deaths(pop_df, death_dict, instanciated_charts)

    if 'land use' in chart_list:

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'surface [Gha]',
                                             chart_name='Surface for forest and food production vs available land over time', stacked_bar=True)

        # total crop surface
        surface_df = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name('food_land_surface_df')[0])
        years = surface_df[GlossaryCore.Years].values.tolist()
        for key in surface_df.keys():
            if key == GlossaryCore.Years:
                pass
            elif key.startswith('total'):
                pass
            else:
                new_series = InstanciatedSeries(
                    years, (surface_df[key]).values.tolist(), key, InstanciatedSeries.BAR_DISPLAY)

                new_chart.add_series(new_series)

        # total food and forest surface, food should be at the bottom to be compared with crop surface
        land_surface_detailed = execution_engine.dm.get_value(execution_engine.dm.get_all_namespaces_from_var_name(LandUseV2.LAND_SURFACE_DETAIL_DF)[0])
        column = 'Forest Surface (Gha)'
        legend = column.replace(' (Gha)', '')
        new_series = InstanciatedSeries(
            years, (land_surface_detailed[column]).values.tolist(), legend, InstanciatedSeries.BAR_DISPLAY)
        new_chart.add_series(new_series)

        column = 'Food Surface (Gha)'
        legend = column.replace(' (Gha)', '')
        new_series = InstanciatedSeries(
            years, (land_surface_detailed[column]).values.tolist(), legend, InstanciatedSeries.LINES_DISPLAY)
        new_chart.add_series(new_series)

        # total land available
        total_land_available = list(land_surface_detailed['Available Agriculture Surface (Gha)'].values + \
                                    land_surface_detailed['Available Forest Surface (Gha)'].values + \
                                    land_surface_detailed['Available Shrub Surface (Gha)'])

        total_land_available_series = InstanciatedSeries(
            years, list(np.ones(len(years)) * total_land_available),
            'Total land available', InstanciatedSeries.LINES_DISPLAY
        )

        new_chart.add_series(total_land_available_series)

        instanciated_charts.append(new_chart)

    if 'energy mix' in chart_list:
        energy_production_detailed = execution_engine.dm.get_value(
            execution_engine.dm.get_all_namespaces_from_var_name(GlossaryEnergy.EnergyProductionDetailedValue)[0])
        energy_mean_price = execution_engine.dm.get_value(
            execution_engine.dm.get_all_namespaces_from_var_name(GlossaryEnergy.EnergyMeanPriceValue)[0])

        years = energy_production_detailed[GlossaryEnergy.Years].values.tolist()

        chart_name = 'Net Energies production/consumption and mean price out of energy mix'

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for reactant in energy_production_detailed.columns:
            if reactant not in [GlossaryEnergy.Years, GlossaryEnergy.TotalProductionValue, 'Total production (uncut)'] \
                    and 'carbon_capture' not in reactant \
                    and 'carbon_storage' not in reactant:
                energy_twh = energy_production_detailed[reactant].values
                legend_title = f'{reactant}'.replace(
                    "(TWh)", "").replace('production', '')

                fig.add_trace(go.Bar(
                    x=years,
                    y=energy_twh.tolist(),
                    #marker_color=color[column],
                    opacity=0.7,
                    name=legend_title,
                ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=years,
            y=energy_mean_price[GlossaryEnergy.EnergyPriceValue].values.tolist(),
            name='mean energy prices [$/MWh]',
            line=dict(color=qualitative.Set1[0]),
        ), secondary_y=True)
        fig.update_layout(
            barmode='stack', )

        fig.update_yaxes(title_text="Net Energy [TWh]", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Prices [$/MWh]", secondary_y=True, rangemode="tozero",
                         color=qualitative.Set1[0])

        new_chart = InstantiatedPlotlyNativeChart(
        fig = fig, chart_name = chart_name)

        instanciated_charts.append(new_chart)


    return instanciated_charts


