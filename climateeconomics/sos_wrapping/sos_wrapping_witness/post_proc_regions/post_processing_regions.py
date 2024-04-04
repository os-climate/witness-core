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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline as MacroEconomics
import climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline as Population
from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.glossarycore import GlossaryCore
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, \
    InstanciatedSeries
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


def get_scenario_value(execution_engine, var_name, scenario_name):
    """returns the value of a variable for the specified scenario"""
    all_scenario_varnames = execution_engine.dm.get_all_namespaces_from_var_name(var_name)
    if len(all_scenario_varnames) > 1:
        # multiscenario case
        scenario_name = scenario_name.split('.')[2]
        selected_scenario_varname = list(filter(lambda x: scenario_name in x, all_scenario_varnames))[0]
    else:
        # not multiscenario case
         selected_scenario_varname = all_scenario_varnames[0]
    value_selected_scenario = execution_engine.dm.get_value(selected_scenario_varname)
    return value_selected_scenario

def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = [
        GlossaryCore.ChartGDPPerGroup,
        GlossaryCore.ChartPercentagePerGroup,
        GlossaryCore.ChartGDPBiggestEconomies,
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    # execution_engine.dm.get_all_namespaces_from_var_name('temperature_df')[0]
    NUMBERCOUNTRIESTOPLOT = 10

    instanciated_charts = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    if GlossaryCore.ChartGDPPerGroup in chart_list:

        # get variable with total gdp per region
        total_gdp_per_region_df = get_scenario_value(execution_engine, GlossaryCore.TotalGDPGroupDFName, scenario_name)

        years = list(total_gdp_per_region_df[GlossaryCore.Years])

        chart_name = 'GDP-PPP adjusted per group of countries in T$2020'
        # create new chart
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'GDP [trillion $2020]',
                                             chart_name=chart_name, y_min_zero=True)

        visible_line = True
        # loop on each column to show GDP per group of countries
        for col in total_gdp_per_region_df.columns:
            if col != GlossaryCore.Years:
                ordonate_data = list(total_gdp_per_region_df[col].values)
                # refactor name of column (group name)
                column_name_updated = col.replace('_', ' ').title()
                # add new serie
                new_series = InstanciatedSeries(
                    years, ordonate_data, f'{column_name_updated}', InstanciatedSeries.LINES_DISPLAY, visible_line)
                new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    if GlossaryCore.ChartPercentagePerGroup in chart_list:

        # get variable with total gdp per region
        total_percentage_per_region_df = get_scenario_value(execution_engine, GlossaryCore.PercentageGDPGroupDFName, scenario_name)

        chart_name = 'Percentage of GDP-PPP adjusted per group of countries in [%]'
        # create new chart
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Percentage of GDP per group of countries in [%]',
                                             chart_name=chart_name, stacked_bar=True)

        visible_line = True
        # loop on each column to show GDP per group of countries
        for col in total_percentage_per_region_df.columns:
            if col != GlossaryCore.Years:
                ordonate_data = list(total_percentage_per_region_df[col].values)
                # refactor name of column (group name)
                column_name_updated = col.replace('_', ' ').title()
                # add new serie
                new_series = InstanciatedSeries(
                    years, ordonate_data, f'{column_name_updated}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

        instanciated_charts.append(new_chart)

    # The ten of countries with the highest GDP per year
    if GlossaryCore.ChartGDPBiggestEconomies in chart_list:
        # get variable with total GDP per countries
        total_gdp_per_countries_df = get_scenario_value(execution_engine, GlossaryCore.GDPCountryDFName, scenario_name)
        economics_df = get_scenario_value(execution_engine, GlossaryCore.EconomicsDetailDfValue, scenario_name)
        # Take the year 2020 as a reference to determine the ten biggest countries in terms of GDP
        # Rank GDP in descending order to select the x countries with the biggest GDP
        # Find the name of the x biggest  countries
        year_start = years[0]
        list_biggest_countries = \
        total_gdp_per_countries_df[total_gdp_per_countries_df[GlossaryCore.Years] == year_start].sort_values(by='gdp',
                                                                                                             ascending=False)[
            'country_name'].values[:NUMBERCOUNTRIESTOPLOT + 1]

        chart_name = f'The {NUMBERCOUNTRIESTOPLOT} biggest countries GDP-PPP adjusted per year in [G$]'

        # create new chart
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                             f'The {NUMBERCOUNTRIESTOPLOT} biggest countries GDP-PPP adjusted per year in [G$]',
                                             chart_name=chart_name, stacked_bar=True)
        visible_line = True

        # loop on each column to show GDP per group of countries
        for country in list_biggest_countries:
            ordonate_data = total_gdp_per_countries_df[total_gdp_per_countries_df[GlossaryCore.CountryName] == country][
                GlossaryCore.GDPName].to_list()
            # refactor name of column (country name)
            country_name = country.title()
            # add new serie
            new_series = InstanciatedSeries(
                years, ordonate_data, f'{country_name}', display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

        # add total gdp line
        # convert T$ to G$
        gdp_net_of_damage = 1000. * economics_df[GlossaryCore.OutputNetOfDamage].values
        new_series = InstanciatedSeries(
            years, list(gdp_net_of_damage),
            f"Total GDP [G$]",
            InstanciatedSeries.LINES_DISPLAY, visible_line)
        new_chart.add_series(new_series)

        instanciated_charts.append(new_chart)

    return instanciated_charts


