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
        GlossaryCore.TotalEmissions,
        GlossaryCore.EnergyEmissions,
        GlossaryCore.NonEnergyEmissions,
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    instanciated_charts = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    total_emission_macroeco = get_scenario_value(execution_engine, GlossaryCore.EconomicsEmissionDfValue, scenario_name)
    years = list(total_emission_macroeco[GlossaryCore.Years].values)

    if GlossaryCore.TotalEmissions in chart_list:
        chart_name = f"Breakdown of emissions per sector [{GlossaryCore.EmissionDf['unit']}]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.EmissionDf['unit'], chart_name=chart_name, stacked_bar=True)
        for sector in GlossaryCore.DefaultSectorListGHGEmissions:
            sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            new_series = InstanciatedSeries(years,
                                            list(sector_emissions[GlossaryCore.TotalEmissions].values),
                                            sector, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
        new_series = InstanciatedSeries(years,
                                        list(total_emission_macroeco[GlossaryCore.TotalEmissions].values),
                                        "Total", display_type=InstanciatedSeries.LINES_DISPLAY)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    if GlossaryCore.EnergyEmissions in chart_list:
        chart_name = f"Breakdown of energy emissions per sector [{GlossaryCore.EmissionDf['unit']}]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.EmissionDf['unit'], chart_name=chart_name, stacked_bar=True)
        for sector in GlossaryCore.DefaultSectorListGHGEmissions:
            sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            new_series = InstanciatedSeries(years,
                                            list(sector_emissions[GlossaryCore.EnergyEmissions].values),
                                            sector, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
        new_series = InstanciatedSeries(years,
                                        list(total_emission_macroeco[GlossaryCore.EnergyEmissions].values),
                                        "Total energy emissions", display_type=InstanciatedSeries.LINES_DISPLAY)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    if GlossaryCore.NonEnergyEmissions in chart_list:
        chart_name = f"Breakdown of non-energy emissions per sector [{GlossaryCore.EmissionDf['unit']}]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.EmissionDf['unit'], chart_name=chart_name, stacked_bar=True)
        for sector in GlossaryCore.DefaultSectorListGHGEmissions:
            sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            new_series = InstanciatedSeries(years,
                                            list(sector_emissions[GlossaryCore.NonEnergyEmissions].values),
                                            sector, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
        new_series = InstanciatedSeries(years,
                                        list(total_emission_macroeco[GlossaryCore.NonEnergyEmissions].values),
                                        "Total non energy emissions", display_type=InstanciatedSeries.LINES_DISPLAY)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    return instanciated_charts
