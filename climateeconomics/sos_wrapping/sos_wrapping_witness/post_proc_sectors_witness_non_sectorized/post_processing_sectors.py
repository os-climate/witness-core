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
        GlossaryCore.SectorGdpPart,
        GlossaryCore.SectionGdpPart,
        GlossaryCore.SectionEnergyEmissionPart,
        GlossaryCore.SectionNonEnergyEmissionPart,
        GlossaryCore.SectionEnergyConsumptionPartTWh,
        GlossaryCore.SectionEmissionPart,
        GlossaryCore.ChartTotalEmissionsGt,
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

    damage_df = get_scenario_value(execution_engine, GlossaryCore.DamageDfValue, scenario_name)
    years = list(damage_df[GlossaryCore.Years].values)
    if GlossaryCore.TotalEmissions in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            breakdown_emission_sector = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)

            chart_name = f"Breakdown of emissions for sector {sector} [{GlossaryCore.EmissionDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, f"Emissions [{GlossaryCore.EmissionDf['unit']}]",
                                                 chart_name=chart_name, stacked_bar=True)

            new_series = InstanciatedSeries(
                years, list(breakdown_emission_sector[GlossaryCore.TotalEmissions].values), 'Total emissions', 'lines', True)

            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(breakdown_emission_sector[GlossaryCore.EnergyEmissions].values), 'Energy emissions', 'bar', True)

            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(breakdown_emission_sector[GlossaryCore.NonEnergyEmissions].values), 'Non energy emissions', 'bar', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

    if GlossaryCore.SectionEnergyEmissionPart in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            breakdown_emission_sector = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            sections_energy_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}", scenario_name)

            chart_name = f"Breakdown of energy emission per section for {sector} sector [{GlossaryCore.SectionEnergyEmissionDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEnergyEmissionPart, chart_name=chart_name, stacked_bar=True)
            new_series = InstanciatedSeries(
                years, list(breakdown_emission_sector[GlossaryCore.EnergyEmissions].values), f'Total energy emissions', display_type=InstanciatedSeries.LINES_DISPLAY)
            new_chart.add_series(new_series)

            # loop on all sections of the sector
            for section in GlossaryCore.SectionDictSectors[sector]:
                new_series = InstanciatedSeries(
                    years, list(sections_energy_emission[section].values), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.SectionNonEnergyEmissionPart in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            breakdown_emission_sector = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            sections_non_energy_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionNonEnergyEmissionDfValue}", scenario_name)

            chart_name = f"Breakdown of non energy emission per section for {sector} sector [{GlossaryCore.SectionNonEnergyEmissionDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionNonEnergyEmissionPart,
                                                 chart_name=chart_name, stacked_bar=True)
            new_series = InstanciatedSeries(
                years, list(breakdown_emission_sector[GlossaryCore.NonEnergyEmissions].values), f'Total Non energy emissions',
                display_type=InstanciatedSeries.LINES_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
            new_chart.add_series(new_series)

            for section in GlossaryCore.SectionDictSectors[sector]:
                new_series = InstanciatedSeries(
                    years, list(sections_non_energy_emission[section].values), f'{section}',
                    display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.SectionEnergyConsumptionPartTWh in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            sections_energy_consumption = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}", scenario_name)

            # conversion : dataframe is in PWh but TWh are displayed
            chart_name = f"Breakdown of energy consumption per section for {sector} sector [TWh]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEnergyConsumptionPartTWh,
                                                 chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section in GlossaryCore.SectionDictSectors[sector]:
                new_series = InstanciatedSeries(
                    years, list(sections_energy_consumption[section].values * 1000), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.SectionEmissionPart in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            sections_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEmissionDfValue}", scenario_name)
            chart_name = f"Breakdown of emission per section for {sector} sector [{GlossaryCore.SectionEmissionDf['unit']}]"
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEmissionPart, chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section in GlossaryCore.SectionDictSectors[sector]:
                new_series = InstanciatedSeries(
                    years, list(sections_emission[section].values), f'{section}',
                    display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

        for sector in GlossaryCore.SectorsPossibleValues:
            sections_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEmissionDfValue}", scenario_name)
            sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)

            chart_name = f"Breakdown of emission share per section for {sector} sector [%]"
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "Share percentage per section", chart_name=chart_name, stacked_bar=True)
            for section in GlossaryCore.SectionDictSectors[sector]:
                share_of_section = list(sections_emission[section].values / sector_emissions[GlossaryCore.TotalEmissions].values * 100.)
                new_series = InstanciatedSeries(years, share_of_section, f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.ChartTotalEmissionsGt in chart_list:
        chart_name = f"Breakdown of emissions per sector [GtCO2eq]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ChartTotalEmissionsGt, chart_name=chart_name, stacked_bar=True)
        for sector in GlossaryCore.SectorsPossibleValues:
            sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
            new_series = InstanciatedSeries(years,
                                            list(sector_emissions[GlossaryCore.TotalEmissions].values),
                                            sector, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

        instanciated_charts.append(new_chart)

    if GlossaryCore.SectionGdpPart in chart_list:
        for sector in GlossaryCore.SectorsPossibleValues:
            sections_gdp = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionGdpDfValue}", scenario_name)
            sector_gdp = sections_gdp[GlossaryCore.SectionDictSectors[sector]].sum(axis=1).values
            chart_name = f"Breakdown of GDP per section for {sector} sector [T$]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionGdpPart,
                                                 chart_name=chart_name, stacked_bar=True)

            for section in GlossaryCore.SectionDictSectors[sector]:
                new_series = InstanciatedSeries(
                    years, list(sections_gdp[section].values), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # plot total gdp for the current sector in line
            new_series = InstanciatedSeries(
                years, list(sector_gdp),
                f"Total GDP for {sector} sector",
                'lines', True)
            new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legen, otherwise show it
            if len(GlossaryCore.SectionDictSectors[sector]) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    return instanciated_charts


