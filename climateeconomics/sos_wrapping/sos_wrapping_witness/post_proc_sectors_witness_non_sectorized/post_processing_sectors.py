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
        GlossaryCore.SectorGdpPart,
        GlossaryCore.SectionGdpPart,
        GlossaryCore.SectionEnergyEmissionPart,
        GlossaryCore.SectionNonEnergyEmissionPart,
        GlossaryCore.SectionEnergyConsumptionPartTWh,
        GlossaryCore.SectionEmissionPart,
        GlossaryCore.ChartTotalEmissionsGt,
        GlossaryCore.ChartTotalEnergyConsumptionSector
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
    sector_gdp_df = get_scenario_value(execution_engine, GlossaryCore.SectorGdpDfValue, scenario_name)
    years = list(damage_df[GlossaryCore.Years].values)
    if GlossaryCore.SectionEnergyEmissionPart in chart_list:
        sections_energy_emission_dict = get_scenario_value(execution_engine, GlossaryCore.SectionEnergyEmissionsDictName, scenario_name)
        list_sectors_to_plot = [sector_name for sector_name in sections_energy_emission_dict.keys() if
                                sector_name != "total"]

        for sector_name in list_sectors_to_plot:
            sections_energy_emission = sections_energy_emission_dict[sector_name]["detailed"]
            sections_energy_emission = sections_energy_emission.drop('years', axis=1)
            energy_emissions = list(
                sections_energy_emission_dict[sector_name]["total"][
                    GlossaryCore.TotalEnergyEmissionsSectorName].values / 1000)

            chart_name = f'Breakdown of energy emission per section for {sector_name} sector [GtCO2eq]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEnergyEmissionPart,
                                                 chart_name=chart_name, stacked_bar=True)
            new_series = InstanciatedSeries(
                years, energy_emissions, f'Total energy emissions',
                display_type=InstanciatedSeries.LINES_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
            new_chart.add_series(new_series)

            # loop on all sections of the sector
            for section, section_value in sections_energy_emission.items():
                new_series = InstanciatedSeries(
                    years, list(section_value / 1000), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_energy_emission.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.SectionNonEnergyEmissionPart in chart_list:
        sections_non_energy_emission_dict = get_scenario_value(execution_engine, GlossaryCore.SectionNonEnergyEmissionsDictName, scenario_name)
        list_sectors_to_plot = [sector_name for sector_name in sections_non_energy_emission_dict.keys() if
                                sector_name != "total"]

        for sector_name in list_sectors_to_plot:
            sections_non_energy_emission = sections_non_energy_emission_dict[sector_name]["detailed"]
            sections_non_energy_emission = sections_non_energy_emission.drop('years', axis=1)

            # check sum of all values of dataframe is not equal to 0
            if sections_non_energy_emission.sum().sum() != 0:
                chart_name = f'Breakdown of non energy emission per section for {sector_name} sector [GtCO2eq]'

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionNonEnergyEmissionPart,
                                                     chart_name=chart_name, stacked_bar=True)
                non_energy_emissions = list(sections_non_energy_emission_dict[sector_name]["total"][
                                                GlossaryCore.TotalNonEnergyEmissionsSectorName].values / 1000)
                new_series = InstanciatedSeries(
                    years, non_energy_emissions, f'Total Non energy emissions',
                    display_type=InstanciatedSeries.LINES_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
                new_chart.add_series(new_series)

                # loop on all sections of the sector
                for section, section_value in sections_non_energy_emission.items():
                    new_series = InstanciatedSeries(
                        years, list(section_value / 1000), f'{section}',
                        display_type=InstanciatedSeries.BAR_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
                    new_chart.add_series(new_series)

                # have a full label on chart (for long names)
                fig = new_chart.to_plotly()
                fig.update_traces(hoverlabel=dict(namelength=-1))
                # if dictionaries has big size, do not show legend, otherwise show it
                if len(list(sections_non_energy_emission.keys())) > 5:
                    fig.update_layout(showlegend=False)
                else:
                    fig.update_layout(showlegend=True)
                instanciated_charts.append(InstantiatedPlotlyNativeChart(
                    fig, chart_name=chart_name,
                    default_title=True, default_legend=False))

    if GlossaryCore.SectionEnergyConsumptionPartTWh in chart_list:
        sections_energy_consumption_dict = get_scenario_value(execution_engine, GlossaryCore.SectionEnergyConsumptionDictName, scenario_name)
        list_sectors_to_plot = [sector_name for sector_name in sections_energy_consumption_dict.keys() if
                                sector_name != "total"]

        for sector_name in list_sectors_to_plot:
            sections_energy_consumption = sections_energy_consumption_dict[sector_name]["detailed"]
            sections_energy_consumption = sections_energy_consumption.drop('years', axis=1)

            chart_name = f'Breakdown of energy consumption per section for {sector_name} sector [TWh]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEnergyConsumptionPartTWh,
                                                 chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section, section_value in sections_energy_consumption.items():
                new_series = InstanciatedSeries(
                    years, list(section_value), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_energy_consumption.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.SectionEmissionPart in chart_list:
        total_emissions_dict = get_scenario_value(execution_engine, GlossaryCore.SectorTotalEmissionsDictName, scenario_name)
        # get sectors available in dictionnaries
        list_sectors_to_plot = [sector_name for sector_name in total_emissions_dict.keys() if
                                sector_name != "total"]
        for sector_name in list_sectors_to_plot:
            sections_emission = total_emissions_dict[sector_name]["detailed"]
            sections_emission = sections_emission.drop('years', axis=1)

            chart_name = f'Breakdown of emission per section for {sector_name} sector [GtCO2eq]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEmissionPart,
                                                 chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section, section_value in sections_emission.items():
                new_series = InstanciatedSeries(
                    years, list(section_value / 1000.), f'{section}',
                    display_type=InstanciatedSeries.BAR_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_emission.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

        ########## percentage chart
        for sector_name in list_sectors_to_plot:
            sections_emission = total_emissions_dict[sector_name]["detailed"]
            sections_emission = sections_emission.drop('years', axis=1)
            chart_name = f'Breakdown of emission share per section for {sector_name} sector [%]'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "Share percentage per section",
                                                 chart_name=chart_name, stacked_bar=True)
            # compute total sections_emission
            total_emissions = sections_emission.sum(axis=1).values
            # loop on all sections of the sector
            for section, section_value in sections_emission.items():
                new_series = InstanciatedSeries(
                    years, list(100. * section_value.values / total_emissions), f'{section}',
                    display_type=InstanciatedSeries.BAR_DISPLAY)  # TODO change this and move it to model if we decide that it is in Gt
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_emission.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    if GlossaryCore.ChartTotalEmissionsGt in chart_list:
        total_emissions_dict = get_scenario_value(execution_engine, GlossaryCore.SectorTotalEmissionsDictName, scenario_name)
        list_sectors_to_plot = [sector_name for sector_name in total_emissions_dict.keys() if
                                sector_name != "total"]
        chart_name = f'Breakdown of emissions per sector [GtCO2eq]'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ChartTotalEmissionsGt,
                                             chart_name=chart_name, stacked_bar=True)
        for sector_name in list_sectors_to_plot:
            sector_emissions = total_emissions_dict[sector_name]["total"]
            sector_emissions = sector_emissions.drop(GlossaryCore.Years, axis=1)
            new_series = InstanciatedSeries(years,
                                            list(sector_emissions[GlossaryCore.TotalEmissionsName].values / 1000.),
                                            # TODO change this and move it to model if we decide that it is in Gt
                                            sector_name, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

        instanciated_charts.append(new_chart)

    if GlossaryCore.ChartTotalEnergyConsumptionSector in chart_list:
        total_consumption_dict = get_scenario_value(execution_engine, GlossaryCore.SectionEnergyConsumptionDictName, scenario_name)
        list_sectors_to_plot = [sector_name for sector_name in total_consumption_dict.keys() if
                                sector_name != "total"]
        chart_name = f'Breakdown of energy consumption per sector [TWh]'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ChartTotalEnergyConsumptionSector,
                                             chart_name=chart_name, stacked_bar=True)
        for sector_name in list_sectors_to_plot:
            sector_emissions = total_consumption_dict[sector_name]["total"]
            sector_emissions = sector_emissions.drop(GlossaryCore.Years, axis=1)
            new_series = InstanciatedSeries(years, list(
                sector_emissions[GlossaryCore.TotalEnergyConsumptionSectorName].values),
                                            sector_name, display_type=InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

        instanciated_charts.append(new_chart)

    if GlossaryCore.SectionGdpPart in chart_list:
        dict_sections_detailed = get_scenario_value(execution_engine, GlossaryCore.SectionGdpDictValue, scenario_name)
        # loop on all sectors to plot a chart per sector
        for sector, dict_section in dict_sections_detailed.items():

            chart_name = f'Breakdown of GDP per section for {sector} sector [T$]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionGdpPart,
                                                 chart_name=chart_name, stacked_bar=True)

            # loop on all sections of current sector
            for section, section_value in dict_section.items():
                new_series = InstanciatedSeries(
                    years, list(section_value), f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # plot total gdp for the current sector in line
            new_series = InstanciatedSeries(
                years, list(sector_gdp_df[sector]),
                f"Total GDP for {sector} sector",
                'lines', True)
            new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legen, otherwise show it
            if len(list(dict_section.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

    return instanciated_charts


