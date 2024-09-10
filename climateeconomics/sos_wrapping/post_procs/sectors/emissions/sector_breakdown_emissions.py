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

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = [
        GlossaryCore.TotalEmissions,
        GlossaryCore.SectionEnergyEmissionPart,
        GlossaryCore.SectionNonEnergyEmissionPart,
        GlossaryCore.SectionEnergyConsumptionPartTWh,
        GlossaryCore.SectionEmissionPart,
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, sector, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''

    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    economics_df = get_scenario_value(execution_engine, GlossaryCore.EconomicsDfValue, scenario_name)
    years = list(economics_df[GlossaryCore.Years].values)
    if GlossaryCore.TotalEmissions in chart_list:

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

    if GlossaryCore.SectionEmissionPart in chart_list:

        sections_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEmissionDfValue}",
                                               scenario_name)
        chart_name = f"Breakdown of emission per section for {sector} sector [{GlossaryCore.SectionEmissionDf['unit']}]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEmissionPart,
                                             chart_name=chart_name, stacked_bar=True)

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

        sections_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEmissionDfValue}",
                                               scenario_name)
        sector_emissions = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}",
                                              scenario_name)

        chart_name = f"Breakdown of emission share per section for {sector} sector [%]"
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "Share percentage per section", chart_name=chart_name,
                                             stacked_bar=True)
        for section in GlossaryCore.SectionDictSectors[sector]:
            share_of_section = list(
                sections_emission[section].values / sector_emissions[GlossaryCore.TotalEmissions].values * 100.)
            new_series = InstanciatedSeries(years, share_of_section, f'{section}',
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

    if GlossaryCore.SectionEnergyEmissionPart in chart_list:

        breakdown_emission_sector = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
        sections_energy_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}", scenario_name)

        chart_name = f"Breakdown of energy emission per section for {sector} sector [{GlossaryCore.SectionEnergyEmissionDf['unit']}]"

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionEnergyEmissionPart, chart_name=chart_name, stacked_bar=True)
        new_series = InstanciatedSeries(
            years, list(breakdown_emission_sector[GlossaryCore.EnergyEmissions].values), 'Total energy emissions', display_type=InstanciatedSeries.LINES_DISPLAY)
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

        breakdown_emission_sector = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.EmissionsDfValue}", scenario_name)
        sections_non_energy_emission = get_scenario_value(execution_engine, f"{sector}.{GlossaryCore.SectionNonEnergyEmissionDfValue}", scenario_name)

        chart_name = f"Breakdown of non energy emission per section for {sector} sector [{GlossaryCore.SectionNonEnergyEmissionDf['unit']}]"

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionNonEnergyEmissionPart,
                                             chart_name=chart_name, stacked_bar=True)
        new_series = InstanciatedSeries(
            years, list(breakdown_emission_sector[GlossaryCore.NonEnergyEmissions].values), 'Total Non energy emissions',
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

    return instanciated_charts
