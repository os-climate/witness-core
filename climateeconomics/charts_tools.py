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
import pandas as pd
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.glossarycore import GlossaryCore


def graph_gross_and_net_output(economics_detail_df: pd.DataFrame,
                               damage_detailed_df: pd.DataFrame,
                               compute_climate_impact_on_gdp: bool,
                               damages_to_productivity: bool,
                               chart_name: str):
    """returns graph for gross and net output in Macroeconomics (non-sectorized & sectorized)"""
    years = list(economics_detail_df[GlossaryCore.Years].values)
    to_plot = [GlossaryCore.OutputNetOfDamage]

    legend = {GlossaryCore.OutputNetOfDamage: 'Net output'}

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                         chart_name=chart_name, stacked_bar=True,
                                         y_min_zero=not compute_climate_impact_on_gdp)

    for key in to_plot:
        visible_line = True

        ordonate_data = list(economics_detail_df[key])

        new_series = InstanciatedSeries(
            years, ordonate_data, legend[key], 'lines', visible_line)

        new_chart.add_series(new_series)

    gross_output = economics_detail_df[GlossaryCore.GrossOutput].values
    new_series = InstanciatedSeries(
        years, list(gross_output), 'Gross output', 'lines', True)

    new_chart.add_series(new_series)
    if compute_climate_impact_on_gdp:
        ordonate_data = list(-damage_detailed_df[GlossaryCore.DamagesFromClimate])
        new_series = InstanciatedSeries(years, ordonate_data, 'Immediate damages from climate', 'bar')
        new_chart.add_series(new_series)
        if damages_to_productivity:
            gdp_without_damage_to_prod = gross_output + damage_detailed_df[
                GlossaryCore.EstimatedDamagesFromProductivityLoss].values
            ordonate_data = list(gdp_without_damage_to_prod)
            new_series = InstanciatedSeries(years, ordonate_data,
                                            'Pessimist estimation of gross output without damage to productivity',
                                            'dash_lines')
            new_chart.add_series(new_series)

            new_chart.series.pop(-1)
            new_chart.series.pop(1)
            new_chart = new_chart.to_plotly()
            import plotly.graph_objects as go

            new_chart.add_trace(go.Scatter(x=years, y=list(gross_output),
                                           mode='lines',
                                           name="Gross output"
                                           ))

            new_chart.add_trace(go.Scatter(
                x=years,
                y=list(gdp_without_damage_to_prod),
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines',
                fillcolor='rgba(200, 200, 200, 0.3)',
                line={'dash': 'dash', 'color': 'rgb(200, 200, 200)'},
                opacity=0.2,
                name='Pessimist estimation of gross output without damage to productivity', ))

            new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

    return new_chart
