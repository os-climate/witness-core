'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

from climateeconomics.glossarycore import GlossaryCore


def get_chart_filter_list(discipline):

    # For the outputs, making a graph for tco vs year for each range and for specific
    # value of ToT with a shift of five year between then

    chart_filters = []

    chart_list = ['Atmospheric temperature evolution', 'Ocean temperature evolution']

    chart_filters.append(ChartFilter(
        'graphs', chart_list, chart_list, 'graphs'))

    return chart_filters

def get_instanciated_charts(discipline, chart_filters=None):
    
    instanciated_charts = []
    graphs_list = []
    
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'graphs':
                graphs_list = chart_filter.selected_values
                
    temperature_df_dict = discipline.get_sosdisc_outputs(
            'temperature_df_dict')

    if 'Atmospheric temperature evolution' in graphs_list:       
        
        chart_name = 'Atmospheric temperature evolution over years'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'temperature evolution (degrees Celsius above preindustrial)',
                                            chart_name = chart_name)
        
        for scenario, temp_df in temperature_df_dict.items():
            years = list(temp_df.index)
#            year_start = years[0]
#            year_end = years[len(years) - 1]
            temp_atmo = temp_df[GlossaryCore.TempAtmo]
            #temp_ocean = temp_df[GlossaryCore.TempOcean]
            
            new_series = InstanciatedSeries( years, temp_atmo.tolist(), f'{scenario} Atmospheric temperature', 'lines')
            new_chart.series.append(new_series)
#             new_series = InstanciatedSeries( years, temp_ocean.tolist(), f'{scenario} Ocean temperature', 'lines')
#             new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)
        
    if 'Ocean temperature evolution' in graphs_list:       
        
        chart_name = 'Ocean temperature evolution over years'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'temperature evolution (degrees Celsius above preindustrial)',
                                            chart_name = chart_name)
        
        for scenario, temp_df in temperature_df_dict.items():
            years = list(temp_df.index)
#            year_start = years[0]
#            year_end = years[len(years) - 1]
            temp_ocean = temp_df[GlossaryCore.TempOcean]
            
            new_series = InstanciatedSeries( years, temp_ocean.tolist(), f'{scenario} Ocean temperature', 'lines')
            new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)
# 
    return instanciated_charts
# 
#    