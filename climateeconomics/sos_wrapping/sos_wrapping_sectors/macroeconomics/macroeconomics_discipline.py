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
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import MacroeconomicsModel
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import AgricultureDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import ServicesDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import IndustrialDiscipline
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd
from copy import deepcopy



class MacroeconomicsDiscipline(ClimateEcoDiscipline):
    ''' Discipline intended to agregate resource parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Macroeconomics Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-industry fa-fw',
        'version': '',
    }

    DESC_IN = {'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
               'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
               'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               'sector_list': {'type': 'string_list', 'default': MacroeconomicsModel.SECTORS_LIST, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 
                               'namespace': 'ns_witness', 'editable': False, 'structuring': True},
               'total_investment_share_of_gdp': {'type': 'dataframe', 'unit': '%', 'dataframe_descriptor': {'years': ('float', None, False),
                                                'share_investment': ('float', None, True)}, 'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_witness'},
                #'scaling_factor_investment': {'type': 'float', 'default': 1e2, 'unit': '-', 'user_level': 2, 'visibility': 'Shared', 'namespace': 'ns_witness'} 
                }

    DESC_OUT = {
        'economics_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'investment_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'economics_detail_df': {'type': 'dataframe'}
            }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.macro_model = MacroeconomicsModel(inputs_dict)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        #dynamic_outputs = {}

        if 'sector_list' in self._data_in:
            sector_list = self.get_sosdisc_inputs('sector_list')
            for sector in sector_list:
                dynamic_inputs[f'{sector}.capital_df'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector]}
                dynamic_inputs[f'{sector}.production_df'] = {
                    'type': 'dataframe', 'unit':  MacroeconomicsModel.SECTORS_OUT_UNIT[sector]}

            self.add_inputs(dynamic_inputs)

    def run(self):

        #-- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.macro_model.configure_parameters(inputs_dict)

        #-- compute
        economics_df, investment_df = self.macro_model.compute(inputs_dict)

        outputs_dict = {'economics_df': economics_df[['years','net_output']],
                        'investment_df': investment_df, 
                        'economics_detail_df': economics_df}

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        net_output and invest wrt sector net_output 
        """
        sector_list = self.get_sosdisc_inputs('sector_list')
        #Gradient wrt each sector production df: same for all sectors 
        grad_netoutput, grad_invest = self.macro_model.get_derivative_sectors()
        for sector in sector_list:
            self.set_partial_derivative_for_other_types(('economics_df', 'net_output'), (f'{sector}.production_df', 'output_net_of_damage'), grad_netoutput)

            self.set_partial_derivative_for_other_types(('investment_df', 'investment'), (f'{sector}.production_df', 'output_net_of_damage'), grad_invest)
        # Gradient wrt share investment 
        grad_invest_share = self.macro_model.get_derivative_dinvest_dshare()
        self.set_partial_derivative_for_other_types(
                ('investment_df', 'investment'), ('total_investment_share_of_gdp','share_investment'), grad_invest_share)   
        

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['output', 'investment', 'capital', 'share capital', 'share output', 'output growth']

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))
        
        return chart_filters
        
    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_df = deepcopy(self.get_sosdisc_outputs('economics_detail_df'))
        investment_df = deepcopy(self.get_sosdisc_outputs('investment_df'))
        sector_list = self.get_sosdisc_inputs('sector_list')
        
        
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'output' in chart_list:

            to_plot = ['output', 'net_output']
            legend = {'output': 'world gross output',
                      'net_output': 'world output net of damage'}
            years = list(economics_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Economics output'
            new_chart = TwoAxesInstanciatedChart('years', 'world output [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True
                ordonate_data = list(economics_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            
        if 'investment' in chart_list:

            to_plot = ['investment']
            years = list(investment_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(investment_df[to_plot])
            chart_name = 'Total investment over years'
            new_chart = TwoAxesInstanciatedChart('years', ' Investment [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(investment_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)
                new_chart.series.append(new_series)
                
            instanciated_charts.append(new_chart)
            
        if 'capital' in chart_list:

            to_plot = ['capital', 'usable_capital']
            legend = {'capital': 'capital stock',
                      'usable_capital': 'usable capital stock'}
            years = list(economics_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total capital stock and usable capital'
            new_chart = TwoAxesInstanciatedChart('years', 'capital stock [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True
                ordonate_data = list(economics_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        
        if 'share capital' in chart_list:
            capital = economics_df['capital'].values
            chart_name = 'Capital distribution between economic sectors'
            new_chart = TwoAxesInstanciatedChart('years', 'share of total capital stock [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name = chart_name)

            
            for sector in sector_list: 
                capital_df = self.get_sosdisc_inputs(f'{sector}.capital_df')
                sector_capital = capital_df['capital'].values
                share = (sector_capital/capital)*100
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total capital stock' , 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        
        if 'share output' in chart_list:
            output = economics_df['net_output'].values
            chart_name = 'Sectors output share of total economics net output'
            new_chart = TwoAxesInstanciatedChart('years', 'share of total net output [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name = chart_name)

            
            for sector in sector_list: 
                production_df = self.get_sosdisc_inputs(f'{sector}.production_df')
                sector_output = production_df['output_net_of_damage'].values
                share = (sector_output/output)*100
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total net output' , 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
            
        if 'output growth' in chart_list:

            to_plot = ['output_growth']
            years = list(economics_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(economics_df[to_plot])
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart('years', ' growth rate [-]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(economics_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)
                new_chart.series.append(new_series)
                
            instanciated_charts.append(new_chart)
       

        return instanciated_charts
