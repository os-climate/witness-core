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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import MacroeconomicsModel
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import AgricultureDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import ServicesDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import IndustrialDiscipline
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
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
        'icon': 'fa-solid fa-city',
        'version': '',
    }

    DESC_IN = {'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
               'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
               'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               'sector_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                               'default': MacroeconomicsModel.SECTORS_LIST,
                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                               'namespace': 'ns_witness', 'editable': False, 'structuring': True},
               'total_investment_share_of_gdp': {'type': 'dataframe', 'unit': '%',
                                                 'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                          'share_investment': ('float', None, True)},
                                                 'dataframe_edition_locked': False, 'visibility': 'Shared',
                                                 'namespace': 'ns_witness'},
               # 'scaling_factor_investment': {'type': 'float', 'default': 1e2, 'unit': '-', 'user_level': 2, 'visibility': 'Shared', 'namespace': 'ns_witness'}
               }

    DESC_OUT = {
        GlossaryCore.EconomicsDfValue: {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                         'namespace': 'ns_witness'},
        'investment_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                          'namespace': 'ns_witness'},
        'sectors_investment_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                          'namespace': 'ns_witness', 'dataframe_descriptor': {},'dynamic_dataframe_columns': True},
        'economics_detail_df': {'type': 'dataframe'},
    }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.macro_model = MacroeconomicsModel(inputs_dict)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        # dynamic_outputs = {}

        if 'sector_list' in self.get_data_in():
            sector_list = self.get_sosdisc_inputs('sector_list')
            df_descriptor = {GlossaryCore.Years: ('float', None, False)}
            df_descriptor.update({col: ('float', None, True)
                                  for col in sector_list})
            dynamic_inputs['sectors_investment_share'] = {'type': 'dataframe', 'unit': '%',
                                                          'dataframe_edition_locked': False, 'visibility': 'Shared',
                                                          'namespace': 'ns_witness',
                                                          'dataframe_descriptor': df_descriptor}
            for sector in sector_list:
                dynamic_inputs[f'{sector}.capital_df'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                             'capital': ('float', None, True),
                                             'usable_capital': ('float', None, True),}
                }
                dynamic_inputs[f'{sector}.production_df'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                             'output': ('float', None, True),
                                             'output_net_of_damage': ('float', None, True),}
                }

            self.add_inputs(dynamic_inputs)

    def run(self):

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.macro_model.configure_parameters(inputs_dict)

        # -- compute
        economics_df, investment_df, sectors_investment_df = self.macro_model.compute(inputs_dict)

        outputs_dict = {GlossaryCore.EconomicsDfValue: economics_df[[GlossaryCore.Years, GlossaryCore.OutputNetOfDamage, 'capital']],
                        'investment_df': investment_df,
                        'sectors_investment_df': sectors_investment_df,
                        'economics_detail_df': economics_df}

        # -- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        net_output and invest wrt sector net_output 
        """
        sector_list = self.get_sosdisc_inputs('sector_list')
        # Gradient wrt share investment
        grad_invest_share = self.macro_model.get_derivative_dinvest_dshare()
        self.set_partial_derivative_for_other_types(
            ('investment_df', 'investment'), ('total_investment_share_of_gdp', 'share_investment'), grad_invest_share)

        # Gradient wrt each sector production df: same for all sectors
        grad_netoutput, grad_invest = self.macro_model.get_derivative_sectors()
        for sector in sector_list:
            self.set_partial_derivative_for_other_types((GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
                                                        (f'{sector}.production_df', 'output_net_of_damage'),
                                                        grad_netoutput)
            self.set_partial_derivative_for_other_types((GlossaryCore.EconomicsDfValue, 'capital'),
                                                        (f'{sector}.capital_df', 'capital'), grad_netoutput)
            self.set_partial_derivative_for_other_types(('investment_df', 'investment'),
                                                        (f'{sector}.production_df', 'output_net_of_damage'),
                                                        grad_invest)
            self.set_partial_derivative_for_other_types( ('sectors_investment_df', f'{sector}'),
                                                         ('sectors_investment_share', f'{sector}'),grad_invest_share)
            #Gradient of sector investment wrt every sectors net output
            for sectorbis in sector_list:
                grad_sector_invest = self.macro_model.get_derivative_dsectinvest_dsectoutput(sector, grad_netoutput)
                self.set_partial_derivative_for_other_types(('sectors_investment_df', f'{sector}'),
                                                        (f'{sectorbis}.production_df', 'output_net_of_damage'),
                                                        grad_sector_invest)


    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['output', 'investment', 'capital', 'share capital', 'share output', 'share investment','output growth']

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
        sectors_investment_df = deepcopy(self.get_sosdisc_outputs('sectors_investment_df'))
        sector_list = self.get_sosdisc_inputs('sector_list')

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'output' in chart_list:

            to_plot = ['output', GlossaryCore.OutputNetOfDamage]
            legend = {'output': 'world gross output',
                      GlossaryCore.OutputNetOfDamage: 'world output net of damage'}
            years = list(economics_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Economics output (Power Purchase Parity)'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
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

            to_plot = sector_list
            years = list(investment_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            chart_name = 'Total investment over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Investment [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 chart_name= chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(sectors_investment_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, f'{key} investment', 'lines', visible_line)
                new_chart.series.append(new_series)

            ordonate_data = list(investment_df['investment'])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'total investment', 'lines', visible_line)
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
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'capital stock [T$]',
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
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total capital stock [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                capital_df = self.get_sosdisc_inputs(f'{sector}.capital_df')
                sector_capital = capital_df['capital'].values
                share = (sector_capital / capital) * 100
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total capital stock', 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'share output' in chart_list:
            output = economics_df[GlossaryCore.OutputNetOfDamage].values
            chart_name = 'Sectors output share of total economics net output'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total net output [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                production_df = self.get_sosdisc_inputs(f'{sector}.production_df')
                sector_output = production_df['output_net_of_damage'].values
                share = (sector_output / output) * 100
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total net output', 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'share investment' in chart_list:
            invest = investment_df['investment'].values
            chart_name = 'Sectors investment share of total investment'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total investment [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                sector_invest = sectors_investment_df[f'{sector}'].values
                share = (sector_invest / invest) * 100
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total investment', 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'output growth' in chart_list:

            to_plot = ['output_growth']
            years = list(economics_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(economics_df[to_plot])
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' growth rate [-]',
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
