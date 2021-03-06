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
from climateeconomics.core.core_sectorization.sectorization_objectives_model import ObjectivesModel
from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import MacroeconomicsModel
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import AgricultureDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import ServicesDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import IndustrialDiscipline
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd
from copy import deepcopy


class ObjectivesDiscipline(ClimateEcoDiscipline):
    ''' Discipline to compute objectives for production function fitting optim 
    '''

    # ontology information
    _ontology_data = {
        'label': 'Objectives Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-bullseye',
        'version': '',
    }

    DESC_IN = {'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
               'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
               'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               'sector_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                               'default': MacroeconomicsModel.SECTORS_LIST,
                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                               'namespace': 'ns_witness', 'editable': False, 'structuring': True},
               #                'data_energy_df': {'type': 'dataframe', 'unit': 'Twh', 'dataframe_descriptor': {'years': ('float', None, False),
               #                                                 'agri_energy': ('float', None, True), 'indus_energy': ('float', None, True),
               #                                                 'services_energy': ('float', None, True), 'total_energy': ('float', None, True)}, 'dataframe_edition_locked': False,} ,
               #                'data_investments_df': {'type': 'dataframe', 'unit': 'T$', 'dataframe_descriptor': {'years': ('float', None, False),
               #                                                 'agri_invest': ('float', None, True), 'indus_invest': ('float', None, True),
               #                                                 'services_invest': ('float', None, True), 'total_invest': ('float', None, True)}, 'dataframe_edition_locked': False,} ,
               #                'data_workforce_df': {'type': 'dataframe', 'unit': 'Million of people', 'dataframe_descriptor': {'years': ('float', None, False),
               #                                                 'Agriculture': ('float', None, True), 'Industry': ('float', None, True),
               #                                                 'services_workforce': ('float', None, True), 'total_workforce': ('float', None, True)}, 'dataframe_edition_locked': False,},
               'historical_gdp': {'type': 'dataframe', 'unit': 'T$',
                                  'dataframe_descriptor': {'years': ('float', None, False),
                                                           'Agriculture': ('float', None, True),
                                                           'Industry': ('float', None, True),
                                                           'Services': ('float', None, True),
                                                           'total': ('float', None, True)},
                                  'dataframe_edition_locked': False, },
               'historical_capital': {'type': 'dataframe', 'unit': 'T$',
                                      'dataframe_descriptor': {'years': ('float', None, False),
                                                               'Agriculture': ('float', None, True),
                                                               'Industry': ('float', None, True),
                                                               'Services': ('float', None, True),
                                                               'total': ('float', None, True)},
                                      'dataframe_edition_locked': False, },
               'historical_energy': {'type': 'dataframe', 'unit': 'PWh',
                                     'dataframe_descriptor': {'years': ('float', None, False),
                                                              'Agriculture': ('float', None, True),
                                                              'Industry': ('float', None, True),
                                                              'Services': ('float', None, True),
                                                              'total': ('float', None, True)},
                                     'dataframe_edition_locked': False, },
               'economics_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                'namespace': 'ns_witness'},

               }

    DESC_OUT = {'error_pib_total': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_obj'},
                'historical_energy_efficiency': {'type': 'dataframe', 'unit': '-',
                                                 'dataframe_descriptor': {'years': ('float', None, False),
                                                                          'Agriculture': ('float', None, True),
                                                                          'Industry': ('float', None, True),
                                                                          'Services': ('float', None, True)},
                                                 'dataframe_edition_locked': False, }
                }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.objectives_model = ObjectivesModel(inputs_dict)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}

        if 'sector_list' in self._data_in:
            sector_list = self.get_sosdisc_inputs('sector_list')
            for sector in sector_list:
                dynamic_inputs[f'{sector}.detailed_capital_df'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_macro'}
                dynamic_inputs[f'{sector}.production_df'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_macro'}
                dynamic_outputs[f'{sector}.gdp_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
                dynamic_outputs[f'{sector}.cap_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
                dynamic_outputs[f'{sector}.energy_eff_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def run(self):

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.objectives_model.configure_parameters(inputs_dict)

        # -- compute
        error_pib_total, sectors_gdp_errors, energy_eff_errors, hist_energy_eff = self.objectives_model.compute_all_errors(inputs_dict)

        # store outputs in a dict
        outputs_dict = {'error_pib_total': np.array([error_pib_total]),
                        'historical_energy_efficiency': hist_energy_eff}
        if 'sector_list' in self._data_in:
            sector_list = self.get_sosdisc_inputs('sector_list')
            for sector in sector_list:
                outputs_dict[f'{sector}.gdp_error'] = np.array([sectors_gdp_errors[sector]])
                outputs_dict[f'{sector}.energy_eff_error'] = np.array([energy_eff_errors[sector]])

        # -- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['Total output', 'Total GDP', 'sectors']

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

        economics_df = deepcopy(self.get_sosdisc_inputs('economics_df'))
        sector_list = self.get_sosdisc_inputs('sector_list')
        historical_gdp = self.get_sosdisc_inputs('historical_gdp')
        historical_capital = self.get_sosdisc_inputs('historical_capital')
        hist_energy_efficiency = self.get_sosdisc_outputs('historical_energy_efficiency')

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Total output' in chart_list:
            ref = historical_gdp['total']
            simu = economics_df['output_net_of_d']

            years = list(economics_df['years'])
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}

            min_values['ref'], max_values['ref'] = self.get_greataxisrange(ref)
            min_values['simu'], max_values['simu'] = self.get_greataxisrange(simu)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total Output fitting comparison with historical data'
            new_chart = TwoAxesInstanciatedChart('years', 'world output [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            visible_line = True
            new_series = InstanciatedSeries(
                years, list(ref), 'historical data', 'scatter', visible_line, marker_symbol='x')
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, list(simu), 'simulated values', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Total capital' in chart_list:
            ref = historical_capital['total']
            simu = economics_df['capital']

            years = list(economics_df['years'])
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}

            min_values['ref'], max_values['ref'] = self.get_greataxisrange(ref)
            min_values['simu'], max_values['simu'] = self.get_greataxisrange(simu)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total capital fitting comparison with historical data'
            new_chart = TwoAxesInstanciatedChart('years', 'world capital stock [T$]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            visible_line = True
            new_series = InstanciatedSeries(
                years, list(ref), 'historical data', 'scatter', visible_line, marker_symbol='x')
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, list(simu), 'simulated values', 'lines', visible_line)

            instanciated_charts.append(new_chart)

        if 'sectors' in chart_list:
            years = list(economics_df['years'])
            for sector in sector_list:
                simu_gdp_sector_df = self.get_sosdisc_inputs(f'{sector}.production_df')
                simu_gdp_sector = simu_gdp_sector_df['output_net_of_damage']
                hist_gdp_sector = historical_gdp[sector]
                simu_capital_sector_df = self.get_sosdisc_inputs(f'{sector}.detailed_capital_df')
                simu_capital_sector = simu_capital_sector_df['capital']
                hist_capital_sector = historical_capital[sector]
                hist_energy_eff_sector = hist_energy_efficiency[sector].values
                simu_energy_eff_sector = simu_capital_sector_df['energy_efficiency']

                new_chart = TwoAxesInstanciatedChart('years', 'capital stock [T$]',
                                                     chart_name=f'{sector} capital fitting comparison with historical data')
                new_series = InstanciatedSeries(
                    years, list(hist_capital_sector), 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                new_series = InstanciatedSeries(
                    years, list(simu_capital_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

                new_chart = TwoAxesInstanciatedChart('years', 'output [T$]',
                                                     chart_name=f'{sector} output fitting comparison with historical data')
                new_series = InstanciatedSeries(
                    years, list(hist_gdp_sector), 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                new_series = InstanciatedSeries(
                    years, list(simu_gdp_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

                new_chart = TwoAxesInstanciatedChart('years', 'energy efficiency [-]',
                                                     chart_name=f'{sector} energy efficiency fitting comparison with historical data')
                new_series = InstanciatedSeries(
                    years, list(hist_energy_eff_sector), 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                new_series = InstanciatedSeries(
                    years, list(simu_energy_eff_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

        return instanciated_charts
