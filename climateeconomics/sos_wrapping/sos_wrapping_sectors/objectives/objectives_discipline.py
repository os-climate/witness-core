'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
from copy import deepcopy

import numpy as np
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import (
    MacroeconomicsModel,
)
from climateeconomics.core.core_sectorization.sectorization_objectives_model import (
    ObjectivesModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


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
    
    default_earlier_energy_eff = pd.DataFrame()

    DESC_IN = {GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
               GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               GlossaryCore.SectorListValue: GlossaryCore.SectorList,
                'historical_gdp': {'type': 'dataframe', 'unit': 'T$',
                                  'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                           GlossaryCore.SectorAgriculture: ('float', None, True),
                                                           GlossaryCore.SectorIndustry: ('float', None, True),
                                                           GlossaryCore.SectorServices: ('float', None, True),
                                                           'total': ('float', None, True)},
                                  'dataframe_edition_locked': False, },
               'historical_capital': {'type': 'dataframe', 'unit': 'T$',
                                      'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                               GlossaryCore.SectorAgriculture: ('float', None, True),
                                                               GlossaryCore.SectorIndustry: ('float', None, True),
                                                               GlossaryCore.SectorServices: ('float', None, True),
                                                               'total': ('float', None, True)},
                                      'dataframe_edition_locked': False, },
               'historical_energy': {'type': 'dataframe', 'unit': 'PWh',
                                     'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                              GlossaryCore.SectorAgriculture: ('float', None, True),
                                                              GlossaryCore.SectorIndustry: ('float', None, True),
                                                              GlossaryCore.SectorServices: ('float', None, True),
                                                              'Total': ('float', None, True),},
                                     'dataframe_edition_locked': False, },
               GlossaryCore.EconomicsDfValue: {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                'namespace': GlossaryCore.NS_WITNESS,
                                'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                         GlossaryCore.Capital: ('float', None, False),
                                                         GlossaryCore.UsableCapital: ('float', None, False),
                                                         GlossaryCore.Output: ('float', None, False),
                                                         GlossaryCore.OutputNetOfDamage: ('float', None, False),}},
               'weights_df': {'type': 'dataframe', 'unit': '-',
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                              'weight': ('float', None, True)},
                                                              'dataframe_edition_locked': False, },
               'data_for_earlier_energy_eff': {'type': 'dataframe', 'unit': '-', 'default': default_earlier_energy_eff,
                                     'dynamic_dataframe_columns': True,
                                     'dataframe_edition_locked': False, },
               'delta_max_gdp': {'type': 'float', 'default': 1, 'user_level': 1, 'unit': '-'},
               'delta_max_energy_eff': {'type': 'float', 'default': 0.1, 'user_level': 1, 'unit': '-'},
               GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool
               }

    DESC_OUT = {'error_pib_total': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_obj'}, 
                'year_min_energy_eff': {'type': 'dict', 'unit': '-'}, 
                }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.objectives_model = ObjectivesModel(inputs_dict)

    def update_default_values(self):
        if GlossaryCore.YearStart in self.get_data_in() and GlossaryCore.YearEnd in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs([GlossaryCore.YearStart, GlossaryCore.YearEnd])
            if year_start is not None and year_end is not None:
                years = np.arange(year_start, year_end + 1)
                default_weight_df = pd.DataFrame({GlossaryCore.Years: years, 'weight': 1.})
                self.update_default_value('weights_df', 'in', default_weight_df)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}
        self.update_default_values()
        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                dynamic_inputs[f'{sector}.{GlossaryCore.DetailedCapitalDfValue}'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_SECTORS,
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                             GlossaryCore.Capital: ('float', None, False),
                                             GlossaryCore.UsableCapital: ('float', None, False),
                                             GlossaryCore.EnergyEfficiency: ('float', None, False),}
                }
                dynamic_inputs[f'{sector}.{GlossaryCore.ProductionDfValue}'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_SECTORS,
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                             GlossaryCore.Output: ('float', None, False),
                                             GlossaryCore.OutputNetOfDamage: ('float', None, False),}
                }
                dynamic_inputs[f'{sector}.longterm_energy_efficiency'] = {
                    'type': 'dataframe', 'unit': MacroeconomicsModel.SECTORS_OUT_UNIT[sector],
                    'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_SECTORS,
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                             GlossaryCore.EnergyEfficiency: ('float', None, False),}
                }
                dynamic_outputs[f'{sector}.gdp_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
                dynamic_outputs[f'{sector}.cap_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
                dynamic_outputs[f'{sector}.energy_eff_error'] = {
                    'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
                dynamic_outputs[f'{sector}.historical_energy_efficiency'] = {
                    'type': 'dataframe', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                    'namespace': 'ns_obj'}
            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def run(self):

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.objectives_model.configure_parameters(inputs_dict)

        # -- compute
        error_pib_total, sectors_gdp_errors, energy_eff_errors, hist_energy_eff, year_min = self.objectives_model.compute_all_errors(inputs_dict)

        # store outputs in a dict
        outputs_dict = {'error_pib_total': np.array([error_pib_total]), 
                        'year_min_energy_eff': year_min}
        
        if GlossaryCore.SectorListValue in self.get_sosdisc_inputs().keys():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                outputs_dict[f'{sector}.gdp_error'] = np.array([sectors_gdp_errors[sector]])
                outputs_dict[f'{sector}.energy_eff_error'] = np.array([energy_eff_errors[sector]])
                outputs_dict[f'{sector}.historical_energy_efficiency'] = hist_energy_eff[sector]

        
        self.store_sos_outputs_values(outputs_dict)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['Total output', 'Total GDP', 'sectors']

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_df = deepcopy(self.get_sosdisc_inputs(GlossaryCore.EconomicsDfValue))
        sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
        historical_gdp = self.get_sosdisc_inputs('historical_gdp')
        historical_capital = self.get_sosdisc_inputs('historical_capital')
        year_min_energy_eff = self.get_sosdisc_outputs('year_min_energy_eff')

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Total output' in chart_list:
            ref = historical_gdp['total']
            simu = economics_df[GlossaryCore.OutputNetOfDamage]

            years = list(economics_df[GlossaryCore.Years])
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}

            min_values['ref'], max_values['ref'] = self.get_greataxisrange(ref)
            min_values['simu'], max_values['simu'] = self.get_greataxisrange(simu)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total Output fitting comparison with historical data'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [T$]',
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
            simu = economics_df[GlossaryCore.Capital]

            years = list(economics_df[GlossaryCore.Years])
            year_start = years[0]
            year_end = years[len(years) - 1]
            max_values = {}
            min_values = {}

            min_values['ref'], max_values['ref'] = self.get_greataxisrange(ref)
            min_values['simu'], max_values['simu'] = self.get_greataxisrange(simu)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total capital fitting comparison with historical data'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world capital stock [T$]',
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
            years = list(economics_df[GlossaryCore.Years])
            for sector in sector_list:
                simu_gdp_sector_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.ProductionDfValue}')
                simu_gdp_sector = simu_gdp_sector_df[GlossaryCore.OutputNetOfDamage]
                hist_gdp_sector = historical_gdp[sector]
                simu_capital_sector_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.DetailedCapitalDfValue}')
                simu_energy_eff_sector_lt = self.get_sosdisc_inputs(f'{sector}.longterm_energy_efficiency')
                simu_capital_sector = simu_capital_sector_df[GlossaryCore.Capital]
                hist_capital_sector = historical_capital[sector]
                hist_energy_eff_sector = self.get_sosdisc_outputs(f'{sector}.historical_energy_efficiency')
                simu_energy_eff_sector = simu_capital_sector_df[GlossaryCore.EnergyEfficiency]

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'capital stock [T$]',
                                                     chart_name=f'{sector} capital fitting comparison with historical data')
                new_series = InstanciatedSeries(
                    years, list(hist_capital_sector), 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                new_series = InstanciatedSeries(
                    years, list(simu_capital_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'output [T$]',
                                                     chart_name=f'{sector} output fitting comparison with historical data')
                new_series = InstanciatedSeries(
                    years, list(hist_gdp_sector), 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                new_series = InstanciatedSeries(
                    years, list(simu_gdp_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'energy efficiency [-]',
                                                     chart_name=f'{sector} energy efficiency fitting comparison with historical data')
                #filter data above year min 
                years_energ_eff = list(hist_energy_eff_sector[GlossaryCore.Years][hist_energy_eff_sector[GlossaryCore.Years]>=year_min_energy_eff[sector]])
                hist_energy_eff_s = list(hist_energy_eff_sector[GlossaryCore.EnergyEfficiency][hist_energy_eff_sector[GlossaryCore.Years]>=year_min_energy_eff[sector]])
                new_series = InstanciatedSeries(
                    years_energ_eff, hist_energy_eff_s, 'historical data', 'scatter', visible_line, marker_symbol='x')
                new_chart.series.append(new_series)
                #If extra data plot all the data and use long term energy efficiency
                if len(years_energ_eff)> len(years): 
                    new_series = InstanciatedSeries(
                        list(simu_energy_eff_sector_lt[GlossaryCore.Years]), list(simu_energy_eff_sector_lt[GlossaryCore.EnergyEfficiency]), 'simulated values', 'lines', visible_line)
                #If not extra data: plot only calculated values in capital_df
                else: 
                    new_series = InstanciatedSeries(
                        years, list(simu_energy_eff_sector), 'simulated values', 'lines', visible_line)
                new_chart.series.append(new_series)
                instanciated_charts.append(new_chart)

        return instanciated_charts
