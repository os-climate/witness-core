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
# coding: utf-8
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.ghg_cycle_model import GHGCycle
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np

import pandas as pd
from copy import deepcopy


class GHGCycleDiscipline(ClimateEcoDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Greenhouse Gas Cycle WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-recycle fa-fw',
        'version': '',
    }
    _maturity = 'Research'

    years = np.arange(2020, 2101)

    # init concentrations in each box from FUND repo in ppm/volume in 1950
    # https://github.com/fund-model/MimiFUND.jl/blob/master/src
    co2_init_conc_fund = np.array([296.002949511, 5.52417779186, 6.65150094285, 2.39635475726, 0.17501699667]) * 412.4/296.002949511

    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'GHG_emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'Gt',
                             'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                      GlossaryCore.TotalCO2Emissions: ('float', None, False),
                                                      'Total N2O emissions': ('float', None, False),
                                                      'Total CH4 emissions': ('float', None, False),
                                                      }},
        'co2_emissions_fractions': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': '-', 'default': [0.13, 0.20, 0.32, 0.25, 0.10], 'user_level': 2},
        'co2_boxes_decays': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': GlossaryCore.Years,
                             'default': [1.0, 0.9972489701005488, 0.9865773841008381, 0.942873143854875, 0.6065306597126334],
                             'user_level': 2},
        'co2_boxes_init_conc': {'type': 'array', 'unit': 'ppm', 'default': co2_init_conc_fund, 'user_level': 2},
        'co2_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': 280, 'user_level': 2},
        'ch4_emis_to_conc': {'type': 'float', 'unit': 'ppm/Mt', 'default': 0.3597, 'user_level': 2},
        'ch4_decay_rate': {'type': 'float', 'unit': '-', 'default': 1/12, 'user_level': 2},
        'ch4_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': 790, 'user_level': 2},
        'ch4_init_conc': {'type': 'float', 'unit': 'ppm', 'default': 1222, 'user_level': 2},
        'n2o_emis_to_conc': {'type': 'float', 'unit': 'ppm/Mt', 'default': 0.2079, 'user_level': 2},
        'n2o_decay_rate': {'type': 'float', 'unit': '-', 'default':  1/114, 'user_level': 2},
        'n2o_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': 285, 'user_level': 2},
        'n2o_init_conc': {'type': 'float', 'unit': 'ppm', 'default': 296, 'user_level': 2},
        'rockstrom_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 490, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'minimum_ppm_limit': {'type': 'float', 'unit': 'ppm', 'default': 250, 'user_level': 2},
        'minimum_ppm_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 10, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'GHG_global_warming_potential20': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'},
                                           'unit': 'kgCO2eq/kg',
                                           'default': ClimateEcoDiscipline.GWP_20_default,
                                           'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                           'namespace': 'ns_witness', 'user_level': 3},
        'GHG_global_warming_potential100': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'},
                                            'unit': 'kgCO2eq/kg',
                                            'default': ClimateEcoDiscipline.GWP_100_default,
                                            'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                            'namespace': 'ns_witness', 'user_level': 3},

    }

    DESC_OUT = {
        'ghg_cycle_df': {'type': 'dataframe', 'unit': 'ppm', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'ghg_cycle_df_detailed': {'type': 'dataframe', 'unit': 'ppm', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'gwp20_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'gwp100_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'rockstrom_limit_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'minimum_ppm_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'}
    }

    def init_execution(self):
        param_in = self.get_sosdisc_inputs()
        self.ghg_cycle = GHGCycle(param_in)

    def run(self):
        # get input of discipline
        param_in = self.get_sosdisc_inputs()

        # compute output
        self.ghg_cycle.compute(param_in)

        dict_values = {
            'ghg_cycle_df': self.ghg_cycle.ghg_cycle_df[[GlossaryCore.Years, 'co2_ppm', 'ch4_ppm', 'n2o_ppm']],
            'ghg_cycle_df_detailed': self.ghg_cycle.ghg_cycle_df,
            'gwp20_objective': self.ghg_cycle.gwp20_obj,
            'gwp100_objective': self.ghg_cycle.gwp100_obj,
            'rockstrom_limit_constraint': self.ghg_cycle.rockstrom_limit_constraint,
            'minimum_ppm_constraint': self.ghg_cycle.minimum_ppm_constraint}

        # store data
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute:
        """
        d_ghg_ppm_d_emissions = self.ghg_cycle.d_ppm_d_ghg()

        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'co2_ppm'), ('GHG_emissions_df', GlossaryCore.TotalCO2Emissions), d_ghg_ppm_d_emissions['CO2'])
        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'ch4_ppm'), ('GHG_emissions_df', 'Total CH4 emissions'), d_ghg_ppm_d_emissions['CH4'])
        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'n2o_ppm'), ('GHG_emissions_df', 'Total N2O emissions'), d_ghg_ppm_d_emissions['N2O'])

        # derivative gwp20 objective
        d_gwp20_objective_d_total_co2_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['CO2'],
            specie='CO2')
        d_gwp20_objective_d_total_ch4_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['CH4'],
            specie='CH4')
        d_gwp20_objective_d_total_n2o_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['N2O'],
            specie='N2O')

        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), ('GHG_emissions_df', GlossaryCore.TotalCO2Emissions), d_gwp20_objective_d_total_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), ('GHG_emissions_df', 'Total CH4 emissions'), d_gwp20_objective_d_total_ch4_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), ('GHG_emissions_df', 'Total N2O emissions'), d_gwp20_objective_d_total_n2o_emissions)

        # derivative gwp100 objective
        d_gwp100_objective_d_total_co2_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['CO2'],
            specie='CO2')
        d_gwp100_objective_d_total_ch4_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['CH4'],
            specie='CH4')
        d_gwp100_objective_d_total_n2o_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions['N2O'],
            specie='N2O')

        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), ('GHG_emissions_df', GlossaryCore.TotalCO2Emissions),
            d_gwp100_objective_d_total_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), ('GHG_emissions_df', 'Total CH4 emissions'),
            d_gwp100_objective_d_total_ch4_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), ('GHG_emissions_df', 'Total N2O emissions'),
            d_gwp100_objective_d_total_n2o_emissions)

        self.set_partial_derivative_for_other_types(
            ('rockstrom_limit_constraint',), ('GHG_emissions_df', GlossaryCore.TotalCO2Emissions),
            -d_ghg_ppm_d_emissions['CO2'] / self.ghg_cycle.rockstrom_constraint_ref)
        self.set_partial_derivative_for_other_types(
            ('minimum_ppm_constraint',), ('GHG_emissions_df', GlossaryCore.TotalCO2Emissions),
            d_ghg_ppm_d_emissions['CO2'] / self.ghg_cycle.minimum_ppm_constraint_ref)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Atmospheric concentrations parts per million']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values
        ghg_cycle_df = deepcopy(self.get_sosdisc_outputs('ghg_cycle_df_detailed'))

        if 'Atmospheric concentrations parts per million' in chart_list:

            ppm = ghg_cycle_df['co2_ppm']
            years = list(ppm.index)
            chart_name = 'CO2 Atmospheric concentrations parts per million'
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(ppm)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 Atmospheric concentrations parts per million',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value], chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)
            new_chart.series.append(new_series)

            # Rockstrom Limit

            ordonate_data = [450] * int(len(years) / 5)
            abscisse_data = np.linspace(year_start, year_end, int(len(years) / 5))
            new_series = InstanciatedSeries(abscisse_data.tolist(), ordonate_data, 'Rockstrom limit', 'scatter')

            note = {'Rockstrom limit': 'Scientifical limit of the Earth'}

            new_chart.series.append(new_series)

            # Minimum PPM constraint
            ordonate_data = [self.get_sosdisc_inputs('minimum_ppm_limit')] * int(len(years) / 5)
            abscisse_data = np.linspace(year_start, year_end, int(len(years) / 5))
            new_series = InstanciatedSeries(abscisse_data.tolist(), ordonate_data, 'Minimum ppm limit', 'scatter')
            note['Minimum ppm limit'] = 'used in constraint calculation'
            new_chart.annotation_upper_left = note

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            ppm = ghg_cycle_df['ch4_ppm']
            years = list(ppm.index)
            chart_name = 'CH4 Atmospheric concentrations parts per million'
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(ppm)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CH4 Atmospheric concentrations parts per million',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value], chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            ppm = ghg_cycle_df['n2o_ppm']
            years = list(ppm.index)
            chart_name = 'N2O Atmospheric concentrations parts per million'
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(ppm)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'N2O Atmospheric concentrations parts per million',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value], chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
