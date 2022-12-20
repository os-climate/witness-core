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
        'GHG_emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'Gt'},
        'co2_emissions_fractions': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': '-', 'default': [0.13, 0.20, 0.32, 0.25, 0.10], 'user_level': 2},
        'co2_boxes_decays': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': 'years',
                             'default': [1.0, 0.9972489701005488, 0.9865773841008381, 0.942873143854875, 0.6065306597126334],
                             'user_level': 2},
        'co2_boxes_init_conc': {'type': 'array', 'unit': 'ppm', 'default': co2_init_conc_fund, 'user_level': 2},
        'ch4_emis_to_conc': {'type': 'float', 'unit': 'ppm/Mt', 'default': 0.3597, 'user_level': 2},
        'ch4_decay_rate': {'type': 'float', 'unit': '-', 'default': 1/12, 'user_level': 2},
        'ch4_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': 790, 'user_level': 2},
        'ch4_init_conc': {'type': 'float', 'unit': 'ppm', 'default': 1222, 'user_level': 2},
        'n2o_emis_to_conc': {'type': 'float', 'unit': 'ppm/Mt', 'default': 0.2079, 'user_level': 2},
        'n2o_decay_rate': {'type': 'float', 'unit': '-', 'default':  1/114, 'user_level': 2},
        'n2o_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': 285, 'user_level': 2},
        'n2o_init_conc': {'type': 'float', 'unit': 'ppm', 'default': 296, 'user_level': 2},



        'ppm_ref': {'type': 'float', 'unit': 'ppm', 'default': 280, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'rockstrom_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 490, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'alpha': ClimateEcoDiscipline.ALPHA_DESC_IN,
        'beta': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-',
                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'minimum_ppm_limit': {'type': 'float', 'unit': 'ppm', 'default': 250, 'user_level': 2},
        'minimum_ppm_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 10, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},

    }

    DESC_OUT = {
        'ghg_cycle_df': {'type': 'dataframe', 'unit': 'ppm', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'ghg_cycle_df_detailed': {'type': 'dataframe', 'unit': 'ppm', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'ppm_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'rockstrom_limit_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'minimum_ppm_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'}
    }

    def init_execution(self, proxy):
        param_in = self.get_sosdisc_inputs()
        self.ghg_cycle = GHGCycle(param_in)

    def run(self):
        # get input of discipline
        param_in = self.get_sosdisc_inputs()

        # compute output
        self.ghg_cycle.compute(param_in)

        dict_values = {
            'ghg_cycle_df': self.ghg_cycle.ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']],
            'ghg_cycle_df_detailed': self.ghg_cycle.ghg_cycle_df,
            'ppm_objective': self.ghg_cycle.ppm_obj,
            'rockstrom_limit_constraint': self.ghg_cycle.rockstrom_limit_constraint,
            'minimum_ppm_constraint': self.ghg_cycle.minimum_ppm_constraint}

        # store data
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute:
        """
        d_co2_ppm_d_emissions = self.ghg_cycle.compute_dco2_ppm_d_emissions()
        d_ghg_ppm_d_emissions = self.ghg_cycle.d_ppm_d_other_ghg()

        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'co2_ppm'), ('GHG_emissions_df', 'Total CO2 emissions'), d_co2_ppm_d_emissions)
        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'ch4_ppm'), ('GHG_emissions_df', 'Total CH4 emissions'), d_ghg_ppm_d_emissions['CH4'])
        self.set_partial_derivative_for_other_types(
            ('ghg_cycle_df', 'n2o_ppm'), ('GHG_emissions_df', 'Total N2O emissions'), d_ghg_ppm_d_emissions['N2O'])

        d_ppm_objective_d_totalemissions = self.ghg_cycle.compute_d_objective(d_co2_ppm_d_emissions)
        self.set_partial_derivative_for_other_types(
            ('ppm_objective',), ('GHG_emissions_df', 'Total CO2 emissions'), d_ppm_objective_d_totalemissions)

        self.set_partial_derivative_for_other_types(
            ('rockstrom_limit_constraint',), ('GHG_emissions_df', 'Total CO2 emissions'),
            -d_co2_ppm_d_emissions / self.ghg_cycle.rockstrom_constraint_ref)
        self.set_partial_derivative_for_other_types(
            ('minimum_ppm_constraint',), ('GHG_emissions_df', 'Total CO2 emissions'),
            d_co2_ppm_d_emissions / self.ghg_cycle.minimum_ppm_constraint_ref)

    def get_chart_filter_list(self, proxy):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Atmospheric concentrations parts per million']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, proxy, chart_filters=None):

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
            new_chart = TwoAxesInstanciatedChart('years', 'CO2 Atmospheric concentrations parts per million',
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
            new_chart = TwoAxesInstanciatedChart('years', 'CH4 Atmospheric concentrations parts per million',
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
            chart_name = 'N20 Atmospheric concentrations parts per million'
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(ppm)
            new_chart = TwoAxesInstanciatedChart('years', 'N2O Atmospheric concentrations parts per million',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value], chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
