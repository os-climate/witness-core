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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from copy import deepcopy
import pandas as pd
import numpy as np
from climateeconomics.core.core_emissions.indus_emissions_model import IndusEmissions
from climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline import GHGemissionsDiscipline


class IndusemissionsDiscipline(ClimateEcoDiscipline):
    "indusemissions discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Industrial Emission WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-smog fa-fw',
        'version': '',
    }
    years = np.arange(2020, 2101)

    name = f'{GHGemissionsDiscipline.name}.Industry'
    _maturity = 'Research'
    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'init_gr_sigma': {'type': 'float', 'default': -0.0152, 'user_level': 2, 'unit': '-'},
        'decline_rate_decarbo': {'type': 'float', 'default': -0.001, 'user_level': 2, 'unit': '-'},
        'init_indus_emissions': {'type': 'float', 'default': 34, 'unit': 'GtCO2 per year', 'user_level': 2},
        'init_gross_output': {'type': 'float', 'default': 130.187, 'unit': 'T$', 'user_level': 2,
                              'visibility': 'Shared', 'namespace': 'ns_witness'},
        'init_cum_indus_emissions': {'type': 'float', 'default': 577.31, 'unit': 'GtCO2', 'user_level': 2},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'energy_emis_share': {'type': 'float', 'default': 0.9, 'user_level': 2, 'unit': '-'},
        'land_emis_share': {'type': 'float', 'default': 0.0636, 'user_level': 2, 'unit': '-'}
    }
    DESC_OUT = {
        'CO2_indus_emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'Gt'},
        'CO2_indus_emissions_df_detailed': {'type': 'dataframe', 'unit': 'Gt or -'},
    }

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.emissions_model = IndusEmissions(in_dict)

    def run(self):
        # Get inputs
        in_dict = self.get_sosdisc_inputs()

        # Compute de emissions_model
        CO2_indus_emissions_df = self.emissions_model.compute(in_dict)
        # Store output data
        dict_values = {
            'CO2_indus_emissions_df': CO2_indus_emissions_df[['years', 'indus_emissions']],
            'CO2_indus_emissions_df_detailed': CO2_indus_emissions_df}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        CO2_emissions_df
          - 'indus_emissions':
                - economics_df, 'gross_output'
          - 'cum_indus_emissions'
                - economics_df, 'gross_output'
        """

        d_indus_emissions_d_gross_output, d_cum_indus_emissions_d_gross_output, d_cum_indus_emissions_d_total_CO2_emitted = self.emissions_model.compute_d_indus_emissions()

        # fill jacobians
        self.set_partial_derivative_for_other_types(
            ('CO2_indus_emissions_df', 'indus_emissions'), ('economics_df', 'gross_output'),  d_indus_emissions_d_gross_output)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Industrial emissions',
                      'Cumulated Industrial emissions', 'Sigma']
        #chart_list = ['sectoral energy carbon emissions cumulated']
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

        if 'Industrial emissions' in chart_list:
            new_chart = self.get_chart_co2_emissions()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Cumulated Industrial emissions' in chart_list:
            new_chart = self.get_chart_cumulated_co2_emissions()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Sigma' in chart_list:
            new_chart = self.get_chart_sigma()
            if new_chart is not None:
                instanciated_charts.append(new_chart)
        return instanciated_charts

    def get_chart_co2_emissions(self):

        CO2_indus_emissions_df_detailed = self.get_sosdisc_outputs(
            'CO2_indus_emissions_df_detailed')

        total_emission = CO2_indus_emissions_df_detailed['indus_emissions']

        years = list(CO2_indus_emissions_df_detailed['years'].values)

        chart_name = 'Total Industrial emissions'

        new_chart = TwoAxesInstanciatedChart(
            'years', 'CO2 emissions [Gt]', chart_name=chart_name)

        c_emission = list(total_emission.values)

        new_series = InstanciatedSeries(
            years, c_emission, '', 'lines')

        new_chart.series.append(new_series)

        return new_chart

    def get_chart_cumulated_co2_emissions(self):

        CO2_indus_emissions_df_detailed = self.get_sosdisc_outputs(
            'CO2_indus_emissions_df_detailed')

        cum_total_emissions = CO2_indus_emissions_df_detailed['cum_indus_emissions']

        years = list(CO2_indus_emissions_df_detailed['years'].values)

        chart_name = f'Cumulated industrial CO2 emissions since {years[0]}'

        new_chart = TwoAxesInstanciatedChart('years', 'Cumulative CO2 emissions [Gt]',
                                             chart_name=chart_name)

        new_series = InstanciatedSeries(
            years, cum_total_emissions.values.tolist(), 'lines')

        new_chart.series.append(new_series)

        return new_chart

    def get_chart_sigma(self):

        CO2_indus_emissions_df_detailed = self.get_sosdisc_outputs(
            'CO2_indus_emissions_df_detailed')

        sigma = CO2_indus_emissions_df_detailed['sigma']

        years = list(CO2_indus_emissions_df_detailed['years'].values)

        chart_name = f'Sigma to compute industrial CO2 emissions'

        new_chart = TwoAxesInstanciatedChart('years', 'Sigma [-]',
                                             chart_name=chart_name)

        new_series = InstanciatedSeries(
            years, sigma.values.tolist(), 'lines')

        new_chart.series.append(new_series)

        return new_chart
