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
from climateeconomics.core.core_witness.carbon_emissions_model import CarbonEmissions
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from copy import deepcopy
import pandas as pd
import numpy as np


class CarbonemissionsDiscipline(ClimateEcoDiscipline):
    "carbonemissions discipline for DICE"
    years = np.arange(2020, 2101)
    _maturity = 'Research'
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 2020, 'possible_values': years, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100, 'possible_values': years, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'unit': 'years per period', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'init_land_emissions': {'type': 'float', 'default': 2.85, 'unit': 'GtCO2 per year', 'user_level': 2},
        'decline_rate_land_emissions': {'type': 'float', 'default': 0.115, 'user_level': 2},
        'init_cum_land_emisisons': {'type': 'float', 'default': 117.13, 'unit': 'GtCO2', 'user_level': 2},
        'init_gr_sigma': {'type': 'float', 'default': -0.0152, 'user_level': 2},
        'decline_rate_decarbo': {'type': 'float', 'default': -0.001, 'user_level': 2},
        'init_indus_emissions': {'type': 'float', 'default': 34, 'unit': 'GtCO2 per year', 'user_level': 2},
        'init_gross_output': {'type': 'float', 'default': 130.187, 'unit': 'trillions $', 'user_level': 2,
                              'visibility': 'Shared', 'namespace': 'ns_witness'},
        'init_cum_indus_emissions': {'type': 'float', 'default': 577.31, 'unit': 'GtCO2', 'user_level': 2},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_emis_share': {'type': 'float', 'default': 0.9, 'user_level': 2},
        'land_emis_share': {'type': 'float', 'default': 0.0636, 'user_level': 2},
        'co2_emissions_Gt': {'type': 'dataframe', 'unit': 'Gt', 'visibility': 'Shared', 'namespace': 'ns_energy_mix'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-',
                  'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'beta': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-',
                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'min_co2_objective': {'type': 'float', 'default': -1000., 'unit': 'Gt', 'user_level': 2},
        'total_emissions_ref': {'type': 'float', 'default': 39.6, 'unit': 'Gt', 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        # Ref in 2020 is around 34 Gt, the objective is normalized with this
        # reference

    }
    DESC_OUT = {
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'emissions_detail_df': {'type': 'dataframe'},
        'CO2_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness'}
    }

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.emissions_model = CarbonEmissions(in_dict)

    def run(self):
        # Get inputs
        in_dict = self.get_sosdisc_inputs()

        # Compute de emissions_model
        emissions_df, CO2_objective = self.emissions_model.compute(in_dict)
        # Store output data
        dict_values = {'emissions_detail_df': emissions_df,
                       'emissions_df': emissions_df[['years', 'total_emissions', 'cum_total_emissions']],
                       'CO2_objective': CO2_objective}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        emissions_df
          - 'indus_emissions':
                - economics_df, 'gross_output'
                - co2_emissions_Gt, 'Total CO2 emissions'
          -'cum_indus_emissions'
                - economics_df, 'gross_output'
                - co2_emissions_Gt, 'Total CO2 emissions'
          - 'total_emissions',
                - emissions_df, land_emissions
                - economics_df, 'gross_output'
                - co2_emissions_Gt, Total CO2 emissions
          - 'cum_total_emissions'
                - emissions_df, land_emissions
                - economics_df, 'gross_output'
                - co2_emissions_Gt, Total CO2 emissions
          - 'CO2_objective'
                - total_emissions:
                    - emissions_df, land_emissions
                    - economics_df, 'gross_output'
                    - co2_emissions_Gt, Total CO2 emissions
        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(
            inputs_dict['year_start'], inputs_dict['year_end'] + 1, inputs_dict['time_step'])

        d_indus_emissions_d_gross_output, d_cum_indus_emissions_d_gross_output, d_cum_indus_emissions_d_total_CO2_emitted = self.emissions_model.compute_d_indus_emissions()
        d_CO2_obj_d_total_emission = self.emissions_model.compute_d_CO2_objective()
        dobjective_exp_min = self.emissions_model.compute_dobjective_with_exp_min()
        # fill jacobians
        self.set_partial_derivative_for_other_types(
            ('emissions_df', 'total_emissions'), ('economics_df', 'gross_output'),  d_indus_emissions_d_gross_output)

        self.set_partial_derivative_for_other_types(
            ('emissions_df', 'cum_total_emissions'), ('economics_df', 'gross_output'),  d_cum_indus_emissions_d_gross_output)

        self.set_partial_derivative_for_other_types(
            ('emissions_df', 'total_emissions'), ('co2_emissions_Gt', 'Total CO2 emissions'),  np.identity(len(years)))

        self.set_partial_derivative_for_other_types(
            ('emissions_df', 'cum_total_emissions'), ('co2_emissions_Gt', 'Total CO2 emissions'), d_cum_indus_emissions_d_total_CO2_emitted)

        self.set_partial_derivative_for_other_types(
            ('CO2_objective',), ('co2_emissions_Gt', 'Total CO2 emissions'),  d_CO2_obj_d_total_emission * dobjective_exp_min)

        self.set_partial_derivative_for_other_types(
            ('CO2_objective',), ('economics_df', 'gross_output'), dobjective_exp_min * d_CO2_obj_d_total_emission.dot(d_indus_emissions_d_gross_output))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['carbon emission']
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
        emissions_df = deepcopy(
            self.get_sosdisc_outputs('emissions_detail_df'))

        if 'carbon emission' in chart_list:

            to_plot = ['total_emissions', 'land_emissions', 'indus_emissions']
            #emissions_df = discipline.get_sosdisc_outputs('emissions_df')

            total_emission = emissions_df['total_emissions']
            land_emissions = emissions_df['land_emissions']
            indus_emissions = emissions_df['indus_emissions']

            years = list(emissions_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value_e, max_value_e = self.get_greataxisrange(total_emission)
            min_value_l, max_value_l = self.get_greataxisrange(land_emissions)
            min_value_i, max_value_i = self.get_greataxisrange(indus_emissions)
            min_value = min(min_value_e, min_value_l, min_value_i)
            max_value = max(max_value_e, max_value_l, max_value_i)

            chart_name = 'total carbon emissions'

            new_chart = TwoAxesInstanciatedChart('years', 'carbon emissions (Gtc)',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(emissions_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
