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
from climateeconomics.core.core_witness.utility_model import UtilityModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from copy import deepcopy
import pandas as pd
import numpy as np


class UtilityModelDiscipline(ClimateEcoDiscipline):
    "UtilityModel discipline for DICE"

    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 2020, 'possible_values': years, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100, 'possible_values': years, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'gamma': {'type': 'float', 'range': [0., 1.], 'default': 0, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'welfare_obj_option': {'type': 'string', 'default': 'welfare', 'possible_values': ['last_utility', 'welfare'], 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_mean_price': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_energy_mix', 'unit': '$/MWh'},
        'initial_raw_energy_price': {'type': 'float', 'unit': '$/MWh', 'default': 110, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_discounted_utility': {'type': 'float', 'unit': '-', 'default': 3400, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
        'init_period_utility_pc': {'type': 'float', 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'discounted_utility_ref': {'type': 'float', 'unit': '-', 'default': 1700, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
    }
    DESC_OUT = {
        'utility_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'welfare_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'min_utility_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness'}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.utility_m = UtilityModel(inp_dict)

    def run(self):
        # get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # compute utility
        economics_df = inp_dict.pop('economics_df')
        energy_mean_price = inp_dict['energy_mean_price']
        population_df = inp_dict.pop('population_df')
        # rescaling from billion to million
        cols = [col for col in population_df.columns if col != 'years']
        population_df[cols] = population_df[cols] * 1e3

        utility_df = self.utility_m.compute(
            economics_df, energy_mean_price, population_df)

        # Compute objective function
        obj_option = inp_dict['welfare_obj_option']
        if obj_option in ['last_utility', 'welfare']:
            welfare_objective = self.utility_m.compute_welfare_objective()
        else:
            raise ValueError('obj_option = ' + str(obj_option) + ' not in ' +
                             str(self.DESC_IN['welfare_obj_option']['possible_values']))
        min_utility_objective = self.utility_m.compute_min_utility_objective()

        # store output data
        dict_values = {'utility_df': utility_df,
                       'welfare_objective': welfare_objective,
                       'min_utility_objective': min_utility_objective, }
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        utility_df
          - 'period_utility_pc':
                - economics_df, 'pc_consumption'
                - energy_mean_price : 'energy_price'
          - 'discounted_utility',
                - economics_df, 'pc_consumption'
                - energy_mean_price : 'energy_price'
          - 'welfare'
                - economics_df, 'pc_consumption'
                - energy_mean_price : 'energy_price'
        """
        inputs_dict = self.get_sosdisc_inputs()
        obj_option = inputs_dict['welfare_obj_option']

        d_period_utility_d_pc_consumption, d_discounted_utility_d_pc_consumption, d_discounted_utility_d_population,\
            d_welfare_d_pc_consumption, d_welfare_d_population = self.utility_m.compute_gradient()
        d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, \
            d_welfare_d_energy_price = self.utility_m.compute_gradient_energy_mean_price()

        d_obj_d_welfare, d_obj_d_period_utility_pc = self.utility_m.compute_gradient_objective()

        # fill jacobians
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('economics_df', 'pc_consumption'),  d_period_utility_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('energy_mean_price', 'energy_price'),  d_period_utility_d_energy_price)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('economics_df', 'pc_consumption'),  d_discounted_utility_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('energy_mean_price', 'energy_price'),  d_discounted_utility_d_energy_price)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('population_df', 'population'),  d_discounted_utility_d_population)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('economics_df', 'pc_consumption'),  d_welfare_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('population_df', 'population'),  d_welfare_d_population)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('energy_mean_price', 'energy_price'),  d_welfare_d_energy_price)

        if obj_option == 'last_utility':
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('economics_df', 'pc_consumption'), d_obj_d_period_utility_pc.dot(d_period_utility_d_pc_consumption))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('energy_mean_price', 'energy_price'), d_obj_d_period_utility_pc.dot(d_period_utility_d_energy_price))

        elif obj_option == 'welfare':

            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('population_df', 'population'),  np.dot(d_obj_d_welfare, d_welfare_d_population))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('economics_df', 'pc_consumption'), np.dot(d_obj_d_welfare, d_welfare_d_pc_consumption))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('energy_mean_price', 'energy_price'), np.dot(d_obj_d_welfare, d_welfare_d_energy_price))
        else:
            pass
        d_obj_d_discounted_utility, d_obj_d_period_utility_pc = self.utility_m.compute_gradient_min_utility_objective()

        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('population_df', 'population'),  np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_population))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('economics_df', 'pc_consumption'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_pc_consumption))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('energy_mean_price', 'energy_price'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_energy_price))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Utility', 'Utility of pc consumption']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Utility' in chart_list:

            to_plot = ['discounted_utility']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_df'))

            discounted_utility = utility_df['discounted_utility']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(discounted_utility)

            chart_name = 'Utility'

            new_chart = TwoAxesInstanciatedChart('years', 'Discounted Utility (trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Utility of pc consumption' in chart_list:

            to_plot = ['period_utility_pc']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_df'))

            utility = utility_df['period_utility_pc']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(utility)

            chart_name = 'Utility of per capita consumption'

            new_chart = TwoAxesInstanciatedChart('years', 'Utility of pc consumption',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
