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

from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from climateeconomics.core.core_dice.utility_model import UtilityModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter

import pandas as pd


class UtilityModelDiscipline(SoSDiscipline):
    "UtilityModel discipline for DICE"


    # ontology information
    _ontology_data = {
        'label': 'Utility DICE Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-child fa-fw',
        'version': '',
    }
    _maturity = 'Research'
    DESC_IN = {
        'year_start': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'year_end': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'time_step': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'init_rate_time_pref': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'scaleone': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': 0.0302455265681763},
        'scaletwo': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': -10993.704},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        'temperature_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
    }
    DESC_OUT = {
        'utility_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}
    }

    def run(self):
        # get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # compute utility
        economics_df = inp_dict.pop('economics_df')
        emissions_df = inp_dict.pop('emissions_df')
        temperature_df = inp_dict.pop('temperature_df')
        utility_m = UtilityModel(inp_dict)
        utility_df = utility_m.compute(
            economics_df, emissions_df, temperature_df)

        # store output data
        dict_values = {'utility_df': utility_df}
        self.store_sos_outputs_values(dict_values)

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
            utility_df = self.get_sosdisc_outputs('utility_df')
            utility_df = resize_df(utility_df)

            discounted_utility = utility_df['discounted_utility']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = discounted_utility.values.max()

            chart_name = 'Utility'

            new_chart = TwoAxesInstanciatedChart('years', 'Discounted Utility (trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Utility of pc consumption' in chart_list:

            to_plot = ['period_utility']
            utility_df = self.get_sosdisc_outputs('utility_df')
            utility_df = resize_df(utility_df)

            utility = utility_df['period_utility']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = utility.values.max()

            chart_name = 'Utility of per capita consumption'

            new_chart = TwoAxesInstanciatedChart('years', 'Utility of pc consumption',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts


def resize_df(df):

    index = df.index
    i = len(index) - 1
    key = df.keys()
    to_check = df.loc[index[i], key[0]]

    while to_check == 0:
        i = i - 1
        to_check = df.loc[index[i], key[0]]

    size_diff = len(index) - i
    new_df = pd.DataFrame()

    if size_diff == 0:
        new_df = df
    else:
        for element in key:
            new_df[element] = df[element][0:i + 1]
            new_df.index = index[0: i + 1]

    return new_df


def resize_array(array):

    i = len(array) - 1
    to_check = array[i]

    while to_check == 0:
        i = i - 1
        to_check = to_check = array[i]

    size_diff = len(array) - i
    new_array = array[0:i]

    return new_array


def resize_index(index, array):

    l = len(array)
    new_index = index[0:l]
    return new_index
