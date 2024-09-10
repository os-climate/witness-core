'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_dice.utility_model import UtilityModel
from climateeconomics.glossarycore import GlossaryCore


class UtilityModelDiscipline(SoSWrapp):
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
        GlossaryCore.YearStart: {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.YearEnd: {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.TimeStep: {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'init_rate_time_pref': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'scaleone': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 0.0302455265681763},
        'scaletwo': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': -10993.704},
        GlossaryCore.EconomicsDfValue: GlossaryCore.set_namespace(GlossaryCore.EconomicsDf, 'ns_scenario'),
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        GlossaryCore.TemperatureDfValue: GlossaryCore.set_namespace(GlossaryCore.TemperatureDf, 'ns_scenario'),
    }
    DESC_OUT = {
        GlossaryCore.UtilityDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}
    }

    def run(self):
        # get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # compute utility
        economics_df = inp_dict.pop(GlossaryCore.EconomicsDfValue)
        emissions_df = inp_dict.pop('emissions_df')
        temperature_df = inp_dict.pop(GlossaryCore.TemperatureDfValue)
        utility_m = UtilityModel(inp_dict)
        utility_df = utility_m.compute(
            economics_df, emissions_df, temperature_df)

        # store output data
        dict_values = {GlossaryCore.UtilityDfValue: utility_df}
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
        chart_list = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Utility' in chart_list:

            to_plot = [GlossaryCore.DiscountedUtility]
            utility_df = self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue)
            utility_df = resize_df(utility_df)

            discounted_utility = utility_df[GlossaryCore.DiscountedUtility]

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = discounted_utility.values.max()

            chart_name = 'Utility'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Discounted Utility (trill $)',
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
            utility_df = self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue)
            utility_df = resize_df(utility_df)

            utility = utility_df['period_utility']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = utility.values.max()

            chart_name = 'Utility of per capita consumption'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Utility of pc consumption',
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
    length = len(array)
    new_index = index[0:length]
    return new_index
