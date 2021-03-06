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
from climateeconomics.core.core_dice.damage_model import DamageModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter

import pandas as pd


class DamageDiscipline(SoSDiscipline):
    "     Temperature evolution"


    # ontology information
    _ontology_data = {
        'label': 'Damage DICE Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-exclamation-triangle fa-fw',
        'version': '',
    }
    DESC_IN = {
        'year_start': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'year_end': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'time_step': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'init_damag_int': {'type': 'float', 'default': 0},
        'damag_int': {'type': 'float', 'default':0},
        'damag_quad': {'type': 'float', 'default': 0.00236},
        'damag_expo': {'type': 'float', 'default': 2},
        'exp_cont_f': {'type': 'float', 'default': 2.6},
        'cost_backstop': {'type': 'float', 'default': 550},
        'init_cost_backstop': {'type': 'float', 'default' : .025},
        'gr_base_carbonprice': {'type': 'float', 'default': .02},
        'init_base_carbonprice': {'type': 'float', 'default': 2},
        'tipping_point': {'type': 'bool', 'default': False},
        'tp_a1': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': 20.46},
        'tp_a2': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': 2},
        'tp_a3': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': 6.081},
        'tp_a4': {'type': 'float', 'visibility': SoSDiscipline.INTERNAL_VISIBILITY, 'default': 6.754},
        'damage_to_productivity': {'type': 'bool', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'frac_damage_prod': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        'temperature_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'},
        'emissions_control_rate': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario',
                                   'dataframe_descriptor': {'year': ('float', None, False), 'value': ('float', None, True)},
                                   'dataframe_edition_locked': False}
    }

    DESC_OUT = {
        'damage_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}}
    _maturity = 'Research'

    def run(self):
        ''' model execution '''
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        economics_df = in_dict.pop('economics_df')
        emissions_df = in_dict.pop('emissions_df')
        temperature_df = in_dict.pop('temperature_df')
        emissions_control_rate = in_dict.pop('emissions_control_rate')

        # model execution
        model = DamageModel(in_dict)
        damage_df = model.compute(economics_df, emissions_df,
                                  temperature_df, emissions_control_rate)

        # store output data
        out_dict = {"damage_df": damage_df}        
        self.store_sos_outputs_values(out_dict)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Damage', 'Abatement cost']  # , 'Abatement cost']
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

        if 'Damage' in chart_list:

            to_plot = ['damages']
            damage_df = self.get_sosdisc_outputs('damage_df')
            damage_df = resize_df(damage_df)

            damage = damage_df['damages']

            years = list(damage_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = damage.values.max()

            chart_name = 'environmental damage'

            new_chart = TwoAxesInstanciatedChart('years', 'Damage (trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

        if 'Abatement cost' in chart_list:

            to_plot = ['abatecost']
            abate_df = self.get_sosdisc_outputs('damage_df')

            abatecost = damage_df['abatecost']

            years = list(abate_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = abatecost.values.max()

            chart_name = 'Abatement cost'

            new_chart = TwoAxesInstanciatedChart('years', 'Abatement cost (Trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)
            for key in to_plot:
                visible_line = True

                c_emission = list(damage_df[key])

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
