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
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_dice.damage_model import DamageModel
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class DamageDiscipline(SoSWrapp):
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
        GlossaryCore.YearStart: {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.YearEnd: {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'init_damag_int': {'type': 'float', 'default': 0},
        'damag_int': {'type': 'float', 'default': 0},
        'damag_quad': {'type': 'float', 'default': 0.00236},
        'damag_expo': {'type': 'float', 'default': 2},
        'exp_cont_f': {'type': 'float', 'default': 2.6},
        'cost_backstop': {'type': 'float', 'default': 550},
        'init_cost_backstop': {'type': 'float', 'default': .025},
        'gr_base_carbonprice': {'type': 'float', 'default': .02},
        'init_base_carbonprice': {'type': 'float', 'default': 2},
        'tipping_point': {'type': 'bool', 'default': False},
        'tp_a1': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 20.46},
        'tp_a2': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 2},
        'tp_a3': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 6.081},
        'tp_a4': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 6.754},
        GlossaryCore.DamageToProductivity: {'type': 'bool', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'visibility': 'Shared',
                                                         'namespace': 'ns_dice'},
        GlossaryCore.EconomicsDfValue: GlossaryCore.set_namespace(GlossaryCore.EconomicsDf, 'ns_scenario'),
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario',
                         'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                  GlossaryCore.TotalCO2Emissions: ('float', None, False),
                                                  GlossaryCore.TotalN2OEmissions: ('float', None, False),
                                                  GlossaryCore.TotalCH4Emissions: ('float', None, False),
                                                  }
                         },
        GlossaryCore.TemperatureDfValue: GlossaryCore.set_namespace(GlossaryCore.TemperatureDf, 'ns_scenario'),
        'emissions_control_rate': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario',
                                   'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                            'value': ('float', None, True)},
                                   'dataframe_edition_locked': False},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
    }

    DESC_OUT = {
        GlossaryCore.DamageDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}}
    _maturity = 'Research'

    def run(self):
        ''' pyworld3 execution '''
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        economics_df = in_dict.pop(GlossaryCore.EconomicsDfValue)
        emissions_df = in_dict.pop('emissions_df')
        temperature_df = in_dict.pop(GlossaryCore.TemperatureDfValue)
        emissions_control_rate = in_dict.pop('emissions_control_rate')

        # pyworld3 execution
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

        chart_list = [GlossaryCore.Damages, 'Abatement cost']  # , 'Abatement cost']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        damage_df = self.get_sosdisc_outputs(GlossaryCore.DamageDfValue)
        years = list(damage_df[GlossaryCore.Years].values)
        if GlossaryCore.Damages in chart_list:
            to_plot = [GlossaryCore.Damages]
            damage_df = resize_df(damage_df)

            damage = damage_df[GlossaryCore.Damages]

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = damage.values.max()

            chart_name = 'environmental damage'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, f'{GlossaryCore.Damages} (trill $)',
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
        if 'Abatement cost' in chart_list:

            to_plot = ['abatecost']
            abate_df = self.get_sosdisc_outputs(GlossaryCore.DamageDfValue)

            abatecost = damage_df['abatecost']

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = abatecost.values.max()

            chart_name = 'Abatement cost'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Abatement cost (Trill $)',
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
    length = len(array)
    new_index = index[0:length]
    return new_index
