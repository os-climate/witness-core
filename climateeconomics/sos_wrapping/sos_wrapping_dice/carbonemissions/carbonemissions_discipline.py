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

from climateeconomics.core.core_dice.geophysical_model import CarbonEmissions
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class CarbonemissionsDiscipline(SoSWrapp):
    "carbonemissions discipline for DICE"


    # ontology information
    _ontology_data = {
        'label': 'Carbon Emissions DICE Model',
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
    _maturity = 'Research'
    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.YearEnd: {'type': 'int', 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.TimeStep: {'type': 'int', 'unit': 'years per period', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'init_land_emissions': {'type': 'float', 'unit': 'GtCO2 per year', 'default': 2.6},
        'decline_rate_land_emissions': {'type': 'float', 'default': .115},
        'init_cum_land_emisisons': {'type': 'float', 'unit': 'GtCO2', 'default': 100},
        'init_gr_sigma': {'type': 'float', 'default': -0.0152},
        'decline_rate_decarbo': {'type': 'float', 'default': -0.001},
        'init_indus_emissions': {'type': 'float', 'unit': 'GtCO2 per year', 'default': 35.745},
        GlossaryCore.InitialGrossOutput['var_name']: {'type': 'float', 'unit': 'trillions $', 'visibility': 'Shared', 'namespace': 'ns_dice', 'default': 105.1},
        'init_cum_indus_emissions': {'type': 'float', 'unit': 'GtCO2', 'default': 400},
        GlossaryCore.EconomicsDfValue: GlossaryCore.set_namespace(GlossaryCore.EconomicsDf, 'ns_scenario'),
        'emissions_control_rate': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario',
                                   'dataframe_descriptor': {'year': ('float', None, False), 'value': ('float', None, True)},
                                   'dataframe_edition_locked': False}
    }
    DESC_OUT = {
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}
    }

    def run(self):
        # Get inputs
        in_dict = self.get_sosdisc_inputs()
        emissions_control_rate = in_dict.pop('emissions_control_rate')
        # Compute de emissions_model
        emissions_model = CarbonEmissions(in_dict)
        emissions_df = emissions_model.compute(in_dict, emissions_control_rate)
        # Warning : float are mandatory for MDA ...
        emissions_df = emissions_df.astype(float)
        # Store output data
        dict_values = {'emissions_df': emissions_df}
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['carbon emission', 'emission control rate']
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
        emissions_df = self.get_sosdisc_outputs('emissions_df')
        emissions_df = resize_df(emissions_df)

        if 'carbon emission' in chart_list:

            to_plot = ['total_emissions', 'land_emissions', 'indus_emissions']
            #emissions_df = discipline.get_sosdisc_outputs('emissions_df')

            total_emission = emissions_df['total_emissions']

            years = list(emissions_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = total_emission.values.max()

            chart_name = 'total carbon emissions'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'carbon emissions (Gtc)',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(emissions_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'emission control rate' in chart_list:

            to_plot = ['emissions_control_rate']

            inputs = self.get_sosdisc_inputs()
            control_rate_df = inputs.pop('emissions_control_rate')
            control_rate = list(control_rate_df['value'])
            emissions_df = self.get_sosdisc_outputs('emissions_df')

            total_emission = emissions_df['total_emissions']

            years = list(total_emission.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = max(control_rate)

            chart_name = 'emission control rate over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'emission control rate',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, control_rate, 'emissions_control_rate', 'lines', visible_line)

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
