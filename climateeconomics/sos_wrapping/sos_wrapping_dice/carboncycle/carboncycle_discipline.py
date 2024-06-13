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

from climateeconomics.core.core_dice.geophysical_model import CarbonCycle
from climateeconomics.glossarycore import GlossaryCore
# coding: utf-8
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class CarbonCycleDiscipline(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'Carbon Cycle DICE Model',
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

    DESC_IN = {

        GlossaryCore.YearStart: {'type': 'int', 'default': 2015, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.YearEnd: {'type': 'int', 'default': GlossaryCore.YearEndDefault, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        GlossaryCore.TimeStep: {'type': 'int', 'default': 5, 'unit': 'year per period', 'visibility': 'Shared', 'namespace': 'ns_dice'},
        'conc_lower_strata': {'type': 'int', 'default': 1720, 'unit': 'Gtc'},
        'conc_upper_strata': {'type': 'int', 'default': 360, 'unit': 'Gtc'},
        'conc_atmo': {'type': 'int', 'default': 588, 'unit': 'Gtc'},
        'init_conc_atmo': {'type': 'int', 'default': 851, 'unit': 'Gtc'},
        'init_upper_strata': {'type': 'int', 'default': 460, 'unit': 'Gtc'},
        'init_lower_strata': {'type': 'int', 'default': 1740, 'unit': 'Gtc'},
        'b_twelve': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 0.12, 'unit': '[-]'},
        'b_twentythree': {'type': 'float', 'visibility': SoSWrapp.INTERNAL_VISIBILITY, 'default': 0.007, 'unit': '[-]'},
        'lo_mat': {'type': 'float', 'default': 10},
        'lo_mu': {'type': 'float', 'default': 100},
        'lo_ml': {'type': 'float', 'default': 1000},
        'emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_scenario'}}

    DESC_OUT = {GlossaryCore.CarbonCycleDfValue: {'type': 'dataframe',
                                   'visibility': 'Shared', 'namespace': 'ns_scenario'}}

    def run(self):
        # get input of discipline
        param_in = self.get_sosdisc_inputs()

        # compute output
        carboncycle = CarbonCycle(param_in)
        carboncycle_df = carboncycle.compute(param_in)
        dict_values = {GlossaryCore.CarbonCycleDfValue: carboncycle_df}

        # store data
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then
        chart_filters = []

        chart_list = ['atmosphere concentration',
                      'Atmospheric concentrations parts per million']
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
        carboncycle_df = self.get_sosdisc_outputs(GlossaryCore.CarbonCycleDfValue)
        carboncycle_df = resize_df(carboncycle_df)

        if 'atmosphere concentration' in chart_list:

            #carboncycle_df = discipline.get_sosdisc_outputs(GlossaryCore.CarbonCycleDfValue)
            atmo_conc = carboncycle_df['atmo_conc']

            years = list(atmo_conc.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = atmo_conc.values.max()

            chart_name = 'atmosphere concentration of carbon'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'carbon concentration (Gtc)',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(atmo_conc)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'atmosphere concentration', 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Atmospheric concentrations parts per million' in chart_list:

            #carboncycle_df = discipline.get_sosdisc_outputs(GlossaryCore.CarbonCycleDfValue)
            ppm = carboncycle_df['ppm']

            years = list(ppm.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = ppm.values.max()

            chart_name = 'Atmospheric concentrations parts per million'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Atmospheric concentrations parts per million',
                                                 [year_start - 5, year_end + 5], [
                                                     0, max_value * 1.1],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(ppm)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)

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
