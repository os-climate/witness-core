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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from climateeconomics.core.core_world3.pollution import Pollution
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np


class PollutionDiscipline(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Pollution World3 Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-smog',
        'version': '',
    }

    _maturity = 'Fake'
    # Define the inputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'default': 1900, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        GlossaryCore.YearEnd: {'type': 'int', 'default': 2100, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        GlossaryCore.TimeStep: {'type': 'float', 'default': 0.5, 'unit': 'year per period', 'visibility': 'Shared',
                      'namespace': 'ns_data'},
        'pyear': {'type': 'int', 'default': 1975, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'pcrum': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pop': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'aiph': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'al': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    # Define the outputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_OUT = {
        'ppol': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppolx': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppgio': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppgao': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppgf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pptd': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppgr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppapr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ahlm': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppasr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ahl': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    POLLUTION_CHARTS = 'pollution charts'

    # Define the init of the SoSWrapp
    def init_execution(self):

        # Instantiate a Ressource object and redifine the args to have array of my parameters with the same size in all my SOSDiscplines
        self.pol = Pollution()


    def run(self):

        # Link the input parameter to the input value entered hough a dictionary
        inputs_dict = self.get_sosdisc_inputs()

        self.pol.set_data(inputs_dict)

        # Follow the instructions from pyworld3.world3.py to define all functions, parameters, constants of the object
        self.pol.init_pollution_constants(ppoli=2.5e7, ppol70=1.36e8, ahl70=1.5, amti=1,
                                          imti=10, imef=0.1, fipm=0.001, frpm=0.02,
                                          ppgf1=1, ppgf2=1, ppgf21=1, pptd1=20, pptd2=20)
        self.pol.init_pollution_variables()
        self.pol.set_pollution_table_functions()
        self.pol.set_pollution_delay_functions()


        # Computation of the values of parameters à step k

        self.pol.redo_loop = True
        for k in range(self.pol.n):
            self.pol.redo_loop = True
            while self.pol.redo_loop:
                self.pol.redo_loop = False
                if k == 0:
                     self.pol.loop0_pollution(alone=False)

                else:
                     self.pol.loopk_pollution(k - 1, k, k - 1, k,alone=False)

        # Store the data at the end of the process in a dictionary
        dict_values = {'ppolx': self.pol.ppolx,
                       'ppol': self.pol.ppol,
                       'ppolx': self.pol.ppolx,
                       'ppgio': self.pol.ppgio,
                       'ppgao': self.pol.ppgao,
                       'ppgf': self.pol.ppgf,
                       'pptd': self.pol.pptd,
                       'ppapr': self.pol.ppapr,
                       'ahlm': self.pol.ahlm,
                       'ppasr': self.pol.ppasr,
                       'ahl': self.pol.ahl,
                       'ppgr' : self.pol.ppgr
                       }

        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            PollutionDiscipline.POLLUTION_CHARTS, 'Pollution Evolution']

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        '''
        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then
        '''
        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if PollutionDiscipline.POLLUTION_CHARTS in chart_list:

            agriculture_df = self.get_sosdisc_outputs()
            years_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
            years_end = self.get_sosdisc_inputs(GlossaryCore.YearEnd)
            time_step = self.get_sosdisc_inputs(GlossaryCore.TimeStep)

            years = np.arange(years_start, years_end, time_step)

            agriculture_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ppol'].tolist(), 'Persistent pollution (in pollution units)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []

            # Plot ppol


            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ppol'].tolist(), 'Persistent pollution (in pollution units)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Persistent pollution (in pollution units)',
                                                 chart_name='Evolution of persistent pollution', stacked_bar=True)
            new_chart.add_series(new_series)


            instanciated_charts.append(new_chart)

            # Plot ppolx


            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ppolx'].tolist(), 'Index of persistent pollution',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Index of persistent pollution',
                                                 chart_name='Evolution of the index of persistent pollution', stacked_bar=True)
            new_chart.add_series(new_series)


            instanciated_charts.append(new_chart)

        return instanciated_charts