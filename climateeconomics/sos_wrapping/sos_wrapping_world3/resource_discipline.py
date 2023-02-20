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
from climateeconomics.core.core_world3.resource import Resource
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np


class ResourceDiscipline(SoSWrapp):
    # Define the inputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 1900, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'year_end': {'type': 'int', 'default': 2100, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'time_step': {'type': 'int', 'default': 0.5, 'unit': 'year per period', 'visibility': 'Shared',
                      'namespace': 'ns_data'},
        'pyear': {'type': 'int', 'default': 1975, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'iopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pop': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}

    }

    # Define the outputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_OUT = {
        'nr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'nrfr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fcaor1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fcaor2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fcaor': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'nruf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pcrum': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'nrur': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    RESOURCES_CHARTS = 'resources charts'

    # Define the init of the SoSWrapp
    def init_execution(self):
        # Instantiate a Ressource object and redifine the args to have array of my parameters with the same size in all my SOSDiscplines
        self.res = Resource()



    def run(self):

        # Link the input parameter to the input value entered hough a dictionary

        inputs_dict = self.get_sosdisc_inputs()

        self.res.set_data(inputs_dict)

        # Follow the instructions from pyworld3.world3.py to define all functions, parameters, constants of the object

        self.res.init_resource_constants(nri=1e12, nruf1=1, nruf2=1)
        self.res.init_resource_variables()
        self.res.set_resource_table_functions()
        self.res.set_resource_delay_functions()


        # Computation of the values of parameters à step k

        self.res.redo_loop = True

        for k in range(0, self.res.n):
            self.res.redo_loop = True
            while self.res.redo_loop:
                self.res.redo_loop = False
                if k==0:
                    self.res.loop0_resource(alone= False)
                else:
                     self.res.loopk_resource(k-1, k, k-1, k, alone=False)

        # Store the data at the end of the process in a dictionary
        dict_values = {'nr': self.res.nr,
        'nrfr': self.res.nrfr,
        'fcaor1': self.res.fcaor1,
        'fcaor2': self.res.fcaor2,
        'fcaor': self.res.fcaor,
        'nruf': self.res.nruf,
        'pcrum': self.res.pcrum,
        'nrur': self.res.nrur
        }


        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            ResourceDiscipline.RESOURCES_CHARTS, 'Resources Evolution']

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

        if ResourceDiscipline.RESOURCES_CHARTS in chart_list:

            agriculture_df = self.get_sosdisc_outputs()
            years_start = ResourceDiscipline._get_sosdisc_inputs(self)['year_start']
            years_end = ResourceDiscipline._get_sosdisc_inputs(self)['year_end']
            time_step = ResourceDiscipline._get_sosdisc_inputs(self)['time_step']

            years = np.arange(years_start, years_end, time_step)

            agriculture_series = InstanciatedSeries(
                years.tolist(), agriculture_df['nr'].tolist(), 'Nonrenewable resources (in resource units)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []

            # Plot nr


            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['nr'].tolist(), 'Nonrenewable resources (in resource units', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Nonrenewable resources (in resource units',
                                                 chart_name='Evolution of nonrenewable resources', stacked_bar=True)
            new_chart.add_series(new_series)


            instanciated_charts.append(new_chart)

            # Plot nrfr

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['nrfr'].tolist(), 'Nonrenewable resource fraction remaining',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Nonrenewable resource fraction remaining',
                                                 chart_name='Evolution of the nonrenewable resource fraction remaining', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot fcaor


            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['fcaor'].tolist(), 'Fraction of capital allocated to obtaining resources',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Fraction of capital allocated to obtaining resources',
                                                 chart_name='Evolution of the fraction of capital allocated to obtaining resources',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

        return instanciated_charts