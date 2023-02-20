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
from climateeconomics.core.core_world3.capital import Capital
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np


class CapitalDiscipline(SoSWrapp):
    # Define the inputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 1900, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'year_end': {'type': 'int', 'default': 2100, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'time_step': {'type': 'int', 'default': 0.5, 'unit': 'year per period', 'visibility': 'Shared',
                      'namespace': 'ns_data'},
        'pyear': {'type': 'int', 'default': 1975, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'fcaor': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pop': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioaa': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'p2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'p3': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'aiph': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'al': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    # Define the outputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_OUT = {
        'lufd': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cuf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ic': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'alic': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'icdr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'icor': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'io': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'iopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioacc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioacv': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioac': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'sc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'isopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'isopc1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'isopc2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'alsc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'scdr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'so': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'sopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioas1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioas2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioas': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'scir': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioai': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'icir': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'jpicu': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pjis': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'jpscu': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pjss': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'jph': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pjas': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'j': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'luf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    CAPITAL_CHARTS = 'capital charts'

    # Define the init of the SoSWrapp
    def init_execution(self):

        # Instantiate a Ressource object and redifine the args to have array of my parameters with the same size in all my SOSDiscplines
        self.cap = Capital()



    def run(self):


        # Link the input parameter to the input value entered hough a dictionary

        inputs_dict = self.get_sosdisc_inputs()

        self.cap.set_data(inputs_dict)


        # Follow the instructions from pyworld3.world3.py to define all functions, parameters, constants of the object
        self.cap.init_capital_constants(ici=2.1e11, sci=1.44e11, iet=4000, iopcd=400,
                                        lfpf=0.75, lufdt=2, icor1=3, icor2=3, scor1=1,
                                        scor2=1, alic1=14, alic2=14, alsc1=20, alsc2=20,
                                        fioac1=0.43, fioac2=0.43)
        self.cap.init_capital_variables()
        self.cap.set_capital_table_functions()
        self.cap.set_capital_delay_functions()

        # Computation of the values of parameters à step k


        self.cap.redo_loop = True
        for k in range(self.cap.n):
            self.cap.redo_loop = True
            while self.cap.redo_loop:
                self.cap.redo_loop = False
                if k == 0:
                    self.cap.loop0_capital(alone= False)
                else:
                    self.cap.loopk_capital(k - 1, k, k - 1, k,alone= False)

        # Store the data at the end of the process in a dictionary
        dict_values = {
            'lufd': self.cap.lufd,
            'cuf': self.cap.cuf,
            'ic': self.cap.ic,
            'alic': self.cap.alic,
            'icdr': self.cap.icdr,
            'icor': self.cap.icor,
            'io': self.cap.io,
            'iopc': self.cap.iopc,
            'fioacc': self.cap.fioacc,
            'fioacv': self.cap.fioacv,
            'fioac': self.cap.fioac,
            'sc': self.cap.sc,
            'isopc': self.cap.isopc,
            'isopc1': self.cap.isopc1,
            'isopc2': self.cap.isopc2,
            'alsc': self.cap.alsc,
            'scdr': self.cap.scdr,
            'so': self.cap.so,
            'sopc': self.cap.sopc,
            'fioas1': self.cap.fioas1,
            'fioas2': self.cap.fioas2,
            'fioas': self.cap.fioas,
            'scir': self.cap.scir,
            'fioai': self.cap.fioai,
            'icir': self.cap.icir,
            'jpicu': self.cap.jpicu,
            'pjis': self.cap.pjis,
            'jpscu': self.cap.jpscu,
            'pjss': self.cap.pjss,
            'jph': self.cap.jph,
            'pjas': self.cap.pjas,
            'j': self.cap.j,
            'lf': self.cap.lf,
            'luf': self.cap.luf
        }

        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            CapitalDiscipline.CAPITAL_CHARTS, 'Agricultural Evolution']

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

        if CapitalDiscipline.CAPITAL_CHARTS in chart_list:

            agriculture_df = self.get_sosdisc_outputs()
            years_start = CapitalDiscipline._get_sosdisc_inputs(self)['year_start']
            years_end = CapitalDiscipline._get_sosdisc_inputs(self)['year_end']
            time_step = CapitalDiscipline._get_sosdisc_inputs(self)['time_step']

            years = np.arange(years_start, years_end, time_step)

            agriculture_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ic'].tolist(), 'Industrial capital (in dollars)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []


            # Plot ic

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ic'].tolist(), 'Industrial capital (in dollars)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Industrial capital (in dollars)',
                                             chart_name='Evolution of industrial capital', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot io

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['io'].tolist(), 'Industrial output (in dollars/year)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Industrial output (in dollars/year)',
                                                 chart_name='Evolution of industrial output', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot fioac

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['fioac'].tolist(), 'Fraction of industrial output allocated to consumption', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Fraction of industrial output allocated to consumption',
                                                 chart_name='Evolution of industrial output', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)


            # Plot sc

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['sc'].tolist(), 'Service capital (in dollars)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Service capital (in dollars)',
                                                 chart_name='Evolution of services capital', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot so

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['so'].tolist(), 'Service output (in dollars/year)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Service output (in dollars/year)',
                                                 chart_name='Evolution of services output', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot fioas

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['fioas'].tolist(), 'Fraction of industrial output allocated to services',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Fraction of industrial output allocated to services',
                                                 chart_name='Evolution of industrial output allocated to service output', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot j

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['j'].tolist(), 'Jobs (in persons)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Jobs (in persons)',
                                                 chart_name='Evolution of industrial output', stacked_bar=True)
            new_chart.add_series(new_series)


            instanciated_charts.append(new_chart)



        return instanciated_charts

