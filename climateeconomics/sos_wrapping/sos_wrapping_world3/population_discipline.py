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
from climateeconomics.core.core_world3.population import Population
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np



class PopulationDiscipline(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Population World3 Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-users fa-fw',
        'version': '',
    }

    # Define the inputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 1900, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'year_end': {'type': 'int', 'default': 2100, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'time_step': {'type': 'int', 'default': 0.5, 'unit': 'year per period', 'visibility': 'Shared',
                      'namespace': 'ns_data'},
        'pyear': {'type': 'int', 'default': 1975, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        'ppolx': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fpc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'iopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'sopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    # Define the outputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_OUT = {
        'p1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'p2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'p3': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'p4': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pop': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fpu': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmp': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cmi': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'hsapc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ehspc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmhs1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmhs2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmhs': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lmc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'm1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'm2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'm3': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'm4': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'le': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mat1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mat2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mat3': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'd1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'd2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'd3': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'd4': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'd': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cdr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'aiopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'diopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fie': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'sfsn': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'dcfs': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'frsn': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ple': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cmple': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'dtf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fm': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mtf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'nfc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fsafc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fcapc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fcfpc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fce': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'tf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cbr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'b': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},

    }

    POPULATION_CHARTS = 'population charts'

    # Define the init of the SoSWrapp
    def init_execution(self):

        # Instantiate a Ressource object and redifine the args to have array of my parameters with the same size in all my SOSDiscplines
        self.pop = Population()



    def run(self):

        # Link the input parameter to the input value entered hough a dictionary

        inputs_dict = self.get_sosdisc_inputs()

        self.pop.set_data(inputs_dict)


        # Follow the instructions from pyworld3.world3.py to define all functions, parameters, constants of the object
        self.pop.init_population_constants(p1i=65e7, p2i=70e7, p3i=19e7, p4i=6e7,
                                           dcfsn=4, fcest=4000, hsid=20, ieat=3, len=28,
                                           lpd=20, mtfn=12, pet=4000, rlt=30, sad=20,
                                           zpgt=4000,iphst = 1940)
        self.pop.init_exogenous_inputs()

        inputs_dict = self.get_sosdisc_inputs()
        self.pop.set_data(inputs_dict)

        self.pop.init_population_variables()
        self.pop.set_population_table_functions()
        self.pop.set_population_delay_functions()

        # Re-define the delay function depending on iopc to change the reference of their input for the new value of iopc instead of an array full of nan
        self.pop.smooth_iopc.in_arr = self.pop.iopc
        self.pop.dlinf3_iopc.in_arr = self.pop.iopc


        # Computation of the values of parameters à step k

        self.pop.redo_loop = True
        for k in range (self.pop.n):
            self.pop.redo_loop = True
            while self.pop.redo_loop:
                self.pop.redo_loop = False
                if k==0:
                    self.pop.loop0_population(alone=True)

                else:
                    self.pop.loopk_population(k - 1, k, k - 1, k,alone= False)

        # Store the data at the end of the process in a dictionary
        dict_values = {
            'p1': self.pop.p1,
            'p2': self.pop.p2,
            'p3': self.pop.p3,
            'p4': self.pop.p4,
            'pop': self.pop.pop,
            'fpu': self.pop.fpu,
            'lmp': self.pop.lmp,
            'lmf': self.pop.lmf,
            'cmi': self.pop.cmi,
            'hsapc': self.pop.hsapc,
            'ehspc': self.pop.ehspc,
            'lmhs1': self.pop.lmhs1,
            'lmhs2': self.pop.lmhs2,
            'lmhs': self.pop.lmhs,
            'lmc': self.pop.lmc,
            'm1': self.pop.m1,
            'm2': self.pop.m2,
            'm3': self.pop.m3,
            'm4': self.pop.m4,
            'le': self.pop.le,
            'mat1': self.pop.mat1,
            'mat2': self.pop.mat2,
            'mat3': self.pop.mat3,
            'd1': self.pop.d1,
            'd2': self.pop.d2,
            'd3': self.pop.d3,
            'd4': self.pop.d4,
            'd': self.pop.d,
            'cdr': self.pop.cdr,
            'aiopc': self.pop.aiopc,
            'diopc': self.pop.diopc,
            'fie': self.pop.fie,
            'sfsn': self.pop.sfsn,
            'dcfs': self.pop.dcfs,
            'frsn': self.pop.frsn,
            'ple': self.pop.ple,
            'cmple': self.pop.cmple,
            'dtf': self.pop.dtf,
            'fm': self.pop.fm,
            'mtf': self.pop.mtf,
            'nfc': self.pop.nfc,
            'fsafc': self.pop.fsafc,
            'fcapc': self.pop.fcapc,
            'fcfpc': self.pop.fcfpc,
            'fce': self.pop.fce,
            'tf': self.pop.tf,
            'cbr': self.pop.cbr,
            'b': self.pop.b
                       }


        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            PopulationDiscipline.POPULATION_CHARTS, 'Pollution Evolution']

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

        if PopulationDiscipline.POPULATION_CHARTS in chart_list:

            agriculture_df = self.get_sosdisc_outputs()
            years_start = PopulationDiscipline._get_sosdisc_inputs(self)['year_start']
            years_end = PopulationDiscipline._get_sosdisc_inputs(self)['year_end']
            time_step = PopulationDiscipline._get_sosdisc_inputs(self)['time_step']

            years = np.arange(years_start, years_end, time_step)

            agriculture_series = InstanciatedSeries(
                years.tolist(), agriculture_df['pop'].tolist(), 'Population (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []

            # Plot pop

            new_chart = TwoAxesInstanciatedChart('Years', 'Population (in persons)',
                                                 chart_name='Evolution of population', stacked_bar=True)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['p1'].tolist(), 'Population between 0-14 (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['p2'].tolist(), 'Population between 15-44 (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['p3'].tolist(), 'Population between 45-64 (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['p4'].tolist(), 'Population over 65 (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # Plot pop

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['pop'].tolist(), 'Population (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Population (in persons)',
                                                 chart_name='Evolution of population', stacked_bar=True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # Plot Deaths rate

            new_chart = TwoAxesInstanciatedChart('Years', 'Deaths rate',
                                                 chart_name='Evolution of deaths', stacked_bar=True)

            d1_bis = [agriculture_df['d1'].tolist()[i] / agriculture_df['p1'].tolist()[i] for i in range(len(agriculture_df['d1'].tolist()))]
            new_series = InstanciatedSeries(
                years.tolist(), d1_bis, 'Deaths rate between 0-14', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            d2_bis = [agriculture_df['d1'].tolist()[i] / agriculture_df['p1'].tolist()[i] for i in range(len(agriculture_df['d2'].tolist()))]
            new_series = InstanciatedSeries(
                years.tolist(), d2_bis, 'Deaths rate between 15-44', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            d3_bis = [agriculture_df['d3'].tolist()[i] / agriculture_df['p3'].tolist()[i] for i in
                      range(len(agriculture_df['d3'].tolist()))]
            new_series = InstanciatedSeries(
                years.tolist(), d3_bis, 'Deaths rate between 45-64', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            d4_bis = [agriculture_df['d4'].tolist()[i] / agriculture_df['p4'].tolist()[i] for i in
                      range(len(agriculture_df['d4'].tolist()))]
            new_series = InstanciatedSeries(
                years.tolist(), d4_bis, 'Deaths rate over 65', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # Plot Deaths

            new_chart = TwoAxesInstanciatedChart('Years', 'Deaths (in persons)',
                                                 chart_name='Evolution of deaths', stacked_bar=True)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['d1'].tolist(), 'Deaths between 0-14 (in persons)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['d2'].tolist(), 'Deaths between 15-44 (in persons)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['d3'].tolist(), 'Deaths between 45-64 (in persons)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['d4'].tolist(), 'Deaths over 65 (in persons)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # Plot d

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['d'].tolist(), 'Deaths (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Deaths (in persons)',
                                                 chart_name='Evolution of deaths', stacked_bar=True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # Plot le

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['le'].tolist(), 'Life expectancy (in years)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Life expectancy (in years)',
                                                 chart_name='Evolution of life expectancy', stacked_bar=True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # Plot B

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['b'].tolist(), 'Births (in persons)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Births (in persons)',
                                                 chart_name='Evolution of births', stacked_bar=True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)




        return instanciated_charts