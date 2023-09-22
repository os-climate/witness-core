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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from climateeconomics.core.core_world3.agriculture import Agriculture
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np


class AgricultureDiscipline(SoSWrapp):


    # ontology information
    _ontology_data = {
        'label': 'Agriculture World3 Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-seedling fa-fw',
        'version': '',
    }


    # Define the inputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'default': 1900, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        GlossaryCore.YearEnd: {'type': 'int', 'default': 2100, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_data'},
        GlossaryCore.TimeStep: {'type': 'float', 'default': 0.5, 'unit': 'year per period', 'visibility': 'Shared',
                      'namespace': 'ns_data'},
        'pyear': {'type': 'int', 'default': 1975, 'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_coupling'},
        'pop': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'iopc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'io': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ppolx': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    # Define the outputs of the SoSWrapp: type, visibility and namespace (useful for coupling)
    DESC_OUT = {
        'al': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pal': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'f': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fpc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ifpc1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ifpc2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ifpc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'tai': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioaa1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioaa2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fioaa': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ldr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'dcph': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'cai': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'alai': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'aiph': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lymc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lyf': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ly': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lymap': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lymap1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lymap2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fiald': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mpld': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mpai': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'mlymc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'all': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'llmy': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'llmy1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'llmy2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'ler': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'uilpc': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'uilr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lrui': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'uil': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfert': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfdr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfd': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'lfrt': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'fr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'},
        'pfr': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_coupling'}
    }

    AGRICULTURE_CHARTS = 'agriculture charts'

    # Define the init of the SoSWrapp
    def init_execution(self):

        # Instantiate a Ressource object and redifine the args to have array of my parameters with the same size in all my SOSDiscplines
        self.agr = Agriculture()


    def run(self):

        # Link the input parameter to the input value entered hough a dictionary

        inputs_dict = self.get_sosdisc_inputs()

        self.agr.set_data(inputs_dict)

        # Follow the instructions from pyworld3.world3.py to define all functions, parameters, constants of the object
        self.agr.init_agriculture_constants(ali=0.9e9, pali=2.3e9, lfh=0.7, palt=3.2e9,
                                            pl=0.1, alai1=2, alai2=2, io70=7.9e11, lyf1=1,
                                            lyf2=1, sd=0.07, uili=8.2e6, alln=6000, uildt=10,
                                            lferti=600, ilf=600, fspd=2, sfpc=230)
        self.agr.init_agriculture_variables()
        self.agr.set_agriculture_table_functions()
        self.agr.set_agriculture_delay_functions()


        # Computation of the values of parameters à step k

        self.agr.redo_loop = True
        for k in range(self.agr.n):
            self.agr.redo_loop = True
            while self.agr.redo_loop:
                self.agr.redo_loop = False
                if k==0:
                    self.agr.loop0_agriculture(alone= False)

                else:
                    self.agr.loopk_agriculture(k - 1, k, k - 1, k,alone= False)

        # Store the data at the end of the process in a dictionary
        dict_values = {'al': self.agr.al,
        'lfc': self.agr.lfc,
        'pal': self.agr.pal,
        'f': self.agr.f,
        'fpc': self.agr.fpc,
        'ifpc1': self.agr.ifpc1,
        'ifpc2': self.agr.ifpc2,
        'ifpc': self.agr.ifpc,
        'tai': self.agr.tai,
        'fioaa1': self.agr.fioaa1,
        'fioaa2': self.agr.fioaa2,
        'fioaa': self.agr.fioaa,
        'ldr': self.agr.ldr,
        'dcph': self.agr.dcph,
        'cai': self.agr.cai,
        'alai': self.agr.alai,
        'aiph': self.agr.aiph,
        'lymc': self.agr.lymc,
        'lyf': self.agr.lyf,
        'ly': self.agr.ly,
        'lymap': self.agr.lymap,
        'lymap1': self.agr.lymap1,
        'lymap2': self.agr.lymap2,
        'fiald': self.agr.fiald,
        'mpld': self.agr.mpld,
        'mpai': self.agr.mpai,
        'mlymc': self.agr.mlymc,
        'all': self.agr.all,
        'llmy': self.agr.llmy,
        'llmy1': self.agr.llmy1,
        'llmy2': self.agr.llmy2,
        'ler': self.agr.ler,
        'uilpc': self.agr.uilpc,
        'uilr': self.agr.uilr,
        'lrui': self.agr.lrui,
        'uil': self.agr.uil,
        'lfert': self.agr.lfert,
        'lfdr': self.agr.lfdr,
        'lfd': self.agr.lfd,
        'lfr': self.agr.lfr,
        'lfrt': self.agr.lfrt,
        'fr': self.agr.fr,
        'pfr': self.agr.pfr
        }

        #for k,v in dict_values.items():
        #    print(k+" ,", norm(v),",")

        #print("***********************")
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)


    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            AgricultureDiscipline.AGRICULTURE_CHARTS, 'Agricultural Evolution']

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

        if AgricultureDiscipline.AGRICULTURE_CHARTS in chart_list:

            agriculture_df = self.get_sosdisc_outputs()
            years_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
            years_end = self.get_sosdisc_inputs(GlossaryCore.YearEnd)
            time_step = self.get_sosdisc_inputs(GlossaryCore.TimeStep)

            years = np.arange(years_start, years_end, time_step)

            agriculture_series = InstanciatedSeries(
                years.tolist(), agriculture_df['al'].tolist(), 'Arable land (in ha)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add = []


            # Plot arable land

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['al'].tolist(), 'Arable land (in ha)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Arable land (in ha)',
                                             chart_name='Evolution of arable lands', stacked_bar=True)
            new_chart.add_series(new_series)


            instanciated_charts.append(new_chart)

            # Plot development costs

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['dcph'].tolist(), 'Development cost per hectare (in dollar/ha)', InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Development cost per hectare (in dollar/ha)',
                                                 chart_name='Evolution of development cost per hectare', stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot food

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['f'].tolist(), 'Food (in vegetable-equivalent kilograms)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Food (in vegetable-equivalent kilograms)',
                                                 chart_name='Evolution of food production',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)


            # Plot food per capita

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['fpc'].tolist(), 'Food per capita (in vegetable-equivalent kilograms/population)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Food (in vegetable-equivalent kilograms/population)',
                                                 chart_name='Evolution of food production per capita',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot fioaa

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['fioaa'].tolist(), 'Fraction of industrial output allocated to agriculture',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Fraction of industrial output allocated to agriculture',
                                                 chart_name='Evolution of industrial output allocated for food production',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot mpld

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['mpld'].tolist(), 'Marginal productivity of land development (in vegetable-equivalent kilograms/dollar)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years', 'Marginal productivity of land development (in vegetable-equivalent kilograms/dollar)',
                                                 chart_name='Evolution of marginal productivity of land development',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot all

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['all'].tolist(),
                'Average life of land (in years)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years',
                                                 'Average life of land (in years)',
                                                 chart_name='Evolution of the average life of land (in years)',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot ler

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['ler'].tolist(),
                'Land erosion rate (in ha/years)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years',
                                                 'Land erosion rate (in ha/years)',
                                                 chart_name='Evolution of the land erosion rate',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

            # Plot lfr

            new_series = InstanciatedSeries(
                years.tolist(), agriculture_df['lfr'].tolist(),
                'Land fertility regeneration (in vegetable-equivalent kilograms/hectare-year-year)',
                InstanciatedSeries.LINES_DISPLAY)

            series_to_add.append(new_series)

            new_chart = TwoAxesInstanciatedChart('Years',
                                                 'Land fertility regeneration (in vegetable-equivalent kilograms/hectare-year-year)',
                                                 chart_name='Evolution of the land fertility regeneration',
                                                 stacked_bar=True)
            new_chart.add_series(new_series)



            instanciated_charts.append(new_chart)

        return instanciated_charts