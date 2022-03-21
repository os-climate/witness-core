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

from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from climateeconomics.core.core_witness.lost_capital_objective_model import LostCapitalObjective


class LostCapitalObjectiveDiscipline(SoSDiscipline):
    "Lost Capital Objective discipline for WITNESS optimization"

    # ontology information
    _ontology_data = {
        'label': 'Lost Capital Objective Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        'year_start': {'type': 'int', 'default': 2020, 'possible_values': years, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100, 'possible_values': years, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_list': {'type': 'string_list', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness', 'user_level': 1, 'structuring': True},
        'lost_capital_obj_ref': {'type': 'float', 'user_level': 2, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},

    }
    DESC_OUT = {
        'lost_capital_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'lost_capital_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'}
    }

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        all_lost_capital_list = []

        # Recover the full techno list to get all lost capital by energy mix
        energy_techno_dict = {}
        if 'energy_list' in self._data_in:
            energy_list = self.get_sosdisc_inputs('energy_list')
            if energy_list is not None:
                for energy in energy_list:
                    dynamic_inputs[f'{energy}.technologies_list'] = {'type': 'string_list',
                                                                     'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                                     'namespace': 'ns_energy_mix',
                                                                     'structuring': True}

                    if f'{energy}.technologies_list' in self._data_in:
                        energy_techno_dict[energy] = self.get_sosdisc_inputs(
                            f'{energy}.technologies_list')

        if len(energy_techno_dict) != 0:
            full_techno_list = compute_full_techno_list(energy_techno_dict)

            # Add the full techno_list to the list of all lost capital
            # the list could be appended with other capital than energy
            all_lost_capital_list.extend(full_techno_list)

        for lost_capital in all_lost_capital_list:
            dynamic_inputs[f'{lost_capital}.lost_capital'] = {'type': 'dataframe',
                                                              'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                                              'namespace': 'ns_energy_mix'}

        self.add_inputs(dynamic_inputs)

    def init_execution(self):

        inp_dict = self.get_sosdisc_inputs()
        self.model = LostCapitalObjective(inp_dict)

    def run(self):
        # get inputs

        inp_dict = self.get_sosdisc_inputs()

        self.model.compute(inp_dict)

        lost_capital_objective = self.model.get_objective()
        lost_capital_df = self.model.get_lost_capital_df()

        # store output data
        dict_values = {'lost_capital_df': lost_capital_df,
                       'lost_capital_objective': lost_capital_objective}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        lost_capital_objective
        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(inputs_dict['year_start'],
                          inputs_dict['year_end'] + 1)
        lost_capital_obj_ref = inputs_dict['lost_capital_obj_ref']
        input_capital_list = [
            key for key in inputs_dict.keys() if key.endswith('lost_capital')]
        for lost_capital in input_capital_list:
            column_name = [
                col for col in inputs_dict[lost_capital].columns if col != 'years'][0]
            self.set_partial_derivative_for_other_types(
                ('lost_capital_objective', ), (lost_capital, column_name), np.ones(len(years)) / lost_capital_obj_ref)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Lost Capitals']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Lost Capitals' in chart_list:

            lost_capital_df = self.get_sosdisc_outputs('lost_capital_df')

            years = list(lost_capital_df['years'].values)

            chart_name = 'Capital lost'

            new_chart = TwoAxesInstanciatedChart('years', 'Lost Capitals (M$)',
                                                 chart_name=chart_name, stacked_bar=True)
            for industry in lost_capital_df.columns:
                if industry not in ['years', 'Sum of lost capital'] and not (lost_capital_df[industry] == 0.0).all():
                    new_series = InstanciatedSeries(
                        years, lost_capital_df[industry].values.tolist(), industry, 'bar')

                    new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, lost_capital_df['Sum of lost capital'].values.tolist(), 'Sum of lost capital', 'lines')

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        return instanciated_charts


def compute_full_techno_list(energy_techno_dict):
    '''
    Get the full list of technologies with a dictionary of energy_techno_dict
    '''
    full_techno_list = []
    for energy, techno_list in energy_techno_dict.items():
        full_techno_list.extend(
            [f'{energy}.{techno}' for techno in techno_list])

    return full_techno_list
