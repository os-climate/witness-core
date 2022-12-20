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

from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from climateeconomics.core.core_witness.non_use_capital_objective_model import NonUseCapitalObjective
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from energy_models.core.energy_mix.energy_mix import EnergyMix


class NonUseCapitalObjectiveDiscipline(SoSWrapp):
    "Non Use Capital Objective discipline for WITNESS optimization"

    # ontology information
    _ontology_data = {
        'label': 'Non Use Capital Objective Model',
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
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'energy_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                        'possible_values': EnergyMix.energy_list,
                        'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_witness', 'user_level': 1,
                        'structuring': True, 'unit': '-'},
        'ccs_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'possible_values': EnergyMix.ccs_list,
                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_witness', 'user_level': 1,
                     'structuring': True, 'unit': '-'},
        'agri_capital_techno_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_witness',
                                     'user_level': 1, 'structuring': True, 'unit': '-'},
        'non_use_capital_obj_ref': {'type': 'float', 'default': 50000., 'unit': 'G$', 'user_level': 2,
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'unit': '-', 'default': 0.5, 'visibility': 'Shared',
                  'namespace': 'ns_witness', 'user_level': 1},
        'gamma': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-', 'visibility': 'Shared',
                  'namespace': 'ns_witness',
                  'user_level': 1},
        'non_use_capital_cons_ref': {'type': 'float', 'default': 20000., 'unit': 'G$', 'user_level': 2,
                                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'non_use_capital_cons_limit': {'type': 'float', 'default': 40000., 'unit': 'G$', 'user_level': 2,
                                       'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        # WIP is_dev to remove once its validated on dev processes
        'is_dev': {'type': 'bool', 'default': False, 'user_level': 2, 'structuring': True,
                   'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_public'},

    }
    DESC_OUT = {
        'non_use_capital_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'G$'},
        'non_use_capital_df': {'type': 'dataframe', 'unit': 'G$'},
        'techno_capital_df': {'type': 'dataframe', 'unit': 'G$'},
        'energy_capital': {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'non_use_capital_cons': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'G$'},
        'forest_lost_capital_cons': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'G$'},
    }

    def setup_sos_disciplines(self, proxy):

        dynamic_inputs = {}
        all_non_use_capital_list = []

        # Recover the full techno list to get all non_use capital by energy mix
        energy_techno_dict = {}
        if 'is_dev' in proxy.get_data_in():
            is_dev = proxy.get_sosdisc_inputs('is_dev')
            if is_dev:
                dynamic_inputs['forest_lost_capital'] = {'type': 'dataframe',
                                                         'unit': 'G$',
                                                         'user_level': 2,
                                                         'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                         'namespace': 'ns_forest',
                                                         'structuring': True}
                dynamic_inputs['forest_lost_capital_cons_ref'] = {'type': 'float',
                                                                  'unit': 'G$',
                                                                  'default': 20.,
                                                                  'user_level': 2,
                                                                  'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                  'namespace': 'ns_ref',
                                                                  'structuring': True}
                dynamic_inputs['forest_lost_capital_cons_limit'] = {'type': 'float',
                                                                    'unit': 'G$',
                                                                    'default': 40.,
                                                                    'user_level': 2,
                                                                    'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                    'namespace': 'ns_ref',
                                                                    'structuring': True}
        if 'energy_list' in proxy.get_data_in():
            energy_list = proxy.get_sosdisc_inputs('energy_list')
            if energy_list is not None:
                for energy in energy_list:
                    if energy == BiomassDry.name and is_dev == True:
                        pass
                    else:
                        dynamic_inputs[f'{energy}.technologies_list'] = {'type': 'list',
                                                                         'subtype_descriptor': {'list': 'string'},
                                                                         'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                         'namespace': 'ns_energy',
                                                                         'structuring': True,
                                                                         'possible_values': EnergyMix.stream_class_dict[
                                                                             energy].default_techno_list}

                        if f'{energy}.technologies_list' in proxy.get_data_in():
                            techno_list = proxy.get_sosdisc_inputs(
                                f'{energy}.technologies_list')
                            if techno_list is not None:
                                energy_techno_dict[energy] = {'namespace': 'ns_energy',
                                                              'value': techno_list}
        if 'ccs_list' in proxy.get_data_in():
            ccs_list = proxy.get_sosdisc_inputs('ccs_list')
            if ccs_list is not None:
                for ccs in ccs_list:
                    dynamic_inputs[f'{ccs}.technologies_list'] = {'type': 'list',
                                                                  'subtype_descriptor': {'list': 'string'},
                                                                  'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                  'namespace': 'ns_ccs',
                                                                  'structuring': True,
                                                                  'possible_values': EnergyMix.stream_class_dict[
                                                                      ccs].default_techno_list}

                    if f'{ccs}.technologies_list' in proxy.get_data_in():
                        techno_list = proxy.get_sosdisc_inputs(
                            f'{ccs}.technologies_list')
                        if techno_list is not None:
                            energy_techno_dict[ccs] = {'namespace': 'ns_ccs',
                                                       'value': techno_list}

        if 'agri_capital_techno_list' in proxy.get_data_in():
            agriculture_techno_list = proxy.get_sosdisc_inputs(
                'agri_capital_techno_list')
            if agriculture_techno_list is not None:
                energy_techno_dict['Agriculture'] = {'namespace': 'ns_forest',
                                                     'value': agriculture_techno_list}

        if len(energy_techno_dict) != 0:
            full_techno_list = compute_full_techno_list(energy_techno_dict)

            # Add the full techno_list to the list of all non_use capital
            # the list could be appended with other capital than energy
            all_non_use_capital_list.extend(full_techno_list)

        for non_use_capital_tuple in all_non_use_capital_list:
            dynamic_inputs[f'{non_use_capital_tuple[0]}non_use_capital'] = {'type': 'dataframe',
                                                                            'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                            'namespace': non_use_capital_tuple[1],
                                                                            'unit': 'G$'}
            dynamic_inputs[f'{non_use_capital_tuple[0]}techno_capital'] = {'type': 'dataframe',
                                                                           'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                           'namespace': non_use_capital_tuple[1],
                                                                           'unit': 'G$'}

        proxy.add_inputs(dynamic_inputs)

    def init_execution(self, proxy):

        inp_dict = proxy.get_sosdisc_inputs()
        self.model = NonUseCapitalObjective(inp_dict)

    def run(self):
        # get inputs

        inp_dict = self.get_sosdisc_inputs()

        self.model.compute(inp_dict)

        non_use_capital_objective = self.model.get_objective()
        non_use_capital_df = self.model.get_non_use_capital_df()
        techno_capital_df = self.model.get_techno_capital_df()
        energy_capital = self.model.get_energy_capital_trillion_dollars()
        non_use_capital_cons = self.model.get_constraint()
        forest_lost_capital_cons = self.model.get_reforestation_constraint()
        # store output data
        dict_values = {'non_use_capital_df': non_use_capital_df,
                       'techno_capital_df': techno_capital_df,
                       'energy_capital': energy_capital,
                       'non_use_capital_objective': non_use_capital_objective,
                       'non_use_capital_cons': non_use_capital_cons,
                       'forest_lost_capital_cons': forest_lost_capital_cons}

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        non_use_capital_objective
        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(inputs_dict['year_start'],
                          inputs_dict['year_end'] + 1)
        non_use_capital_obj_ref = inputs_dict['non_use_capital_obj_ref']
        alpha, gamma = inputs_dict['alpha'], inputs_dict['gamma']
        non_use_capital_cons_ref = inputs_dict['non_use_capital_cons_ref']
        is_dev = inputs_dict['is_dev']
        outputs_dict = self.get_sosdisc_outputs()
        non_use_capital_df = outputs_dict['non_use_capital_df']
        input_nonusecapital_list = [
            key for key in inputs_dict.keys() if key.endswith('non_use_capital')]
        delta_years = len(non_use_capital_df['years'].values)
        for non_use_capital in input_nonusecapital_list:
            column_name = [
                col for col in inputs_dict[non_use_capital].columns if col != 'years'][0]
            self.set_partial_derivative_for_other_types(
                ('non_use_capital_objective',), (non_use_capital, column_name),
                np.ones(len(years)) * alpha * (1 - gamma) / non_use_capital_obj_ref / delta_years)
            self.set_partial_derivative_for_other_types(
                ('non_use_capital_cons',), (non_use_capital, column_name),
                - np.ones(len(years)) / non_use_capital_cons_ref / delta_years)
        input_capital_list = [
            key for key in inputs_dict.keys() if key.endswith('techno_capital')]

        for capital in input_capital_list:
            column_name = [
                col for col in inputs_dict[capital].columns if col != 'years'][0]
            self.set_partial_derivative_for_other_types(
                ('energy_capital', 'energy_capital'), (capital, column_name), np.identity(len(years)) / 1.e3)
        if is_dev:
            forest_lost_capital_cons_ref = inputs_dict['forest_lost_capital_cons_ref']
            self.set_partial_derivative_for_other_types(
                ('forest_lost_capital_cons',
                 ), ('forest_lost_capital', 'reforestation'),
                - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)
            self.set_partial_derivative_for_other_types(
                ('forest_lost_capital_cons',), ('forest_lost_capital', 'managed_wood'),
                - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)
            self.set_partial_derivative_for_other_types(
                ('forest_lost_capital_cons',
                 ), ('forest_lost_capital', 'deforestation'),
                - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)

    def get_chart_filter_list(self, proxy):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Non Use Capitals', 'Energy Mix Total Capital',
                      'Forest Management Lost Capital']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, proxy, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        is_dev = proxy.get_sosdisc_inputs('is_dev')
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Non Use Capitals' in chart_list:

            non_use_capital_df = self.get_sosdisc_outputs('non_use_capital_df')

            years = list(non_use_capital_df['years'].values)

            chart_name = 'Non-use Capital per year'

            new_chart = TwoAxesInstanciatedChart('years', 'non_use Capital [G$]',
                                                 chart_name=chart_name, stacked_bar=True)
            for industry in non_use_capital_df.columns:
                if industry not in ['years', 'Sum of non use capital'] and not (
                        non_use_capital_df[industry] == 0.0).all():
                    new_series = InstanciatedSeries(
                        years, non_use_capital_df[industry].values.tolist(), industry, 'bar')

                    new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, non_use_capital_df['Sum of non use capital'].values.tolist(), 'Sum of non-use capital', 'lines')

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Energy Mix Total Capital' in chart_list:
            techno_capital_df = self.get_sosdisc_outputs('techno_capital_df')

            non_use_capital_df = self.get_sosdisc_outputs('non_use_capital_df')

            years = list(techno_capital_df['years'].values)

            chart_name = 'Energy Mix total capital vs non-use capital per year'

            new_chart = TwoAxesInstanciatedChart('years', 'Total Capital [G$]',
                                                 chart_name=chart_name)

            new_series = InstanciatedSeries(
                years, techno_capital_df['Sum of techno capital'].values.tolist(), 'Energy Mix Capital', 'lines')

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, non_use_capital_df['Sum of non use capital'].values.tolist(), 'Non-use Capital', 'bar')

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Forest Management Lost Capital' in chart_list and is_dev:
            forest_lost_capital = proxy.get_sosdisc_inputs(
                'forest_lost_capital')

            years = list(forest_lost_capital['years'].values)

            chart_name = 'Forest Management Lost Capital'

            new_chart = TwoAxesInstanciatedChart('years', 'Total Capital [G$]',
                                                 chart_name=chart_name, stacked_bar=True)

            new_serie_reforest = InstanciatedSeries(
                years, forest_lost_capital['reforestation'].values.tolist(), 'Reforestation Lost Capital', 'bar')

            new_serie_managed_wood = InstanciatedSeries(
                years, forest_lost_capital['managed_wood'].values.tolist(), 'Managed Wood Lost Capital', 'bar')

            new_serie_deforestation = InstanciatedSeries(
                years, forest_lost_capital['deforestation'].values.tolist(), 'Deforestation Lost Capital', 'bar')

            new_chart.series.append(new_serie_reforest)
            new_chart.series.append(new_serie_managed_wood)
            new_chart.series.append(new_serie_deforestation)
            instanciated_charts.append(new_chart)

        return instanciated_charts


def compute_full_techno_list(energy_techno_dict):
    '''
    Get the full list of technologies with a dictionary of energy_techno_dict
    '''
    full_techno_list = []
    for energy, techno_dict in energy_techno_dict.items():
        if energy == 'Agriculture':
            full_techno_list.extend(
                [(f'', techno_dict['namespace']) for techno in techno_dict['value']])
        else:
            full_techno_list.extend(
                [(f'{energy}.{techno}.', techno_dict['namespace']) for techno in techno_dict['value']])

    return full_techno_list
