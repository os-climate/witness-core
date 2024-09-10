'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/21-2023/11/03 Copyright 2023 Capgemini

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
import numpy as np
from energy_models.core.energy_mix.energy_mix import EnergyMix
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.non_use_capital_objective_model import (
    NonUseCapitalObjective,
)
from climateeconomics.glossarycore import GlossaryCore


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

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.energy_list: {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                        'possible_values': EnergyMix.energy_list,
                        'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 1,
                        'structuring': True, 'unit': '-'},
        GlossaryCore.ccs_list: {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'possible_values': EnergyMix.ccs_list,
                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 1,
                     'structuring': True, 'unit': '-'},
        'agri_capital_techno_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                                     'user_level': 1, 'structuring': True, 'unit': '-'},
        'non_use_capital_obj_ref': {'type': 'float', 'default': 50000., 'unit': 'G$', 'user_level': 2,
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'non_use_capital_cons_ref': {'type': 'float', 'default': 20000., 'unit': 'G$', 'user_level': 2,
                                     'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'non_use_capital_cons_limit': {'type': 'float', 'default': 40000., 'unit': 'G$', 'user_level': 2,
                                       'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'forest_lost_capital': {'type': 'dataframe', 'unit': 'G$', 'user_level': 2, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                'namespace': 'ns_forest', 'dataframe_descriptor':{
                                                         GlossaryCore.Years: ('float', None, False),
                                                         'reforestation': ('float', None, True),
                                                         'managed_wood': ('float', None, True),
                                                         'deforestation': ('float', None, True),
                                                     }
                                                 },
        'forest_lost_capital_cons_ref': {'type': 'float',  'unit': 'G$', 'default': 20., 'user_level': 2,
                                         'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE,},
        'forest_lost_capital_cons_limit': {'type': 'float', 'unit': 'G$', 'default': 40., 'user_level': 2,
                                           'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE,
                                           }

    }
    DESC_OUT = {
        'non_use_capital_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': 'G$'},
        'non_use_capital_df': {'type': 'dataframe', 'unit': 'G$'},
        'techno_capital_df': {'type': 'dataframe', 'unit': 'G$'},
        GlossaryCore.EnergyCapitalDfValue: {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'non_use_capital_cons': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': 'G$'},
        'forest_lost_capital_cons': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': 'G$'},
    }

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        all_non_use_capital_list = []

        # Recover the full techno list to get all non_use capital by energy mix
        energy_techno_dict = {}

        if GlossaryCore.energy_list in self.get_data_in():
            energy_list = self.get_sosdisc_inputs(GlossaryCore.energy_list)
            if energy_list is not None:
                for energy in energy_list:
                    if energy == BiomassDry.name:
                        pass
                    else:
                        dynamic_inputs[f'{energy}.{GlossaryCore.techno_list}'] = {'type': 'list',
                                                                         'subtype_descriptor': {'list': 'string'},
                                                                         'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                         'namespace': 'ns_energy',
                                                                         'structuring': True,
                                                                         'possible_values': EnergyMix.stream_class_dict[
                                                                             energy].default_techno_list}

                        if f'{energy}.{GlossaryCore.techno_list}' in self.get_data_in():
                            techno_list = self.get_sosdisc_inputs(
                                f'{energy}.{GlossaryCore.techno_list}')
                            if techno_list is not None:
                                energy_techno_dict[energy] = {'namespace': 'ns_energy',
                                                              'value': techno_list}
        if GlossaryCore.ccs_list in self.get_data_in():
            ccs_list = self.get_sosdisc_inputs(GlossaryCore.ccs_list)
            if ccs_list is not None:
                for ccs in ccs_list:
                    dynamic_inputs[f'{ccs}.{GlossaryCore.techno_list}'] = {'type': 'list',
                                                                  'subtype_descriptor': {'list': 'string'},
                                                                  'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                  'namespace': GlossaryCore.NS_CCS,
                                                                  'structuring': True,
                                                                  'possible_values': EnergyMix.stream_class_dict[
                                                                      ccs].default_techno_list}

                    if f'{ccs}.{GlossaryCore.techno_list}' in self.get_data_in():
                        techno_list = self.get_sosdisc_inputs(
                            f'{ccs}.{GlossaryCore.techno_list}')
                        if techno_list is not None:
                            energy_techno_dict[ccs] = {'namespace': GlossaryCore.NS_CCS,
                                                       'value': techno_list}

        if 'agri_capital_techno_list' in self.get_data_in():
            agriculture_techno_list = self.get_sosdisc_inputs(
                'agri_capital_techno_list')
            if agriculture_techno_list is not None:
                energy_techno_dict[GlossaryCore.SectorAgriculture] = {'namespace': 'ns_forest',
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
                                                                            'unit': 'G$',
                                                                            "dynamic_dataframe_columns": True,
                                                                             }
            dynamic_inputs[f'{non_use_capital_tuple[0]}techno_capital'] = {'type': 'dataframe',
                                                                           'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                                           'namespace': non_use_capital_tuple[1],
                                                                           'unit': 'G$',
                                                                           "dynamic_dataframe_columns": True,
                                                                           }

        self.add_inputs(dynamic_inputs)

    def init_execution(self):

        inp_dict = self.get_sosdisc_inputs()
        self.model = NonUseCapitalObjective(inp_dict)

    def run(self):
        # get inputs

        inp_dict = self.get_sosdisc_inputs()

        self.model.compute(inp_dict)

        dict_values = {'non_use_capital_df': self.model.non_use_capital_df,
                       'techno_capital_df': self.model.techno_capital_df,
                       GlossaryCore.EnergyCapitalDfValue: self.model.get_energy_capital_trillion_dollars(),
                       'non_use_capital_objective': self.model.non_use_capital_objective,
                       'non_use_capital_cons': self.model.non_use_capital_cons,
                       'forest_lost_capital_cons': self.model.forest_lost_capital_cons}

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        non_use_capital_objective
        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(inputs_dict[GlossaryCore.YearStart],
                          inputs_dict[GlossaryCore.YearEnd] + 1)
        non_use_capital_obj_ref = inputs_dict['non_use_capital_obj_ref']
        non_use_capital_cons_ref = inputs_dict['non_use_capital_cons_ref']
        outputs_dict = self.get_sosdisc_outputs()
        non_use_capital_df = outputs_dict['non_use_capital_df']
        input_nonusecapital_list = [key for key in inputs_dict.keys() if key.endswith('non_use_capital')]
        delta_years = len(non_use_capital_df[GlossaryCore.Years].values)

        for non_use_capital in input_nonusecapital_list:
            column_name = [col for col in inputs_dict[non_use_capital].columns if col != GlossaryCore.Years][0]
            self.set_partial_derivative_for_other_types(
                ('non_use_capital_objective',),
                (non_use_capital, column_name),
                np.ones(len(years))  / non_use_capital_obj_ref / delta_years)
            self.set_partial_derivative_for_other_types(
                ('non_use_capital_cons',),
                (non_use_capital, column_name),
                - np.ones(len(years)) / non_use_capital_cons_ref / delta_years)
        input_capital_list = [key for key in inputs_dict.keys() if key.endswith(GlossaryEnergy.TechnoCapitalValue)]

        for capital in input_capital_list:
            column_name = [col for col in inputs_dict[capital].columns if col != GlossaryCore.Years][0]
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EnergyCapitalDfValue, GlossaryCore.Capital),
                (capital, column_name),
                np.identity(len(years)) / 1.e3)

        forest_lost_capital_cons_ref = inputs_dict['forest_lost_capital_cons_ref']
        self.set_partial_derivative_for_other_types(
            ('forest_lost_capital_cons',),
            ('forest_lost_capital', 'reforestation'),
            - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)
        self.set_partial_derivative_for_other_types(
            ('forest_lost_capital_cons',),
            ('forest_lost_capital', 'managed_wood'),
            - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)
        self.set_partial_derivative_for_other_types(
            ('forest_lost_capital_cons',),
            ('forest_lost_capital', 'deforestation'),
            - np.ones(len(years)) / forest_lost_capital_cons_ref / delta_years)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Non Use Capitals', 'Energy Mix Total Capital',
                      'Forest Management Lost Capital']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Non Use Capitals' in chart_list:

            non_use_capital_df = self.get_sosdisc_outputs('non_use_capital_df')

            years = list(non_use_capital_df[GlossaryCore.Years].values)

            chart_name = 'Non-use Capital per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'non_use Capital [G$]',
                                                 chart_name=chart_name, stacked_bar=True)
            for industry in non_use_capital_df.columns:
                if industry not in [GlossaryCore.Years, 'Sum of non use capital'] and not (
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

            years = list(techno_capital_df[GlossaryCore.Years].values)

            chart_name = 'Energy Mix total capital vs non-use capital per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Capital [G$]',
                                                 chart_name=chart_name)

            new_series = InstanciatedSeries(
                years, techno_capital_df['Sum of techno capital'].values.tolist(), 'Energy Mix Capital', 'lines')

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, non_use_capital_df['Sum of non use capital'].values.tolist(), 'Non-use Capital', 'bar')

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Forest Management Lost Capital' in chart_list:
            forest_lost_capital = self.get_sosdisc_inputs(
                'forest_lost_capital')

            years = list(forest_lost_capital[GlossaryCore.Years].values)

            chart_name = 'Forest Management Lost Capital'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Capital [G$]',
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
        if energy == GlossaryCore.SectorAgriculture:
            full_techno_list.extend(
                [('', techno_dict['namespace']) for techno in techno_dict['value']])
        else:
            full_techno_list.extend(
                [(f'{energy}.{techno}.', techno_dict['namespace']) for techno in techno_dict['value']])

    return full_techno_list
