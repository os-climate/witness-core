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
import numpy as np
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class InvestDiscipline(ClimateEcoDiscipline):
    "Macroeconomics discipline for WITNESS"

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Investissement Model',
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
        'energy_investment_macro': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.EnergyInvestmentsValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_ENERGY_MIX},
        'invest_norm': {'type': 'float', 'default': 10.0},
        'formulation': {'type': 'string', 'default': 'objective', 'possile_values': ['objective', 'constraint']},
        'max_difference': {'type': 'float', 'default': 1.0e-1},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        'invest_objective': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'diff_norm': {'type': 'array'}
    }

    def run(self):
        # Get inputs
        inputs = self.get_sosdisc_inputs()

        difference = np.linalg.norm(inputs['energy_investment_macro'][GlossaryCore.EnergyInvestmentsValue].values -
                                    inputs[GlossaryCore.EnergyInvestmentsValue][GlossaryCore.EnergyInvestmentsValue].values) / inputs['invest_norm']

        if inputs['formulation'] == 'objective':
            invest_objective = difference
        elif inputs['formulation'] == 'constraint':
            invest_objective = inputs['max_difference'] - difference
        else:
            raise Exception("formulation type should be either objective or constraint")
        # Store output data
        dict_values = {'invest_objective': pd.DataFrame(
            {'norm': [invest_objective]}),
            'diff_norm': difference}

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradiant of coupling variable to compute
        """
        inputs = self.get_sosdisc_inputs()
        invest_objective = self.get_sosdisc_outputs(
            'invest_objective')['norm'].values[0]
        dinvestment = (inputs['energy_investment_macro'][GlossaryCore.EnergyInvestmentsValue].values -
                       inputs[GlossaryCore.EnergyInvestmentsValue][GlossaryCore.EnergyInvestmentsValue].values) / invest_objective / inputs['invest_norm']**2

        self.set_partial_derivative_for_other_types(
            ('invest_objective', 'norm'), ('energy_investment_macro', GlossaryCore.EnergyInvestmentsValue), dinvestment)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            ('invest_objective', 'norm'), (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), -dinvestment)  # Invest from T$ to G$

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Difference of investments']
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

        if 'Difference of investments' in chart_list:

            energy_investment_macro = self.get_sosdisc_inputs(
                'energy_investment_macro')

            energy_investment = self.get_sosdisc_inputs(GlossaryCore.EnergyInvestmentsValue)

            years = list(energy_investment_macro[GlossaryCore.Years].values)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy investments between macroeconomy output and energy input'

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, 'Investments', chart_name=chart_name)

            energy_investment_series = InstanciatedSeries(
                years, list(energy_investment[GlossaryCore.EnergyInvestmentsValue].values), 'energy investment (energy)', 'lines')

            new_chart.series.append(energy_investment_series)

            energy_investment_macro_series = InstanciatedSeries(
                years, list(energy_investment_macro[GlossaryCore.EnergyInvestmentsValue].values), 'energy_investment (macroeconomy)', 'lines')

            new_chart.series.append(energy_investment_macro_series)
            instanciated_charts.append(new_chart)

            norm = self.get_sosdisc_outputs('diff_norm')
            chart_name = 'Differences between energy investments'

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, 'Differences of investments', chart_name=chart_name)

            energy_investment_series = InstanciatedSeries(
                years, list(energy_investment_macro[GlossaryCore.EnergyInvestmentsValue].values - energy_investment[GlossaryCore.EnergyInvestmentsValue].values), '', 'lines')

            new_chart.series.append(energy_investment_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
