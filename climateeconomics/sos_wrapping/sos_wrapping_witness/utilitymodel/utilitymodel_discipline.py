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
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.utility_model import UtilityModel
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from copy import deepcopy
import pandas as pd
import numpy as np


class UtilityModelDiscipline(ClimateEcoDiscipline):
    "UtilityModel discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Utility WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-child fa-fw',
        'version': '',
    }
    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: ClimateEcoDiscipline.YEAR_END_DESC_IN,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'alpha': {'type': 'float', 'range': [0., 1.], 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'gamma': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'welfare_obj_option': {'type': 'string', 'default': GlossaryCore.Welfare, 'possible_values': ['last_utility', GlossaryCore.Welfare], 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        GlossaryCore.EconomicsDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-',
                         'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                  GlossaryCore.GrossOutput: ('float', None, False),
                                                  GlossaryCore.PopulationValue: ('float', None, False),
                                                  GlossaryCore.Productivity: ('float', None, False),
                                                  GlossaryCore.ProductivityGrowthRate: ('float', None, False),
                                                  'energy_productivity_gr': ('float', None, False),
                                                  'energy_productivity': ('float', None, False),
                                                  GlossaryCore.Consumption: ('float', None, False),
                                                  GlossaryCore.Capital: ('float', None, False),
                                                  GlossaryCore.InvestmentsValue: ('float', None, False),
                                                  'interest_rate': ('float', None, False),
                                                  GlossaryCore.OutputGrowth: ('float', None, False),
                                                  GlossaryCore.EnergyInvestmentsValue: ('float', None, False),
                                                  GlossaryCore.PerCapitaConsumption: ('float', None, False),
                                                  GlossaryCore.OutputNetOfDamage: ('float', None, False),
                                                  GlossaryCore.NetOutput: ('float', None, False),
                                                  }},
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.EnergyMeanPriceValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_energy_mix', 'unit': '$/MWh',
                              'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False), GlossaryCore.EnergyPriceValue: ('float', None, True)}},
        'initial_raw_energy_price': {'type': 'float', 'unit': '$/MWh', 'default': 110, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_discounted_utility': {'type': 'float', 'unit': '-', 'default': 3400, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
        'init_period_utility_pc': {'type': 'float', 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'discounted_utility_ref': {'type': 'float', 'unit': '-', 'default': 1700, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
    }
    DESC_OUT = {
        GlossaryCore.UtilityDfValue: GlossaryCore.UtilityDf,
        'welfare_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'negative_welfare_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'min_utility_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.utility_m = UtilityModel(inp_dict)

    def run(self):
        # get inputs
        inp_dict = self.get_sosdisc_inputs()

        # compute utility
        economics_df = deepcopy(inp_dict[GlossaryCore.EconomicsDfValue])
        energy_mean_price = deepcopy(inp_dict[GlossaryCore.EnergyMeanPriceValue])
        population_df = deepcopy(inp_dict[GlossaryCore.PopulationDfValue])

        utility_df = self.utility_m.compute(
            economics_df, energy_mean_price, population_df)

        # Compute objective function
        obj_option = inp_dict['welfare_obj_option']
        if obj_option in ['last_utility', GlossaryCore.Welfare]:
            welfare_objective = self.utility_m.compute_welfare_objective()
        else:
            raise ValueError('obj_option = ' + str(obj_option) + ' not in ' +
                             str(self.DESC_IN['welfare_obj_option']['possible_values']))
        min_utility_objective = self.utility_m.compute_min_utility_objective()
        negative_welfare_objective = self.utility_m.compute_negative_welfare_objective()
        # store output data
        dict_values = {GlossaryCore.UtilityDfValue: utility_df,
                       'welfare_objective': welfare_objective,
                       'min_utility_objective': min_utility_objective,
                       'negative_welfare_objective': negative_welfare_objective
                       }
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        utility_df
          - GlossaryCore.PeriodUtilityPerCapita:
                - economics_df, GlossaryCore.PerCapitaConsumption
                - energy_mean_price : GlossaryCore.EnergyPriceValue
          - GlossaryCore.DiscountedUtility,
                - economics_df, GlossaryCore.PerCapitaConsumption
                - energy_mean_price : GlossaryCore.EnergyPriceValue
          - GlossaryCore.Welfare
                - economics_df, GlossaryCore.PerCapitaConsumption
                - energy_mean_price : GlossaryCore.EnergyPriceValue
        """
        inputs_dict = self.get_sosdisc_inputs()
        obj_option = inputs_dict['welfare_obj_option']

        d_period_utility_d_pc_consumption, d_discounted_utility_d_pc_consumption, d_discounted_utility_d_population,\
            d_welfare_d_pc_consumption, d_welfare_d_population = self.utility_m.compute_gradient()
        d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, \
            d_welfare_d_energy_price = self.utility_m.compute_gradient_energy_mean_price()

        d_obj_d_welfare, d_obj_d_period_utility_pc = self.utility_m.compute_gradient_objective()

        # fill jacobians
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),  d_period_utility_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_period_utility_d_energy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),  d_discounted_utility_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_discounted_utility_d_energy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_discounted_utility_d_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),  d_welfare_d_pc_consumption)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_welfare_d_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_welfare_d_energy_price)

        if obj_option == 'last_utility':
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), d_obj_d_period_utility_pc.dot(d_period_utility_d_pc_consumption))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), d_obj_d_period_utility_pc.dot(d_period_utility_d_energy_price))

        elif obj_option == GlossaryCore.Welfare:

            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  np.dot(d_obj_d_welfare, d_welfare_d_population))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), np.dot(d_obj_d_welfare, d_welfare_d_pc_consumption))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), np.dot(d_obj_d_welfare, d_welfare_d_energy_price))
        else:
            pass

        d_neg_obj_d_welfare, x = self.utility_m.compute_gradient_negative_objective()
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue), np.dot(d_neg_obj_d_welfare, d_welfare_d_population))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_pc_consumption))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_energy_price))

        d_obj_d_discounted_utility, d_obj_d_period_utility_pc = self.utility_m.compute_gradient_min_utility_objective()

        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_population))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_pc_consumption))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_energy_price))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Utility', 'Utility of pc consumption',
                      'Energy price effect on utility']
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

        if 'Utility' in chart_list:

            to_plot = [GlossaryCore.DiscountedUtility]
            utility_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue))

            discounted_utility = utility_df[GlossaryCore.DiscountedUtility]

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(discounted_utility)

            chart_name = 'Utility'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Discounted Utility (trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Utility of pc consumption' in chart_list:

            to_plot = [GlossaryCore.PeriodUtilityPerCapita]
            utility_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue))

            utility = utility_df[GlossaryCore.PeriodUtilityPerCapita]

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(utility)

            chart_name = 'Utility of per capita consumption'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Utility of pc consumption',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Energy price effect on utility' in chart_list:

            utility_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue))

            discounted_utility_final = utility_df[GlossaryCore.DiscountedUtility].values

            energy_mean_price = self.get_sosdisc_inputs(GlossaryCore.EnergyMeanPriceValue)[
                GlossaryCore.EnergyPriceValue].values

            energy_price_ref = self.get_sosdisc_inputs(
                'initial_raw_energy_price')

            energy_price_ratio = energy_price_ref / energy_mean_price

            discounted_utility_before = discounted_utility_final / energy_price_ratio

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy price ratio effect on discounted utility'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Discounted Utility (trill $)',
                                                 chart_name=chart_name)

            visible_line = True

            new_series = InstanciatedSeries(
                years, discounted_utility_before.tolist(), 'Before energy price ratio effect', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, discounted_utility_final.tolist(), 'After energy price ratio effect', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)
        return instanciated_charts
