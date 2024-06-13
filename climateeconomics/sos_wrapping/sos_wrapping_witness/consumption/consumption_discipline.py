'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
from copy import deepcopy

import numpy as np

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.consumption_model import ConsumptionModel
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class ConsumptionDiscipline(ClimateEcoDiscipline):
    "ConsumptionModel discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Consumption WITNESS Model',
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
    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'alpha': {'type': 'float', 'range': [0., 1.], 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 1},
        'gamma': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 1},
        'welfare_obj_option': {'type': 'string', 'default': GlossaryCore.Welfare, 'possible_values': ['last_utility', GlossaryCore.Welfare], 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.EnergyMeanPriceValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_ENERGY_MIX, 'unit': '$/MWh',
                              'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                       GlossaryCore.EnergyPriceValue: ('float', None, True)}},
        'initial_raw_energy_price': {'type': 'float', 'unit': '$/MWh', 'default': 110, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        'init_discounted_utility': {'type': 'float', 'unit': '-', 'default': 3400, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_REFERENCE, 'user_level': 2},
        'init_period_utility_pc': {'type': 'float', 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        'discounted_utility_ref': {'type': 'float', 'unit': '-', 'default': 1700, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_REFERENCE, 'user_level': 2},
        'lo_conso': {'type': 'float', 'unit': 'T$', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        GlossaryCore.InvestmentDfValue: GlossaryCore.InvestmentDf,
        'residential_energy_conso_ref' : {'type': 'float', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_REFERENCE, 'unit': 'MWh', 'default': 24.3816},
        GlossaryCore.ResidentialEnergyConsumptionDfValue : GlossaryCore.ResidentialEnergyConsumptionDf,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }
    DESC_OUT = {
        'utility_detail_df': {'type': 'dataframe', 'unit': '-'},
        GlossaryCore.UtilityDfValue: GlossaryCore.UtilityDf,
        GlossaryCore.WelfareObjective: {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.NegativeWelfareObjective: {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'min_utility_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.conso_m = ConsumptionModel(inp_dict)

    def run(self):
        # get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        

        # compute utility
        economics_df = inp_dict.pop(GlossaryCore.EconomicsDfValue)
        energy_mean_price = inp_dict[GlossaryCore.EnergyMeanPriceValue]
        population_df = inp_dict.pop(GlossaryCore.PopulationDfValue)
        investment_df = inp_dict.pop(
            GlossaryCore.InvestmentDfValue)
        residential_energy = inp_dict.pop(
            GlossaryCore.ResidentialEnergyConsumptionDfValue)

        utility_inputs = {GlossaryCore.EconomicsDfValue: economics_df[[GlossaryCore.Years, GlossaryCore.OutputNetOfDamage]],
                          GlossaryCore.PopulationDfValue: population_df[[GlossaryCore.Years, GlossaryCore.PopulationValue]],
                          GlossaryCore.EnergyMeanPriceValue: energy_mean_price,
                          GlossaryCore.InvestmentDfValue: investment_df,
                          GlossaryCore.ResidentialEnergyConsumptionDfValue: residential_energy
                        }
        utility_df = self.conso_m.compute(utility_inputs)

        # Compute objective function
        obj_option = inp_dict['welfare_obj_option']
        if obj_option in ['last_utility', GlossaryCore.Welfare]:
            welfare_objective = self.conso_m.compute_welfare_objective()
        else:
            raise ValueError('obj_option = ' + str(obj_option) + ' not in ' +
                             str(self.DESC_IN['welfare_obj_option']['possible_values']))
        min_utility_objective = self.conso_m.compute_min_utility_objective()
        negative_welfare_objective = self.conso_m.compute_negative_welfare_objective()
        # store output data
        dict_values = {'utility_detail_df': utility_df,
                       GlossaryCore.UtilityDfValue: utility_df[[GlossaryCore.Years, GlossaryCore.UtilityDiscountRate, GlossaryCore.PeriodUtilityPerCapita, GlossaryCore.DiscountedUtility, GlossaryCore.Welfare, GlossaryCore.PerCapitaConsumption]],
                       GlossaryCore.WelfareObjective: welfare_objective,
                       'min_utility_objective': min_utility_objective,
                       GlossaryCore.NegativeWelfareObjective : negative_welfare_objective
                       }
        

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        utility_df
          - GlossaryCore.Consumption:
                - economics_df : GlossaryCore.OutputNetOfDamage
                - energy_mean_price : GlossaryCore.EnergyPriceValue
                - residential_energy : 'residential_energy'
                - investment : 'investment'
          - GlossaryCore.PerCapitaConsumption:
                - economics_df : GlossaryCore.OutputNetOfDamage
                - energy_mean_price : GlossaryCore.EnergyPriceValue
                - residential_energy : 'residential_energy'
                - GlossaryCore.InvestmentDfValue : GlossaryCore.InvestmentsValue
          - GlossaryCore.PeriodUtilityPerCapita:
                - economics_df : GlossaryCore.OutputNetOfDamage
                - energy_mean_price : GlossaryCore.EnergyPriceValue
                - residential_energy : 'residential_energy'
                - GlossaryCore.InvestmentDfValue : GlossaryCore.InvestmentsValue
          - GlossaryCore.DiscountedUtility,
                - economics_df : GlossaryCore.OutputNetOfDamage
                - energy_mean_price : GlossaryCore.EnergyPriceValue
                - residential_energy : 'residential_energy'
                - GlossaryCore.InvestmentDfValue : GlossaryCore.InvestmentsValue
          - GlossaryCore.Welfare
                - economics_df : GlossaryCore.OutputNetOfDamage
                - energy_mean_price : GlossaryCore.EnergyPriceValue
                - residential_energy : 'residential_energy'
                - GlossaryCore.InvestmentDfValue : GlossaryCore.InvestmentsValue
        """
        inputs_dict = self.get_sosdisc_inputs()
        obj_option = inputs_dict['welfare_obj_option']
        d_pc_consumption_d_output_net_of_d, d_pc_consumption_d_investment, d_pc_consumption_d_population, \
        d_period_utility_pc_d_output_net_of_d, d_period_utility_pc_d_investment, d_period_utility_d_population,\
        d_discounted_utility_d_output_net_of_d, d_discounted_utility_d_investment, d_discounted_utility_d_population, \
         d_welfare_d_output_net_of_d, d_welfare_d_investment, d_welfare_d_population = self.conso_m.compute_gradient()
        
        # d_pc_consumption_d_output_net_of_d, d_pc_consumption_d__investment, \
        # d_period_utility_d_pc_consumption, d_discounted_utility_d_pc_consumption, d_discounted_utility_d_population,\
        #     d_welfare_d_pc_consumption, d_welfare_d_population = self.conso_m.compute_gradient()
            
        d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, \
            d_welfare_d_energy_price = self.conso_m.compute_gradient_energy_mean_price()

        d_period_utility_d_residential_energy, d_discounted_utility_d_residential_energy, \
            d_welfare_d_residential_energy = self.conso_m.compute_gradient_residential_energy()
        d_obj_d_welfare, d_obj_d_period_utility_pc = self.conso_m.compute_gradient_objective()

        # fill jacobians
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),  d_pc_consumption_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),  d_pc_consumption_d_investment)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_pc_consumption_d_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),  d_period_utility_pc_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),  d_period_utility_pc_d_investment)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_period_utility_d_energy_price)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue),  d_period_utility_d_residential_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_period_utility_d_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),  d_discounted_utility_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),  d_discounted_utility_d_investment)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_discounted_utility_d_energy_price)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue),  d_discounted_utility_d_residential_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_discounted_utility_d_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),  d_welfare_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),  d_welfare_d_investment)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_welfare_d_population)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),  d_welfare_d_energy_price)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.Welfare), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue),  d_welfare_d_residential_energy)

        if obj_option == 'last_utility':
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), d_obj_d_period_utility_pc.dot(d_period_utility_pc_d_output_net_of_d))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue), d_obj_d_period_utility_pc.dot(d_period_utility_pc_d_investment))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), d_obj_d_period_utility_pc.dot(d_period_utility_d_energy_price))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue), d_obj_d_period_utility_pc.dot(d_period_utility_d_residential_energy))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  d_obj_d_period_utility_pc.dot(d_period_utility_d_population))

        elif obj_option == GlossaryCore.Welfare:
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), np.dot(d_obj_d_welfare, d_welfare_d_output_net_of_d))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue), np.dot(d_obj_d_welfare, d_welfare_d_investment))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), np.dot(d_obj_d_welfare, d_welfare_d_energy_price))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue), np.dot(d_obj_d_welfare, d_welfare_d_residential_energy))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.WelfareObjective,), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  np.dot(d_obj_d_welfare, d_welfare_d_population))

        else:
            pass

        d_neg_obj_d_welfare, x = self.conso_m.compute_gradient_negative_objective()

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_output_net_of_d))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_investment))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_energy_price))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_residential_energy))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue), np.dot(d_neg_obj_d_welfare, d_welfare_d_population))


        d_obj_d_discounted_utility, d_obj_d_period_utility_pc = self.conso_m.compute_gradient_min_utility_objective()

        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_output_net_of_d))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_investment))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_energy_price))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.ResidentialEnergyConsumptionDfValue, GlossaryCore.TotalProductionValue), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_residential_energy))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),  np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_population))
    
    
    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Consumption', 'Consumption PC', 'Utility', 'Utility of pc consumption',
                      'Energy effects on utility', 'Energy ratios']
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

        if 'Energy ratios' in chart_list:

            energy_mean_price = self.get_sosdisc_inputs(GlossaryCore.EnergyMeanPriceValue)[
                GlossaryCore.EnergyPriceValue].values

            energy_price_ref = self.get_sosdisc_inputs(
                'initial_raw_energy_price')
            
            
            residential_energy = self.get_sosdisc_inputs(GlossaryCore.ResidentialEnergyConsumptionDfValue)[
                GlossaryCore.TotalProductionValue].values

            residential_energy_conso_ref = self.get_sosdisc_inputs(
                'residential_energy_conso_ref')

            residential_energy_ratio = residential_energy / residential_energy_conso_ref

            energy_price_ratio = energy_price_ref / energy_mean_price

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy ratios'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '',
                                                 chart_name=chart_name)

            visible_line = True

            new_series = InstanciatedSeries(
                years, residential_energy_ratio.tolist(), 'Residential energy availability ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, energy_price_ratio.tolist(), 'Energy price ratio', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Energy effects on utility' in chart_list:

            utility_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue))

            discounted_utility_final = utility_df[GlossaryCore.DiscountedUtility].values

            energy_mean_price = self.get_sosdisc_inputs(GlossaryCore.EnergyMeanPriceValue)[
                GlossaryCore.EnergyPriceValue].values

            energy_price_ref = self.get_sosdisc_inputs(
                'initial_raw_energy_price')
            
            
            residential_energy = self.get_sosdisc_inputs(GlossaryCore.ResidentialEnergyConsumptionDfValue)[
                GlossaryCore.TotalProductionValue].values

            residential_energy_conso_ref = self.get_sosdisc_inputs(
                'residential_energy_conso_ref')

            residential_energy_ratio = residential_energy / residential_energy_conso_ref

            energy_price_ratio = energy_price_ref / energy_mean_price

            discounted_utility_residential_ratio = discounted_utility_final / energy_price_ratio

            discounted_utility_price_ratio = discounted_utility_final / residential_energy_ratio

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy price ratio effect on discounted utility'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Discounted Utility (trill $)',
                                                 chart_name=chart_name)

            visible_line = True

            new_series = InstanciatedSeries(
                years, discounted_utility_residential_ratio.tolist(), 'Discounted Utility without price ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, discounted_utility_price_ratio.tolist(), 'Discounted Utility without residential energy ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, discounted_utility_final.tolist(), 'Discounted Utility', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Consumption' in chart_list:

            to_plot = [GlossaryCore.Consumption]
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_detail_df'))
            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                utility_df[to_plot])

            chart_name = 'Global consumption over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' global consumption [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Consumption PC' in chart_list:

            to_plot = [GlossaryCore.PerCapitaConsumption]
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_detail_df'))
            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                utility_df[to_plot])

            chart_name = 'Per capita consumption over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Per capita consumption [thousand $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
