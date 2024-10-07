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

import numpy as np
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.utility_model import UtilityModel
from climateeconomics.glossarycore import GlossaryCore


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
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        'strech_scurve': {'type': 'float', 'default': 1.7},
        'shift_scurve': {'type': 'float', 'default': -0.2},
        'init_rate_time_pref': {'type': 'float', 'default': 0., 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.EnergyMeanPriceValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_ENERGY_MIX, 'unit': '$/MWh',
                              'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False), GlossaryCore.EnergyPriceValue: ('float', None, True)}},
        'initial_raw_energy_price': {'type': 'float', 'unit': '$/MWh', 'default': 110, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        'init_discounted_utility': {'type': 'float', 'unit': '-', 'default': 3400, 'user_level': 2},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.UtilityDfValue: GlossaryCore.UtilityDf,
        GlossaryCore.QuantityObjectiveValue: GlossaryCore.QuantityObjective,
        GlossaryCore.LastYearUtilityObjectiveValue: GlossaryCore.LastYearUtilityObjective,
        GlossaryCore.DecreasingGdpIncrementsObjectiveValue: GlossaryCore.DecreasingGdpIncrementsObjective,
        GlossaryCore.NetGdpGrowthRateObjectiveValue: GlossaryCore.NetGdpGrowthRateObjective,
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.utility_m = UtilityModel(inp_dict)

    def run(self):
        """run"""
        inp_dict = self.get_sosdisc_inputs()
        economics_df = inp_dict[GlossaryCore.EconomicsDfValue]
        energy_mean_price = inp_dict[GlossaryCore.EnergyMeanPriceValue]
        population_df = inp_dict[GlossaryCore.PopulationDfValue]

        utility_df = self.utility_m.compute(economics_df, energy_mean_price, population_df)

        dict_values = {
            GlossaryCore.UtilityDfValue: utility_df[GlossaryCore.UtilityDf['dataframe_descriptor'].keys()],
            GlossaryCore.QuantityObjectiveValue: self.utility_m.discounted_utility_quantity_objective,
            GlossaryCore.LastYearUtilityObjectiveValue: self.utility_m.last_year_utility_objective,
            GlossaryCore.DecreasingGdpIncrementsObjectiveValue: self.utility_m.decreasing_gpd_obj,
            GlossaryCore.NetGdpGrowthRateObjectiveValue: self.utility_m.net_gdp_growth_rate_obj
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

        d_utility_denergy_price, d_utility_dpcc, \
        d_discounted_utility_quantity_denergy_price, d_discounted_utility_quantity_dpcc, \
        d_pop_discounted_utility_quantity_denergy_price, d_pop_discounted_utility_quantity_dpcc, d_pop_discounted_utility_quantity_dpop, \
        d_utility_obj_d_energy_price, d_utility_obj_dpcc, d_utility_obj_dpop,\
        d_ly_utility_obj_d_energy_price, d_ly_utility_obj_dpcc, d_ly_utility_obj_dpop= self.utility_m.d_utility_quantity()

        d_decreasing_obj_d_economic = self.utility_m.d_decreasing_gdp_obj()

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaUtilityQuantity),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_utility_denergy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaUtilityQuantity),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_utility_dpcc)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtilityQuantityPerCapita),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_discounted_utility_quantity_denergy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtilityQuantityPerCapita),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_discounted_utility_quantity_dpcc)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedQuantityUtilityPopulation),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_pop_discounted_utility_quantity_denergy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedQuantityUtilityPopulation),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_pop_discounted_utility_quantity_dpcc)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedQuantityUtilityPopulation),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_pop_discounted_utility_quantity_dpop)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.QuantityObjectiveValue,),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_utility_obj_d_energy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.QuantityObjectiveValue,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_utility_obj_dpcc)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.QuantityObjectiveValue,),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_utility_obj_dpop)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearUtilityObjectiveValue,),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_ly_utility_obj_d_energy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearUtilityObjectiveValue,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_ly_utility_obj_dpcc)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearUtilityObjectiveValue,),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_ly_utility_obj_dpop)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.DecreasingGdpIncrementsObjectiveValue,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            d_decreasing_obj_d_economic)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NetGdpGrowthRateObjectiveValue,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            self.utility_m.d_net_gdp_growth_rate_obj())


    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.QuantityObjectiveValue]
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

        utility_df = self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue)
        economics_df = self.get_sosdisc_inputs(GlossaryCore.EconomicsDfValue)
        population = self.get_sosdisc_inputs(GlossaryCore.PopulationDfValue)[GlossaryCore.PopulationValue].values
        energy_price = self.get_sosdisc_inputs(GlossaryCore.EnergyMeanPriceValue)[GlossaryCore.EnergyPriceValue].values
        years = list(utility_df[GlossaryCore.Years].values)
        self.utility_m = None
        self.utility_m = UtilityModel(self.get_sosdisc_inputs())

        if GlossaryCore.QuantityObjectiveValue in chart_list:
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Utility gain',
                                                 chart_name='Quantity utility')

            values = utility_df[GlossaryCore.PerCapitaUtilityQuantity].values
            new_series = InstanciatedSeries(
                years, list(values), 'Utility gain', 'lines', True)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.QuantityObjectiveValue in chart_list:
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, f'Variation since {years[0]}[%]',
                                                 chart_name=f'Population utility composants variation since {years[0]}')

            population_ratio = (population / population[0] - 1) * 100
            energy_price_ratio = (energy_price / energy_price[0] - 1) * 100
            new_series = InstanciatedSeries(
                years, list(energy_price_ratio), 'Energy price', 'lines', True)
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, list(population_ratio), 'Population', 'lines', True)
            new_chart.series.append(new_series)
            pcc = economics_df[GlossaryCore.PerCapitaConsumption].values
            pcc_var = (pcc / pcc[0] - 1) * 100
            new_series = InstanciatedSeries(
                years, list(pcc_var), 'Per capita consumption', 'lines', True)
            new_chart.series.append(new_series)

            quantity_consumed = pcc / energy_price
            quantity_consumed_var = (quantity_consumed / quantity_consumed[0] - 1) * 100
            new_series = InstanciatedSeries(
                years, list(quantity_consumed_var), "Quantity of 'things' consumed per capita", 'bar', True)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.QuantityObjectiveValue in chart_list:

            n = 200
            ratios = np.linspace(-0.2, 4, n)

            new_chart = TwoAxesInstanciatedChart(f'Variation of quantity of things consumed per capita since {years[0]} [%]', 'Utility gain per capita', chart_name='Model visualisation : Quantity utility per capita function')
            new_series = InstanciatedSeries(list((ratios -1)*100), list(self.utility_m.s_curve_function(ratios)), 'welfare quantity', 'lines', True)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)
            #new_chart.to_plotly().show()

        return instanciated_charts

