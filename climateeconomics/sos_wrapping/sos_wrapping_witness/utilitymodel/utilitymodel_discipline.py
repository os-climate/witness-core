"""
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
"""

from copy import deepcopy

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
        "label": "Utility WITNESS Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fas fa-child fa-fw",
        "version": "",
    }
    _maturity = "Research"
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        "conso_elasticity": {
            "type": "float",
            "default": 1.45,
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "user_level": 2,
        },
        "init_rate_time_pref": {
            "type": "float",
            "default": 0.015,
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
        },
        GlossaryCore.EconomicsDfValue: {
            "type": "dataframe",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "unit": "-",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                GlossaryCore.GrossOutput: ("float", None, False),
                GlossaryCore.PopulationValue: ("float", None, False),
                GlossaryCore.Productivity: ("float", None, False),
                GlossaryCore.ProductivityGrowthRate: ("float", None, False),
                "energy_productivity_gr": ("float", None, False),
                "energy_productivity": ("float", None, False),
                GlossaryCore.Consumption: ("float", None, False),
                GlossaryCore.Capital: ("float", None, False),
                GlossaryCore.InvestmentsValue: ("float", None, False),
                "interest_rate": ("float", None, False),
                GlossaryCore.OutputGrowth: ("float", None, False),
                GlossaryCore.EnergyInvestmentsValue: ("float", None, False),
                GlossaryCore.PerCapitaConsumption: ("float", None, False),
                GlossaryCore.OutputNetOfDamage: ("float", None, False),
                GlossaryCore.NetOutput: ("float", None, False),
            },
        },
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.EnergyMeanPriceValue: {
            "type": "dataframe",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_ENERGY_MIX,
            "unit": "$/MWh",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                GlossaryCore.EnergyPriceValue: ("float", None, True),
            },
        },
        "initial_raw_energy_price": {
            "type": "float",
            "unit": "$/MWh",
            "default": 110,
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "user_level": 2,
        },
        "init_discounted_utility": {
            "type": "float",
            "unit": "-",
            "default": 3400,
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_REFERENCE,
            "user_level": 2,
        },
        GlossaryCore.PerCapitaConsumptionUtilityRefName: GlossaryCore.PerCapitaConsumptionUtilityRef,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.UtilityDfValue: GlossaryCore.UtilityDf,
        GlossaryCore.NormalizedWelfare: {
            "type": "array",
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "description": "Sum of discounted utilities divided by number of year divided by initial discounted utility",
        },
        GlossaryCore.WelfareObjective: {
            "type": "array",
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "description": "inverse of normalized welfare",
        },
        GlossaryCore.NegativeWelfareObjective: {
            "type": "array",
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "description": "minus normalized welfare",
        },
        GlossaryCore.LastYearDiscountedUtilityObjective: {
            "type": "array",
            "unit": "-",
            "visibility": "Shared",
            "namespace": GlossaryCore.NS_WITNESS,
            "description": "- discounted utility at year end / discounted utility at year start",
        },
        GlossaryCore.PerCapitaConsumptionUtilityObjectiveName: GlossaryCore.PerCapitaConsumptionUtilityObjective,
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.utility_m = UtilityModel(inp_dict)

    def run(self):
        """run"""
        inp_dict = self.get_sosdisc_inputs()

        economics_df = deepcopy(inp_dict[GlossaryCore.EconomicsDfValue])
        energy_mean_price = deepcopy(inp_dict[GlossaryCore.EnergyMeanPriceValue])
        population_df = deepcopy(inp_dict[GlossaryCore.PopulationDfValue])

        utility_df = self.utility_m.compute(economics_df, energy_mean_price, population_df)

        dict_values = {
            GlossaryCore.UtilityDfValue: utility_df,
            GlossaryCore.NormalizedWelfare: self.utility_m.normalized_welfare,
            GlossaryCore.WelfareObjective: self.utility_m.inverse_welfare_objective,
            GlossaryCore.NegativeWelfareObjective: self.utility_m.negative_welfare_objective,
            GlossaryCore.LastYearDiscountedUtilityObjective: self.utility_m.last_year_utility_objective,
            GlossaryCore.PerCapitaConsumptionUtilityObjectiveName: self.utility_m.per_capita_consumption_objective,
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

        d_pc_consumption_utility_d_per_capita_consumption = (
            self.utility_m.d_pc_consumption_utility_d_per_capita_consumption()
        )
        d_pc_consumption_utility_obj_d_per_capita_consumption = (
            self.utility_m.d_pc_consumption_utility_objective_d_per_capita_consumption()
        )
        d_energy_price_ratio_d_energy_price = self.utility_m.d_energy_price_ratio_d_energy_price()

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PerCapitaConsumptionUtility),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_pc_consumption_utility_d_per_capita_consumption,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.PerCapitaConsumptionUtilityObjectiveName,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_pc_consumption_utility_obj_d_per_capita_consumption,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.EnergyPriceRatio),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_energy_price_ratio_d_energy_price,
        )

        d_utility_d_energy_price = self.utility_m.d_utility_d_energy_price()
        d_utility_d_per_capita_consumption = self.utility_m.d_utility_d_per_capita_consumption()

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_utility_d_energy_price,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.PeriodUtilityPerCapita),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_utility_d_per_capita_consumption,
        )

        d_discounted_utility_d_population = self.utility_m.d_discounted_utility_d_population()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_discounted_utility_d_population,
        )

        d_discounted_utility_d_per_capita_consumption = self.utility_m.d_discounted_utility_d_user_input(
            d_utility_d_per_capita_consumption
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_discounted_utility_d_per_capita_consumption,
        )

        d_discounted_utility_d_energy_price = self.utility_m.d_discounted_utility_d_user_input(d_utility_d_energy_price)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.UtilityDfValue, GlossaryCore.DiscountedUtility),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_discounted_utility_d_energy_price,
        )

        d_negative_welfare_objective_d_energy_price, d_inverse_welfare_objective_d_energy_price = (
            self.utility_m.d_objectives_d_user_input(d_discounted_utility_d_energy_price)
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WelfareObjective,),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_inverse_welfare_objective_d_energy_price,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_negative_welfare_objective_d_energy_price,
        )

        d_negative_welfare_objective_d_pop, d_inverse_welfare_objective_d_pop = (
            self.utility_m.d_objectives_d_user_input(d_discounted_utility_d_population)
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WelfareObjective,),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_inverse_welfare_objective_d_pop,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_negative_welfare_objective_d_pop,
        )

        d_negative_welfare_objective_d_pc_consumption, d_inverse_welfare_objective_d_pc_consumption = (
            self.utility_m.d_objectives_d_user_input(d_discounted_utility_d_per_capita_consumption)
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WelfareObjective,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_inverse_welfare_objective_d_pc_consumption,
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.NegativeWelfareObjective,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_negative_welfare_objective_d_pc_consumption,
        )

        d_last_utility_objective_d_pop = self.utility_m.d_last_utility_objective_d_user_input(
            d_discounted_utility_d_population
        )
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearDiscountedUtilityObjective,),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_last_utility_objective_d_pop,
        )

        d_last_utility_objective_d_energy_price = self.utility_m.d_last_utility_objective_d_user_input(
            d_discounted_utility_d_energy_price
        )
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearDiscountedUtilityObjective,),
            (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
            d_last_utility_objective_d_energy_price,
        )

        d_last_utility_objective_d_pc_consumption = self.utility_m.d_last_utility_objective_d_user_input(
            d_discounted_utility_d_per_capita_consumption
        )
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.LastYearDiscountedUtilityObjective,),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            d_last_utility_objective_d_pc_consumption,
        )

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.DiscountedUtility, "Utility of pc consumption", GlossaryCore.EnergyPriceRatio]
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == "charts":
                    chart_list = chart_filter.selected_values

        utility_df = self.get_sosdisc_outputs(GlossaryCore.UtilityDfValue)
        years = list(utility_df[GlossaryCore.Years].values)

        if GlossaryCore.DiscountedUtility in chart_list:

            chart_name = "Discounted utility"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, "Discounted Utility (trill $)", chart_name=chart_name
            )
            new_series = InstanciatedSeries(
                years, list(utility_df[GlossaryCore.DiscountedUtility]), chart_name, "lines", True
            )

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if "Utility of pc consumption" in chart_list:

            chart_name = "Utility of per capita consumption"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, "Utility of per capita consumption", chart_name=chart_name
            )
            new_series = InstanciatedSeries(
                years, list(utility_df[GlossaryCore.PeriodUtilityPerCapita]), chart_name, "lines", True
            )

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyPriceRatio in chart_list:

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, "Variation [%]", chart_name="Energy price variation since year start"
            )

            values = (1 / utility_df[GlossaryCore.EnergyPriceRatio].values - 1) * 100
            new_series = InstanciatedSeries(years, list(values), "Energy price variation", "lines", True)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)
        return instanciated_charts
