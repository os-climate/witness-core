"""
Copyright 2022 Airbus SAS
Modifications on 2023/08/07-2023/11/03 Copyright 2023 Capgemini

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

import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_dice.macroeconomics_model import MacroEconomics
from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsDiscipline(SoSWrapp):
    "Macroeconomics discipline for DICE"

    # ontology information
    _ontology_data = {
        "label": "MacroEconomics DICE Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fas fa-industry fa-fw",
        "version": "",
    }
    _maturity = "Research"
    DESC_IN = {
        GlossaryCore.DamageDfValue: {"type": "dataframe", "visibility": "Shared", "namespace": "ns_scenario"},
        GlossaryCore.YearStart: {"type": "int", "visibility": "Shared", "unit": "year", "namespace": "ns_dice"},
        GlossaryCore.YearEnd: {"type": "int", "visibility": "Shared", "unit": "year", "namespace": "ns_dice"},
        GlossaryCore.TimeStep: {"type": "int", "visibility": "Shared", "unit": "year", "namespace": "ns_dice"},
        "productivity_start": {"type": "float", "default": 5.115},
        GlossaryCore.InitialGrossOutput["var_name"]: {
            "type": "float",
            "visibility": "Shared",
            "namespace": "ns_dice",
            "unit": "trillions $",
        },
        "capital_start": {"type": "float", "unit": "trillions $", "default": 223},
        "pop_start": {"type": "float", "unit": "millions", "default": 7403},
        "output_elasticity": {"type": "float", "default": 0.3},
        "popasym": {"type": "float", "unit": "millions of people", "default": 11500},
        "population_growth": {"type": "float", "default": 0.134},
        "productivity_gr_start": {"type": "float", "default": 0.076},
        "decline_rate_tfp": {"type": "float", "default": 0.005},
        "depreciation_capital": {"type": "float", "default": 0.1},
        "init_rate_time_pref": {"type": "float", "visibility": "Shared", "namespace": "ns_dice", "default": 0.015},
        "conso_elasticity": {"type": "float", "visibility": "Shared", "namespace": "ns_dice", "default": 1.45},
        "lo_capital": {"type": "float", "unit": "trillions $", "default": 1},
        "lo_conso": {"type": "float", "unit": "trillions $", "default": 2},
        "lo_per_capita_conso": {"type": "float", "unit": "trillions $", "default": 0.01},
        GlossaryCore.DamageToProductivity: {"type": "bool", "visibility": "Shared", "namespace": "ns_dice"},
        GlossaryCore.FractionDamageToProductivityValue: {
            "type": "float",
            "visibility": "Shared",
            "namespace": "ns_dice",
        },
        "saving_rate": {"type": "float", "unit": "%", "default": 0.2},
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDfValue: {"type": "dataframe", "visibility": "Shared", "namespace": "ns_scenario"}
    }

    def run(self):
        # Get inputs
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        damage_df = param.pop(GlossaryCore.DamageDfValue)

        damage_inputs = {
            GlossaryCore.DamageFractionOutput: damage_df[GlossaryCore.DamageFractionOutput],
            "abatecost": damage_df["abatecost"],
        }

        # Model execution
        macro_model = MacroEconomics(param, damage_inputs)
        economics_df = macro_model.compute(damage_inputs)

        # Store output data
        dict_values = {GlossaryCore.EconomicsDfValue: economics_df}
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [
            "economic output",
            GlossaryCore.PopulationValue,
            GlossaryCore.Productivity,
            GlossaryCore.Consumption,
        ]
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == "charts":
                    chart_list = chart_filter.selected_values

        economics_df = self.get_sosdisc_outputs(GlossaryCore.EconomicsDfValue)
        economics_df = resize_df(economics_df)

        if "economic output" in chart_list:

            to_plot = [GlossaryCore.GrossOutput, GlossaryCore.OutputNetOfDamage]

            legend = {
                GlossaryCore.GrossOutput: "world gross output",
                GlossaryCore.OutputNetOfDamage: "world output net of damage",
            }

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = 0
            min_value = 0

            for key in to_plot:
                max_value = max(economics_df[key].values.max(), max_value)
                min_value = min(economics_df[key].values.min(), min_value)

            chart_name = "Economic output (Power Purchase Parity)"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "world output [trillion $2020]",
                [year_start - 5, year_end + 5],
                [min_value * 0.9, max_value * 1.1],
                chart_name,
            )

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, legend[key], "lines", visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.PopulationValue in chart_list:

            to_plot = [GlossaryCore.PopulationValue]

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = 0
            min_value = 0

            for key in to_plot:
                max_value = max(economics_df[key].values.max(), max_value)
                min_value = min(economics_df[key].values.min(), min_value)

            chart_name = "population evolution over the years"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                " population (million)",
                [year_start - 5, year_end + 5],
                [min_value * 0.9, max_value * 1.1],
                chart_name,
            )

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, key, "lines", visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.PopulationValue in chart_list:

            to_plot = [GlossaryCore.Productivity]

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = 0
            min_value = 0

            for key in to_plot:
                max_value = max(economics_df[key].values.max(), max_value)
                min_value = min(economics_df[key].values.min(), min_value)

            chart_name = "Total Factor Productivity"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "global productivity",
                [year_start - 5, year_end + 5],
                [min_value * 0.9, max_value * 1.1],
                chart_name,
            )

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, key, "lines", visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Consumption in chart_list:

            to_plot = [GlossaryCore.Consumption]

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = 0
            min_value = 0

            for key in to_plot:
                max_value = max(economics_df[key].values.max(), max_value)
                min_value = min(economics_df[key].values.min(), min_value)

            chart_name = "global consumption over the years"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                " global consumption (trill $)",
                [year_start - 5, year_end + 5],
                [min_value * 0.9, max_value * 1.1],
                chart_name,
            )

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, key, "lines", visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        return instanciated_charts


def resize_df(df):

    index = df.index
    i = len(index) - 1
    key = df.keys()
    to_check = df.loc[index[i], key[0]]

    while to_check == 0:
        i = i - 1
        to_check = df.loc[index[i], key[0]]

    size_diff = len(index) - i
    new_df = pd.DataFrame()

    if size_diff == 0:
        new_df = df
    else:
        for element in key:
            new_df[element] = df[element][0 : i + 1]
            new_df.index = index[0 : i + 1]

    return new_df


def resize_array(array):

    i = len(array) - 1
    to_check = array[i]

    while to_check == 0:
        i = i - 1
        to_check = to_check = array[i]

    size_diff = len(array) - i
    new_array = array[0:i]

    return new_array


def resize_index(index, array):

    l = len(array)
    new_index = index[0:l]
    return new_index
