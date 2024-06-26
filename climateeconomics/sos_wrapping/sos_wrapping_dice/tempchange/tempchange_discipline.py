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

import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_dice.tempchange_model import TempChange
from climateeconomics.glossarycore import GlossaryCore


class TempChangeDiscipline(SoSWrapp):
    "Temperature evolution"

    # ontology information
    _ontology_data = {
        "label": "Temperature Change DICE Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fas fa-thermometer-three-quarters fa-fw",
        "version": "",
    }
    DESC_IN = {
        GlossaryCore.YearStart: {"type": "int", "visibility": "Shared", "namespace": "ns_dice"},
        GlossaryCore.YearEnd: {"type": "int", "visibility": "Shared", "namespace": "ns_dice"},
        GlossaryCore.TimeStep: {"type": "int", "visibility": "Shared", "namespace": "ns_dice"},
        "init_temp_ocean": {"type": "float", "default": 0.00687},
        "init_temp_atmo": {"type": "float", "default": 0.85},
        "eq_temp_impact": {"type": "float", "default": 3.1},
        "init_forcing_nonco": {"type": "float", "default": 0.5},
        "hundred_forcing_nonco": {"type": "float", "default": 1},
        "climate_upper": {"type": "float", "default": 0.1005},
        "transfer_upper": {"type": "float", "default": 0.088},
        "transfer_lower": {"type": "float", "default": 0.025},
        "forcing_eq_co2": {"type": "float", "default": 3.6813},
        "lo_tocean": {"type": "float", "default": -1},
        "up_tatmo": {"type": "float", "default": 12},
        "up_tocean": {"type": "float", "default": 20},
        GlossaryCore.CarbonCycleDfValue: {
            "type": "dataframe",
            "visibility": "Shared",
            "namespace": "ns_scenario",
        },
    }

    DESC_OUT = {GlossaryCore.TemperatureDfValue: GlossaryCore.set_namespace(GlossaryCore.TemperatureDf, "ns_scenario")}

    _maturity = "Research"

    def run(self):
        """pyworld3 execution"""
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        #         carboncycle_df = in_dict.pop(GlossaryCore.CarbonCycleDfValue)

        # pyworld3 execution
        model = TempChange()
        temperature_df = model.compute(in_dict)

        # store output data
        out_dict = {GlossaryCore.TemperatureDfValue: temperature_df}
        self.store_sos_outputs_values(out_dict)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ["temperature evolution"]
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

        if "temperature evolution" in chart_list:

            to_plot = [GlossaryCore.TempAtmo, GlossaryCore.TempOcean]
            temperature_df = self.get_sosdisc_outputs(GlossaryCore.TemperatureDfValue)
            temperature_df = resize_df(temperature_df)

            legend = {GlossaryCore.TempAtmo: "atmosphere temperature", GlossaryCore.TempOcean: "ocean temperature"}

            years = list(temperature_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_value = 0
            min_value = 0

            for key in to_plot:
                max_value = max(temperature_df[key].values.max(), max_value)
                min_value = min(temperature_df[key].values.min(), min_value)

            chart_name = "temperature evolution over the years"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "temperature evolution (degrees Celsius above preindustrial)",
                [year_start - 5, year_end + 5],
                [min_value * 0.9, max_value * 1.1],
                chart_name,
            )

            for key in to_plot:
                visible_line = True

                ordonate_data = list(temperature_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, legend[key], "lines", visible_line)

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
    length = len(array)
    new_index = index[0:length]
    return new_index
