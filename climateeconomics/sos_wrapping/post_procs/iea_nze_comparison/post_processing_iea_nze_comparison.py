"""
Copyright 2024 Capgemini

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

import logging
import os.path

import pandas as pd

from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

DATA_FILE_LIST = [
    "IEA_NZE_CO2_100.csv",
    "IEA_NZE_CO2_tax.csv",
    "IEA_NZE_crop_detailed_mix_production.csv",
    "IEA_NZE_electricity_Technologies_Mix_prices.csv",
    "IEA_NZE_EnergyMix.biogas.Energy Production Quantity Detailed.csv",
    "IEA_NZE_EnergyMix.electricity.Hydropower.techno_production.csv",
    "IEA_NZE_EnergyMix.electricity.Nuclear.techno_production.csv",
    "IEA_NZE_EnergyMix.electricity.SolarPv.techno_production.csv",
    "IEA_NZE_EnergyMix.electricity.WindXXshore.techno_production.csv",
    "IEA_NZE_EnergyMix.energy_production_brut_detailed.csv",
    "IEA_NZE_EnergyMix_solid_fuel_CoalExtraction_techno_production.csv",
    "IEA_NZE_forest_Technology_Production_Quantities.csv",
    "IEA_NZE_output_net_of_d.csv",
    "IEA_NZE_population.csv",
    "IEA_NZE_temp_atmo.csv",
]


def create_df_from_csv(filename: str, data_dir=DATA_DIR, **kwargs):
    """Creates a pandas DataFrame from a given filename"""
    return pd.read_csv(os.path.join(data_dir, filename), **kwargs)


def get_comp_chart_from_df(comp_df, y_axis_name, chart_name):
    """
    Create comparison chart from df with all series to compare.
    """
    years = comp_df[GlossaryCore.Years].values.tolist()
    series = comp_df.loc[:, comp_df.columns != GlossaryCore.Years]
    min_x = min(years)
    max_x = max(years)
    min_y = series.min()
    max_y = series.max()
    new_chart = TwoAxesInstanciatedChart(
        GlossaryCore.Years,
        y_axis_name,
        [min_x - 5, max_x + 5],
        [min_y - max_y * 0.05, max_y * 1.05],
        chart_name,
    )
    for sc in series.columns:
        new_series = InstanciatedSeries(years, series[sc].values.tolist(), sc, "lines")
        new_chart.series.append(new_series)
    return new_chart


def get_comp_chart_from_dfs(
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        y_axis_name: str,
        chart_name: str,
        args_0: dict = None,
        args_1: dict = None,
        args_2: dict = None,
):
    """
    Create comparison chart from two df's.
    """
    years_1 = df_1[GlossaryCore.Years].values.tolist()
    years_2 = df_2[GlossaryCore.Years].values.tolist()

    series_1 = df_1.loc[:, df_1.columns != GlossaryCore.Years]
    series_2 = df_2.loc[:, df_2.columns != GlossaryCore.Years]

    min_x = min(years_1, years_2)
    max_x = max(years_1, years_2)
    min_y = min(series_1.min(), series_2.min())
    max_y = max(series_1.max(), series_2.max())

    if args_0 is None:
        args_0 = {}

    new_chart = TwoAxesInstanciatedChart(
        GlossaryCore.Years,
        y_axis_name,
        [min_x - 5, max_x + 5],
        [min_y - max_y * 0.05, max_y * 1.05],
        chart_name,
        **args_0,
    )

    for years, series, args in zip(
            [years_1, years_2], [series_1, series_2], [args_1, args_2]
    ):
        if args is None:
            args = {}
        for col in series.columns:
            new_series = InstanciatedSeries(
                years, series[col].values.tolist(), col, **args
            )
            new_chart.series.append(new_series)
    return new_chart


def post_processing_filters(execution_engine, namespace):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    chart_filters = []

    chart_list = ["Test"]
    # First filter to deal with the view : program or actor
    chart_filters.append(
        ChartFilter("Charts_grad", chart_list, chart_list, "Charts_grad")
    )  # name 'Charts' is already used by ssp_comparison post-proc

    return chart_filters


def post_processings(execution_engine, namespace, filters):
    """
    Instantiate postprocessing charts.
    """
    logging.info("TEST NODE POST_PROC")

    instanciated_charts = []
    chart_list = []

    # example: 3 methods to recover the dataframe of the variable 'invest_mix'
    # method 1: if invest_mix occurs in different disciplines, first list the full variable name including namespace value
    list_of_variables_with_full_namespace = (
        execution_engine.dm.get_all_namespaces_from_var_name("invest_mix")
    )
    # then recover the values for each occurences of the variable
    invest_mix_dict = {}
    for var in list_of_variables_with_full_namespace:
        invest_mix_dict[var] = execution_engine.dm.get_value(var)
    # method 2: if only one occurrence of the variable, it gets it automatically:
    invest_mix = get_scenario_value(execution_engine, "invest_mix", namespace)

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == "Charts_grad":
                chart_list = chart_filter.selected_values

    # Collect data into easily usable format
    iea_data = {
        fname.replace("IEA_NZE_", "").replace(".csv", ""): create_df_from_csv(fname)
        for fname in DATA_FILE_LIST
    }

    # CO2_100 plot
    if "CO2_100" in chart_list:
        df_iea = iea_data["CO2_100"]
        y_axis_name = "CO2"
        df_witness =

        pass

    if "Test" not in chart_list:
        # if not in coarse, add primary energy chart
        pass

    return instanciated_charts
