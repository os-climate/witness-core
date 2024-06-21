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
from pathlib import Path

import pandas as pd

from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

DATA_DIR = Path(__file__).parents[3] / "data"

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
    return pd.read_csv(str(data_dir / filename), **kwargs)


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

    # min_x = min(years_1, years_2)
    # max_x = max(years_1, years_2)
    # min_y = series_1.min()
    # max_y = max(series_1.max(), series_2.max())

    if args_0 is None:
        args_0 = {}

    new_chart = TwoAxesInstanciatedChart(
        GlossaryCore.Years,
        y_axis_name,
        # [min_x - 5, max_x + 5],
        # [min_y - max_y * 0.05, max_y * 1.05],
        chart_name=chart_name,
        **args_0,
    )

    for years, series, args in zip(
        [years_1, years_2], [series_1, series_2], [args_1, args_2]
    ):
        if args is None:
            args = {}
        col_suffix = args.pop("col_suffix", "")
        for col in series.columns:
            new_series = InstanciatedSeries(
                years, series[col].values.tolist(), col + f" {col_suffix}", **args
            )
            new_chart.series.append(new_series)
    return new_chart


def post_processing_filters(execution_engine, namespace):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    chart_filters = []

    chart_list = ["Coal", "Nuclear", "Hydro"]
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

    def get_variable_from_namespace(
        var_name: str, namespace_str: str = None, is_single_occurence: bool = False
    ):
        if is_single_occurence and namespace_str is not None:
            return get_scenario_value(execution_engine, var_name, namespace_str)
        else:
            list_of_variables_with_full_namespace = (
                execution_engine.dm.get_all_namespaces_from_var_name(var_name)
            )
            # then recover the values for each occurences of the variable
            var_dict = {}
            for var in list_of_variables_with_full_namespace:
                var_dict[var] = execution_engine.dm.get_value(var)
            return var_dict

    instanciated_charts = []
    chart_list = []

    # # example: 2 methods to recover the dataframe of the variable 'invest_mix'
    # # method 1: if invest_mix occurs in different disciplines, first list the full variable name including namespace value
    # list_of_variables_with_full_namespace = (
    #     execution_engine.dm.get_all_namespaces_from_var_name("invest_mix")
    # )
    # # then recover the values for each occurences of the variable
    # invest_mix_dict = {}
    # for var in list_of_variables_with_full_namespace:
    #     invest_mix_dict[var] = execution_engine.dm.get_value(var)
    # # method 2: if only one occurrence of the variable, it gets it automatically:
    # invest_mix = get_scenario_value(execution_engine, "invest_mix", namespace)

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

    def create_chart_comparing_WITNESS_and_IEA(
            chart_name: str,
            y_axis_name: str,
            iea_key: str,
            witness_variable: str,
            witness_var_path: str,
            columns_to_plot: list,
            args_to_plot: dict,
            sum_columns: str = None
    ):

        df_iea = iea_data[iea_key]

        witness_data_dict = get_variable_from_namespace(witness_variable)

        # Find dataframe whose namespace "path" contains witness_var_path
        df_witness = None
        for ns, df in witness_data_dict.items():
            if witness_var_path in ns:
                df_witness = df
                break

        # Assume we arrive here with df_witness having been found
        # Select only column(s) we are interested in
        df_witness = df_witness[[GlossaryEnergy.Years] + columns_to_plot]

        if sum_columns is not None:
            dff=pd.DataFrame(data={GlossaryEnergy.Years: df_witness[GlossaryEnergy.Years]})
            dff[sum_columns] = df_witness[columns_to_plot].sum(axis=1)
            df_witness = dff

        # Create chart
        return get_comp_chart_from_dfs(
            df_witness, df_iea, y_axis_name, chart_name, **args_to_plot
        )

    # Coal Plot
    if "Coal" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
                chart_name="Energy from Coal",
                y_axis_name= "Energy (TWh)",
                iea_key= "EnergyMix_solid_fuel_CoalExtraction_techno_production",
                witness_variable= "techno_production",
                witness_var_path= "EnergyMix.solid_fuel.CoalExtraction",
                columns_to_plot= ["solid_fuel (TWh)"],
                args_to_plot= {"args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
        )
        instanciated_charts.append(new_chart)
    if "Nuclear" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
                chart_name="Energy from Nuclear",
                y_axis_name= "Energy (TWh)",
                iea_key= "EnergyMix.electricity.Nuclear.techno_production",
                witness_variable= "techno_production",
                witness_var_path= "EnergyMix.electricity.Nuclear",
                columns_to_plot= ["electricity (TWh)", "heat.hightemperatureheat (TWh)"],
                args_to_plot= {"args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
                sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)
    if "Hydro" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
                chart_name="Energy from Hydro",
                y_axis_name= "Electricity (TWh)",
                iea_key= "EnergyMix.electricity.Hydropower.techno_production",
                witness_variable= "techno_production",
                witness_var_path= "EnergyMix.electricity.Hydropower",
                columns_to_plot= ["electricity (TWh)"],
                args_to_plot= {"args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
                sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)


    if "Test" not in chart_list:
        # if not in coarse, add primary energy chart
        pass

    return instanciated_charts
