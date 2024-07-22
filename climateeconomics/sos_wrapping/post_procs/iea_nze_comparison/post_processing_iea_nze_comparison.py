'''
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
'''

import logging
from typing import Union

import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation_discipline import (
    IEADataPreparationDiscipline,
)

IEA_NAME = IEADataPreparationDiscipline.IEA_NAME
END_YEAR_NAME = 'Ending year'
# to plot interpolated IEA data, use SUFFIX_VAR_IEA = IEADataPreparationDiscipline.SUFFIX_VAR_INTERPOLATED
# to use raw IEA data, use SUFFIX_VAR_IEA = ''
SUFFIX_VAR_IEA = '' #IEADataPreparationDiscipline.SUFFIX_VAR_INTERPOLATED


def get_shared_value(execution_engine, short_name_var: str):
    """returns the value of a variables common to all scenarios"""
    var_full_name = execution_engine.dm.get_all_namespaces_from_var_name(short_name_var)[0]
    value = execution_engine.dm.get_value(var_full_name)
    return value, var_full_name

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
        new_chart.add_series(new_series)
    return new_chart


def get_comp_chart_from_dfs(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    y_axis_name: str,
    chart_name: str,
    year_end: int,
    args_0: dict = None,
    args_1: dict = None,
    args_2: dict = None,
):
    """
    Create comparison chart from two df's.
    """
    years_1 = df_1.loc[df_1[GlossaryCore.Years] <= year_end][GlossaryCore.Years].values.tolist()
    years_2 = df_2.loc[df_2[GlossaryCore.Years] <= year_end][GlossaryCore.Years].values.tolist()

    series_1 = (df_1.loc[df_1[GlossaryCore.Years] <= year_end]).loc[:, df_1.columns != GlossaryCore.Years]
    series_2 = (df_2.loc[df_2[GlossaryCore.Years] <= year_end]).loc[:, df_2.columns != GlossaryCore.Years]

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
        df_label = args.pop("df_label", None)
        for col in series.columns:
            df_label = col + f" {col_suffix}" if df_label is None else df_label
            new_series = InstanciatedSeries(
                years, series[col].values.tolist(), df_label, **args
            )
            new_chart.add_series(new_series)
    return new_chart


def post_processing_filters(execution_engine, namespace):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    chart_filters = []

    chart_list = [
        "Population",
        "Economics",
        "Temperature",
        "CO2_emissions",
        "CO2_taxes",
        "Energy_production",
        "Energy_prices",
        "Land_use",
    ]
    year_start, _ = get_shared_value(execution_engine, GlossaryCore.YearStart)
    year_end, _ = get_shared_value(execution_engine, GlossaryCore.YearEnd)
    years_list = list(np.arange(year_start, year_end + 1))

    # First filter to deal with the view : program or actor
    chart_filters.append(
        ChartFilter("Charts", chart_list, chart_list, "Charts")
    )
    chart_filters.append(ChartFilter(END_YEAR_NAME, years_list, year_end, END_YEAR_NAME, multiple_selection=False)) # by default shows all years

    return chart_filters


def post_processings(execution_engine, namespace, filters):
    """
    Instantiate postprocessing charts.
    """
    logging.debug("post_processing iea nze vs witness")
    year_end, _ = get_shared_value(execution_engine, GlossaryCore.YearEnd)


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

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == "Charts":
                chart_list = chart_filter.selected_values
            if chart_filter.filter_key == END_YEAR_NAME:
                year_end = chart_filter.selected_values

    def get_df_from_var_name(var):
        data_dict = get_variable_from_namespace(var)
        dff = None
        for ns, df in data_dict.items():
            if var in ns:
                dff = df
                break
        return dff

    def create_chart_comparing_WITNESS_and_IEA(
        chart_name: str,
        y_axis_name: str,
        iea_variable: str,
        witness_variable: Union[str, list[str]],
        columns_to_plot: Union[list[str], list[list[str]]],
        args_to_plot: dict,
        sum_columns: str = None,
    ):
        if isinstance(witness_variable, str):
            witness_variable = [witness_variable]
        if len(witness_variable) == 1:
            columns_to_plot = [columns_to_plot]

        # Find dataframe whose namespace "path" contains witness_var_path
        df_witness_list = []
        for wv in witness_variable:
            df = get_df_from_var_name(wv)
            if df is None:
                logging.warning(f"No data found for {wv} in {namespace}")
                return None
            else:
                df_witness_list.append(df)

        df_iea = get_df_from_var_name(iea_variable)
        if df_iea is None:
            logging.warning(f"No data found for {iea_variable} in {namespace}")
            return None
        # in case of multiple plots on the same graph or in case of multiple columns in the iea df, must select the ones
        # to plots
        if sum_columns is None:
            df_iea = df_iea[[GlossaryCore.Years] + columns_to_plot[0]]

        # Check if there is any None in df_witness_list
        if any(elem is None for elem in df_witness_list):
            logging.warning(f"No data found for {witness_variable} in {namespace}")
            return None

        # Create empty dataframe with years
        df = pd.DataFrame(
            data={
                GlossaryCore.Years: df_witness_list[0][
                    GlossaryEnergy.Years
                ].values.tolist()
            }
        )

        for c_to_plot, df_witness in zip(columns_to_plot, df_witness_list):
            # Check if columns exist in  df_witness
            if not all(elem in df_witness.columns for elem in c_to_plot):
                logging.warning(
                    f"Columns {c_to_plot} not found in {df_witness} for {witness_variable} in {namespace}"
                )
                return None

            # Select only column(s) we are interested in
            dff = df_witness[c_to_plot].copy()

            # Sum culimns into a column named sum_columns
            if sum_columns is not None:
                if sum_columns in df.columns:
                    df[sum_columns] += dff.sum(axis=1)
                else:
                    df[sum_columns] = dff.sum(axis=1)

            else:  # no forced sum, just add the dataframes together
                for col in c_to_plot:
                    if col in df.columns:
                        df[col] = df[col].to_numpy() + dff[col].to_numpy()
                    else:
                        df[col] = dff[col].to_numpy()

        # Create chart
        return get_comp_chart_from_dfs(df, df_iea, y_axis_name, chart_name, year_end, **args_to_plot)

    if "Population" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Population",
            y_axis_name="Population (Millions)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.PopulationDfValue}{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.population_df",
            columns_to_plot=["population"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)
    if "Economics" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Economics",
            y_axis_name="GDP net of damage (G$)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.EconomicsDfValue}{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.Macroeconomics.economics_detail_df",
            columns_to_plot=["output_net_of_d"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

    if "Temperature" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Temperature",
            y_axis_name="Increase in atmospheric temperature (°C)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.TemperatureDfValue}{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.temperature_df",
            columns_to_plot=["temp_atmo"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

    if "CO2_emissions" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="CO2 emissions of the energy sector",
            y_axis_name="Total CO2 emissions (Gt)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.CO2EmissionsGtValue}{SUFFIX_VAR_IEA}",
            witness_variable="EnergyMix.co2_emissions_Gt",
            columns_to_plot=["Total CO2 emissions"],
            args_to_plot={
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

    if "CO2_taxes" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="CO2 Taxes",
            y_axis_name="CO2 taxes ($/ton of CO2)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.CO2TaxesValue}{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.CO2_taxes",
            columns_to_plot=["CO2_tax"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

    if "Energy_production" in chart_list:
        # GDP vs energy for IEA, witness and historical data
        new_chart = TwoAxesInstanciatedChart(
            "World Raw energy production (PWh)",
            "World GDP net of damage (T$)",
            chart_name="GDP vs energy production evolution",
        )
        x_witness_df, _ = get_shared_value(execution_engine, f"EnergyMix.{GlossaryEnergy.EnergyProductionValue}")
        y_witness_df, _ = get_shared_value(execution_engine, "WITNESS.Macroeconomics.economics_detail_df")
        x_witness = x_witness_df.loc[x_witness_df[GlossaryCore.Years] <= year_end][GlossaryCore.TotalProductionValue]
        y_witness = y_witness_df.loc[y_witness_df[GlossaryCore.Years] <= year_end]["output_net_of_d"]
        new_series = InstanciatedSeries(
            x_witness.values.tolist(),
            y_witness.values.tolist(),
            "WITNESS", display_type="scatter",
            text=y_witness_df.loc[y_witness_df[GlossaryCore.Years] <= year_end][GlossaryCore.Years].values.tolist()
        )
        new_chart.add_series(new_series)

        x_iea_df, _ = get_shared_value(execution_engine, f"{IEA_NAME}.{GlossaryEnergy.EnergyProductionValue}{SUFFIX_VAR_IEA}")
        y_iea_df, _ = get_shared_value(execution_engine, f"{IEA_NAME}.{GlossaryEnergy.EconomicsDfValue}{SUFFIX_VAR_IEA}")
        # iea Data are not always provided at the same years for different quantities => only keep the data for the common
        # years for gdp and energy production. Witness data are provided for the same years
        years_x = x_iea_df.loc[x_iea_df[GlossaryCore.Years] <= year_end][GlossaryCore.Years]
        years_y = y_iea_df.loc[y_iea_df[GlossaryCore.Years] <= year_end][GlossaryCore.Years]
        common_years = sorted(list(set(years_x).intersection(set(years_y))))
        x_iea = x_iea_df.loc[x_iea_df[GlossaryCore.Years].isin(common_years)][GlossaryCore.TotalProductionValue]
        y_iea = y_iea_df.loc[y_iea_df[GlossaryCore.Years].isin(common_years)]["output_net_of_d"]
        new_series = InstanciatedSeries(
            x_iea.values.tolist(),
            y_iea.values.tolist(),
            "IEA", display_type="scatter",
            text=common_years
        )
        new_chart.add_series(new_series)

        df_historical_df = pd.read_csv(join(Path(__file__).parents[3], "data", 'world_gdp_vs_net_energy_consumption.csv'))
        years_historical = df_historical_df['years']
        x_historical = df_historical_df['Net energy consumption [PWh]']
        y_historical = df_historical_df['World GDP [T$]']
        new_series = InstanciatedSeries(
            x_historical.values.tolist(),
            y_historical.values.tolist(),
            "Historical", display_type="scatter",
            text=years_historical
        )
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # total
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Raw Energy Production",
            y_axis_name="Energy (PWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.EnergyProductionValue}{SUFFIX_VAR_IEA}",
            witness_variable=f"EnergyMix.{GlossaryEnergy.EnergyProductionValue}",
            columns_to_plot=[GlossaryCore.TotalProductionValue],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

        # Coal
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Coal",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.solid_fuel}_{GlossaryEnergy.CoalExtraction}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="EnergyMix.solid_fuel.CoalExtraction.techno_detailed_production",
            columns_to_plot=["solid_fuel (TWh)"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)
        # Nuclear = sum of heat + electricity
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Nuclear",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Nuclear}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="EnergyMix.electricity.Nuclear.techno_detailed_production",
            columns_to_plot=["electricity (TWh)", "heat.hightemperatureheat (TWh)"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
            sum_columns="WITNESS",
        )
        instanciated_charts.append(new_chart)
        # "Hydro"
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Hydro",
            y_axis_name="Electricity (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Hydropower}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="EnergyMix.electricity.Hydropower.techno_detailed_production",
            columns_to_plot=["electricity (TWh)"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
            sum_columns="WITNESS",
        )
        instanciated_charts.append(new_chart)
        # "Solar" (coming from two different WITNESS variables)
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Solar",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Solar}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable=[
                "EnergyMix.electricity.SolarPv.techno_detailed_production",
                "EnergyMix.electricity.SolarThermal.techno_detailed_production",
            ],
            columns_to_plot=[["electricity (TWh)"], ["electricity (TWh)"]],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
            sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)
        # "Wind" (coming from two different WITNESS variables)
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Wind",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.WindOnshoreAndOffshore}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable=[
                "EnergyMix.electricity.WindOnshore.techno_detailed_production",
                "EnergyMix.electricity.WindOffshore.techno_detailed_production",
            ],
            columns_to_plot=[["electricity (TWh)"], ["electricity (TWh)"]],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
            sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)
        # "Modern gaseous bioenergy"
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from gaseous bionergy",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.biogas}_{GlossaryEnergy.AnaerobicDigestion}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="EnergyMix.biogas.energy_production_detailed",
            columns_to_plot=["biogas AnaerobicDigestion (TWh)"],
            args_to_plot={"args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
            # sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)

        # "Forest energy"
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from Forest",
            y_axis_name="Energy (PWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.ForestProduction}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.AgricultureMix.Forest.techno_production",
            columns_to_plot=["biomass_dry (TWh)"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
            # sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)

        # "crop energy"
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Energy from crop",
            y_axis_name="Energy (TWh)",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.CropEnergy}_techno_production{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.AgricultureMix.Crop.mix_detailed_production",
            columns_to_plot=["Total (TWh)"],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
            # sum_columns="WITNESS"
        )
        instanciated_charts.append(new_chart)
        
        # SSR: I do not know which one goes here
        # # "Oil" 
        # new_chart = create_chart_comparing_WITNESS_and_IEA(
        #     chart_name="Energy from Oil",
        #     y_axis_name="Energy (TWh)",
        #     iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Hydropower}_techno_production",
        #     witness_variable="EnergyMix.energy_production_brut_detailed",
        #     columns_to_plot=["production fuel.liquid_fuel (TWh)"],
        #     args_to_plot={"args_2": {"display_type": "scatter", "col_suffix": "IEA"}},
        #     # sum_columns="WITNESS"
        # )
        # instanciated_charts.append(new_chart)

    if "Energy_prices" in chart_list:
        new_chart = create_chart_comparing_WITNESS_and_IEA(
            chart_name="Natural gas price",
            y_axis_name="$/MWh",
            iea_variable=f"{IEA_NAME}.{GlossaryEnergy.methane}_{GlossaryEnergy.EnergyPricesValue}{SUFFIX_VAR_IEA}",
            witness_variable="WITNESS.EnergyMix.methane.FossilGas.techno_prices",
            columns_to_plot=[GlossaryEnergy.FossilGas],
            args_to_plot={
                "args_0": {'y_min_zero': True},
                "args_1": {"df_label": "WITNESS"},
                "args_2": {"display_type": "scatter", "df_label": "IEA"},
            },
        )
        instanciated_charts.append(new_chart)

        # get the technos
        electricity_prices_df = get_scenario_value(execution_engine, f'{GlossaryEnergy.electricity}_{GlossaryEnergy.EnergyPricesValue}', namespace + IEA_NAME)
        for techno in [var for var in electricity_prices_df.keys() if var != GlossaryEnergy.Years]:
            new_chart = create_chart_comparing_WITNESS_and_IEA(
                chart_name=f"Electricity price for {techno}",
                y_axis_name="$/MWh",
                iea_variable=f"{IEA_NAME}.{GlossaryEnergy.electricity}_{GlossaryEnergy.EnergyPricesValue}{SUFFIX_VAR_IEA}",
                witness_variable=f"WITNESS.EnergyMix.electricity.{techno}.techno_prices",
                columns_to_plot=[techno],
                args_to_plot={
                    "args_0": {'y_min_zero': True},
                    "args_1": {"df_label": "WITNESS"},
                    "args_2": {"display_type": "scatter", "df_label": "IEA"},
                },
            )
            instanciated_charts.append(new_chart)

    if "Land_use" in chart_list:
        land_use_df = get_scenario_value(execution_engine, f"{IEA_NAME}.{LandUseV2.LAND_SURFACE_DETAIL_DF}", namespace + IEA_NAME)
        for surface in [var for var in land_use_df.keys() if var != GlossaryEnergy.Years]:
            new_chart = create_chart_comparing_WITNESS_and_IEA(
                chart_name="Land use",
                y_axis_name=f"{surface}",
                iea_variable=f"{IEA_NAME}.{LandUseV2.LAND_SURFACE_DETAIL_DF}{SUFFIX_VAR_IEA}",
                witness_variable="WITNESS.Land_Use.land_surface_detail_df",
                columns_to_plot=[surface],
                args_to_plot={
                    "args_0": {'y_min_zero': True},
                    "args_1": {"df_label": "WITNESS"},
                    "args_2": {"display_type": "scatter", "df_label": "IEA"},
                },
            )
            instanciated_charts.append(new_chart)


    if "Test" not in chart_list:
        # if not in coarse, add primary energy chart
        pass

    return instanciated_charts
