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
import os.path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

WITNESS_SERIES_NAME = 'WITNESS'
WITNESS_YEARS = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)

# CSV files keys
REGION = 'Region'
SCENARIO = 'Scenario'
MODEL = 'Model'
CSV_SEP = ';'
CSV_DEC = ','
CSV_YRS = [str(_yr) for _yr in range(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 10)]
YEARS = GlossaryCore.Years

# POSTPROCESSING DICTS KEYS (INTERNAL USAGE)
FILE_NAME = 'file_name'
VAR_NAME = 'var_name'
COLUMN = 'column'
WITNESS_VARS_COLS = f'({VAR_NAME}, {COLUMN})_list'
CHART_TITLE = 'chart_title'
UNIT_CONV_FACTOR = 'unit_conv_factor'
Y_AXIS = 'y_axis'

# POSTPROCESSING DICTS WITH INFOS TO PRODUCE COMPARISON GRAPHS
GDP = 'GDP'
_gdp = {FILE_NAME: 'gdp_ppp_usd2020.csv',
        VAR_NAME: GlossaryCore.EconomicsDfValue,
        COLUMN: GlossaryCore.GrossOutput,
        CHART_TITLE: 'GDP: WITNESS vs. SSP scenarios (IPCC)',
        UNIT_CONV_FACTOR: 1E-3,
        Y_AXIS: 'World Output [trillion $]'}

POPULATION = 'Population'
_population = {FILE_NAME: 'population.csv',
               VAR_NAME: GlossaryCore.PopulationDfValue,
               COLUMN: GlossaryCore.PopulationValue,
               CHART_TITLE: 'Population: WITNESS vs. SSP scenarios (IPCC)',
               UNIT_CONV_FACTOR: 1.0,
               Y_AXIS: 'World Population [million people]'}

CONSUMPTION = 'Consumption'
_consumption = {FILE_NAME: 'consumption_global_usd2020.csv',
                VAR_NAME: 'Macroeconomics.economics_detail_df',
                COLUMN: GlossaryCore.Consumption,
                CHART_TITLE: 'Consumption: WITNESS vs. SSP scenarios (IPCC)',
                UNIT_CONV_FACTOR: 1E-3,
                Y_AXIS: 'Global Consumption [trillion $]'}

MEAN_TEMPERATURE = 'Mean_temperature'
_mean_temperature = {FILE_NAME: 'mean_temperature_global.csv',
                     VAR_NAME: GlossaryCore.TemperatureDfValue,
                     COLUMN: GlossaryCore.TempAtmo,
                     CHART_TITLE: 'Atmospheric temperature: WITNESS vs. SSP scenarios (IPCC)',
                     UNIT_CONV_FACTOR: 1.0,
                     Y_AXIS: 'Mean Temperature [ºC above pre-industrial]'}

FORCING = 'Forcing'
_forcing = {FILE_NAME: 'forcing_global.csv',
            VAR_NAME: 'Temperature_change.temperature_detail_df',
            COLUMN: GlossaryCore.Forcing,
            CHART_TITLE: 'Total radiative forcing: WITNESS vs. SSP scenarios (IPCC)',
            UNIT_CONV_FACTOR: 1.0,
            Y_AXIS: 'Total Forcing [W/m^2]'}

CO2_CONCENTRATION = 'CO2_concentration'
_co2_concentration = {FILE_NAME: 'CO2_concentration_global.csv',
                      VAR_NAME: GlossaryCore.GHGCycleDfValue,
                      COLUMN: GlossaryCore.CO2Concentration,
                      CHART_TITLE: 'Atmospheric CO2 concentration: WITNESS vs. SSP scenarios (IPCC)',
                      UNIT_CONV_FACTOR: 1.0,
                      Y_AXIS: 'CO2 concentration [ppm]'}

CH4_CONCENTRATION = 'CH4_concentration'
_ch4_concentration = {FILE_NAME: 'CH4_concentration_global.csv',
                      VAR_NAME: GlossaryCore.GHGCycleDfValue,
                      COLUMN: GlossaryCore.CH4Concentration,
                      CHART_TITLE: 'Atmospheric CH4 concentration: WITNESS vs. SSP scenarios (IPCC)',
                      UNIT_CONV_FACTOR: 1.0,
                      Y_AXIS: 'CH4 concentration [ppm]'}

N2O_CONCENTRATION = 'N2O_concentration'
_n2o_concentration = {FILE_NAME: 'N2O_concentration_global.csv',
                      VAR_NAME: GlossaryCore.GHGCycleDfValue,
                      COLUMN: GlossaryCore.N2OConcentration,
                      CHART_TITLE: 'Atmospheric N2O concentration: WITNESS vs. SSP scenarios (IPCC)',
                      UNIT_CONV_FACTOR: 1.0,
                      Y_AXIS: 'N2O concentration [ppm]'}

CO2_EMISSIONS = 'CO2_emissions'
_co2_emissions = {FILE_NAME: 'CO2_emissions_global.csv',
                  VAR_NAME: 'GHGEmissions.GHG_emissions_detail_df',
                  COLUMN: GlossaryCore.TotalCO2Emissions,
                  CHART_TITLE: 'Total CO2 Emissions: WITNESS vs. SSP scenarios (IPCC)',
                  UNIT_CONV_FACTOR: 1e-3,
                  Y_AXIS: 'Total CO2 Emissions [GtCO2]'}

CH4_EMISSIONS = 'CH4_emissions'
_ch4_emissions = {FILE_NAME: 'CH4_emissions_global.csv',
                  VAR_NAME: 'GHGEmissions.GHG_emissions_detail_df',
                  COLUMN: GlossaryCore.TotalCH4Emissions,
                  CHART_TITLE: 'Total CH4 Emissions: WITNESS vs. SSP scenarios (IPCC)',
                  UNIT_CONV_FACTOR: 1e-3,
                  Y_AXIS: 'Total CH4 Emissions [GtCH4]'}

N2O_EMISSIONS = 'N2O_emissions'
_n2o_emissions = {FILE_NAME: 'N2O_emissions_global.csv',
                  VAR_NAME: 'GHGEmissions.GHG_emissions_detail_df',
                  COLUMN: GlossaryCore.TotalN2OEmissions,
                  CHART_TITLE: 'Total N2O Emissions: WITNESS vs. SSP scenarios (IPCC)',
                  UNIT_CONV_FACTOR: 1e-6,
                  Y_AXIS: 'Total N2O Emissions [GtN2O]'}

FOREST_SURFACE = 'Forest_surface'
_forest_surface = {FILE_NAME: 'land_cover_forest_global.csv',
                   VAR_NAME: 'Land_Use.land_surface_detail_df',
                   COLUMN: 'Total Forest Surface (Gha)',
                   CHART_TITLE: 'Total Forest Surface: WITNESS vs. SSP scenarios (IPCC)',
                   UNIT_CONV_FACTOR: 1e-3,
                   Y_AXIS: 'Forest Surface [Gha]'}

AGRICULTURE_SURFACE = 'Agriculture_surface'
_agriculture_surface = {FILE_NAME: 'land_cover_pasture+cropland_global.csv',
                        VAR_NAME: 'Land_Use.land_surface_detail_df',
                        COLUMN: 'Total Agriculture Surface (Gha)',
                        CHART_TITLE: 'Total Agriculture Surface: WITNESS vs. SSP scenarios (IPCC)',
                        UNIT_CONV_FACTOR: 1e-3,
                        Y_AXIS: 'Agriculture Surface [Gha]'}

FINAL_ENERGY = 'energy'
_final_energy = {FILE_NAME: 'final_energy_global.csv',
                 VAR_NAME: f'EnergyMix.{GlossaryCore.EnergyProductionValue}',
                 COLUMN: GlossaryCore.TotalProductionValue,
                 CHART_TITLE: 'World Energy Production: WITNESS vs. SSP scenarios (IPCC)',
                 UNIT_CONV_FACTOR: 0.27777777777,
                 Y_AXIS: 'Final Energy Production [PWh]'}

CHARTS_DATA = {
    GDP: _gdp,
    POPULATION: _population,
    CONSUMPTION: _consumption,
    MEAN_TEMPERATURE: _mean_temperature,
    FORCING: _forcing,
    CO2_CONCENTRATION: _co2_concentration,
    CH4_CONCENTRATION: _ch4_concentration,
    N2O_CONCENTRATION: _n2o_concentration,
    CO2_EMISSIONS: _co2_emissions,
    CH4_EMISSIONS: _ch4_emissions,
    N2O_EMISSIONS: _n2o_emissions,
    FOREST_SURFACE: _forest_surface,
    AGRICULTURE_SURFACE: _agriculture_surface,
    FINAL_ENERGY: _final_energy
    }


# special chart: primary energy fractions
PRIMARY_ENERGY = 'Primary energy fraction'
COAL = 'Coal'
_coal = {FILE_NAME: 'coal_fraction.csv', UNIT_CONV_FACTOR: 1.0,
         WITNESS_VARS_COLS: [(f'EnergyMix.solid_fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'solid_fuel CoalExtraction (TWh)'),
                             # ('EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity CoalGen (TWh)'),
                             ]}
OIL_GAS = 'Oil & Gas'
_oil_gas = {FILE_NAME: 'oil_gas_fraction.csv', UNIT_CONV_FACTOR: 1.0,
            WITNESS_VARS_COLS: [(f'EnergyMix.methane.{GlossaryCore.EnergyProductionDetailedValue}', 'methane FossilGas (TWh)'),
                                # (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.biodiesel Transesterification (TWh)'),
                                # (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.ethanol BiomassFermentation (TWh)'),
                                (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.hydrotreated_oil_fuel HefaDecarboxylation (TWh)'),
                                (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.hydrotreated_oil_fuel HefaDeoxygenation (TWh)'),
                                (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.liquid_fuel FischerTropsch (TWh)'),
                                (f'EnergyMix.fuel.{GlossaryCore.EnergyProductionDetailedValue}', 'fuel.liquid_fuel Refinery (TWh)'),
                                # (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity CombinedCycleGasTurbine (TWh)'),
                                # (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity GasTurbine (TWh)'),
                                # (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity OilGen (TWh)'),
                                    ]
            }
NON_FOSSIL = 'Non-fossil'
_non_fossil = {FILE_NAME: 'non-fossil_fraction.csv', UNIT_CONV_FACTOR: 1.0,}

HYDROGEN = 'Hydrogen'
_hydrogen = {WITNESS_VARS_COLS: [(f'EnergyMix.methane.{GlossaryCore.EnergyProductionDetailedValue}', 'methane Methanation (TWh)'),
                                 ('EnergyMix.energy_production_brut_detailed', 'production hydrogen.gaseous_hydrogen (TWh)'),
                                 ('EnergyMix.energy_production_brut_detailed', 'production hydrogen.liquid_hydrogen (TWh)'),
                                 ]}
PRIMARY_ENERGY_DATA = {
    COAL: _coal,
    OIL_GAS: _oil_gas,
    NON_FOSSIL: _non_fossil
}
WITNESS_PRIMARY_ENERGY_DATA = {
    COAL: _coal,
    OIL_GAS: _oil_gas,
    # HYDROGEN: _hydrogen
} # NON-FOSSIL is deduced from total
WITNESS_BRUT_ENERGY_TOTAL = ('EnergyMix.energy_production_brut_detailed', GlossaryCore.TotalProductionValue)
WITNESS_BRUT_ENERGY_TOTAL_MINUS = [
    (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity CoalGen (TWh)'),
    (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity CombinedCycleGasTurbine (TWh)'),
    (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity GasTurbine (TWh)'),
    (f'EnergyMix.electricity.{GlossaryCore.EnergyProductionDetailedValue}', 'electricity OilGen (TWh)'),

    (f'EnergyMix.methane.{GlossaryCore.EnergyProductionDetailedValue}', 'methane Methanation (TWh)'),
    ('EnergyMix.energy_production_brut_detailed', 'production hydrogen.gaseous_hydrogen (TWh)'),
    ('EnergyMix.energy_production_brut_detailed', 'production hydrogen.liquid_hydrogen (TWh)'),
]
CHART_LIST = list(CHARTS_DATA.keys()) 

def get_ssp_data(data_name, data_dict, region='World'):
    """
    Get ssp dataframes for each variable.
    """
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    var_df = pd.read_csv(os.path.join(data_dir, data_dict[data_name][FILE_NAME]), sep=CSV_SEP, decimal=CSV_DEC)
    var_df = var_df[var_df[REGION] == region]
    var_df[SCENARIO] = [f"{_sc.split('-Baseline')[0]} ({_model})" for _sc, _model in var_df[[SCENARIO, MODEL]].values.tolist()]
    var_df = var_df[[SCENARIO] + CSV_YRS].set_index(SCENARIO, drop=True).transpose().reset_index().rename(columns={'index': YEARS})
    var_df[YEARS] = pd.to_numeric(var_df[YEARS])
    var_df.loc[:, var_df.columns != YEARS] *= data_dict[data_name][UNIT_CONV_FACTOR]
    var_df = var_df.reindex(columns=[YEARS] + sorted(set(var_df.columns) - {YEARS}))  # sort the scenarios by name for clarity
    var_df.columns.name = None
    return var_df

def post_processing_filters(execution_engine, namespace):

    # get energy list 
    energy_list = execution_engine.dm.get_value(f'{namespace}.{GlossaryCore.energy_list}')
    # if renewable not in energy list, we are not in coarse 
    chart_l = CHART_LIST

    if 'renewable' not in energy_list:
        # if not in coarse, add primary energy chart
        chart_l = chart_l + [PRIMARY_ENERGY]

    return [ChartFilter('Charts', chart_l, chart_l, 'Charts')]

def get_comp_chart_from_df(comp_df, y_axis_name, chart_name):
    """
    Create comparison chart from df with all series to compare.
    """
    years = comp_df[YEARS].values.tolist()
    series = comp_df.loc[:, comp_df.columns != YEARS]
    min_x = min(years)
    max_x = max(years)
    min_y = series.min()
    max_y = series.max()
    new_chart = TwoAxesInstanciatedChart(YEARS, y_axis_name,
                                         [min_x - 5, max_x + 5], [
                                         min_y - max_y * 0.05, max_y * 1.05],
                                         chart_name)
    for sc in series.columns:
        new_series = InstanciatedSeries(
            years, series[sc].values.tolist(), sc, 'lines')
        new_chart.series.append(new_series)
    return new_chart

def get_witness_primary_energy_chart(execution_engine, namespace):
    """
    Create the primary energy fractions chart for witness scenario.
    """
    varname, colname = WITNESS_BRUT_ENERGY_TOTAL
    var_f_name = f'{namespace}.{varname}'
    total_brut_energy = execution_engine.dm.get_value(var_f_name)[colname].to_numpy(copy=True)
    for varname, colname in WITNESS_BRUT_ENERGY_TOTAL_MINUS:
        var_f_name = f'{namespace}.{varname}'
        total_brut_energy -= execution_engine.dm.get_value(var_f_name)[colname].to_numpy(copy=True)

    energy_df = pd.DataFrame({YEARS: WITNESS_YEARS,
                              COAL: 0.,
                              OIL_GAS: 0.,
                              NON_FOSSIL: 1.,
                              # HYDROGEN: 0.
                              })

    for energy_type, _energy_dict in WITNESS_PRIMARY_ENERGY_DATA.items():
        for varname, colname in _energy_dict[WITNESS_VARS_COLS]:
            var_f_name = f'{namespace}.{varname}'
            contr = execution_engine.dm.get_value(var_f_name)[colname].to_numpy(copy=True)
            energy_df.loc[:, energy_type] += contr
        energy_df.loc[:, energy_type] /= total_brut_energy
        energy_df.loc[:, NON_FOSSIL] -= energy_df[energy_type]

    min_x = min(WITNESS_YEARS)
    max_x = max(WITNESS_YEARS)
    min_y = 0.0
    max_y = 1.0

    new_chart = TwoAxesInstanciatedChart(YEARS, f'{PRIMARY_ENERGY} [-]',
                                         [min_x - 5, max_x + 5], [
                                         min_y - max_y * 0.05, max_y * 1.05],
                                         f'{PRIMARY_ENERGY} for {WITNESS_SERIES_NAME}',
                                         stacked_bar=True)

    for energy_type in energy_df.columns[1:]:
        series = InstanciatedSeries(list(WITNESS_YEARS), energy_df[energy_type].values.tolist(),
                                    energy_type, InstanciatedSeries.BAR_DISPLAY)
        new_chart.add_series(series)
    return new_chart

def get_ssp_primary_energy_charts():
    """
    Create the primary energy fractions charts for ssp.
    """
    primary_energy_charts = []
    var_dfs = {}
    for key, value in PRIMARY_ENERGY_DATA.items():
        var_dfs[key] = get_ssp_data(key, PRIMARY_ENERGY_DATA)

    scenario_dfs = {}
    for energy_type, energy_type_df in var_dfs.items():
        scenarios = set(energy_type_df.columns) - {YEARS}
        for scenario in scenarios:
            if scenario not in scenario_dfs:
                scenario_dfs[scenario] = pd.DataFrame({YEARS: WITNESS_YEARS})
            f_interp = interp1d(energy_type_df[YEARS], energy_type_df[scenario])
            scenario_energy_type_data = f_interp(WITNESS_YEARS)
            scenario_dfs[scenario][energy_type] = scenario_energy_type_data
    min_x = min(WITNESS_YEARS)
    max_x = max(WITNESS_YEARS)
    min_y = 0.0
    max_y = 1.0
    for scenario, scenario_df in scenario_dfs.items():
        new_chart = TwoAxesInstanciatedChart(YEARS, f'{PRIMARY_ENERGY} [-]',
                                             [min_x - 5, max_x + 5], [
                                             min_y - max_y * 0.05, max_y * 1.05],
                                             f'{PRIMARY_ENERGY} for {scenario}',
                                             stacked_bar=True)
        for energy_type in var_dfs:
            series = InstanciatedSeries(list(WITNESS_YEARS), scenario_df[energy_type].values.tolist(),
                                        energy_type, InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(series)
        primary_energy_charts.append(new_chart)
    return primary_energy_charts

def post_processings(execution_engine, namespace, filters):
    """
    Instantiate postprocessing charts.
    """
    def get_comparison_chart(data_name):
        """
        Gets ssp data, gets witness data, interpolates the former and instantiates the graph.
        """
        var_f_name = f"{namespace}.{CHARTS_DATA[data_name][VAR_NAME]}"
        column = CHARTS_DATA[data_name][COLUMN]
        witness_data = execution_engine.dm.get_value(var_f_name)[
            [column]].copy().rename(columns={column: WITNESS_SERIES_NAME})
        # [YEARS, column]].rename(columns={column: WITNESS_SERIES_NAME})
        witness_data[YEARS] = WITNESS_YEARS  # Not all witness vars include years
        ssp_data = get_ssp_data(data_name, CHARTS_DATA, region='World')
        for scenario in ssp_data.columns:
            if scenario != YEARS:
                f_interp = interp1d(ssp_data[YEARS], ssp_data[scenario])
                scenario_data = f_interp(witness_data[YEARS])
                witness_data[scenario] = scenario_data
        return get_comp_chart_from_df(witness_data, CHARTS_DATA[data_name][Y_AXIS], CHARTS_DATA[data_name][CHART_TITLE])

    instanciated_charts = []
    energy_list = execution_engine.dm.get_value(f'{namespace}.{GlossaryCore.energy_list}')
    # if renewable not in energy list, we are not in coarse 
    chart_l = CHART_LIST

    if 'renewable' not in energy_list:
        # if not in coarse, add primary energy chart
        chart_l = chart_l + [PRIMARY_ENERGY]

    graphs_list = chart_l
    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list = chart_filter.selected_values
            # if chart_filter.filter_key == 'Scenarios':
            #     selected_scenarios = chart_filter.selected_values

    _graphs_set = set(graphs_list)
    _primary_energy_chart = False
    if PRIMARY_ENERGY in _graphs_set:
        _primary_energy_chart = True
        _graphs_set.discard(PRIMARY_ENERGY)

    instanciated_charts.extend(map(get_comparison_chart, _graphs_set))
    if _primary_energy_chart:
        instanciated_charts.extend(get_ssp_primary_energy_charts())
        instanciated_charts.append(get_witness_primary_energy_chart(execution_engine, namespace))
    return instanciated_charts
