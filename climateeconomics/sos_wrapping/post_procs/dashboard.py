'''
Copyright 2023 Capgemini

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

from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from plotly.subplots import make_subplots
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

import climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline as MacroEconomics
import climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline as Population
from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ["gdp vs energy evolution", 'temperature and ghg evolution', 'population and death', 'gdp breakdown',
                  'energy mix', 'investment distribution', 'land use',
                  'KPI1', 'KPI2', 'KPI4', 'KPI5', 'KPI6']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'Charts'))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    CROP_DISC = 'Crop'
    LANDUSE_DISC = 'Land Use'
    ENERGYMIX_DISC = 'EnergyMix'
    DAMAGE_DISC = 'Damage'

    sectorization: bool = len(execution_engine.dm.get_all_namespaces_from_var_name(
        f"{GlossaryEnergy.SectorServices}.{GlossaryEnergy.DamageDetailedDfValue}")) > 0
    # execution_engine.dm.get_all_namespaces_from_var_name('temperature_df')[0]

    instanciated_charts = []
    chart_list = []
    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts':
                chart_list = chart_filter.selected_values

    if "gdp vs energy evolution" in chart_list:

        raw_data_dict = {
            "WITNESS": {
                "data_type": "variable",
                "scenario_name": scenario_name,
                "x_var_name": "EnergyMix.energy_production_brut",
                "x_column_name": GlossaryCore.TotalProductionValue,
                "x_data_scale": 1e-3,
                "y_var_name": "Macroeconomics.economics_detail_df",
                "y_column_name": "output_net_of_d",
                "text_column": GlossaryCore.Years,
            },
            "Historical Data": {
                "data_type": "csv",
                "filename": join(Path(__file__).parents[2], "data", 'primary-energy-consumption_vs_gdp.csv'),
                "x_column_name": 'Primary energy consumption [PWh]',
                "y_column_name": 'World GDP [T$]',
                "marker_symbol": "triangle-up",
                "text_column": "years",
            }
        }

        net_data_dict = {
            "WITNESS": {
                "data_type": "variable",
                "scenario_name": scenario_name,
                "x_var_name": f"EnergyMix.{GlossaryEnergy.EnergyProductionValue}",
                "x_column_name": GlossaryCore.TotalProductionValue,
                "y_var_name": "Macroeconomics.economics_detail_df",
                "y_column_name": "output_net_of_d",
                "text_column": GlossaryCore.Years,
            },
            "Historical Data": {
                "data_type": "csv",
                "filename": join(Path(__file__).parents[2], "data", 'world_gdp_vs_net_energy_consumption.csv'),
                "x_column_name": 'Net energy consumption [PWh]',
                "y_column_name": 'World GDP [T$]',
                "marker_symbol": "triangle-up",
                "text_column": "years",
            }
        }

        new_chart = create_xy_chart(execution_engine, chart_name="GDP vs Raw energy production",
                                    x_axis_name="World's raw energy production (PWh)",
                                    y_axis_name="World's GDP net of damage (T$)", data_dict=raw_data_dict)
        new_chart.post_processing_section_name = "Key performance indicators"
        instanciated_charts.append(new_chart)

        new_chart = create_xy_chart(execution_engine, chart_name="GDP vs Net energy production",
                                    x_axis_name="World's net energy production (PWh)",
                                    y_axis_name="World's GDP net of damage (T$)", data_dict=net_data_dict)
        new_chart.post_processing_section_name = "Key performance indicators"
        instanciated_charts.append(new_chart)

    if 'temperature and ghg evolution' in chart_list:
        temperature_df = get_scenario_value(execution_engine, GlossaryCore.TemperatureDfValue, scenario_name)
        total_ghg_df = get_scenario_value(execution_engine, GlossaryCore.GHGEmissionsDfValue, scenario_name)
        carbon_captured = get_scenario_value(execution_engine, GlossaryEnergy.CarbonCapturedValue, scenario_name)
        co2_emissions = get_scenario_value(execution_engine, 'co2_emissions_ccus_Gt', scenario_name)
        years = temperature_df[GlossaryEnergy.Years].values.tolist()

        chart_name = 'Temperature and CO2 evolution over the years'

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=years,
            y=temperature_df[GlossaryCore.TempAtmo].values.tolist(),
            name='Temperature',
        ), secondary_y=True)

        # Creating list of values according to CO2 storage limited by CO2 captured
        graph_gross_co2 = []
        graph_dac = []
        graph_flue_gas = []
        for year_index, year in enumerate(years):
            storage_limit = co2_emissions['carbon_storage Limited by capture (Gt)'][year_index]
            graph_gross_co2.append(total_ghg_df['Total CO2 emissions'][year_index] + storage_limit)
            captured_total = carbon_captured['DAC'][year_index] * 0.001 + carbon_captured['flue gas'][
                year_index] * 0.001
            if captured_total > 0.0:
                proportion_stockage = storage_limit / captured_total
                graph_dac.append(proportion_stockage * carbon_captured['DAC'][year_index] * 0.001)
                graph_flue_gas.append(proportion_stockage * carbon_captured['flue gas'][year_index] * 0.001)
            else:
                graph_dac.append(0)
                graph_flue_gas.append(0)

        fig.add_trace(go.Scatter(
            x=years,
            y=total_ghg_df['Total CO2 emissions'].to_list(),
            fill='tonexty',  # fill area between trace0 and trace1
            mode='lines',
            fillcolor='rgba(200, 200, 200, 0.0)',
            name='Net CO2 emissions',
            stackgroup='one',
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=years,
            y=graph_dac,
            name='CO2 captured by DAC and stored',
            stackgroup='one',
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=years,
            y=graph_flue_gas,
            name='CO2 captured by flue gas and stored',
            stackgroup='one',
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=years,
            y=graph_gross_co2,
            name='Total CO2 emissions',
        ), secondary_y=False)

        fig.update_yaxes(title_text='Temperature evolution (degrees Celsius above preindustrial)', secondary_y=True,
                         rangemode="tozero")
        fig.update_yaxes(title_text='CO2 emissions [Gt]', rangemode="tozero", secondary_y=False)

        new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if 'population and death' in chart_list:
        pop_df = get_scenario_value(execution_engine, 'population_detail_df', scenario_name)
        death_dict = get_scenario_value(execution_engine, 'death_dict', scenario_name)
        instanciated_charts = Population.graph_model_world_pop_and_cumulative_deaths(pop_df, death_dict,
                                                                                     instanciated_charts)

    if 'gdp breakdown' in chart_list:
        economics_df = get_scenario_value(execution_engine, f'Macroeconomics.{GlossaryCore.EconomicsDetailDfValue}', scenario_name)

        compute_climate_impact_on_gdp = get_scenario_value(execution_engine, 'assumptions_dict', scenario_name)[
            'compute_climate_impact_on_gdp']
        damages_to_productivity = get_scenario_value(execution_engine, GlossaryCore.DamageToProductivity,
                                                     scenario_name) and compute_climate_impact_on_gdp
        if sectorization:
            damage_df = get_scenario_value(execution_engine, f"Macroeconomics.{GlossaryCore.DamageDetailedDfValue}",
                                           scenario_name)
            df_non_energy_invest = get_scenario_value(execution_engine, GlossaryCore.RedistributionInvestmentsDfValue,
                                                      scenario_name)
            df_energy_invest = get_scenario_value(execution_engine, GlossaryCore.EnergyInvestmentsWoTaxValue,
                                                  scenario_name)
            df_consumption = get_scenario_value(execution_engine, "consumption_detail_df", scenario_name)
            economics_df = complete_economics_df_for_sectorization(economics_df, df_non_energy_invest, df_energy_invest,
                                                                   df_consumption)
        else:
            damage_df = get_scenario_value(execution_engine, f"Macroeconomics.{GlossaryCore.DamageDetailedDfValue}", scenario_name)
        new_chart = MacroEconomics.breakdown_gdp(economics_df, damage_df, compute_climate_impact_on_gdp,
                                                 damages_to_productivity)
        instanciated_charts.append(new_chart)

    if 'energy mix' in chart_list:
        energy_production_detailed = get_scenario_value(execution_engine,
                                                        f'{ENERGYMIX_DISC}.{GlossaryEnergy.StreamProductionDetailedValue}',
                                                        scenario_name)
        energy_mean_price = get_scenario_value(execution_engine, GlossaryCore.EnergyMeanPriceValue, scenario_name)
        years = energy_production_detailed[GlossaryEnergy.Years].values.tolist()

        chart_name = 'Net Energies production/consumption and mean price out of energy mix'

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for reactant in energy_production_detailed.columns:
            if reactant not in [GlossaryEnergy.Years, GlossaryEnergy.TotalProductionValue, 'Total production (uncut)'] \
                    and GlossaryEnergy.carbon_capture not in reactant \
                    and GlossaryEnergy.carbon_storage not in reactant:
                energy_twh = energy_production_detailed[reactant].values
                legend_title = f'{reactant}'.replace(
                    "(TWh)", "").replace('production', '')

                fig.add_trace(go.Scatter(
                    x=years,
                    y=energy_twh.tolist(),
                    opacity=0.7,
                    line=dict(width=1.25),
                    name=legend_title,
                    stackgroup='one',
                ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=years,
            y=energy_mean_price[GlossaryEnergy.EnergyPriceValue].values.tolist(),
            name='Mean energy prices',
            # line=dict(color=qualitative.Set1[0]),
        ), secondary_y=True)

        fig.update_yaxes(title_text="Net Energy [TWh]", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Prices [$/MWh]", secondary_y=True, rangemode="tozero")

        new_chart = InstantiatedPlotlyNativeChart(
            fig=fig, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if 'investment distribution' in chart_list:
        forest_investment = get_scenario_value(execution_engine, GlossaryEnergy.ReforestationInvestmentValue, scenario_name)
        years = forest_investment[GlossaryEnergy.Years]

        chart_name_energy = 'Distribution of investments on each energy '

        new_chart_energy = TwoAxesInstanciatedChart(GlossaryEnergy.Years, 'Invest [G$]',
                                                    chart_name=chart_name_energy, stacked_bar=True)
        energy_list = get_scenario_value(execution_engine, GlossaryCore.energy_list, scenario_name)
        ccs_list = get_scenario_value(execution_engine, GlossaryCore.ccs_list, scenario_name)

        new_chart_energy = new_chart_energy.to_plotly()

        # add a chart per energy with breakdown of investments in every technology of the energy
        for energy in energy_list + ccs_list:
            list_energy = []
            if energy != BiomassDry.name:
                techno_list = get_scenario_value(execution_engine, f'{energy}.{GlossaryEnergy.TechnoListName}',
                                                 scenario_name)
                for techno in techno_list:
                    invest_level = get_scenario_value(execution_engine,
                                                      f'{energy}.{techno}.{GlossaryEnergy.InvestLevelValue}',
                                                      scenario_name)
                    list_energy.append(invest_level[f'{GlossaryEnergy.InvestValue}'].values)

                total_invest = list(np.sum(list_energy, axis=0))
                new_chart_energy.add_trace(go.Scatter(
                    x=years.tolist(),
                    y=total_invest,
                    opacity=0.7,
                    line=dict(width=1.25),
                    name=energy,
                    stackgroup='one',
                ))

        new_chart_energy = InstantiatedPlotlyNativeChart(fig=new_chart_energy, chart_name=chart_name_energy)

        instanciated_charts.append(new_chart_energy)

    if 'land use' in chart_list:
        chart_name = 'Surface for forest and food production vs available land over time'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Surface [Gha]',
                                             chart_name=chart_name, stacked_bar=True)

        new_chart = new_chart.to_plotly()

        # total crop surface
        surface_df = get_scenario_value(execution_engine, f'{CROP_DISC}.food_land_surface_df', scenario_name)
        years = surface_df[GlossaryCore.Years].values.tolist()
        for key in surface_df.keys():
            if key == GlossaryCore.Years:
                pass
            elif key.startswith('total'):
                pass
            else:
                new_chart.add_trace(go.Scatter(
                    x=years,
                    y=(surface_df[key]).values.tolist(),
                    opacity=0.7,
                    line=dict(width=1.25),
                    name=key,
                    stackgroup='one',
                ))

        # total food and forest surface, food should be at the bottom to be compared with crop surface
        land_surface_detailed = get_scenario_value(execution_engine,
                                                   f'{LANDUSE_DISC}.{LandUseV2.LAND_SURFACE_DETAIL_DF}', scenario_name)
        column = 'Forest Surface (Gha)'
        legend = column.replace(' (Gha)', '')
        new_chart.add_trace(go.Scatter(
            x=years,
            y=(land_surface_detailed[column]).values.tolist(),
            opacity=0.7,
            line=dict(width=1.25),
            name=legend,
            stackgroup='one',
        ))

        column = 'total surface (Gha)'
        legend = column.replace(' (Gha)', '')
        new_chart.add_trace(go.Scatter(
            x=years,
            y=(surface_df[column]).values.tolist(),
            mode='lines',
            name=legend,
        ))

        # total land available
        total_land_available = list(land_surface_detailed['Available Agriculture Surface (Gha)'].values +
                                    land_surface_detailed['Available Forest Surface (Gha)'].values +
                                    land_surface_detailed['Available Shrub Surface (Gha)'])

        # shrub surface cannot be <0
        shrub_surface = np.maximum(np.zeros(len(years)), total_land_available[0] * np.ones(len(years)) -
                                   (land_surface_detailed['Total Forest Surface (Gha)'] +
                                    surface_df['total surface (Gha)']).values)

        column = 'Shrub Surface (Gha)'
        legend = column.replace(' (Gha)', '')
        new_chart.add_trace(go.Scatter(
            x=years,
            y=(list(shrub_surface)),
            opacity=0.7,
            line=dict(width=1.25),
            name=legend,
            stackgroup='one',
        ))

        new_chart.add_trace(go.Scatter(
            x=years,
            y=list(np.ones(len(years)) * total_land_available),
            mode='lines',
            name='Total land available',
        ))

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if 'KPI1' in chart_list:
        # KPI1 is the clean energy growth rate, ie the PWh added in 10 years
        # clean technologies dictionary
        green_energies_and_technos = {f'{GlossaryEnergy.heat}.{GlossaryEnergy.hightemperatureheat}':
                                          [GlossaryEnergy.GeothermalHighHeat, GlossaryEnergy.HeatPumpHighHeat],
                                      f'{GlossaryEnergy.heat}.{GlossaryEnergy.mediumtemperatureheat}':
                                          [GlossaryEnergy.GeothermalMediumHeat, GlossaryEnergy.HeatPumpMediumHeat],
                                      f'{GlossaryEnergy.heat}.{GlossaryEnergy.lowtemperatureheat}':
                                          [GlossaryEnergy.GeothermalLowHeat, GlossaryEnergy.HeatPumpLowHeat],
                                      GlossaryEnergy.electricity: [GlossaryEnergy.Geothermal,
                                                                   GlossaryEnergy.Nuclear,
                                                                   GlossaryEnergy.RenewableElectricitySimpleTechno,
                                                                   GlossaryEnergy.SolarPv, GlossaryEnergy.SolarThermal,
                                                                   GlossaryEnergy.WindOffshore,
                                                                   GlossaryEnergy.WindOnshore],
                                      GlossaryEnergy.clean_energy: [GlossaryEnergy.CleanEnergySimpleTechno]}

        # dataframe of energy production by energy in TWh
        energy_production_detailed = get_scenario_value(execution_engine,
                                                        f'{ENERGYMIX_DISC}.{GlossaryEnergy.StreamProductionDetailedValue}',
                                                        scenario_name)
        years = energy_production_detailed[GlossaryEnergy.Years].values.tolist()

        energy_list = get_scenario_value(execution_engine, GlossaryCore.energy_list, scenario_name)

        # creation of clean technologies dataframe
        clean_energy_df = pd.DataFrame()
        clean_energy_df[GlossaryEnergy.Years] = energy_production_detailed[GlossaryEnergy.Years]

        for technos in green_energies_and_technos.values():
            for techno in technos:
                clean_energy_df[techno] = 0.0

        # for energy in green_energies_and_technos:
        #     for techno in green_energies_and_technos[energy]:
        #         clean_energy_df[techno] = 0.0

        # getting energy production values for clean technologies
        for energy in energy_list:
            if energy in green_energies_and_technos:
                techno_list = get_scenario_value(execution_engine, f'{energy}.{GlossaryEnergy.TechnoListName}',
                                                 scenario_name)
                energy_production_df = get_scenario_value(execution_engine,
                                                          f'{energy}.{GlossaryEnergy.StreamProductionDetailedValue}',
                                                          scenario_name)
                for techno in techno_list:
                    if techno in green_energies_and_technos[energy]:
                        clean_energy_df[techno] += energy_production_df[f'{energy} {techno} (TWh)']

        # total clean energy production
        clean_energy_df['Total'] = clean_energy_df.drop(columns=GlossaryEnergy.Years).sum(axis=1)

        # creation of clean energy growth dataframe for 10 years intervals
        clean_energy_growth_df = pd.DataFrame()
        clean_energy_growth_df[GlossaryEnergy.Years] = None
        clean_energy_growth_df['clean energy growth (PWh)'] = None
        year_start = clean_energy_df[GlossaryEnergy.Years].iloc[0]
        year_end = clean_energy_df[GlossaryEnergy.Years].iloc[-1]
        year = year_start
        # computing growth in an interval
        while (year + 9) <= year_end:
            growth = 0
            # growth between two consecutive years
            for i in range(0, 10):
                value1 = clean_energy_df.loc[clean_energy_df[GlossaryEnergy.Years] == year + i, 'Total'].values[0]
                value2 = clean_energy_df.loc[clean_energy_df[GlossaryEnergy.Years] == year + i + 1, 'Total'].values[0]
                growth = growth + value2 - value1
            # growth is in TWh so it needs to be converted into PWh
            growth /= 1000
            # applying an equal share of the growth to each year in the interval
            year_growth = growth / 10
            for i in range(0, 10):
                year_interval = pd.DataFrame(
                    {GlossaryEnergy.Years: [year + i], 'clean energy growth (PWh)': year_growth})
                clean_energy_growth_df = pd.concat([clean_energy_growth_df, year_interval], ignore_index=True)
            # switching to next 10 year interval
            year += 10

        # default value is 13 PWh
        clean_energy_growth_df['default (PWh)'] = 13

        chart_name = 'Clean energy growth'

        new_chart = TwoAxesInstanciatedChart('years intervals', 'PWh added in 10 years',
                                             chart_name=chart_name, stacked_bar=True,
                                             y_min_zero=False)

        new_chart = new_chart.to_plotly()

        years_intervals = clean_energy_growth_df[GlossaryEnergy.Years].to_list()
        computed_data = clean_energy_growth_df['clean energy growth (PWh)'].to_list()

        new_chart.add_trace(go.Scatter(
            x=years_intervals,
            y=computed_data,
            name='clean energy growth'
        ))

        new_chart.add_trace(go.Scatter(
            x=list(range(2020, 2051)),
            y=[13] * len(range(2020, 2051)),
            mode='lines',
            line=dict(
                dash='dash',
                width=2
            ),
            name='default value (2020-2050) - Y. Caseau, CCEM 2024',
        ))

        new_chart.add_trace(go.Scatter(
            x=list(range(2020, 2051)),
            y=[25] * len(range(2020, 2051)),
            mode='lines',
            line=dict(
                dash='dash',
                width=2
            ),
            name='default value (2020-2050) - IRENA 1.5°C scenario',
        ))

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        new_chart.post_processing_section_name = "Key performance indicators"

        instanciated_charts.append(new_chart)

    if 'KPI2' in chart_list:
        # KPI2 is the energy efficiency, ie the variation of GDP/TotalEnergyProduction
        #  dataframe of energy production in PWh
        energy_production = get_scenario_value(execution_engine,
                                               f'{ENERGYMIX_DISC}.{GlossaryEnergy.EnergyProductionValue}',
                                               scenario_name)
        years = energy_production[GlossaryEnergy.Years].values.tolist()
        gdp = get_scenario_value(execution_engine, GlossaryCore.EconomicsDfValue, scenario_name)
        gdp = gdp.reset_index(drop=True)
        energy_efficiency = pd.DataFrame()
        energy_efficiency[GlossaryEnergy.Years] = years
        energy_efficiency['energy efficiency'] = gdp[GlossaryCore.OutputNetOfDamage] / energy_production[
            GlossaryEnergy.TotalProductionValue]
        energy_efficiency['variation'] = 0
        # computing variation of energy efficiency
        for i in range(1, len(years)):
            previous_year_efficiency = energy_efficiency.loc[i - 1, 'energy efficiency']
            current_year_efficiency = energy_efficiency.loc[i, 'energy efficiency']
            energy_efficiency.loc[i, 'variation'] = (
                                                            current_year_efficiency - previous_year_efficiency) / previous_year_efficiency * 100

        chart_name = "Variation of energy efficiency"

        new_chart = TwoAxesInstanciatedChart(GlossaryEnergy.Years,
                                             'variation of GDP / energy production (%)',
                                             chart_name=chart_name)

        new_chart = new_chart.to_plotly()

        new_chart.add_trace(go.Scatter(
            x=years,
            y=energy_efficiency['variation'].to_list(),
            name="variation of energy efficiency",
        ))

        # default value is 1.2%
        new_chart.add_trace(go.Scatter(
            x=list(range(2020, 2051)),
            y=[1.2] * len(range(2020, 2051)),
            mode='lines',
            line=dict(
                dash='dash',
                width=2
            ),
            name="default value (2020-2050) - Y. Caseau, CCEM 2024",
        ))

        new_chart.add_trace(go.Scatter(
            x=list(range(2020, 2051)),
            y=[2.7] * len(range(2020, 2051)),
            mode='lines',
            line=dict(
                dash='dash',
                width=2
            ),
            name="default value (2020-2050) - IRENA 1.5°C scenario",
        ))

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        new_chart.post_processing_section_name = "Key performance indicators"

        instanciated_charts.append(new_chart)

    if 'KPI4' in chart_list:
        # KPI4 is the electrification of energy, ie ElectricityProduction/TotalEnergyProduction
        # dataframe of energy production by energy in TWh
        energy_production_detailed = get_scenario_value(execution_engine,
                                                        f'{ENERGYMIX_DISC}.{GlossaryEnergy.StreamProductionDetailedValue}',
                                                        scenario_name)
        years = energy_production_detailed[GlossaryEnergy.Years].values.tolist()
        if f'production {GlossaryEnergy.electricity} (TWh)' in energy_production_detailed.columns:
            energy_electrification = pd.DataFrame()
            energy_electrification[GlossaryEnergy.Years] = years
            energy_electrification['value'] = energy_production_detailed[
                                                  f'production {GlossaryEnergy.electricity} (TWh)'] / \
                                              energy_production_detailed[
                                                  f'{GlossaryEnergy.TotalProductionValue} (uncut)'] * 100

            chart_name = "Electrification of energy"

            new_chart = TwoAxesInstanciatedChart(GlossaryEnergy.Years,
                                                 'electricity production / energy production (%)',
                                                 chart_name=chart_name)

            new_chart = new_chart.to_plotly()

            new_chart.add_trace(go.Scatter(
                x=years,
                y=energy_electrification['value'].to_list(),
                name="electrification of energy",
            ))

            # default values
            new_chart.add_trace(go.Scatter(
                x=[2020, 2050],
                y=[16, 48],
                mode='lines',
                line=dict(
                    dash='dash',
                    width=2
                ),
                name="default values (2020 & 2050) - Y. Caseau, CCEM 2024",
            ))

            new_chart.add_trace(go.Scatter(
                x=list(range(2020, 2051)),
                y=[80] * len(range(2020, 2051)),
                mode='lines',
                line=dict(
                    dash='dash',
                    width=2
                ),
                name="default values (2020 & 2050) - IRENA 1.5°C scenario",
            ))

            new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

            new_chart.post_processing_section_name = "Key performance indicators"

            instanciated_charts.append(new_chart)

    if 'KPI5' in chart_list:
        # KPI5 is the return on investment, ie GDPVariation/Investment
        economics_detailed_df = get_scenario_value(execution_engine, f'Macroeconomics.{GlossaryCore.EconomicsDetailDfValue}', scenario_name)
        if sectorization:
            df_non_energy_invest = get_scenario_value(execution_engine, GlossaryCore.RedistributionInvestmentsDfValue,
                                                      scenario_name)
            df_energy_invest = get_scenario_value(execution_engine, GlossaryCore.EnergyInvestmentsWoTaxValue,
                                                  scenario_name)
            df_consumption = get_scenario_value(execution_engine, "consumption_detail_df", scenario_name)
            economics_detailed_df = complete_economics_df_for_sectorization(economics_detailed_df, df_non_energy_invest,
                                                                            df_energy_invest, df_consumption)
        economics_detailed_df = economics_detailed_df.reset_index(drop=True)
        years = economics_detailed_df[GlossaryCore.Years].values.tolist()
        roi = pd.DataFrame()
        roi[GlossaryCore.Years] = years
        roi['yearly_gdp_variation'] = 0
        # computing GDP variation
        for i in range(1, len(years)):
            previous_year_gdp = economics_detailed_df.loc[i - 1, GlossaryCore.OutputNetOfDamage]
            current_year_gdp = economics_detailed_df.loc[i, GlossaryCore.OutputNetOfDamage]
            roi.loc[i, 'yearly_gdp_variation'] = current_year_gdp - previous_year_gdp
        roi['value'] = roi['yearly_gdp_variation'] / economics_detailed_df[GlossaryCore.InvestmentsValue] * 100

        chart_name = "Return on Investment"

        new_chart = TwoAxesInstanciatedChart(GlossaryEnergy.Years,
                                             'GDP variation / investments (%)',
                                             chart_name=chart_name)

        new_chart = new_chart.to_plotly()

        new_chart.add_trace(go.Scatter(
            x=years,
            y=roi['value'].to_list(),
            name="return on Investment",
        ))

        # default value is 9.3%
        new_chart.add_trace(go.Scatter(
            x=list(range(2020, 2051)),
            y=[9.3] * len(range(2020, 2051)),
            mode='lines',
            line=dict(
                dash='dash',
                width=2
            ),
            name="default value (2020-2050) - Y. Caseau, CCEM 2024",
        ))

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        new_chart.post_processing_section_name = "Key performance indicators"

        instanciated_charts.append(new_chart)

    if 'KPI6' in chart_list:
        # KPI6 is the global warming impact, ie the damages as % of GDP at tipping point 3°C
        tipping_point_model = get_scenario_value(execution_engine, f'{DAMAGE_DISC}.tipping_point', scenario_name)
        tp_a1 = get_scenario_value(execution_engine, f'{DAMAGE_DISC}.tp_a1', scenario_name)
        tp_a2 = get_scenario_value(execution_engine, f'{DAMAGE_DISC}.tp_a2', scenario_name)
        tp_a3 = get_scenario_value(execution_engine, f'{DAMAGE_DISC}.tp_a3', scenario_name)
        tp_a4 = get_scenario_value(execution_engine, f'{DAMAGE_DISC}.tp_a4', scenario_name)

        def damage_fraction(damage):
            return damage / (1 + damage) * 100

        def damage_function_tipping_point_weitzmann(temp_increase):
            return (temp_increase / tp_a1) ** tp_a2 + (temp_increase / tp_a3) ** tp_a4

        temperature_increase = 3

        value = damage_fraction(damage_function_tipping_point_weitzmann(temperature_increase))

        chart_name = "Global warming impact"

        new_chart = TwoAxesInstanciatedChart('Temperature increase (°C)',
                                             'Impact on GDP (%)',
                                             chart_name=chart_name)

        new_chart = new_chart.to_plotly()

        new_chart.add_trace(go.Bar(
            x=[3],
            y=[value],
            opacity=1,
            name="tipping point damage model (Weitzman, 2009)" + ' (selected model)' * tipping_point_model,
        ))

        # default value
        new_chart.add_trace(go.Bar(
            x=[2.6],
            y=[6.7],
            opacity=0.5,
            name="default value - Y. Caseau, CCEM 2024",
        ))

        new_chart.add_trace(go.Bar(
            x=[3],
            y=[8],
            opacity=0.5,
            name="default value - Schroders",
        ))

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        new_chart.post_processing_section_name = "Key performance indicators"

        instanciated_charts.append(new_chart)

    return instanciated_charts


def complete_economics_df_for_sectorization(economics_detail_df, df_non_energy_invest, df_energy_invest,
                                            df_consumption):
    economics_detail_df_completed = economics_detail_df.copy()
    economics_detail_df_completed[GlossaryCore.Consumption] = df_consumption[GlossaryCore.Consumption].values
    economics_detail_df_completed[GlossaryCore.NonEnergyInvestmentsValue] = df_non_energy_invest[
        GlossaryCore.InvestmentsValue].values
    economics_detail_df_completed[GlossaryCore.EnergyInvestmentsValue] = df_energy_invest[
        GlossaryCore.EnergyInvestmentsWoTaxValue].values
    economics_detail_df_completed[GlossaryCore.InvestmentsValue] = economics_detail_df_completed[
                                                                       GlossaryCore.EnergyInvestmentsValue] + \
                                                                   economics_detail_df_completed[
                                                                       GlossaryCore.NonEnergyInvestmentsValue]

    return economics_detail_df_completed


def create_xy_chart(execution_engine, chart_name, x_axis_name,
                    y_axis_name, data_dict, **kwargs) -> TwoAxesInstanciatedChart:
    """Create XY chart from data dictionary"""
    new_chart = TwoAxesInstanciatedChart(
        x_axis_name,
        y_axis_name,
        chart_name=chart_name,
        **kwargs
    )

    for data_name, data in data_dict.items():

        x_data_df = None
        y_data_df = None

        if data["data_type"] == "variable":
            x_data_df = get_scenario_value(execution_engine, data["x_var_name"], data["scenario_name"], split_scenario_name=False)
            y_data_df = get_scenario_value(execution_engine, data["y_var_name"], data["scenario_name"], split_scenario_name=False)
        elif data["data_type"] == "csv":
            data_df = pd.read_csv(data["filename"])
            x_data_df = y_data_df = data_df

        if x_data_df is not None and y_data_df is not None:
            new_series = InstanciatedSeries(
                x_data_df[data["x_column_name"]] * data.get("x_data_scale", 1.0),
                y_data_df[data["y_column_name"]] * data.get("y_data_scale", 1.0),
                data_name, display_type="scatter",
                marker_symbol=data.get("marker_symbol", "circle"),
                text=y_data_df[data["text_column"]].values.tolist(),
                **data.get("kwargs", {})
            )
            new_chart.add_series(new_series)

    return new_chart
