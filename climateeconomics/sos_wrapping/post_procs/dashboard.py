"""
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
"""

import numpy as np
import plotly.graph_objects as go
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from plotly.subplots import make_subplots
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
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
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    chart_filters = []

    chart_list = [
        "temperature and ghg evolution",
        "population and death",
        "gdp breakdown",
        "energy mix",
        "investment distribution",
        "land use",
    ]
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "Charts"))

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    """
    CROP_DISC = "Crop"
    LANDUSE_DISC = "Land_Use"
    ENERGYMIX_DISC = "EnergyMix"

    # execution_engine.dm.get_all_namespaces_from_var_name('temperature_df')[0]

    instanciated_charts = []
    chart_list = []
    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == "Charts":
                chart_list = chart_filter.selected_values

    if "temperature and ghg evolution" in chart_list:
        temperature_df = get_scenario_value(execution_engine, "temperature_detail_df", scenario_name)
        total_ghg_df = get_scenario_value(execution_engine, GlossaryCore.GHGEmissionsDfValue, scenario_name)
        carbon_captured = get_scenario_value(execution_engine, GlossaryEnergy.CarbonCapturedValue, scenario_name)
        co2_emissions = get_scenario_value(execution_engine, "co2_emissions_ccus_Gt", scenario_name)
        years = temperature_df[GlossaryEnergy.Years].values.tolist()

        chart_name = "Temperature and CO2 evolution over the years"

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=years,
                y=temperature_df[GlossaryCore.TempAtmo].values.tolist(),
                name="Temperature",
            ),
            secondary_y=True,
        )

        # Creating list of values according to CO2 storage limited by CO2 captured
        graph_gross_co2 = []
        graph_dac = []
        graph_flue_gas = []
        for year_index, year in enumerate(years):
            storage_limit = co2_emissions["carbon_storage Limited by capture (Gt)"][year_index]
            graph_gross_co2.append(total_ghg_df[f"Total CO2 emissions"][year_index] + storage_limit)
            captured_total = (
                carbon_captured["DAC"][year_index] * 0.001 + carbon_captured["flue gas"][year_index] * 0.001
            )
            if captured_total > 0.0:
                proportion_stockage = storage_limit / captured_total
                graph_dac.append(proportion_stockage * carbon_captured["DAC"][year_index] * 0.001)
                graph_flue_gas.append(proportion_stockage * carbon_captured["flue gas"][year_index] * 0.001)
            else:
                graph_dac.append(0)
                graph_flue_gas.append(0)

        fig.add_trace(
            go.Scatter(
                x=years,
                y=total_ghg_df[f"Total CO2 emissions"].to_list(),
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                fillcolor="rgba(200, 200, 200, 0.0)",
                name="Net CO2 emissions",
                stackgroup="one",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=graph_dac,
                name="CO2 captured by DAC and stored",
                stackgroup="one",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=graph_flue_gas,
                name="CO2 captured by flue gas and stored",
                stackgroup="one",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=graph_gross_co2,
                name="Total CO2 emissions",
            ),
            secondary_y=False,
        )

        fig.update_yaxes(
            title_text="Temperature evolution (degrees Celsius above preindustrial)",
            secondary_y=True,
            rangemode="tozero",
        )
        fig.update_yaxes(title_text=f"CO2 emissions [Gt]", rangemode="tozero", secondary_y=False)

        new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if "population and death" in chart_list:
        pop_df = get_scenario_value(execution_engine, "population_detail_df", scenario_name)
        death_dict = get_scenario_value(execution_engine, "death_dict", scenario_name)
        instanciated_charts = Population.graph_model_world_pop_and_cumulative_deaths(
            pop_df, death_dict, instanciated_charts
        )

    if "gdp breakdown" in chart_list:
        economics_df = get_scenario_value(execution_engine, GlossaryCore.EconomicsDetailDfValue, scenario_name)
        damage_df = get_scenario_value(execution_engine, GlossaryCore.DamageDetailedDfValue, scenario_name)
        compute_climate_impact_on_gdp = get_scenario_value(execution_engine, "assumptions_dict", scenario_name)[
            "compute_climate_impact_on_gdp"
        ]
        damages_to_productivity = (
            get_scenario_value(execution_engine, GlossaryCore.DamageToProductivity, scenario_name)
            and compute_climate_impact_on_gdp
        )
        new_chart = MacroEconomics.breakdown_gdp(
            economics_df, damage_df, compute_climate_impact_on_gdp, damages_to_productivity
        )
        instanciated_charts.append(new_chart)

    if "energy mix" in chart_list:
        energy_production_detailed = get_scenario_value(
            execution_engine, f"{ENERGYMIX_DISC}.{GlossaryEnergy.EnergyProductionDetailedValue}", scenario_name
        )
        energy_mean_price = get_scenario_value(execution_engine, GlossaryCore.EnergyMeanPriceValue, scenario_name)
        years = energy_production_detailed[GlossaryEnergy.Years].values.tolist()

        chart_name = "Net Energies production/consumption and mean price out of energy mix"

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for reactant in energy_production_detailed.columns:
            if (
                reactant not in [GlossaryEnergy.Years, GlossaryEnergy.TotalProductionValue, "Total production (uncut)"]
                and "carbon_capture" not in reactant
                and "carbon_storage" not in reactant
            ):
                energy_twh = energy_production_detailed[reactant].values
                legend_title = f"{reactant}".replace("(TWh)", "").replace("production", "")

                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=energy_twh.tolist(),
                        opacity=0.7,
                        line=dict(width=1.25),
                        name=legend_title,
                        stackgroup="one",
                    ),
                    secondary_y=False,
                )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=energy_mean_price[GlossaryEnergy.EnergyPriceValue].values.tolist(),
                name="Mean energy prices",
                # line=dict(color=qualitative.Set1[0]),
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Net Energy [TWh]", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Prices [$/MWh]", secondary_y=True, rangemode="tozero")

        new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    if "investment distribution" in chart_list:
        forest_investment = get_scenario_value(execution_engine, GlossaryEnergy.ForestInvestmentValue, scenario_name)
        years = forest_investment[GlossaryEnergy.Years]

        chart_name_energy = f"Distribution of investments on each energy "

        new_chart_energy = TwoAxesInstanciatedChart(
            GlossaryEnergy.Years, "Invest [G$]", chart_name=chart_name_energy, stacked_bar=True
        )
        energy_list = get_scenario_value(execution_engine, GlossaryCore.energy_list, scenario_name)
        ccs_list = get_scenario_value(execution_engine, GlossaryCore.ccs_list, scenario_name)

        new_chart_energy = new_chart_energy.to_plotly()

        # add a chart per energy with breakdown of investments in every technology of the energy
        for energy in energy_list + ccs_list:
            list_energy = []
            if energy != BiomassDry.name:
                techno_list = get_scenario_value(
                    execution_engine, f"{energy}.{GlossaryEnergy.TechnoListName}", scenario_name
                )
                for techno in techno_list:
                    invest_level = get_scenario_value(
                        execution_engine, f"{energy}.{techno}.{GlossaryEnergy.InvestLevelValue}", scenario_name
                    )
                    list_energy.append(invest_level[f"{GlossaryEnergy.InvestValue}"].values)

                total_invest = list(np.sum(list_energy, axis=0))
                new_chart_energy.add_trace(
                    go.Scatter(
                        x=years.tolist(),
                        y=total_invest,
                        opacity=0.7,
                        line=dict(width=1.25),
                        name=energy,
                        stackgroup="one",
                    )
                )

        new_chart_energy = InstantiatedPlotlyNativeChart(fig=new_chart_energy, chart_name=chart_name_energy)

        instanciated_charts.append(new_chart_energy)

    if "land use" in chart_list:
        chart_name = "Surface for forest and food production vs available land over time"
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, "Surface [Gha]", chart_name=chart_name, stacked_bar=True
        )

        new_chart = new_chart.to_plotly()

        # total crop surface
        surface_df = get_scenario_value(execution_engine, f"{CROP_DISC}.food_land_surface_df", scenario_name)
        years = surface_df[GlossaryCore.Years].values.tolist()
        for key in surface_df.keys():
            if key == GlossaryCore.Years:
                pass
            elif key.startswith("total"):
                pass
            else:
                new_chart.add_trace(
                    go.Scatter(
                        x=years,
                        y=(surface_df[key]).values.tolist(),
                        opacity=0.7,
                        line=dict(width=1.25),
                        name=key,
                        stackgroup="one",
                    )
                )

        # total food and forest surface, food should be at the bottom to be compared with crop surface
        land_surface_detailed = get_scenario_value(
            execution_engine, f"{LANDUSE_DISC}.{LandUseV2.LAND_SURFACE_DETAIL_DF}", scenario_name
        )
        column = "Forest Surface (Gha)"
        legend = column.replace(" (Gha)", "")
        new_chart.add_trace(
            go.Scatter(
                x=years,
                y=(land_surface_detailed[column]).values.tolist(),
                opacity=0.7,
                line=dict(width=1.25),
                name=legend,
                stackgroup="one",
            )
        )

        column = "total surface (Gha)"
        legend = column.replace(" (Gha)", "")
        new_chart.add_trace(
            go.Scatter(
                x=years,
                y=(surface_df[column]).values.tolist(),
                mode="lines",
                name=legend,
            )
        )

        # total land available
        total_land_available = list(
            land_surface_detailed["Available Agriculture Surface (Gha)"].values
            + land_surface_detailed["Available Forest Surface (Gha)"].values
            + land_surface_detailed["Available Shrub Surface (Gha)"]
        )

        # shrub surface cannot be <0
        shrub_surface = np.maximum(
            np.zeros(len(years)),
            total_land_available[0] * np.ones(len(years))
            - (land_surface_detailed["Total Forest Surface (Gha)"] + surface_df["total surface (Gha)"]).values,
        )

        column = "Shrub Surface (Gha)"
        legend = column.replace(" (Gha)", "")
        new_chart.add_trace(
            go.Scatter(
                x=years,
                y=(list(shrub_surface)),
                opacity=0.7,
                line=dict(width=1.25),
                name=legend,
                stackgroup="one",
            )
        )

        new_chart.add_trace(
            go.Scatter(
                x=years,
                y=list(np.ones(len(years)) * total_land_available),
                mode="lines",
                name="Total land available",
            )
        )

        new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

        instanciated_charts.append(new_chart)

    return instanciated_charts
