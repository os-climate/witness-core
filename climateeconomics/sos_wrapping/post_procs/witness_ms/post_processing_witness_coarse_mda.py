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

import logging
from math import floor
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from energy_models.core.energy_mix.energy_mix import EnergyMix
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_chart_template import (
    SeriesTemplate,
)
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorized_ms_optim_process._usecase import (
    Study as study_ms_mdo_sect,
)
from climateeconomics.sos_wrapping.post_procs.dashboard import create_xy_chart

TAX_NAME = 'with tax'
DAMAGE_NAME = 'with damage'
DAMAGE_AND_TAX_NAME = 'with damage and tax'
ALL_SCENARIOS = 'all filtered scenarios'
effects_list = [ALL_SCENARIOS, TAX_NAME, DAMAGE_NAME, DAMAGE_AND_TAX_NAME]
EFFECT_NAME = 'Effects on filtered scenarios'
CHART_NAME = 'Charts'
END_YEAR_NAME = 'Ending year'
SCENARIO_NAME = 'Scenarios'
SCATTER_SCENARIO = 'mda_scenarios'
TIPPING_POINT = "Tipping point"
SEP = ' '
UNIT = "deg C"
# list of graphs to be plotted and filtered
graphs_list = ['GDP vs Energy',
               'Temperature',
               'CO2 emissions',
               'Population',
               'Cumulative climate deaths',
               'GDP',
               'Investements in energy',
               'Invest in Energy + CCUS',
               'CO2 tax',
               'Utility',
               'Total energy production',
               'Fossil production',
               'Clean energy production',
               'Mean energy price',
               'Consumption'
               ]


def get_shared_value(execution_engine, short_name_var: str):
    """returns the value of a variables common to all scenarios"""
    var_full_name = execution_engine.dm.get_all_namespaces_from_var_name(short_name_var)[0]
    value = execution_engine.dm.get_value(var_full_name)
    return value, var_full_name


def get_all_scenarios_values(execution_engine, short_name_var: str):
    var_full_names = execution_engine.dm.get_all_namespaces_from_var_name(short_name_var)
    values = {var_full_name: execution_engine.dm.get_value(var_full_name) for var_full_name in var_full_names}
    return values


def post_processing_filters(execution_engine, namespace):

    filters = []

    samples_df, varfullname_samples_df = get_shared_value(execution_engine, 'samples_df')
    scenario_list = samples_df['scenario_name'].tolist()

    # recover year start and year end arbitrarily from the first scenario
    year_start, _ = get_shared_value(execution_engine, GlossaryCore.YearStart)
    year_end, _ = get_shared_value(execution_engine, GlossaryCore.YearEnd)
    years_list = np.arange(year_start, year_end + 1).tolist()

    filters.append(ChartFilter(CHART_NAME, graphs_list, graphs_list, CHART_NAME))
    filters.append(ChartFilter(END_YEAR_NAME, years_list, year_end, END_YEAR_NAME, multiple_selection=False))  # by default shows all years
    # filter on effects applies on the list of scenarios already filtered (ie it's a logical AND between the filters)
    filters.append(ChartFilter(SCENARIO_NAME, scenario_list,
                               scenario_list, SCENARIO_NAME))
    filters.append(ChartFilter(EFFECT_NAME, effects_list,
                               ALL_SCENARIOS, EFFECT_NAME, multiple_selection=False))  # by default shows all studies, ie does not apply any filter
    # specific case of tipping point study => filter will apply if at least one scenario has TIPPING_POINT in its name
    if True in [TIPPING_POINT in scenario for scenario in scenario_list]:
        # recover the values of tipping points:
        tp_dict = get_all_scenarios_values(execution_engine, 'Damage.tp_a3')
        tipping_point_list = list(set(tp_dict.values()))
        filters.append(ChartFilter(TIPPING_POINT, tipping_point_list,
                                   tipping_point_list, TIPPING_POINT))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    samples_df, _ = get_shared_value(execution_engine, 'samples_df')
    scenario_list = samples_df['scenario_name'].tolist()

    selected_scenarios = scenario_list
    sectorization: bool = len(execution_engine.dm.get_all_namespaces_from_var_name(f"{GlossaryEnergy.SectorServices}.{GlossaryEnergy.DamageDetailedDfValue}")) > 0
    year_start, _ = get_shared_value(execution_engine, GlossaryCore.YearStart)
    year_end, _ = get_shared_value(execution_engine, GlossaryCore.YearEnd)

    damage_tax_activation_status_dict = get_scenario_damage_tax_activation_status(execution_engine, scenario_list)
    graphs_list = []
    if filters is not None:
        for chart_filter in filters:  # filter on "scenarios" must occur before filter on "Effects" otherwise filter "Effects" does not work
            if chart_filter.filter_key == CHART_NAME:
                graphs_list = chart_filter.selected_values
            if chart_filter.filter_key == END_YEAR_NAME:
                year_end = chart_filter.selected_values
            if chart_filter.filter_key == SCENARIO_NAME:
                selected_scenarios = chart_filter.selected_values
            if chart_filter.filter_key == EFFECT_NAME:
                # performs a "OR" operation on the filter criteria. If no effect is selected for filtering, all scenarios
                # are shown. Then, restricts the scenarios shown to those respecting at least one of the filtered condition(s)
                effect = chart_filter.selected_values
                if effect != ALL_SCENARIOS:
                    if effect == DAMAGE_AND_TAX_NAME:
                        selected_scenarios = [scenario for scenario in selected_scenarios
                                                    if (damage_tax_activation_status_dict[scenario][TAX_NAME] and
                                                        damage_tax_activation_status_dict[scenario][DAMAGE_NAME])]
                    else:
                        selected_scenarios = [scenario for scenario in selected_scenarios
                                              if damage_tax_activation_status_dict[scenario][effect]]
            if chart_filter.filter_key == TIPPING_POINT:
                # Keep scenarios with selected tipping points + scenarios without tipping point defined (ex: reference scenario without damage)
                # => remove from selected scenarios the "tipping point scenarios" that do not respect the filtering condition
                tp_dict = get_all_scenarios_values(execution_engine, 'Damage.tp_a3')
                tipping_point_list = list(set(tp_dict.values()))
                tipping_points_to_drop = [tp for tp in tipping_point_list if tp not in chart_filter.selected_values]
                scenarios_to_drop = []
                for scenario in selected_scenarios:
                    for tipping_point in tipping_points_to_drop:
                        # tipping point scenario name ends with the following pattern as defined in usecase_witness_ms_mda_four_scenarios_tipping_points
                        if TIPPING_POINT + SEP + tipping_point + UNIT in scenario:
                            scenarios_to_drop.append(scenario)
                selected_scenarios = [scenario for scenario in selected_scenarios if scenario not in scenarios_to_drop]

    years = np.arange(year_start, year_end + 1).tolist()
    """
        -------------
        -------------
        SCENARIO COMPARISON CHART
        -------------
        -------------
    """
    # put in a box the symbols used for tax and damage filtering
    note = {'______': 'tax + damage',
            '............': 'with tax',
            '__ . __': 'with damage',
            }

    if 'Temperature' in graphs_list:

        x_axis_name = 'Years'
        y_axis_name = 'Temperature (degrees Celsius above preindustrial)'

        df_paths = [
            f'Temperature change.{GlossaryCore.TemperatureDetailedDfValue}', 'tp_a3']
        (temperature_detail_df_dict, tipping_points_dict) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        tipping_ptt_title_msg = ""
        if len(set(tipping_points_dict.values())) == 1:
            tipping_ptt_title_msg = f' (tipping point {list(tipping_points_dict.values())[0]}°C)'
        chart_name = 'Atmosphere temperature evolution' + tipping_ptt_title_msg
        temperature_dict = {}
        for scenario in scenario_list:
            temperature_dict[scenario] = temperature_detail_df_dict[scenario][GlossaryCore.TempAtmo].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note

        instanciated_charts.append(new_chart)

    if 'GDP' in graphs_list:

        chart_name = 'World GDP Net of Damage'
        x_axis_name = 'Years'
        y_axis_name = 'World GDP Net of Damage (Trillion $2020)'

        df_paths = ['Macroeconomics.' + GlossaryCore.EconomicsDetailDfValue, ]
        (gdp_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        gdp_dict = {}
        for scenario in scenario_list:
            gdp_dict[scenario] = gdp_df_dict[scenario][GlossaryCore.OutputNetOfDamage].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, gdp_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'CO2 emissions' in graphs_list:

        chart_name = 'CO2 emissions'
        x_axis_name = 'Years'
        y_axis_name = f'Carbon emissions [{GlossaryCore.GHGEmissionsDf["unit"]}]'

        df_paths = [
            GlossaryCore.GHGEmissionsDfValue]
        (co2_emissions_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_emissions_dict = {}
        for scenario in scenario_list:
            co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario][GlossaryCore.CO2].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_emissions_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Population' in graphs_list:

        chart_name = 'World population'
        x_axis_name = 'Years'
        y_axis_name = 'World Population'

        df_paths = ['Population.population_detail_df', ]
        (pop_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        pop_dict = {}
        for scenario in scenario_list:
            pop_dict[scenario] = pop_df_dict[scenario]['total'].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, pop_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Cumulative climate deaths' in graphs_list:

        chart_name = 'Cumulative climate deaths'
        x_axis_name = 'Years'
        y_axis_name = 'Cumulative climate deaths'

        df_paths = ['Population.death_dict', ]
        (death_dict_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        death_dict = {}
        for scenario in scenario_list:
            death_dict[scenario] = death_dict_dict[scenario]['climate']['cum_total'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, death_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Invesments in energy' in graphs_list:
        chart_name = 'Energy investments'
        x_axis_name = 'Years'
        y_axis_name = f'Investments [{GlossaryCore.EnergyInvestments["unit"]}]'

        # Get the total energy investment

        df_paths = [
            f'{GlossaryCore.EnergyInvestmentsValue}']
        (energy_investment_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        energy_investment_dict = {}
        for scenario in scenario_list:
            energy_investment_dict[scenario] = energy_investment_df_dict[
                scenario][GlossaryCore.EnergyInvestmentsValue].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_investment_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Investements in energy' in graphs_list:

        chart_name = 'Energy investments without tax'
        x_axis_name = 'Years'
        y_axis_name = f'Energy investments wo tax [{GlossaryCore.EnergyInvestmentsWoTax["unit"]}]'

        df_paths = [GlossaryEnergy.EnergyInvestmentsWoTaxValue, ]
        (invest_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        invest_dict = {}
        for scenario in scenario_list:
            invest_dict[scenario] = invest_df_dict[scenario][GlossaryEnergy.EnergyInvestmentsWoTaxValue].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, invest_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Invest in Energy + CCUS' in graphs_list:

        energy_list, _ = get_shared_value(execution_engine, GlossaryCore.energy_list)
        ccs_list, _ = get_shared_value(execution_engine, GlossaryCore.ccs_list)

        for energy in energy_list + ccs_list:
            # will sum in list_energy all the invests of all the technos of a given energy
            list_energy = []
            if energy in energy_list:
                energy_disc = EnergyMix.name
            else:
                energy_disc = GlossaryEnergy.CCUS
            if energy != GlossaryCore.biomass_dry:
                techno_list, _ = get_shared_value(execution_engine, f"{energy}.{GlossaryEnergy.TechnoListName}")

                for techno in techno_list:
                    df_paths = [f'{energy_disc}.{energy}.{techno}.{GlossaryEnergy.InvestLevelValue}', ]
                    (invest_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
                    invest_dict = {}
                    for scenario in scenario_list:
                        invest_dict[scenario] = invest_df_dict[scenario][GlossaryEnergy.InvestValue].values.tolist()
                    list_energy.append(invest_dict)

                invest_per_energy = {}
                for scenario in scenario_list:
                    invest_per_energy[scenario] = list(np.sum([invest_dict[scenario] for invest_dict in list_energy], axis=0))

            chart_name = f'Distribution of investments for {energy}'
            x_axis_name = GlossaryEnergy.Years
            y_axis_name = f'Investments in {energy} (Billion $2020)'

            new_chart = get_scenario_comparison_chart(years, invest_per_energy,
                                                      chart_name=chart_name,
                                                      x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                      selected_scenarios=selected_scenarios,
                                                      status_dict=damage_tax_activation_status_dict)

            new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

    if 'CO2 tax' in graphs_list:

        chart_name = 'CO2 tax'
        x_axis_name = 'Years'
        y_axis_name = f'Price [{GlossaryCore.CO2Taxes["unit"]}]'

        df_paths = [f'{GlossaryCore.CO2TaxesValue}', ]
        (co2_taxes_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        co2_tax_dict = {}
        for scenario in scenario_list:
            co2_tax_dict[scenario] = co2_taxes_df_dict[scenario][GlossaryCore.CO2Tax].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_tax_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Welfare' in graphs_list:

        chart_name = 'Welfare'
        y_axis_name = f'Welfare in {year_end}'

        df_paths = [f'{GlossaryCore.UtilityDfValue}',
                    ]
        (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        welfare_dict = {}
        for scenario in scenario_list:
            welfare_dict[scenario] = utility_df_dict[scenario][GlossaryCore.Welfare][year_end]

        min_y = min(list(welfare_dict.values()))
        max_y = max(list(welfare_dict.values()))

        new_chart = TwoAxesInstanciatedChart('', y_axis_name,
                                             [], [
                                                 min_y * 0.95, max_y * 1.05],
                                             chart_name)

        for scenario, welfare in welfare_dict.items():
            if scenario in selected_scenarios:
                serie = InstanciatedSeries(
                    [''],
                    [welfare], scenario, 'bar')

                new_chart.series.append(serie)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Utility' in graphs_list and not sectorization:

        chart_name = 'Utility'
        x_axis_name = 'Years'
        y_axis_name = 'Discounted Utility per capita [-]'

        df_paths = [f'{GlossaryCore.UtilityDfValue}', ]
        (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        utility_dict = {}
        for scenario in scenario_list:
            utility_dict[scenario] = utility_df_dict[scenario][GlossaryCore.DiscountedUtilityQuantityPerCapita].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, utility_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'ppm' in graphs_list:

        chart_name = 'CO2 Atmospheric concentrations'
        x_axis_name = 'Years'
        y_axis_name = GlossaryCore.CO2Concentration

        df_paths = [
            GlossaryCore.GHGCycleDfValue]
        (carboncycle_detail_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_ppm_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario][GlossaryCore.CO2Concentration].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_ppm_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        # Rockstrom Limit
        ordonate_data = [450] * int(len(years) / 5)
        abscisse_data = np.linspace(
            year_start, year_end, int(len(years) / 5))
        new_series = InstanciatedSeries(
            abscisse_data.tolist(), ordonate_data, 'Rockstrom limit', 'scatter')

        note = {'Rockstrom limit': 'Scientifical limit of the Earth'}
        new_chart.annotation_upper_left = note

        new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

    if 'Total energy production' in graphs_list:

        chart_name = 'Total Net Energy production'
        x_axis_name = 'Years'
        y_axis_name = GlossaryCore.TotalProductionValue + ' [TWh]'

        df_paths = [
            f'{EnergyMix.name}.{GlossaryCore.StreamProductionDetailedValue}']
        (energy_production_detailed_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        energy_production_detailed_dict = {}
        for scenario in scenario_list:
            energy_production_detailed_dict[scenario] = energy_production_detailed_df_dict[
                scenario]['Total production (uncut)'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_production_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Fossil production' in graphs_list:

        chart_name = 'Total Net Fossil Energy production'
        x_axis_name = 'Years'
        y_axis_name = 'Fossil energy production [TWh]'

        df_paths = [f'{EnergyMix.name}.{GlossaryCore.StreamProductionDetailedValue}']
        (energy_production_brut_detailed_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        energy_production_brut_detailed_dict = {}
        for scenario in scenario_list:
            energy_production_brut_detailed_dict[scenario] = energy_production_brut_detailed_df_dict[
                scenario]['production fossil (TWh)'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_production_brut_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

    if 'Clean energy production' in graphs_list:

        chart_name = 'Total Net Clean energy production'
        x_axis_name = 'Years'
        y_axis_name = '[TWh]'

        df_paths = [f'{EnergyMix.name}.{GlossaryCore.StreamProductionDetailedValue}']
        (energy_production_brut_detailed_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        energy_production_brut_detailed_dict = {}
        for scenario in scenario_list:
            energy_production_brut_detailed_dict[scenario] = energy_production_brut_detailed_df_dict[
                scenario][f'production {GlossaryCore.clean_energy} (TWh)'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_production_brut_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

        if 'Mean energy price' in graphs_list:

            chart_name = 'Mean Energy price'
            x_axis_name = 'Years'
            y_axis_name = f"[{GlossaryCore.EnergyMeanPrice['unit']}]"

            df_paths = [f'{GlossaryCore.EnergyMeanPriceValue}']
            (mean_energy_price_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

            mean_energy_price_dict = {}
            for scenario in scenario_list:
                mean_energy_price_dict[scenario] = mean_energy_price_df_dict[
                    scenario][GlossaryCore.EnergyPriceValue].values.tolist()

            new_chart = get_scenario_comparison_chart(years, mean_energy_price_dict,
                                                      chart_name=chart_name,
                                                      x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                      selected_scenarios=selected_scenarios,
                                                      status_dict=damage_tax_activation_status_dict)

            new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

        if 'Consumption' in graphs_list and not sectorization:

            chart_name = 'Consumption'
            x_axis_name = 'Years'
            y_axis_name = '[G$]'

            df_paths = [GlossaryCore.EconomicsDetailDfValue]
            (economics_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

            consumption_dict = {}
            for scenario in scenario_list:
                consumption_dict[scenario] = economics_df_dict[
                    scenario][GlossaryCore.Consumption].values.tolist()

            new_chart = get_scenario_comparison_chart(years, consumption_dict,
                                                      chart_name=chart_name,
                                                      x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                      selected_scenarios=selected_scenarios,
                                                      status_dict=damage_tax_activation_status_dict)

            new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

    if "GDP vs Energy":

        raw_iea_df = pd.merge(
            DatabaseWitnessCore.IEANZEEnergyProduction.value,
            DatabaseWitnessCore.IEANZEGDPNetOfDamage.value,
            on="years",
            how="inner",
        )

        net_iea_df = pd.merge(
            DatabaseWitnessCore.IEANZEFinalEnergyConsumption.value,
            DatabaseWitnessCore.IEANZEGDPNetOfDamage.value,
            on="years",
            how="inner",
        )

        raw_data_dict = {scenario_name: {
                "data_type": "variable",
                "scenario_name": scenario_name,
                "x_var_name": f"{GlossaryEnergy.EnergyMixRawProductionValue}",
                "x_column_name": "Total",
                "x_data_scale": 1e-3,
                "y_var_name": "Macroeconomics.economics_detail_df",
                "y_column_name": "output_net_of_d",
                "text_column": GlossaryCore.Years,
            }
            for scenario_name in scenario_list
        }
        raw_data_dict["Historical Data"] = {
                "data_type": "csv",
                "filename": join(Path(__file__).parents[3], "data", 'primary-energy-consumption_vs_gdp.csv'),
                "x_column_name": 'Primary energy consumption [PWh]',
                "y_column_name": 'World GDP [T$]',
                "marker_symbol": "triangle-up",
                "text_column": "years",
            }

        raw_data_dict["IEA"] = {
                "data_type": "dataframe",
                "data": raw_iea_df,
                "x_column_name": "Total production",
                "y_column_name": "output_net_of_d",
                "marker_symbol": "square",
                "text_column": "years",
            }

        net_data_dict = {
            scenario_name: {
                "data_type": "variable",
                "scenario_name": scenario_name,
                "x_var_name": f"EnergyMix.{GlossaryEnergy.StreamProductionValue}",
                "x_column_name": GlossaryCore.TotalProductionValue,
                "y_var_name": "Macroeconomics.economics_detail_df",
                "y_column_name": "output_net_of_d",
                "text_column": GlossaryCore.Years,
            }
            for scenario_name in scenario_list
        }

        net_data_dict["Historical Data"] = {
                "data_type": "csv",
                "filename": join(Path(__file__).parents[3], "data", 'world_gdp_vs_net_energy_consumption.csv'),
                "x_column_name": 'Net energy consumption [PWh]',
                "y_column_name": 'World GDP [T$]',
                "marker_symbol": "triangle-up",
                "text_column": "years",
            }

        net_data_dict["IEA"] = {
                "data_type": "dataframe",
                "data": net_iea_df,
                "x_column_name": "Final Consumption",
                "y_column_name": "output_net_of_d",
                "marker_symbol": "square",
                "text_column": "years",
            }

        new_chart = create_xy_chart(execution_engine, chart_name="GDP vs Raw energy production",
                                    x_axis_name="World's raw energy production (PWh)",
                                    y_axis_name="World's GDP net of damage (T$)", data_dict=raw_data_dict)
        instanciated_charts.append(new_chart)

        new_chart = create_xy_chart(execution_engine, chart_name="GDP vs Net energy production",
                                    x_axis_name="World's net energy production (PWh)",
                                    y_axis_name="World's GDP net of damage (T$)", data_dict=net_data_dict)
        instanciated_charts.append(new_chart)

    return instanciated_charts


def get_scenario_damage_tax_activation_status(execution_engine, scenario_list):
    '''
    Determines for each scenario if the damage and the taxes are activated
    assumes that tax is activated when ccs_price_percentage > 0 and co2_damage_price_percentage > 0 in case of damage
        NB: if damage are deactivated, co2_damage_price_percentage can be set to 0 as it has no effect
        In case there is no damage, co2_damage_price_percentage > 0 does not activate the tax but ccs_price_percentage > 0 does

    assumes that damage are activated when damage_to_productivity and compute_climate_impact_on_gdp and
                                          activate_climate_effect_population and
                                          activate_pandemic_effects are true
    '''
    df_paths = ['assumptions_dict']
    (assumption_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['ccs_price_percentage', ]
    (ccs_price_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['co2_damage_price_percentage', ]
    (co2_damage_price_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['damage_to_productivity', ]
    (damage_to_productivity_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    status_dict = {}
    for scenario in scenario_list:
        status_dict[scenario] = {}
        status_dict[scenario][DAMAGE_NAME] = damage_to_productivity_dict[scenario] and \
                                          assumption_dict[scenario]['compute_climate_impact_on_gdp'] and \
                                          assumption_dict[scenario]['activate_climate_effect_population']
        status_dict[scenario][TAX_NAME] = ccs_price_dict[scenario] > 25. or (co2_damage_price_dict[scenario] > 0 and status_dict[scenario][DAMAGE_NAME])

    return status_dict


def get_shade_of_color(color, weight):
    '''
    gives the rgb and hex codes for a color interpolated between a dark and light shade of a color
    Args:
        color: [str] color to chose from
        weight: [float] between 0 and 1. 0 => the light color. 1 => the dark color
    returns the rgb code (tuple) and the hex code (string)
    '''
    if color == 'green':
        light = (118, 215, 196)
        dark = (20, 143, 119)
    elif color == 'orange':
        light = (255, 243, 205)
        dark = (255, 140, 0)
    else:
        color = 'green'
        light = (118, 215, 196)
        dark = (20, 143, 119)
        logging.info(f'color={color} is not in available list. Imposing color=green')
    if weight > 1. or weight < 0.:
        logging.info(f'weight must be between 0 and 1. weight={weight}. Imposing dark shade of {color}')
    rgb_array = weight * np.array(light) + (1. - weight) * np.array(dark)
    rgb = tuple([floor(rgb_array[0]), floor(rgb_array[1]), floor(rgb_array[2])])
    hex_code = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    return [rgb, hex_code]


def get_scenario_comparison_chart(x_list, y_dict, chart_name, x_axis_name, y_axis_name, selected_scenarios, status_dict=None):
    min_x = min(x_list)
    max_x = max(x_list)
    # Graphs ordinate should start at 0, except for CO2 emissions that could go <0
    min_y = min(0, min([min(list(y)) for y in y_dict.values()]))
    max_y = max([max(list(y)) for y in y_dict.values()])

    # Initialize new_chart with default values
    new_chart = TwoAxesInstanciatedChart(x_axis_name, y_axis_name, [min_x - 5, max_x + 5], [min_y - max_y * 0.05, max_y * 1.05], chart_name)

    # Define colors for curves
    color_mapping = {

        # MDO multiscenario coarse sectorisé:
        study_ms_mdo_sect.UC1: dict(color='red'),
        study_ms_mdo_sect.UC2: dict(color='red'),
        study_ms_mdo_sect.UC3: dict(color='orange'),
        study_ms_mdo_sect.UC4: dict(color='green'),

    }
    """
    usecase_ms_mda_tipping_point.USECASE2: dict(color='red'),  # Red
    usecase_ms_mda_tipping_point.USECASE4_TP2: dict(color='#FF5733'),  # Dark orange
    usecase_ms_mda_tipping_point.USECASE4_TP1: dict(color='#FFA533'),  # Orange
    usecase_ms_mda_tipping_point.USECASE4_TP_REF: dict(color='#FFD633'),  # Light Orange
    usecase_ms_mda_tipping_point.USECASE7_TP2: dict(color='#2E8B57'),  # Dark green
    usecase_ms_mda_tipping_point.USECASE7_TP1: dict(color='#32CD32'),  # Green
    usecase_ms_mda_tipping_point.USECASE7_TP_REF: dict(color='#7FFF00'),  # Light Green

    # the lower the TP, the darker the color
    usecase_ms_mdo_iamc.UC1: dict(color='red'),  # Red
    usecase_ms_mdo_iamc.UC3_tp1: dict(color='#FFD633'),  # Light Orange
    usecase_ms_mdo_iamc.UC3_tp2: dict(color='#FFA533'),  # orange
    usecase_ms_mdo_iamc.UC4_tp1: dict(color='#89CFF0'),  # Light blue
    usecase_ms_mdo_iamc.UC4_tp2: dict(color='#0047AB'),  # Dark blue
    usecase_ms_mdo_iamc.UC_NZE_tp1: dict(color='#7FFF00'),  # Light green
    usecase_ms_mdo_iamc.UC_NZE_tp2: dict(color='#2E8B57'),  # Dark Green

    usecase_ms_mdo_with_nze.NO_CCUS: dict(color='#FFD633'),  # Light Orange
    usecase_ms_mdo_with_nze.ALL_TECHNOS: dict(color='#0047AB'),  # Dark blue
    usecase_ms_mdo_with_nze.ALL_TECHNOS_NZE: dict(color='#7FFF00'),  # Light green
    """
    line_color = None

    for scenario, y_values in y_dict.items():
        if scenario in color_mapping.keys():
            line_color = color_mapping[scenario]

        if line_color is not None:  # Check if line_color is assigned
            marker_symbol = 'circle'
            lines = SeriesTemplate.LINES_DISPLAY
            if status_dict is not None:
                if status_dict[scenario][TAX_NAME] is False and status_dict[scenario][DAMAGE_NAME] is False:
                    lines = SeriesTemplate.DASH_LINES_DISPLAY
                elif status_dict[scenario][TAX_NAME] is True and status_dict[scenario][DAMAGE_NAME] is False:
                    lines = SeriesTemplate.DOT_LINES_DISPLAY
                elif status_dict[scenario][TAX_NAME] is False and status_dict[scenario][DAMAGE_NAME] is True:
                    lines = SeriesTemplate.DASH_DOT_LINES_DISPLAY

            if scenario in selected_scenarios:
                new_series = InstanciatedSeries(x_list, y_values, scenario, lines, True, marker_symbol=marker_symbol, line=line_color)
                new_chart.series.append(new_series)

    return new_chart


def get_df_per_scenario_dict(execution_engine, var_names, scenario_list=[]):
    '''! Function to retrieve dataframes from all the scenarios given a specified path
    @param execution_engine: Execution_engine, object from which the data is gathered
    @param var_names: list of string, containing the paths to access the df

    @return df_per_scenario_dict: list of dict, with {key = scenario_name: value= requested_dataframe}
    '''
    df_per_scenario_dicts = [{} for _ in var_names]
    if not scenario_list:
        samples_df, _ = get_shared_value(execution_engine, 'samples_df')
        scenario_list = samples_df['scenario_name']

    for i, var_name in enumerate(var_names):
        all_scenarios_variable_values = get_all_scenarios_values(execution_engine, var_name)
        for scenario in scenario_list:
            scenario_var_name = list(filter(lambda x: scenario in x, all_scenarios_variable_values.keys()))[0]
            df_per_scenario_dicts[i][scenario] = all_scenarios_variable_values[scenario_var_name]
    return df_per_scenario_dicts
