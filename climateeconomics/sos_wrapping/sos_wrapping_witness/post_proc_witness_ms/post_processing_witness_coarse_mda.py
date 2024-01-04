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
import numpy as np

from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.two_axes_chart_template import SeriesTemplate
from energy_models.core.energy_mix.energy_mix import EnergyMix
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.core.ccus.ccus import CCUS

TAX_NAME = 'with tax'
DAMAGE_NAME = 'with damage'
ALL_NAME = 'all scenarios'
effects_list = [TAX_NAME, DAMAGE_NAME, ALL_NAME]
EFFECT_NAME = 'Effects'
CHART_NAME = 'Charts'
YEARS_NAME = 'YEARS'
SCATTER_SCENARIO = 'mda_scenarios'
# list of graphs to be plotted and filtered
graphs_list = ['Temperature per scenario',
               'CO2 emissions per scenario',
               'Population per scenario',
               'Cumulative climate deaths per scenario',
               'GDP per scenario',
               # 'invest per scenario',
               'invest in energy per scenario',
               'invest in energy and ccus per scenario',
               'CO2 tax per scenario',
               'Utility per scenario',
               'Total production per scenario',
               'Fossil production per scenario',
               'Renewable production per scenario'
               ]
def post_processing_filters(execution_engine, namespace):

    filters = []

    namespace_w = f'{execution_engine.study_name}.{SCATTER_SCENARIO}'
    scenario_list = execution_engine.dm.get_value(f'{namespace_w}.scenario_df')['scenario_name'].tolist()

    # recover year start and year end arbitrarily from the first scenario
    year_start = execution_engine.dm.get_value(f'{namespace_w}.{scenario_list[0]}.{GlossaryCore.YearStart}')
    year_end = execution_engine.dm.get_value(f'{namespace_w}.{scenario_list[0]}.{GlossaryCore.YearEnd}')
    years_list = np.arange(year_start, year_end + 1).tolist()

    filters.append(ChartFilter(CHART_NAME, graphs_list, graphs_list, CHART_NAME))
    filters.append(ChartFilter(YEARS_NAME, years_list, years_list, YEARS_NAME))
    filters.append(ChartFilter(EFFECT_NAME, effects_list,
                               effects_list, EFFECT_NAME))

    return filters


def post_processings(execution_engine, namespace, filters):

    instanciated_charts = []

    namespace_w = f'{execution_engine.study_name}.{SCATTER_SCENARIO}'
    scenario_list = execution_engine.dm.get_value(f'{namespace_w}.scenario_df')['scenario_name'].tolist()

    selected_scenarios = scenario_list

    df_paths = [f'{GlossaryCore.YearStart}',
                f'{GlossaryCore.YearEnd}', ]
    year_start_dict, year_end_dict = get_df_per_scenario_dict(
        execution_engine, df_paths, scenario_list)
    year_start, year_end = year_start_dict[scenario_list[0]
                                           ], year_end_dict[scenario_list[0]]
    years = np.arange(year_start, year_end + 1).tolist()

    damage_tax_activation_status_dict = get_scenario_damage_tax_activation_status(execution_engine, scenario_list)

    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == CHART_NAME:
                graphs_list = chart_filter.selected_values
            if chart_filter.filter_key == YEARS_NAME:
                years = chart_filter.selected_values
            if chart_filter.filter_key == EFFECT_NAME:
                effects_list_filtered = chart_filter.selected_values
                if ALL_NAME in effects_list_filtered: #disregards the filters on damage and tax
                    selected_scenarios = scenario_list
                else:
                    selected_scenarios = []
                    for scenario in scenario_list:
                        # convert list into dictionnary to compare effects in an easier way
                        effects_filtered_dict = dict.fromkeys(damage_tax_activation_status_dict[scenario].keys(), False)
                        for key in effects_list_filtered:
                            if key != ALL_NAME:
                                effects_filtered_dict[key] = True
                        if effects_filtered_dict == damage_tax_activation_status_dict[scenario]:
                            selected_scenarios.append(scenario)

    """
        -------------
        -------------
        SCENARIO COMPARISON CHART
        -------------
        -------------
    """

    if 'Temperature per scenario' in graphs_list:

        chart_name = 'Atmosphere temperature evolution per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Temperature (degrees Celsius above preindustrial)'

        df_paths = [
            'Temperature_change.temperature_detail_df', ]
        (temperature_detail_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)
        temperature_dict = {}
        for scenario in scenario_list:
            temperature_dict[scenario] = temperature_detail_df_dict[scenario][GlossaryCore.TempAtmo].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, temperature_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        instanciated_charts.append(new_chart)

    if 'GDP per scenario' in graphs_list:

        chart_name = 'World GDP Net of Damage over years per scenario'
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

        instanciated_charts.append(new_chart)

    if 'CO2 emissions per scenario' in graphs_list:

        chart_name = 'CO2 emissions per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Carbon emissions (Gtc)'

        df_paths = [
            'GHG_emissions_df']
        (co2_emissions_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_emissions_dict = {}
        for scenario in scenario_list:
            co2_emissions_dict[scenario] = co2_emissions_df_dict[scenario][GlossaryCore.TotalCO2Emissions].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, co2_emissions_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        instanciated_charts.append(new_chart)


    if 'Population per scenario' in graphs_list:

        chart_name = 'World population over years per scenario'
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

        instanciated_charts.append(new_chart)

    if 'Cumulative climate deaths per scenario' in graphs_list:

        chart_name = 'Cumulative climate deaths over years per scenario'
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

        instanciated_charts.append(new_chart)

    if 'invest per scenario' in graphs_list:
        chart_name = f'investments per scenario'
        x_axis_name = 'Years'
        y_axis_name = f'total energy investment'

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

        instanciated_charts.append(new_chart)



    if 'invest in energy per scenario' in graphs_list:

        chart_name = 'Energy investments without tax over years per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Energy investments wo tax (Trillion $2020)'

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

        instanciated_charts.append(new_chart)

    if 'invest in energy and ccus per scenario' in graphs_list:

        namespace_w = f'{execution_engine.study_name}.{SCATTER_SCENARIO}.{scenario_list[0]}'
        energy_list = execution_engine.dm.get_value(f'{namespace_w}.{GlossaryEnergy.energy_list}')
        ccs_list = execution_engine.dm.get_value(f'{namespace_w}.{GlossaryEnergy.ccs_list}')

        for energy in energy_list + ccs_list:
            # will sum in list_energy all the invests of all the technos of a given energy
            list_energy = []
            if energy in energy_list:
                energy_disc = EnergyMix.name
            else:
                energy_disc = CCUS.name
            if energy != BiomassDry.name:
                techno_list = execution_engine.dm.get_value(f'{namespace_w}.{energy_disc}.{energy}.{GlossaryEnergy.TechnoListName}')

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

            chart_name = f'Distribution of investments for {energy} vs years'
            x_axis_name = GlossaryEnergy.Years
            y_axis_name = f'Investments in {energy} (Billion $2020)'

            new_chart = get_scenario_comparison_chart(years, invest_per_energy,
                                                      chart_name=chart_name,
                                                      x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                      selected_scenarios=selected_scenarios,
                                                      status_dict=damage_tax_activation_status_dict)

            instanciated_charts.append(new_chart)


    if 'CO2 tax per scenario' in graphs_list:

        chart_name = 'CO2 tax per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Price ($/tCO2)'

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

        instanciated_charts.append(new_chart)

    if 'Welfare per scenario' in graphs_list:

        chart_name = 'Welfare per scenario'
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

        instanciated_charts.append(new_chart)

    if 'Utility per scenario' in graphs_list:

        chart_name = 'Utility per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Discounted Utility [-]'

        df_paths = [f'{GlossaryCore.UtilityDfValue}', ]
        (utility_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        utility_dict = {}
        for scenario in scenario_list:
            utility_dict[scenario] = utility_df_dict[scenario][GlossaryCore.DiscountedUtility].values.tolist(
            )

        new_chart = get_scenario_comparison_chart(years, utility_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        instanciated_charts.append(new_chart)



    if 'ppm per scenario' in graphs_list:

        chart_name = 'Atmospheric concentrations parts per million per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Atmospheric concentrations parts per million'

        df_paths = [
            'ghg_cycle_df']
        (carboncycle_detail_df_dict,) = get_df_per_scenario_dict(
            execution_engine, df_paths)

        co2_ppm_dict, welfare_dict = {}, {}
        for scenario in scenario_list:
            co2_ppm_dict[scenario] = carboncycle_detail_df_dict[scenario]['co2_ppm'].values.tolist(
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

    if 'Total production per scenario' in graphs_list:

        chart_name = 'Total Net Energy production per scenario'
        x_axis_name = 'Years'
        y_axis_name = GlossaryCore.TotalProductionValue + ' [TWh]'

        df_paths = [
            f'{EnergyMix.name}.energy_production_detailed']
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

        instanciated_charts.append(new_chart)



    if 'Fossil production per scenario' in graphs_list:

        chart_name = 'Total Net Fossil Energy production per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Fossil energy production [TWh]'

        df_paths = [f'{EnergyMix.name}.energy_production_detailed']
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

        instanciated_charts.append(new_chart)

    if 'Renewable production per scenario' in graphs_list:

        chart_name = 'Total Net Renewable Energy production per scenario'
        x_axis_name = 'Years'
        y_axis_name = 'Renewable net energy production [TWh]'

        df_paths = [f'{EnergyMix.name}.energy_production_detailed']
        (energy_production_brut_detailed_df_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)

        energy_production_brut_detailed_dict = {}
        for scenario in scenario_list:
            energy_production_brut_detailed_dict[scenario] = energy_production_brut_detailed_df_dict[
                scenario]['production renewable (TWh)'].values.tolist()

        new_chart = get_scenario_comparison_chart(years, energy_production_brut_detailed_dict,
                                                  chart_name=chart_name,
                                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name,
                                                  selected_scenarios=selected_scenarios,
                                                  status_dict=damage_tax_activation_status_dict)

        instanciated_charts.append(new_chart)


    return instanciated_charts


def get_scenario_damage_tax_activation_status(execution_engine, scenario_list):
    '''
    Determines for each scenario if the damage and the taxes are activated
    assumes that tax is activated when ccs_price_percentage > 0 and co2_damage_price_percentage > 0 in case of damage
        NB: if damage are deactivated, co2_damage_price_percentage can be set to 0 as it has no effect
    assumes that damage are activated when damage_to_productivity and compute_climate_impact_on_gdp and
                                          activate_climate_effect_population are true
    '''
    df_paths = ['assumptions_dict']
    (assumption_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['ccs_price_percentage', ]
    (ccs_price_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['co2_damage_price_percentage', ]
    (co2_damage_price_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    df_paths = ['Macroeconomics.damage_to_productivity', ]
    (damage_to_productivity_dict,) = get_df_per_scenario_dict(execution_engine, df_paths)
    status_dict = {}
    for scenario in scenario_list:
        status_dict[scenario] = {}
        status_dict[scenario][DAMAGE_NAME] = damage_to_productivity_dict[scenario] and \
                                          assumption_dict[scenario]['compute_climate_impact_on_gdp'] and \
                                          assumption_dict[scenario]['activate_climate_effect_population']
        status_dict[scenario][TAX_NAME] = (ccs_price_dict[scenario] > 0. and (co2_damage_price_dict[scenario] > 0. or \
                                                                             (co2_damage_price_dict[scenario] <= 0. and \
                                                                              not status_dict[scenario][DAMAGE_NAME])))

    return status_dict


def get_scenario_comparison_chart(x_list, y_dict, chart_name, x_axis_name, y_axis_name, selected_scenarios, status_dict=None):
    min_x = min(x_list)
    max_x = max(x_list)
    # graphs ordinate should start at 0, except for CO2 emissions that could go <0
    min_y = min(0, min([min(list(y)) for y in y_dict.values()]))
    max_y = max([max(list(y)) for y in y_dict.values()])

    new_chart = TwoAxesInstanciatedChart(x_axis_name, y_axis_name,
                                         [min_x - 5, max_x + 5], [
                                             min_y - max_y * 0.05, max_y * 1.05],
                                         chart_name)

    for scenario, y_values in y_dict.items():
        '''
        For ease of understanding of the plots, scenarios without damage are in dashed line(solid otherwise) and scenarios
        with tax have circles on the line. wether or not damage and taxes are activated is provided in status_dict
        '''
        lines = SeriesTemplate.LINES_DISPLAY
        marker = 'circle'
        if status_dict is not None:
            if status_dict[scenario][TAX_NAME] == False:
                marker = ''

            if status_dict[scenario][DAMAGE_NAME] == False:
                lines = SeriesTemplate.DASH_LINES_DISPLAY

        if scenario in selected_scenarios:
            new_series = InstanciatedSeries(
                x_list, y_values, scenario, lines, True, marker_symbol=marker)

            new_chart.series.append(new_series)

    return new_chart


def get_df_per_scenario_dict(execution_engine, df_paths, scenario_list=None):
    '''! Function to retrieve dataframes from all the scenarios given a specified path
    @param execution_engine: Execution_engine, object from which the data is gathered
    @param df_paths: list of string, containing the paths to access the df

    @return df_per_scenario_dict: list of dict, with {key = scenario_name: value= requested_dataframe} 
    '''
    df_per_scenario_dicts = [{} for _ in df_paths]
    namespace_w = f'{execution_engine.study_name}.{SCATTER_SCENARIO}'
    if not scenario_list:
        scenario_list = execution_engine.dm.get_value(f'{namespace_w}.scenario_df')['scenario_name'].tolist()

    for scenario in scenario_list:
        for i, df_path in enumerate(df_paths):
            df_per_scenario_dicts[i][scenario] = execution_engine.dm.get_value(
                f'{namespace_w}.{scenario}.{df_path}')
    return df_per_scenario_dicts
