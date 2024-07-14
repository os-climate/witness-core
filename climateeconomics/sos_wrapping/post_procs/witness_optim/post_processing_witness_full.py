'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from matplotlib.pyplot import cm
from plotly import graph_objects as go
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.glossarycore import GlossaryCore


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    filters = []

    chart_list = ['Breakdown price energies', 'Energies CO2 intensity',
                  'Global CO2 breakdown sankey']
    # The filters are set to False by default since the graphs are not yet
    # mature
    filters.append(ChartFilter('Charts', chart_list, chart_list, 'Charts'))

    return filters


def post_processings(execution_engine, namespace, filters):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    instanciated_charts = []

    # Overload default value with chart filter
    graphs_list = []
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'Charts':
                graphs_list.extend(chart_filter.selected_values)

    # ---
    if 'Energies CO2 intensity' in graphs_list:
        chart_name = 'Energies CO2 intensity summary'
        new_chart = get_chart_green_energies(
            execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

        chart_name = 'Energies CO2 intensity by years'
        new_chart = get_chart_green_energies(
            execution_engine, namespace, chart_name=chart_name, summary=False)
        if new_chart is not None:
            instanciated_charts.append(new_chart)
    # ---
    if 'Global CO2 breakdown sankey' in graphs_list:
        chart_name = 'Global CO2 breakdown sankey summary'
        new_chart = get_chart_Global_CO2_breakdown_sankey(
            execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

        chart_name = 'Global CO2 breakdown sankey by years'
        new_chart = get_chart_Global_CO2_breakdown_sankey(
            execution_engine, namespace, chart_name=chart_name, summary=False)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

    # #---
    # if 'Breakdown price energies' in graphs_list:
    # chart_name = f'Breakdown price energies'
    # new_chart = get_chart_breakdown_price_energies(
    # execution_engine, namespace, chart_name=chart_name)
    # if new_chart is not None:
    # instanciated_charts.append(new_chart)

    return instanciated_charts


def get_chart_green_energies(execution_engine, namespace, chart_name='Energies CO2 intensity', summary=True):
    '''! Function to create the green_techno/_energy scatter chart
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc
    @param summary:Boolean, switch from summary (True) to detailed by years via sliders (False)

    @return new_chart: InstantiatedPlotlyNativeChart Scatter plot
    '''

    # Prepare data
    multilevel_df, years, updated_ns = get_multilevel_df(
        execution_engine, namespace,
        columns=['price_per_kWh', 'price_per_kWh_wotaxes', 'CO2_per_kWh', 'production', GlossaryCore.InvestValue])
    energy_list = list(set(multilevel_df.index.droplevel(1)))
    namespace = updated_ns

    # Create Figure
    chart_name = f'{chart_name}'
    fig = go.Figure()
    # Get min and max CO2 emissions for colorscale and max of production for
    # marker size
    array_of_cmin, array_of_cmax, array_of_pmax, array_of_pintmax = [], [], [], []
    for (array_c, array_p) in multilevel_df[['CO2_per_kWh', 'production']].values:
        array_of_cmin += [array_c.min()]
        array_of_cmax += [array_c.max()]
        array_of_pmax += [array_p.max()]
        array_of_pintmax += [array_p.sum()]
    cmin, cmax = np.min(array_of_cmin), np.max(array_of_cmax)
    pmax, pintmax = np.max(array_of_pmax), np.max(array_of_pintmax)
    if summary:
        # Create a graph to aggregate the informations on all years
        price_per_kWh, price_per_kWh_wotaxes, CO2_per_kWh, production, invest = [], [], [], [], []
        total_CO2, total_price, total_price_wotaxes = [], [], []
        EnergyMix = execution_engine.dm.get_disciplines_with_name(
            f'{namespace}.EnergyMix')[0]
        CO2_taxes, CO2_taxes_array = EnergyMix.get_sosdisc_inputs(GlossaryEnergy.CO2Taxes['var_name'])[
            GlossaryCore.CO2Tax].values, []
        for i, energy in enumerate(energy_list):
            # energies level
            production += [np.sum(multilevel_df.groupby(level=0)
                                  ['production'].sum()[energy]), ]
            invest += [np.sum(multilevel_df.groupby(level=0)
                              [GlossaryCore.InvestValue].sum()[energy]), ]
            total_CO2 += [np.sum((multilevel_df.loc[energy]['CO2_per_kWh']
                                  * multilevel_df.loc[energy]['production']).sum()), ]
            total_price += [np.sum((multilevel_df.loc[energy]['price_per_kWh']
                                    * multilevel_df.loc[energy]['production']).sum()), ]
            total_price_wotaxes += [np.sum((multilevel_df.loc[energy]['price_per_kWh_wotaxes']
                                            * multilevel_df.loc[energy]['production']).sum()), ]
            CO2_per_kWh += [np.divide(total_CO2[i], production[i]), ]
            price_per_kWh += [np.divide(total_price[i], production[i]), ]
            price_per_kWh_wotaxes += [
                np.divide(total_price_wotaxes[i], production[i]), ]
            CO2_taxes_array += [np.mean(CO2_taxes), ]
        customdata = [energy_list, price_per_kWh, CO2_per_kWh,
                      production, invest, total_CO2, CO2_taxes_array,
                      price_per_kWh_wotaxes]
        hovertemplate = '<br>Name: %{customdata[0]}' + \
                        '<br>Mean Price per kWh before CO2 tax: %{customdata[7]: .2e}' + \
                        '<br>Mean CO2 per kWh: %{customdata[2]: .2e}' + \
                        '<br>Integrated Total CO2: %{customdata[5]: .2e}' + \
                        '<br>Integrated Production: %{customdata[3]: .2e}' + \
                        '<br>Integrated Invest: %{customdata[4]: .2e}' + \
                        '<br>Mean Price per kWh: %{customdata[1]: .2e}' + \
                        '<br>Mean CO2 taxes: %{customdata[6]: .2e}'
        marker_sizes = np.multiply(production, 20.0) / \
                       pintmax + 10.0
        scatter_mean = go.Scatter(x=list(price_per_kWh_wotaxes), y=list(CO2_per_kWh),
                                  customdata=list(np.asarray(customdata).T),
                                  hovertemplate=hovertemplate,
                                  text=energy_list,
                                  textposition="top center",
                                  mode='markers+text',
                                  marker=dict(color=CO2_per_kWh,
                                              cmin=cmin, cmax=cmax,
                                              colorscale='RdYlGn_r', size=list(marker_sizes),
                                              colorbar=dict(title='CO2 per kWh', thickness=20)),
                                  visible=False)
        fig.add_trace(scatter_mean)
    else:
        # Fill figure with data by year
        for i_year in range(len(years)):
            ################
            # -energy level-#
            ################
            price_per_kWh, price_per_kWh_wotaxes, CO2_per_kWh, production, invest = [], [], [], [], []
            total_CO2, total_price, total_price_wotaxes = [], [], []
            EnergyMix = execution_engine.dm.get_disciplines_with_name(
                f'{namespace}.EnergyMix')[0]
            CO2_taxes, CO2_taxes_array = EnergyMix.get_sosdisc_inputs(GlossaryEnergy.CO2Taxes['var_name'])[
                GlossaryCore.CO2Tax].values, []
            for i, energy in enumerate(energy_list):
                # energies level
                production += [multilevel_df.groupby(level=0)
                               ['production'].sum()[energy][i_year], ]
                invest += [multilevel_df.groupby(level=0)
                           [GlossaryCore.InvestValue].sum()[energy][i_year], ]
                total_CO2 += [(multilevel_df.loc[energy]['CO2_per_kWh'] *
                               multilevel_df.loc[energy]['production']).sum()[i_year], ]
                total_price += [(multilevel_df.loc[energy]['price_per_kWh']
                                 * multilevel_df.loc[energy]['production']).sum()[i_year], ]
                total_price_wotaxes += [(multilevel_df.loc[energy]['price_per_kWh_wotaxes']
                                         * multilevel_df.loc[energy]['production']).sum()[i_year], ]
                CO2_per_kWh += [np.divide(total_CO2[i], production[i]), ]
                price_per_kWh += [np.divide(total_price[i], production[i]), ]
                price_per_kWh_wotaxes += [
                    np.divide(total_price_wotaxes[i], production[i]), ]
                CO2_taxes_array += [CO2_taxes[i], ]
            customdata = [energy_list, price_per_kWh, CO2_per_kWh,
                          production, invest, total_CO2, CO2_taxes_array,
                          price_per_kWh_wotaxes]
            hovertemplate = '<br>Name: %{customdata[0]}' + \
                            '<br>Price per kWh before CO2 tax: %{customdata[7]: .2e}' + \
                            '<br>CO2 per kWh: %{customdata[2]: .2e}' + \
                            '<br>Total CO2: %{customdata[5]: .2e}' + \
                            '<br>Production: %{customdata[3]: .2e}' + \
                            '<br>Invest: %{customdata[4]: .2e}' + \
                            '<br>Price per kWh: %{customdata[1]: .2e}' + \
                            '<br>CO2 taxes: %{customdata[6]: .2e}'
            marker_sizes = np.multiply(production, 20.0) / \
                           pmax + 10.0
            scatter = go.Scatter(x=list(price_per_kWh_wotaxes), y=list(CO2_per_kWh),
                                 customdata=list(np.asarray(customdata).T),
                                 hovertemplate=hovertemplate,
                                 text=energy_list,
                                 textposition="top center",
                                 mode='markers+text',
                                 marker=dict(color=CO2_per_kWh,
                                             cmin=cmin, cmax=cmax,
                                             colorscale='RdYlGn_r', size=list(marker_sizes),
                                             colorbar=dict(title='CO2 per kWh', thickness=20)),
                                 visible=False)
            fig.add_trace(scatter)
        # Prepare year slider and layout updates
        steps = []
        for i in range(int(len(fig.data)) - 1):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
                label=str(2020 + i)
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Year: "},
            steps=steps), ]
        fig.update_layout(
            sliders=sliders
        )
    fig.update_xaxes(title_text='Price per MWh [$/MWh] (before CO2 taxes)')
    fig.update_yaxes(title_text='CO2 per kWh [kgCO2/kWh]')
    fig.data[0].visible = True

    new_chart = InstantiatedPlotlyNativeChart(
        fig, chart_name=chart_name, default_title=True)
    return new_chart


def get_multilevel_df(execution_engine, namespace, columns=None):
    '''! Function to create the dataframe with all the data necessary for the graphs in a multilevel [energy, technologies]
    @param execution_engine: Current execution engine object, from which the data is extracted
    @param namespace: Namespace at which the data can be accessed

    @return multilevel_df: Dataframe

    Definitions of the various levels of emissions:
    CO2_from_production = -scope 1- (direct emissions): direct CO2 emissions during the production phase of the techno
            ex: CO2 emitted or needed as part of the chemical reaction used to produced the techno (ex: Sabatier reaction
            requires CO2 to produce methane from H2) + company facilities emissions + company vehicles emissions
            + CO2 emissions of energy used for the production + ...
    CO2_from_other_consumption = -scope 2- (indirect emissions): CO2 emissions during the production of the energies
            used in scope 1, namely of the energies used during the production of the techno = emissions during
            production of purchased electricity, steam, heating & cooling for own use
            NB: in energy_models/core/techno_type/techno_type.py, CO2_from_other_consumption is referred to as
            co2_emissions_frominput_energies
            In https://ghgprotocol.org/sites/default/files/standards/ghg-protocol-revised.pdf p.31, scope 2 definition only accounts for
                indirect CO2 emissions of purchased electricity
    CO2_techno = -scope 1 + scope 2- = CO2_from_production + CO2_from_other_consumption
    CO2_per_use = - scope 3 (partially, ie only the indirect downstream emissions)- : CO2 emitted when using the techno
            ex: CO2 emitted when burning techno=biogas, techno=methane, etc
    CO2_per_kWh_techno 	= scope 1 + scope 2 + scope 3 (downstream emissions only)
                = CO2_after_use
				= total_emissions
				= CO2_per_use + CO2_techno
				= CO2_from_production + CO2_from_input_energies + CO2_per_use
    '''

    ns_list = execution_engine.ns_manager.get_all_namespace_with_name(GlossaryCore.NS_ENERGY_MIX)

    # get ns_object with longest 
    for ns in ns_list:
        if hasattr(ns, 'value') and isinstance(ns.value, str):
            if namespace in ns.value:
                namespace_full = ns.value
                # split namespace to get the name without EnergyMix
                namespace = namespace_full.rsplit('.', 1)[0]

    EnergyMix = execution_engine.dm.get_disciplines_with_name(
        f'{namespace}.EnergyMix')[0]

    # Construct a DataFrame to organize the data on two levels: energy and
    # techno
    idx = pd.MultiIndex.from_tuples([], names=['energy', 'techno'])
    multilevel_df = pd.DataFrame(
        index=idx,
        columns=['production', GlossaryCore.InvestValue, 'CO2_per_kWh', 'price_per_kWh', 'price_per_kWh_wotaxes',
                 'CO2_from_production', 'CO2_per_use', 'CO2_after_use', 'CO2_from_other_consumption'])

    energy_list = EnergyMix.get_sosdisc_inputs(GlossaryCore.energy_list)

    years = np.arange(EnergyMix.get_sosdisc_inputs(
        GlossaryCore.YearStart), EnergyMix.get_sosdisc_inputs(GlossaryCore.YearEnd) + 1, 1)
    total_carbon_emissions = None
    for energy in energy_list:
        if energy == 'biomass_dry':
            namespace_disc = f'{namespace}.AgricultureMix'
        else:
            namespace_disc = f'{namespace}.EnergyMix.{energy}'
        energy_disc = execution_engine.dm.get_disciplines_with_name(namespace_disc)[0]
        techno_list = energy_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(
                f'{namespace_disc}.{techno}')[0]
            production_techno = techno_disc.get_sosdisc_outputs(
                'techno_production')[f'{energy} (TWh)'].values * \
                                techno_disc.get_sosdisc_inputs(
                                    'scaling_factor_techno_production')  # TODO: check if scaling required
            # crop had not invest_level but crop_investment. Same for Forest
            # data_fuel_dict is missing in Forest and is a copy of biomass_dry for Crop
            # no detailed CO2_emissions_df for Crop and Forest => CO2_from_other_consumption = 0
            if 'Forest' in techno:
                data_fuel_dict = energy_disc.get_sosdisc_inputs('data_fuel_dict')
                invest_techno = techno_disc.get_sosdisc_inputs('forest_investment')[
                                    'forest_investment'].values * \
                                techno_disc.get_sosdisc_inputs('scaling_factor_forest_investment')
                carbon_emissions = techno_disc.get_sosdisc_outputs(
                    'CO2_emissions')
            elif 'Crop' in techno:
                data_fuel_dict = techno_disc.get_sosdisc_inputs('data_fuel_dict')
                invest_techno = techno_disc.get_sosdisc_inputs('crop_investment')[
                                    'investment'].values * \
                                techno_disc.get_sosdisc_inputs('scaling_factor_crop_investment')
                carbon_emissions = techno_disc.get_sosdisc_outputs(
                    'CO2_emissions')
            else:
                data_fuel_dict = techno_disc.get_sosdisc_inputs('data_fuel_dict')
                invest_techno = techno_disc.get_sosdisc_inputs(GlossaryCore.InvestLevelValue)[
                                    GlossaryCore.InvestValue].values * \
                                techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
                carbon_emissions = techno_disc.get_sosdisc_outputs(
                    'CO2_emissions_detailed')

            # Calculate total CO2 emissions
            CO2_per_use = np.zeros(len(carbon_emissions[GlossaryCore.Years]))
            CO2_from_other_consumption = np.zeros(len(carbon_emissions[GlossaryCore.Years]))
            CO2_from_production = np.zeros(len(carbon_emissions[GlossaryCore.Years]))
            if 'CO2_per_use' in data_fuel_dict and 'high_calorific_value' in data_fuel_dict:
                if data_fuel_dict['CO2_per_use_unit'] == 'kg/kg':
                    CO2_per_use = np.ones(
                        len(carbon_emissions[GlossaryCore.Years])) * data_fuel_dict['CO2_per_use'] / data_fuel_dict[
                                      'high_calorific_value']
                elif data_fuel_dict['CO2_per_use_unit'] == 'kg/kWh':
                    CO2_per_use = np.ones(
                        len(carbon_emissions[GlossaryCore.Years])) * data_fuel_dict['CO2_per_use']
            for emission_type in carbon_emissions:
                if emission_type == GlossaryCore.Years:
                    continue
                elif emission_type == 'production':
                    CO2_from_production = carbon_emissions[emission_type].values
                elif emission_type == techno:
                    total_carbon_emissions = CO2_per_use + \
                                             carbon_emissions[techno].values
                else:
                    CO2_from_other_consumption += carbon_emissions[emission_type].values
            if total_carbon_emissions is None :
                raise Exception("variable total carbon emissions not defined")
            CO2_after_use = total_carbon_emissions
            CO2_per_kWh_techno = total_carbon_emissions
            # Data for scatter plot
            price_per_kWh_techno = techno_disc.get_sosdisc_outputs('techno_prices')[
                f'{techno}'].values
            price_per_kWh_wotaxes_techno = techno_disc.get_sosdisc_outputs('techno_prices')[
                f'{techno}_wotaxes'].values
            idx = pd.MultiIndex.from_tuples(
                [(f'{energy}', f'{techno}')], names=['energy', 'techno'])
            columns_techno = ['energy', 'technology',
                              'production', GlossaryCore.InvestValue,
                              'CO2_per_kWh', 'price_per_kWh',
                              'price_per_kWh_wotaxes', 'CO2_from_production',
                              'CO2_per_use', 'CO2_after_use', 'CO2_from_other_consumption']
            techno_df = pd.DataFrame([(energy, techno, production_techno, invest_techno,
                                       CO2_per_kWh_techno, price_per_kWh_techno, price_per_kWh_wotaxes_techno,
                                       CO2_from_production, CO2_per_use, CO2_after_use, CO2_from_other_consumption)],
                                     index=idx, columns=columns_techno)
            multilevel_df = pd.concat([multilevel_df, techno_df])

    # If columns is not None, return a subset of multilevel_df with selected
    # columns
    if columns is not None and isinstance(columns, list):
        multilevel_df = pd.DataFrame(multilevel_df[columns])

    return multilevel_df, years, namespace


def get_chart_Global_CO2_breakdown_sankey(execution_engine, namespace, chart_name, summary=True):
    '''! Function to create the CO2 breakdown Sankey diagram
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc

    @return new_chart: InstantiatedPlotlyNativeChart a Sankey Diagram
    '''

    # Prepare data
    columns = ['energy', 'technology', 'production', 'CO2_from_production',
               'CO2_per_use', 'CO2_after_use', 'CO2_from_other_consumption']
    multilevel_df, years, namespace = get_multilevel_df(execution_engine, namespace, columns=columns)
    energy_list = list(set(multilevel_df.index.droplevel(1)))
    technologies_list = list(multilevel_df.index.droplevel(0))
    label_col1 = ['CO2 from production',
                  'CO2 per use', 'CO2 from other consumption']
    label_col2 = energy_list
    label_col3 = [GlossaryCore.TotalCO2Emissions]
    label_list = label_col1 + label_col2 + label_col3

    # Create Figure
    chart_name = f'{chart_name}'
    fig = go.Figure()

    # Fill figure with data by year
    # i_label_dict associates each label with an integer value
    i_label_dict = dict((key, i) for i, key in enumerate(label_list))
    fig = go.Figure()
    cmap_over = cm.get_cmap('Reds')
    cmap_under = cm.get_cmap('Greens')
    if summary:
        source, target = [], []
        source_name, target_name = [], []
        flux, flux_color = [], []
        production_list = []
        for i, energy in enumerate(energy_list):
            # energies level
            production = np.sum(multilevel_df.groupby(
                level=0)['production'].sum()[energy])
            # per kWh
            CO2_from_production = np.mean((multilevel_df.loc[energy]['CO2_from_production'] *
                                           multilevel_df.loc[energy]['production']).sum() / multilevel_df.groupby(
                level=0)['production'].sum()[energy])
            CO2_per_use = np.mean((multilevel_df.loc[energy]['CO2_per_use'] *
                                   multilevel_df.loc[energy]['production']).sum() / multilevel_df.groupby(
                level=0)['production'].sum()[energy])
            CO2_after_use = np.mean((multilevel_df.loc[energy]['CO2_after_use'] *
                                     multilevel_df.loc[energy]['production']).sum() / multilevel_df.groupby(
                level=0)['production'].sum()[energy])
            CO2_from_other_consumption = np.mean((multilevel_df.loc[energy]['CO2_from_other_consumption'] *
                                                  multilevel_df.loc[energy]['production']).sum() /
                                                 multilevel_df.groupby(
                                                     level=0)['production'].sum()[energy])
            # total
            CO2_from_production_tot = np.sum((multilevel_df.loc[energy]['CO2_from_production'] *
                                              multilevel_df.loc[energy]['production']).sum()) + \
                                      1e-20
            CO2_per_use_tot = np.sum((multilevel_df.loc[energy]['CO2_per_use'] *
                                      multilevel_df.loc[energy]['production']).sum()) + \
                              1e-20
            CO2_after_use_tot = np.sum((multilevel_df.loc[energy]['CO2_after_use'] *
                                        multilevel_df.loc[energy]['production']).sum()) + \
                                1e-20
            CO2_from_other_consumption_tot = np.sum((multilevel_df.loc[energy]['CO2_from_other_consumption'] *
                                                     multilevel_df.loc[energy]['production']).sum()) + \
                                             1e-20
            # col1 to col2
            source += [i_label_dict['CO2 from production'], ]
            source_name += ['CO2 from production', ]
            target += [i_label_dict[energy], ]
            target_name += [energy, ]
            flux += [CO2_from_production_tot, ]
            flux_color += [CO2_from_production, ]
            production_list += [production, ]
            source += [i_label_dict['CO2 per use'], ]
            source_name += ['CO2 per use', ]
            target += [i_label_dict[energy], ]
            target_name += [energy, ]
            flux += [CO2_per_use_tot, ]
            flux_color += [CO2_per_use, ]
            production_list += [production, ]
            source += [
                i_label_dict['CO2 from other consumption'], ]
            source_name += ['CO2 from other consumption', ]
            target += [i_label_dict[energy], ]
            target_name += [energy, ]
            flux += [CO2_from_other_consumption_tot, ]
            flux_color += [CO2_from_other_consumption, ]
            production_list += [production, ]
            # col2 to col3
            source += [i_label_dict[energy], ]
            source_name += [energy, ]
            target += [i_label_dict[GlossaryCore.TotalCO2Emissions], ]
            target_name += [GlossaryCore.TotalCO2Emissions, ]
            flux += [CO2_after_use_tot, ]
            flux_color += [CO2_after_use, ]
            production_list += [production, ]
        customdata = list(
            np.array([source_name, target_name, production_list, flux_color,
                      np.where(np.array(flux) <= 1e-20, 0.0, np.array(flux))]).T)
        hovertemplate = '<br>Source: %{customdata[0]}' + \
                        '<br>Target: %{customdata[1]}' + \
                        '<br>Aggregated Production: %{customdata[2]: .2e}' + \
                        '<br>Mean CO2 per kWh (color): %{customdata[3]: .2e}' + \
                        '<br>Aggregated Total CO2 (thickness): %{customdata[4]: .2e}'
        rgba_list_over = cmap_over(
            0.1 + np.abs(flux_color) / np.max(np.abs(flux_color)))
        rgba_list_under = cmap_under(
            0.1 + np.abs(flux_color) / np.max(np.abs(flux_color)))
        color_over = ['rgb' + str(tuple(int((255 * (x * 0.8 + 0.2)))
                                        for x in rgba[0:3])) for rgba in rgba_list_over]
        color_under = ['rgb' + str(tuple(int((255 * (x * 0.8 + 0.2)))
                                         for x in rgba[0:3])) for rgba in rgba_list_under]
        color = np.where(np.array(flux_color) > 0.0, color_over, color_under)
        link = dict(source=source, target=target,
                    value=list(np.where(np.abs(flux) > 0.0, 1.0 +
                                        100.0 * np.abs(flux) / np.max(np.abs(flux)), 0.0)),
                    color=list(color),
                    customdata=customdata,
                    hovertemplate=hovertemplate, )
        sankey = go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                                     label=list(i_label_dict.keys()), color="black"),
                           link=link,
                           visible=False)
        fig.add_trace(sankey)

    else:
        for i_year in range(len(years)):
            source, target = [], []
            source_name, target_name = [], []
            flux, flux_color = [], []
            production_list = []
            for i, energy in enumerate(energy_list):
                # energies level
                production = multilevel_df.groupby(
                    level=0)['production'].sum()[energy][i_year]
                # per kWh
                CO2_from_production = (multilevel_df.loc[energy]['CO2_from_production'] *
                                       multilevel_df.loc[energy]['production']).sum()[i_year] / production
                CO2_per_use = (multilevel_df.loc[energy]['CO2_per_use'] *
                               multilevel_df.loc[energy]['production']).sum()[i_year] / production
                CO2_after_use = (multilevel_df.loc[energy]['CO2_after_use'] *
                                 multilevel_df.loc[energy]['production']).sum()[i_year] / production
                CO2_from_other_consumption = (multilevel_df.loc[energy]['CO2_from_other_consumption'] *
                                              multilevel_df.loc[energy]['production']).sum()[i_year] / production
                # total
                CO2_from_production_tot = (multilevel_df.loc[energy]['CO2_from_production'] *
                                           multilevel_df.loc[energy]['production']).sum()[i_year] + \
                                          1e-20
                CO2_per_use_tot = (multilevel_df.loc[energy]['CO2_per_use'] *
                                   multilevel_df.loc[energy]['production']).sum()[i_year] + \
                                  1e-20
                CO2_after_use_tot = (multilevel_df.loc[energy]['CO2_after_use'] *
                                     multilevel_df.loc[energy]['production']).sum()[i_year] + \
                                    1e-20
                CO2_from_other_consumption_tot = (multilevel_df.loc[energy]['CO2_from_other_consumption'] *
                                                  multilevel_df.loc[energy]['production']).sum()[i_year] + \
                                                 1e-20
                # col1 to col2
                source += [i_label_dict['CO2 from production'], ]
                source_name += ['CO2 from production', ]
                target += [i_label_dict[energy], ]
                target_name += [energy, ]
                flux += [CO2_from_production_tot, ]
                flux_color += [CO2_from_production, ]
                production_list += [production, ]
                source += [i_label_dict['CO2 per use'], ]
                source_name += ['CO2 per use', ]
                target += [i_label_dict[energy], ]
                target_name += [energy, ]
                flux += [CO2_per_use_tot, ]
                flux_color += [CO2_per_use, ]
                production_list += [production, ]
                source += [
                    i_label_dict['CO2 from other consumption'], ]
                source_name += ['CO2 from other consumption', ]
                target += [i_label_dict[energy], ]
                target_name += [energy, ]
                flux += [CO2_from_other_consumption_tot, ]
                flux_color += [CO2_from_other_consumption, ]
                production_list += [production, ]
                # col2 to col3
                source += [i_label_dict[energy], ]
                source_name += [energy, ]
                target += [i_label_dict[GlossaryCore.TotalCO2Emissions], ]
                target_name += [GlossaryCore.TotalCO2Emissions, ]
                flux += [CO2_after_use_tot, ]
                flux_color += [CO2_after_use, ]
                production_list += [production, ]
            customdata = list(
                np.array([source_name, target_name, production_list, flux_color,
                          np.where(np.array(flux) <= 1e-20, 0.0, np.array(flux))]).T)
            hovertemplate = '<br>Source: %{customdata[0]}' + \
                            '<br>Target: %{customdata[1]}' + \
                            '<br>Production: %{customdata[2]: .2e}' + \
                            '<br>CO2 per kWh (color): %{customdata[3]: .2e}' + \
                            '<br>Total CO2 (thickness): %{customdata[4]: .2e}'
            rgba_list_over = cmap_over(
                0.1 + np.abs(flux_color) / np.max(np.abs(flux_color)))
            rgba_list_under = cmap_under(
                0.1 + np.abs(flux_color) / np.max(np.abs(flux_color)))
            color_over = ['rgb' + str(tuple(int((255 * (x * 0.8 + 0.2)))
                                            for x in rgba[0:3])) for rgba in rgba_list_over]
            color_under = ['rgb' + str(tuple(int((255 * (x * 0.8 + 0.2)))
                                             for x in rgba[0:3])) for rgba in rgba_list_under]
            color = np.where(np.array(flux_color) > 0.0,
                             color_over, color_under)
            link = dict(source=source, target=target,
                        value=list(np.where(
                            np.abs(flux) > 0.0, 1.0 + 100.0 * np.abs(flux) / np.max(np.abs(flux)), 0.0)),
                        color=list(color),
                        customdata=customdata,
                        hovertemplate=hovertemplate, )
            sankey = go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                                         label=list(i_label_dict.keys()), color="black"),
                               link=link,
                               visible=False)
            fig.add_trace(sankey)

        # Prepare year slider and layout updates
        steps = []
        for i in range(int(len(fig.data)) - 1):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, ],
                label=str(2020 + i)
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Year: "},
            steps=steps), ]

        fig.update_layout(
            sliders=sliders
        )

    fig.data[0].visible = True

    new_chart = InstantiatedPlotlyNativeChart(
        fig, chart_name=chart_name, default_title=True)
    return new_chart

# def get_chart_breakdown_price_energies(execution_engine, namespace, chart_name):
# '''! Function to create the breakdown_prices_techno/_energy Sankey diagram with the associated scatter plot
# @param execution_engine: Execution engine object from which the data is gathered
# @param namespace: String containing the namespace to access the data
# @param chart_name:String, title of the post_proc
#
# @return new_chart: InstantiatedPlotlyNativeChart a Sankey Diagram and a Scatter plot side to side
# '''
#
# # Prepare data
# multilevel_df, years = get_price_multilevel_df(execution_engine, namespace)
#
# # Create Figure
# chart_name = f'{chart_name}'
# fig = go.Figure()
#
# price_sources = ['transport', 'factory',
# 'energy_cost', GlossaryCore.EnergyInvestmentsValue, 'margin']
# energy_list = list(set(multilevel_df.index.droplevel(1)))
# technologies_list = list(multilevel_df.index.droplevel(0))
# stream_category_list = list(
# ['Energy', 'CCUS.CarbonCapture', 'CCUS.CarbonStorage'])
# label_list = price_sources + technologies_list + \
# energy_list + stream_category_list
#
# # i_label_dict associates each label with an integer value
# i_label_dict = dict((key, i) for i, key in enumerate(label_list))
# x_scatter = GlossaryCore.InvestValue
# y_scatter = 'CO2_per_kWh'
#
# # Create Figure
# chart_name = f'{chart_name}'
# fig = go.Figure()
#
# # Fill figure with data by year
# for i_year in range(len(years)):
# source, target, color = [], [], []
# fluxes = []
# for i, row in multilevel_df.iterrows():
# # Add flux from price_sources to techno
# for price_source in price_sources:
# source += [i_label_dict[price_source], ]
# target += [i_label_dict[row['technology']], ]
# fluxes += [max(row[f'{price_source}'][i_year], 1e-10), ]
# # Add flux from techno to energy
# source += [i_label_dict[row['technology']], ]
# target += [i_label_dict[row['stream']], ]
# fluxes += [max(row['weighted_techno_price'][i_year], 1e-10), ]
# for key, group_df in multilevel_df[['weighted_stream_price', 'stream_category']].groupby(axis=0, level=0):
# # Add flux from energy to stream_category
# source += [i_label_dict[next(group_df.iterrows())[0][0]], ]
# target += [i_label_dict[next(group_df.iterrows())
# [1]['stream_category']], ]
# fluxes += [max(next(group_df.iterrows())[1]
# ['weighted_stream_price'][i_year], 1e-10), ]
# link = dict(source=source, target=target,
# value=fluxes)
# customdata = [i_year]
# sankey = go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
# label=list(i_label_dict.keys()), color="black"),
# link=link,
# customdata=customdata,
# visible=False)
# fig.add_trace(sankey)
#
# fig.data[0].visible = True
#
# # Prepare year slider and layout updates
# steps = []
# for i in range(len(fig.data)):
# step = dict(
# method="update",
# args=[{"visible": [False] * len(fig.data)},
# {"title": 'Year: ' + str(2020 + i)}],
# label=str(2020 + i)
# )
# # visible_index = [True if trace['customdata']
# # [0] == i else False for trace in fig.data]
# step["args"][0]["visible"][i] = True
# steps.append(step)
#
# sliders = [dict(
# active=0,
# currentvalue={"prefix": "Year: "},
# steps=steps), ]
#
# layout = {}
# layout.update({'width': 1000})
# layout.update({'height': 450})
# layout.update({'autosize': False})
# layout.update({'showlegend': False})
# fig.update_layout(layout)
#
# fig.update_layout(
# sliders=sliders
# )
#
# new_chart = InstantiatedPlotlyNativeChart(
# fig, chart_name=chart_name, default_title=True)
# return new_chart
#
#
# def get_price_multilevel_df(execution_engine, namespace):
# '''! Function to create the sankey dataframe with all the data necessary for the 'energy price' Sankey diagram post_proc
# @param execution engine: Current execution engine object, from which the data is extracted
# @param namespace: Namespace at which the data can be accessed
#
# @return multilevel_df: Dataframe
# '''
# EnergyMix = execution_engine.dm.get_disciplines_with_name(
# f'{namespace}.EnergyMix')[0]
# # Construct a DataFrame to organize the data on two levels: energy and
# # techno
# idx = pd.MultiIndex.from_tuples([], names=['stream', 'techno'])
# multilevel_df = pd.DataFrame(
# index=idx,
# columns=['stream', 'technology', 'stream_price', 'transport',
# 'factory', 'energy_cost', GlossaryCore.EnergyInvestmentsValue, 'margin',
# 'production_stream', 'production_techno',
# 'weighted_techno_price', 'weighted_stream_price',
# GlossaryCore.InvestValue, 'CO2_per_kWh', 'stream_category'])
# energy_list = EnergyMix.get_sosdisc_inputs(GlossaryCore.energy_list)
# for energy in energy_list:
# energy_disc = execution_engine.dm.get_disciplines_with_name(
# f'{namespace}.EnergyMix.{energy}')[0]
# techno_list = energy_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
# for techno in techno_list:
# techno_disc = execution_engine.dm.get_disciplines_with_name(
# f'{namespace}.EnergyMix.{energy}.{techno}')[0]
# #---------
# # Data for Sankey diagram
# #---------
# #- Prices
# techno_price = techno_disc.get_sosdisc_outputs('techno_prices')[
# f'{techno}'].values
# stream_price = energy_disc.get_sosdisc_outputs('energy_prices')[
# f'{energy}'].values
# techno_detailed_prices = techno_disc.get_sosdisc_outputs(
# 'techno_detailed_prices')
# transport_techno = techno_detailed_prices['transport'].values
# factory_techno = techno_detailed_prices[f'{techno}_factory'].values
# energy_cost_techno = techno_detailed_prices['energy_costs'].values
# CO2_taxes_techno = techno_detailed_prices['CO2_taxes_factory'].values
# margin_techno = (techno_detailed_prices[['transport', f'{techno}_factory', 'energy_costs', 'CO2_taxes_factory']].sum(axis=1) * techno_disc.get_sosdisc_inputs('margin')[
# 'margin'] / 100).values
# #- Prod
# production_stream = energy_disc.get_sosdisc_outputs(
# GlossaryCore.EnergyProductionValue)[f'{energy}'].values *\
# energy_disc.get_sosdisc_inputs(
# 'scaling_factor_energy_production')
# production_techno = techno_disc.get_sosdisc_outputs(
# 'techno_production')[f'{energy} (TWh)'].values *\
# techno_disc.get_sosdisc_inputs(
# 'scaling_factor_techno_production')
# techno_mix = energy_disc.get_sosdisc_outputs('techno_mix')[
# f'{techno}'].values
# energy_mix = EnergyMix.get_sosdisc_outputs('energy_mix')[
# f'{energy}'].values
# weighted_techno_price = techno_price * techno_mix
# weighted_stream_price = stream_price * energy_mix
# stream_category = 'Energy'
#
# #---------
# # Data for Scatter plot
# #---------
# invest_techno = techno_disc.get_sosdisc_inputs(GlossaryCore.InvestLevelValue)[
# fGlossaryCore.InvestValue].values *\
# techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
# CO2_per_kWh_techno = techno_disc.get_sosdisc_outputs('CO2_emissions')[
# f'{techno}'].values
# idx = pd.MultiIndex.from_tuples(
# [(f'{energy}', f'{techno}')], names=['stream', 'techno'])
# columns = ['stream', 'technology', 'stream_price', 'transport',
# 'factory', 'energy_cost', GlossaryCore.EnergyInvestmentsValue, 'margin',
# 'production_stream', 'production_techno',
# 'weighted_techno_price', 'weighted_stream_price',
# GlossaryCore.InvestValue, 'CO2_per_kWh', 'stream_category']
# techno_df = pd.DataFrame([(energy, techno, stream_price, transport_techno, factory_techno,
# energy_cost_techno, CO2_taxes_techno, margin_techno,
# production_stream, production_techno,
# weighted_techno_price, weighted_stream_price,
# invest_techno, CO2_per_kWh_techno, stream_category)],
# index=idx, columns=columns)
# multilevel_df = pd.concat([multilevel_df,techno_df])
#
# ccs_list = EnergyMix.get_sosdisc_inputs(GlossaryCore.ccs_list)
# for stream in ccs_list:
# stream_disc = execution_engine.dm.get_disciplines_with_name(
# f'{namespace}.CCUS.{stream}')[0]
# techno_list = stream_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
# for techno in techno_list:
# techno_disc = execution_engine.dm.get_disciplines_with_name(
# f'{namespace}.CCUS.{stream}.{techno}')[0]
# #---------
# # Data for Sankey diagram
# #---------
# #- Prices
# stream_price = stream_disc.get_sosdisc_outputs('energy_prices')[
# f'{stream}'].values
# techno_detailed_prices = techno_disc.get_sosdisc_outputs(
# 'techno_detailed_prices')
# transport_techno = techno_detailed_prices['transport'].values
# factory_techno = techno_detailed_prices[f'{techno}_factory'].values
# energy_cost_techno = techno_detailed_prices['energy_costs'].values
# CO2_taxes_techno = techno_detailed_prices['CO2_taxes_factory'].values
# margin_techno = (techno_detailed_prices[['transport', f'{techno}_factory', 'energy_costs', 'CO2_taxes_factory']].sum(axis=1) * techno_disc.get_sosdisc_inputs('margin')[
# 'margin'] / 100).values
# #- Prod
# production_stream = stream_disc.get_sosdisc_outputs(
# GlossaryCore.EnergyProductionValue)[f'{stream}'].values *\
# stream_disc.get_sosdisc_inputs(
# 'scaling_factor_energy_production')
# production_techno = techno_disc.get_sosdisc_outputs(
# 'techno_production')[f'{stream} ({GlossaryEnergy.mass_unit})'].values *\
# techno_disc.get_sosdisc_inputs(
# 'scaling_factor_techno_production')
# techno_mix = stream_disc.get_sosdisc_outputs('techno_mix')[
# f'{techno}'].values
# weighted_techno_price = techno_price * techno_mix
# weighted_stream_price = stream_price
# if stream == 'carbon_capture':
# stream_category = 'CCUS.CarbonCapture'
# if stream == 'carbon_storage':
# stream_category = 'CCUS.CarbonStorage'
# # Data for scatter plot
# invest_techno = techno_disc.get_sosdisc_inputs(GlossaryCore.InvestLevelValue)[
# fGlossaryCore.InvestValue].values *\
# techno_disc.get_sosdisc_inputs('scaling_factor_invest_level')
# CO2_per_kWh_techno = techno_disc.get_sosdisc_outputs('CO2_emissions')[
# f'{techno}'].values
# idx = pd.MultiIndex.from_tuples(
# [(f'{stream}', f'{techno}')], names=['stream', 'techno'])
# columns = ['stream', 'technology', 'stream_price', 'transport',
# 'factory', 'energy_cost', GlossaryCore.EnergyInvestmentsValue, 'margin',
# 'production_stream', 'production_techno',
# 'weighted_techno_price', 'weighted_stream_price',
# GlossaryCore.InvestValue, 'CO2_per_kWh', 'stream_category']
# techno_df = pd.DataFrame([(stream, techno, stream_price, transport_techno, factory_techno,
# energy_cost_techno, CO2_taxes_techno, margin_techno,
# production_stream, production_techno,
# weighted_techno_price, weighted_stream_price,
# invest_techno, CO2_per_kWh_techno, stream_category)],
# index=idx, columns=columns)
# multilevel_df = pd.concat([multilevel_df,techno_df])
#
# years = np.arange(EnergyMix.get_sosdisc_inputs(
# GlossaryCore.YearStart), EnergyMix.get_sosdisc_inputs(GlossaryCore.YearEnd) + 1, 1)
#
# return multilevel_df, years
