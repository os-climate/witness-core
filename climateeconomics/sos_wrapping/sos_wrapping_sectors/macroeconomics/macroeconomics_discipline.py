'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/09 Copyright 2023 Capgemini

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

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.charts_tools import graph_gross_and_net_output
from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import (
    MacroeconomicsModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsDiscipline(ClimateEcoDiscipline):
    ''' Discipline intended to agregate resource parameters'''

    # ontology information
    _ontology_data = {
        'label': 'Macroeconomics Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': "fa-solid fa-money-bill-trend-up",
        'version': '',
    }

    DESC_IN = {
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.ShareMaxInvestName: GlossaryCore.ShareMaxInvest,
        GlossaryCore.MaxInvestConstraintRefName: GlossaryCore.MaxInvestConstraintRef,
        GlossaryCore.DamageToProductivity: {'type': 'bool', 'default': True,
                                            'visibility': 'Shared',
                                            'unit': '-', 'namespace': GlossaryCore.NS_WITNESS},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDfValue: GlossaryCore.SectorizedEconomicsDf,
        GlossaryCore.EconomicsDetailDfValue: GlossaryCore.SectorizedEconomicsDetailDf,
        GlossaryCore.MaxInvestConstraintName: GlossaryCore.MaxInvestConstraint,
        GlossaryCore.InvestmentDfValue: GlossaryCore.InvestmentDf,
        GlossaryCore.DamageDfValue: GlossaryCore.DamageDf,
        GlossaryCore.DamageDetailedDfValue: GlossaryCore.DamageDetailedDf,
    }

    def init_execution(self):
        self.macro_model = MacroeconomicsModel()

    def setup_sos_disciplines(self):
        dynamic_inputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            # ensure sector list is not None before going further for configuration steps
            if sector_list is not None:
                for sector in sector_list:
                    capital_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.CapitalDf)
                    capital_df_disc[self.NAMESPACE] = GlossaryCore.NS_SECTORS

                    dynamic_inputs[f'{sector}.{GlossaryCore.CapitalDfValue}'] = capital_df_disc
                    dynamic_inputs[f'{sector}.{GlossaryCore.ProductionDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.ProductionDf)
                    dynamic_inputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)

                    damage_df = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDf)
                    damage_df.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
                    dynamic_inputs[f'{sector}.{GlossaryCore.DamageDfValue}'] = damage_df
                    damage_detailed_df = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf)
                    damage_detailed_df.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
                    dynamic_inputs[f'{sector}.{GlossaryCore.DamageDetailedDfValue}'] = damage_detailed_df

            self.add_inputs(dynamic_inputs)

    def run(self):
        """run method"""
        inputs_dict = self.get_sosdisc_inputs()
        
        self.macro_model.compute(inputs_dict)

        outputs_dict = {
            GlossaryCore.EconomicsDfValue: self.macro_model.economics_df,
            GlossaryCore.EconomicsDetailDfValue: self.macro_model.economics_detail_df,
            GlossaryCore.MaxInvestConstraintName: self.macro_model.max_invest_constraint,
            GlossaryCore.InvestmentDfValue: self.macro_model.sum_invests_df,
            GlossaryCore.DamageDfValue: self.macro_model.damage_df[GlossaryCore.DamageDf['dataframe_descriptor'].keys()],
            GlossaryCore.DamageDetailedDfValue: self.macro_model.damage_df[GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys()],
        }

        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        net_output and invest wrt sector net_output
        """
        inputs_dict = self.get_sosdisc_inputs()
        sector_list = inputs_dict[GlossaryCore.SectorListValue]
        share_max_invest = inputs_dict[GlossaryCore.ShareMaxInvestName]
        max_invest_ref = inputs_dict[GlossaryCore.MaxInvestConstraintRefName]

        # Generic gradient wrt each sector : same for all sectors
        identity_mat = self.macro_model.get_derivative_sectors()
        for sector in sector_list:
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
                (f'{sector}.{GlossaryCore.ProductionDfValue}', GlossaryCore.GrossOutput),
                identity_mat)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
                (f'{sector}.{GlossaryCore.ProductionDfValue}', GlossaryCore.OutputNetOfDamage),
                identity_mat)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EconomicsDfValue, GlossaryCore.Capital),
                (f'{sector}.{GlossaryCore.CapitalDfValue}', GlossaryCore.Capital),
                identity_mat)
            # wrt output net damage for each sector

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.MaxInvestConstraintName, ),
                (f'{sector}.{GlossaryCore.ProductionDfValue}', GlossaryCore.OutputNetOfDamage),
                identity_mat/100 * share_max_invest / max_invest_ref)

            # gradient of constraint and invest_df wrt invest for each sector (except for energy)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.MaxInvestConstraintName,),
                (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                -1.0 * identity_mat / max_invest_ref)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.InvestmentDfValue,GlossaryCore.InvestmentsValue),
                (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                identity_mat)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.DamageDfValue, GlossaryCore.Damages),
                (f'{sector}.{GlossaryCore.DamageDfValue}', GlossaryCore.Damages),
                identity_mat)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.DamageDfValue, GlossaryCore.EstimatedDamages),
                (f'{sector}.{GlossaryCore.DamageDfValue}', GlossaryCore.EstimatedDamages),
                identity_mat)

        # gradient of constraint and invest_df wrt output net damage for each
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.MaxInvestConstraintName,),
            (f'{GlossaryCore.EnergyInvestmentsWoTaxValue}', GlossaryCore.EnergyInvestmentsWoTaxValue),
            -1.0 * identity_mat / max_invest_ref)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),
            (f'{GlossaryCore.EnergyInvestmentsWoTaxValue}', GlossaryCore.EnergyInvestmentsWoTaxValue),
            identity_mat)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = [GlossaryCore.GrossOutput,
                      GlossaryCore.OutputNetOfDamage,
                      GlossaryCore.Damages,
                      GlossaryCore.DamagesFromClimate,
                      GlossaryCore.DamagesFromProductivityLoss,
                      GlossaryCore.Capital,
                      'share capital',
                      'share output',
                      'output growth',
                      'Investments breakdown by sector']

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_detail_df = self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue)
        inputs_dict = self.get_sosdisc_inputs()
        sector_list = inputs_dict[GlossaryCore.SectorListValue]
        years = list(economics_detail_df[GlossaryCore.Years].values)
        compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
        damages_to_productivity = self.get_sosdisc_inputs(GlossaryCore.DamageToProductivity) and compute_climate_impact_on_gdp
        damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
        investment_df = self.get_sosdisc_outputs(GlossaryCore.InvestmentDfValue)
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if GlossaryCore.GrossOutput in chart_list:

            chart_name = 'Gross and net of damage output per year'
            new_chart = graph_gross_and_net_output(chart_name=chart_name,
                                                   compute_climate_impact_on_gdp=compute_climate_impact_on_gdp,
                                                   damages_to_productivity=damages_to_productivity,
                                                   economics_detail_df=economics_detail_df,
                                                   damage_detailed_df=damage_detailed_df)

            instanciated_charts.append(new_chart)

        if GlossaryCore.GrossOutput in chart_list:
            chart_name = 'Consumption breakdown'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'T$', stacked_bar=True, chart_name=chart_name)


            new_series = InstanciatedSeries(years, economics_detail_df[GlossaryCore.GrossOutput], 'Gross output', 'bar', True)
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(years, - damage_detailed_df[GlossaryCore.DamagesFromClimate], 'Immediate damages from climate', 'bar', True)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(years, - investment_df[GlossaryCore.InvestmentsValue], 'Investments', 'bar', True)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(years, economics_detail_df[GlossaryCore.OutputNetOfDamage].values - investment_df[GlossaryCore.InvestmentsValue].values, 'Consumption', 'lines', True)
            new_chart.add_series(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.OutputNetOfDamage in chart_list:
            chart_name = 'Global Output net of damage breakdown by sector'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'T$', stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                production_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.ProductionDfValue}')
                sector_net_output = list(production_df[GlossaryCore.OutputNetOfDamage].values)

                new_series = InstanciatedSeries(years, sector_net_output,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            net_output = economics_detail_df[GlossaryCore.OutputNetOfDamage].values
            new_series = InstanciatedSeries(years, list(net_output), 'Total', 'lines', True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:

            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            to_plot = {}
            if compute_climate_impact_on_gdp:
                to_plot.update({GlossaryCore.DamagesFromClimate: 'Immediate climate damage (applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (
                                    not damages_to_productivity) + 'applied to gross output)', })
            else:
                to_plot.update({
                                   GlossaryCore.EstimatedDamagesFromClimate: 'Immediate climate damage (estimation not applied to net output)',
                                   GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (
                                       not damages_to_productivity) + 'applied to gross output)', })
            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            all_damages = damage_detailed_df[GlossaryCore.EstimatedDamages].values
            years = list(damage_detailed_df[GlossaryCore.Years].values)
            chart_name = 'Breakdown of damages' + ' (not applied)' * (not compute_climate_impact_on_gdp)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key, legend in to_plot.items():
                ordonate_data = list(damage_detailed_df[key])

                new_series = InstanciatedSeries(years, ordonate_data, legend, 'bar', True)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(all_damages), 'Total all damages', 'lines', True)

            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(applied_damages), 'Total applied', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Capital in chart_list:

            to_plot = [GlossaryCore.Capital, GlossaryCore.UsableCapital]
            legend = {GlossaryCore.Capital: 'capital stock',
                      GlossaryCore.UsableCapital: 'usable capital stock'}
            years = list(economics_detail_df[GlossaryCore.Years].values)

            chart_name = 'Total capital stock and usable capital'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'capital stock [T$]',
                                                 chart_name=chart_name, y_min_zero=True)

            for key in to_plot:
                ordonate_data = list(economics_detail_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', True)
                new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                years, economics_detail_df[GlossaryCore.Capital] * 0.85, '85% of capital stock', 'lines', True)
            new_chart.add_series(new_series)
            instanciated_charts.append(new_chart)

        if 'share capital' in chart_list:
            capital = economics_detail_df[GlossaryCore.Capital].values
            chart_name = 'Capital distribution between economic sectors'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total capital stock [%]', stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                capital_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.CapitalDfValue}')
                sector_capital = capital_df[GlossaryCore.Capital].values
                share = (sector_capital / capital) * 100
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'share output' in chart_list:
            output = economics_detail_df[GlossaryCore.OutputNetOfDamage].values
            chart_name = 'Sectors share of total economics net output'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total net output [%]', stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                production_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.ProductionDfValue}')
                sector_output = production_df[GlossaryCore.OutputNetOfDamage].values
                share = (sector_output / output) * 100
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'output growth' in chart_list:

            to_plot = [GlossaryCore.OutputGrowth]
            years = list(economics_detail_df[GlossaryCore.Years].values)
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' growth rate [-]',
                                                 chart_name=chart_name)
            for key in to_plot:
                ordonate_data = list(economics_detail_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', True)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'Investments breakdown by sector' in chart_list:

            chart_name = ('Investments breakdown per sector in G$ over years')

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Sectors investments [G$]',
                                                 chart_name=chart_name, stacked_bar=True)
            # get investment per sector and add to serie
            for sector in sector_list:
                invest_sector = inputs_dict[f'{sector}.{GlossaryCore.InvestmentDfValue}']
                ordonate_data = list(invest_sector[GlossaryCore.InvestmentsValue])
                new_series = InstanciatedSeries(
                    years, ordonate_data, f'{sector} investments', 'bar')
                new_chart.add_series(new_series)

            # add investments in energy to the chart as well
            invest_energy = inputs_dict[GlossaryCore.EnergyInvestmentsWoTaxValue]
            ordonate_data = list(invest_energy[GlossaryCore.EnergyInvestmentsWoTaxValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'energy investments', 'bar')
            new_chart.add_series(new_series)

            #add total investments
            total_invest = self.get_sosdisc_outputs(GlossaryCore.InvestmentDfValue)
            ordonate_data = list(total_invest[GlossaryCore.InvestmentsValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'total investments', 'lines')
            new_chart.add_series(new_series)

            # add chart to instanciated charts
            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:
            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            years = list(damage_detailed_df[GlossaryCore.Years].values)
            chart_name = 'Applied damages by sector'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)
            for sector in sector_list:
                damage_detailed_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.DamageDetailedDfValue}')
                sector_damage = damage_detailed_df[GlossaryCore.Damages].values
                ordonate_data = list(sector_damage)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(applied_damages), 'Total', 'lines', True)

            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Damages"

            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:
            chart_name = 'All damages by sector (climate + productivity loss)'
            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]', chart_name=chart_name,
                                                 stacked_bar=True)
            years = list(damage_detailed_df[GlossaryCore.Years].values)

            for sector in sector_list:
                damage_detailed_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.DamageDetailedDfValue}')
                sector_damage = damage_detailed_df[GlossaryCore.EstimatedDamagesFromClimate].values + \
                                damage_detailed_df[GlossaryCore.EstimatedDamagesFromProductivityLoss].values
                ordonate_data = list(sector_damage)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            all_damages = damage_detailed_df[GlossaryCore.EstimatedDamagesFromClimate].values + damage_detailed_df[
                GlossaryCore.EstimatedDamagesFromProductivityLoss].values
            new_series = InstanciatedSeries(years, list(all_damages), 'Total', 'lines', True)
            new_chart.add_series(new_series)

            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            new_series = InstanciatedSeries(years, list(applied_damages), 'Total applied', 'lines', True)
            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Damages"

            instanciated_charts.append(new_chart)

        if GlossaryCore.DamagesFromClimate in chart_list:

            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            damages_from_climate = damage_detailed_df[GlossaryCore.DamagesFromClimate].values
            years = list(damage_detailed_df[GlossaryCore.Years].values)
            chart_name = 'Immediate damages from climate by sector' + ' (not applied to net output)' * (
                not compute_climate_impact_on_gdp)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)
            for sector in sector_list:
                damage_detailed_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.DamageDetailedDfValue}')
                sector_damage = damage_detailed_df[GlossaryCore.EstimatedDamagesFromClimate].values
                # share = (sector_capital / capital) * 100
                ordonate_data = list(sector_damage)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(damages_from_climate), 'Total', 'lines', True)

            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Damages"

            instanciated_charts.append(new_chart)

        if GlossaryCore.DamagesFromProductivityLoss in chart_list:

            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            damages_from_productivity_loss = damage_detailed_df[GlossaryCore.DamagesFromProductivityLoss].values
            years = list(damage_detailed_df[GlossaryCore.Years].values)
            chart_name = 'Damages from productivity loss by sector' + ' (not applied to gross output)' * (
                not damages_to_productivity)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)
            for sector in sector_list:
                damage_detailed_df = self.get_sosdisc_inputs(f'{sector}.{GlossaryCore.DamageDetailedDfValue}')
                sector_damage = damage_detailed_df[GlossaryCore.EstimatedDamagesFromProductivityLoss].values
                # share = (sector_capital / capital) * 100
                ordonate_data = list(sector_damage)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                sector, 'bar', True)
                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(damages_from_productivity_loss), 'Total', 'lines', True)

            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Damages"

            instanciated_charts.append(new_chart)

        return instanciated_charts
