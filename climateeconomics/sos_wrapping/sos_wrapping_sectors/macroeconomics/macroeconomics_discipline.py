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
from copy import deepcopy

from climateeconomics.core.core_sectorization.macroeconomics_sectorization_model import MacroeconomicsModel
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart


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
        'icon': 'fa-solid fa-city',
        'version': '',
    }

    DESC_IN = {
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.ShareMaxInvestName: GlossaryCore.ShareMaxInvest,
        GlossaryCore.MaxInvestConstraintRefName: GlossaryCore.MaxInvestConstraintRef
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDfValue: GlossaryCore.SectorizedEconomicsDf,
        GlossaryCore.EconomicsDetailDfValue: GlossaryCore.SectorizedEconomicsDetailDf,
        GlossaryCore.MaxInvestConstraintName: GlossaryCore.MaxInvestConstraint,
        GlossaryCore.InvestmentDfValue: GlossaryCore.InvestmentDf
    }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.macro_model = MacroeconomicsModel(inputs_dict)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            # ensure sector list is not None before going further for configuration steps
            if sector_list is not None:
                for sector in sector_list:
                    capital_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.CapitalDf)
                    capital_df_disc[self.NAMESPACE] = GlossaryCore.NS_MACRO
                    dynamic_inputs[f'{sector}.{GlossaryCore.CapitalDfValue}'] = capital_df_disc
                    dynamic_inputs[f'{sector}.{GlossaryCore.ProductionDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.ProductionDf)
                    # add investment_df for each sector as input
                    dynamic_inputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)

            self.add_inputs(dynamic_inputs)

    def run(self):
        """run method"""
        inputs_dict = self.get_sosdisc_inputs()
        self.macro_model.compute(inputs_dict)

        outputs_dict = {
            GlossaryCore.EconomicsDfValue: self.macro_model.economics_df,
            GlossaryCore.EconomicsDetailDfValue: self.macro_model.economics_detail_df,
            GlossaryCore.MaxInvestConstraintName: self.macro_model.max_invest_constraint,
            GlossaryCore.InvestmentDfValue: self.macro_model.sum_invests_df
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
            self.set_partial_derivative_for_other_types((GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
                                                        (f'{sector}.{GlossaryCore.ProductionDfValue}',
                                                         GlossaryCore.GrossOutput),
                                                        identity_mat)
            self.set_partial_derivative_for_other_types((GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
                                                        (f'{sector}.{GlossaryCore.ProductionDfValue}', GlossaryCore.OutputNetOfDamage),
                                                        identity_mat)
            self.set_partial_derivative_for_other_types((GlossaryCore.EconomicsDfValue, GlossaryCore.Capital),
                                                        (f'{sector}.{GlossaryCore.CapitalDfValue}', GlossaryCore.Capital), identity_mat)

            # gradient of constraint wrt output net damage for each sector

            self.set_partial_derivative_for_other_types((GlossaryCore.MaxInvestConstraintName, ),
                                                        (f'{sector}.{GlossaryCore.ProductionDfValue}',
                                                         GlossaryCore.OutputNetOfDamage),
                                                        identity_mat/100 * share_max_invest / max_invest_ref)

            # gradient of constraint and invest_df wrt invest for each sector (except for energy)

            self.set_partial_derivative_for_other_types((GlossaryCore.MaxInvestConstraintName,),
                                                        (f'{sector}.{GlossaryCore.InvestmentDfValue}',
                                                         GlossaryCore.InvestmentsValue),
                                                        -1.0 * identity_mat / max_invest_ref)

            self.set_partial_derivative_for_other_types((GlossaryCore.InvestmentDfValue,GlossaryCore.InvestmentsValue),
                                                        (f'{sector}.{GlossaryCore.InvestmentDfValue}',
                                                         GlossaryCore.InvestmentsValue), identity_mat)



        # gradient of constraint and invest_df wrt output net damage for each
        self.set_partial_derivative_for_other_types((GlossaryCore.MaxInvestConstraintName,),
                                                    (f'{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                                     GlossaryCore.EnergyInvestmentsWoTaxValue),
                                                    -1.0 * identity_mat / max_invest_ref)

        self.set_partial_derivative_for_other_types((GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),
                                                    (f'{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                                     GlossaryCore.EnergyInvestmentsWoTaxValue), identity_mat)
    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = [GlossaryCore.GrossOutput,
                      GlossaryCore.OutputNetOfDamage,
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

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue))
        inputs_dict = self.get_sosdisc_inputs()
        sector_list = inputs_dict[GlossaryCore.SectorListValue]
        years = list(economics_df.index)
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if GlossaryCore.GrossOutput in chart_list:


            chart_name = 'Breakdown of gross output'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)


            new_series = InstanciatedSeries(
                years, list(economics_df[GlossaryCore.Damages]), 'Damages', 'bar', True)
            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_df[GlossaryCore.OutputNetOfDamage]), 'Gross output net of damage', 'bar', True)
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, list(economics_df[GlossaryCore.GrossOutput]), 'Gross output', 'lines', True)
            new_chart.series.append(new_series)

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
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)


        if GlossaryCore.Capital in chart_list:

            to_plot = [GlossaryCore.Capital, GlossaryCore.UsableCapital]
            legend = {GlossaryCore.Capital: 'capital stock',
                      GlossaryCore.UsableCapital: 'usable capital stock'}
            years = list(economics_df.index)

            chart_name = 'Total capital stock and usable capital'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'capital stock [T$]',
                                                 chart_name=chart_name)

            for key in to_plot:
                ordonate_data = list(economics_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'share capital' in chart_list:
            capital = economics_df[GlossaryCore.Capital].values
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
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'share output' in chart_list:
            output = economics_df[GlossaryCore.OutputNetOfDamage].values
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
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'output growth' in chart_list:

            to_plot = [GlossaryCore.OutputGrowth]
            years = list(economics_df.index)
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' growth rate [-]',
                                                 chart_name=chart_name)
            for key in to_plot:
                ordonate_data = list(economics_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', True)
                new_chart.series.append(new_series)

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
                new_chart.series.append(new_series)

            # add investments in energy to the chart as well
            invest_energy = inputs_dict[GlossaryCore.EnergyInvestmentsWoTaxValue]
            ordonate_data = list(invest_energy[GlossaryCore.EnergyInvestmentsWoTaxValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, f'energy investments', 'bar')
            new_chart.series.append(new_series)

            #add total investments
            total_invest = self.get_sosdisc_outputs(GlossaryCore.InvestmentDfValue)
            ordonate_data = list(total_invest[GlossaryCore.InvestmentsValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, f'total investments', 'lines')
            new_chart.series.append(new_series)

            # add chart to instanciated charts
            instanciated_charts.append(new_chart)

        return instanciated_charts
