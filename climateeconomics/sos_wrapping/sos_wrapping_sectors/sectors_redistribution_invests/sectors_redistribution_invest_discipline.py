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
import copy

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_invests.sectors_redistribution_invests_model import (
    SectorRedistributionInvestsModel,
)


class SectorsRedistributionInvestsDiscipline(SoSWrapp):
    """Discipline redistributing energy production and global investment into sectors"""
    _ontology_data = {
        'label': 'Demand WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-arrows-split-up-and-left',
        'version': '',
    }
    _maturity = 'Research'

    economics_df = copy.deepcopy(GlossaryCore.EconomicsDf)
    del economics_df["dataframe_descriptor"][GlossaryCore.PerCapitaConsumption]
    DESC_IN = {
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.EconomicsDfValue: economics_df,
    }

    DESC_OUT = {
        GlossaryCore.RedistributionInvestmentsDfValue: GlossaryCore.RedistributionInvestmentsDf
    }

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        for sector in GlossaryCore.SectorsPossibleValues:
            dynamic_inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.ShareSectorInvestmentDf)
            dynamic_outputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        """run method"""
        inputs = self.get_sosdisc_inputs()

        model = SectorRedistributionInvestsModel()

        sectors_invests, all_sectors_invests_df = model.compute(inputs)

        outputs = {
            GlossaryCore.RedistributionInvestmentsDfValue: all_sectors_invests_df,
        }

        for sector in GlossaryCore.SectorsPossibleValues:
            outputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = sectors_invests[sector]

        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()

        sectors_list = inputs[GlossaryCore.SectorListValue]
        net_output = inputs[GlossaryCore.EconomicsDfValue][GlossaryCore.OutputNetOfDamage].values

        for sector in sectors_list:
            sector_share_invests = inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'][GlossaryCore.ShareInvestment].values
            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
                np.diag(sector_share_invests/ 100.)
            )

            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                (f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}', GlossaryCore.ShareInvestment),
                np.diag(net_output) / 100.
            )

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = [GlossaryCore.RedistributionInvestmentsDfValue,
                      GlossaryCore.ShareSectorInvestmentDfValue,]

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        all_filters = True
        charts = []

        if filters is not None:
            charts = filters

        instanciated_charts = []

        if all_filters or GlossaryCore.InvestmentsValue:
            # first graph
            all_sectors_invests_df = self.get_sosdisc_outputs(
                GlossaryCore.RedistributionInvestmentsDfValue)

            chart_name = f"Investments breakdown by sectors [{GlossaryCore.RedistributionInvestmentsDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 GlossaryCore.RedistributionInvestmentsDf['unit'],
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            years = list(all_sectors_invests_df[GlossaryCore.Years])
            for sector in GlossaryCore.SectorsPossibleValues:
                sector_invest = all_sectors_invests_df[sector].values
                new_series = InstanciatedSeries(years,
                                                list(sector_invest),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            total_invests = all_sectors_invests_df[GlossaryCore.InvestmentsValue].values
            new_series = InstanciatedSeries(years,
                                            list(total_invests),
                                            'Total', 'lines', True)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

            # second graph
            chart_name = "Share of total investments production allocated to sectors [%]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 '%',
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in GlossaryCore.SectorsPossibleValues:
                sector_invest = all_sectors_invests_df[sector].values
                share_sector = sector_invest / total_invests * 100.
                new_series = InstanciatedSeries(years,
                                                list(share_sector),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts

