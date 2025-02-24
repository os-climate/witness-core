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
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_invests.sectors_redistribution_invests_model import (
    SectorRedistributionInvestsModel,
)


class SectorsRedistributionInvestsDiscipline(ClimateEcoDiscipline):
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

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        "mdo_mode": {"visibility": "Shared", "namespace": GlossaryCore.NS_PUBLIC, "type": "bool", "default": False, 'structuring': True},
        "mdo_sub_sector_mode": {"visibility": "Shared", "namespace": GlossaryCore.NS_PUBLIC, "type": "bool", "default": False, 'structuring': True},
        "sector_list_wo_subsector": GlossaryCore.SectorListWoSubsector,
    }

    DESC_OUT = {
        GlossaryCore.RedistributionInvestmentsDfValue: GlossaryCore.RedistributionInvestmentsDf
    }

    def _run(self):
        self.run()

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}
        values_dict, go = self.collect_var_for_dynamic_setup(['mdo_mode', "sector_list_wo_subsector", GlossaryCore.YearStart, GlossaryCore.YearEnd])
        if go:
            if values_dict["mdo_mode"]:
                for sector in values_dict["sector_list_wo_subsector"]:
                    dynamic_inputs[f'{sector}.invest_mdo_df'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                    dynamic_outputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
            else:
                economics_df = copy.deepcopy(GlossaryCore.EconomicsDf)
                del economics_df["dataframe_descriptor"][GlossaryCore.PerCapitaConsumption]
                dynamic_inputs[GlossaryCore.EconomicsDfValue] = economics_df
                default_values = {
                    GlossaryCore.SectorAgriculture: DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value,
                    GlossaryCore.SectorIndustry: DatabaseWitnessCore.InvestInduspercofgdp2020.value,
                    GlossaryCore.SectorServices: DatabaseWitnessCore.InvestServicespercofgdpYearStart.value,
                }
                for sector in values_dict["sector_list_wo_subsector"]:
                    default_df = pd.DataFrame({
                        GlossaryCore.Years: np.arange(values_dict[GlossaryCore.YearStart], values_dict[GlossaryCore.YearEnd] + 1),
                        sector: default_values[sector]
                    })
                    share_sector_invest_df_var = GlossaryCore.get_dynamic_variable(GlossaryCore.ShareSectorInvestmentDf)
                    share_sector_invest_df_var["default"] = default_df
                    dynamic_inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'] = share_sector_invest_df_var
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

        for sector in inputs["sector_list_wo_subsector"]:
            outputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = sectors_invests[sector]

        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()

        sectors_list = inputs["sector_list_wo_subsector"]

        if inputs["mdo_mode"]:
            for sector in sectors_list:
                self.set_partial_derivative_for_other_types(
                    (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                    (f'{sector}.invest_mdo_df', GlossaryCore.InvestmentsValue),
                    np.eye(len(inputs[f'{sector}.invest_mdo_df'][GlossaryCore.InvestmentsValue].values))
                )
        else:
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
        inputs = self.get_sosdisc_inputs()
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
            for sector in inputs["sector_list_wo_subsector"]:
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

            for sector in inputs["sector_list_wo_subsector"]:
                sector_invest = all_sectors_invests_df[sector].values
                share_sector = sector_invest / total_invests * 100.
                new_series = InstanciatedSeries(years,
                                                list(share_sector),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts

