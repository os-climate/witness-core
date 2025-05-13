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
from copy import deepcopy

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
        GlossaryCore.EconomicsDfValue: GlossaryCore.SectorizedEconomicsDf,
        "mdo_mode_energy": {"visibility": "Shared", "namespace": GlossaryCore.NS_PUBLIC, "type": "bool", 'structuring': True, 'description': "set to true if you optim driver controls raw invests in each energy/ccus techno"},
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
        values_dict, go = self.collect_var_for_dynamic_setup(['mdo_mode_energy', "sector_list_wo_subsector", GlossaryCore.YearStart, GlossaryCore.YearEnd])
        if go:
            # share sector investments
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
                #share_sector_invest_df_var["default"] = default_df
                dynamic_inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'] = share_sector_invest_df_var
                dynamic_outputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)

            if not values_dict["mdo_mode_energy"]:
                share_sector_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.ShareSectorInvestmentDf)
                share_sector_variable["namespace"] = GlossaryCore.NS_WITNESS
                dynamic_inputs[f"{GlossaryCore.CCUS}.{GlossaryCore.ShareSectorInvestmentDfValue}"] = deepcopy(share_sector_variable)
                dynamic_inputs[f"{GlossaryCore.EnergyMix}.{GlossaryCore.ShareSectorInvestmentDfValue}"] = deepcopy(share_sector_variable)

                investments_df_variable = deepcopy(GlossaryCore.InvestmentDf)
                investments_df_variable["namespace"] = GlossaryCore.NS_WITNESS

                dynamic_outputs[f"{GlossaryCore.CCUS}.{GlossaryCore.InvestmentsValue}"] = deepcopy(investments_df_variable)
                dynamic_outputs[f"{GlossaryCore.EnergyMix}.{GlossaryCore.InvestmentsValue}"] = deepcopy(investments_df_variable)


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

        outputs.update(model.outputs)
        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()

        sectors_list = inputs["sector_list_wo_subsector"]

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
                      GlossaryCore.ShareSectorInvestmentDfValue,
                      'Energy & CCUS sectors investments'
                      ]

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        all_filters = True
        charts_list = []
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts_list = chart_filter.selected_values

        instanciated_charts = []
        inputs = self.get_sosdisc_inputs()
        if True:
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

            instanciated_charts.append(new_chart)

        if not self.get_sosdisc_inputs("mdo_mode_energy"):
            if 'Energy & CCUS sectors investments' in charts_list:
                chart_name = 'Energy & CCUS sectors investments'

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.InvestmentDf['unit'],
                                                     stacked_bar=True,
                                                     chart_name=chart_name)

                for sector in [GlossaryCore.CCUS, GlossaryCore.EnergyMix]:
                    sector_invest = self.get_sosdisc_outputs(f"{sector}.{GlossaryCore.InvestmentsValue}")[GlossaryCore.InvestmentsValue]
                    new_series = InstanciatedSeries(years, sector_invest, sector, 'bar', True)
                    new_chart.series.append(new_series)

                instanciated_charts.append(new_chart)


        return instanciated_charts

