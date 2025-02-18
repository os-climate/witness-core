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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class SubSectorDiscipline(AutodifferentiedDisc):
    """Generic sector discipline"""
    subsector_name = 'UndefinedSubSector'  # to overwrite
    sector_name = "UndefinedSector"
    DESC_IN = {
        "mdo_sectors_invest_level": GlossaryCore.MDOSectorsLevel,
    }

    def setup_sos_disciplines(self):  # type: (...) -> None
        dynamic_inputs = {}
        dynamic_outputs = {}

        values_dict, go = self.collect_var_for_dynamic_setup(["mdo_sectors_invest_level", GlossaryCore.YearStart, GlossaryCore.YearEnd])
        if go:
            if values_dict['mdo_sectors_invest_level'] == 0:
                # then we compute invest in subs sector as sub-sector-invest = Net outpput * Share invest sector * Share invest sub sector * Shares invests inside Sub sector

                # requires net output
                economics_df = copy.deepcopy(GlossaryCore.EconomicsDf)
                del economics_df["dataframe_descriptor"][GlossaryCore.PerCapitaConsumption]
                dynamic_inputs[GlossaryCore.EconomicsDfValue] = economics_df

                default_values = {
                    GlossaryCore.SectorAgriculture: DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value,
                    GlossaryCore.SectorIndustry: DatabaseWitnessCore.InvestInduspercofgdp2020.value,
                    GlossaryCore.SectorServices: DatabaseWitnessCore.InvestServicespercofgdpYearStart.value,
                }
                default_df = pd.DataFrame({
                    GlossaryCore.Years: np.arange(values_dict[GlossaryCore.YearStart], values_dict[GlossaryCore.YearEnd] + 1),
                    self.sector_name: default_values[self.sector_name]
                })
                share_sector_invest_df_var = GlossaryCore.get_dynamic_variable(GlossaryCore.SubSectorInvestDf)
                share_sector_invest_df_var["default"] = default_df

                dynamic_inputs[f'{GlossaryCore.ShareSectorInvestmentDfValue}'] = share_sector_invest_df_var
                dynamic_inputs[f'{self.sector_name}.{GlossaryCore.ShareSectorInvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.SubSectorInvestDf)
                dynamic_inputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.SubShareSectorInvestDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.SubShareSectorInvestDf)
                dynamic_outputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.SubSectorInvestDf)
            elif values_dict['mdo_sectors_invest_level'] == 2:
                dynamic_inputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.SubSectorInvestDf)
            else:
                raise NotImplementedError('')
            dynamic_outputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.SubSectorInvestDf)
        di, do = self.add_additionnal_dynamic_variables()
        di.update(dynamic_inputs)
        do.update(dynamic_outputs)
        self.add_inputs(di)
        self.add_outputs(do)

    def add_additionnal_dynamic_variables(self):
        return {}, {}

    def get_invest_details_df(self):
        mdo_level = self.get_sosdisc_inputs("mdo_sectors_invest_level")
        if mdo_level == 2:
            return self.get_sosdisc_inputs(f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}")
        elif mdo_level == 0:
            return self.get_sosdisc_outputs(f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}")
        else:
            raise NotImplementedError

    def get_investment_chart(self):
        sub_sector_invest_details_df = self.get_invest_details_df()
        total_subsector_invest_df = self.get_sosdisc_outputs(f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}")
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Investments [G$]', chart_name=f'Investments in {self.subsector_name}', stacked_bar=True)
        years = sub_sector_invest_details_df[GlossaryCore.Years]
        for col in sub_sector_invest_details_df.columns:
            if col != GlossaryCore.Years:
                new_chart.add_series(InstanciatedSeries(years, sub_sector_invest_details_df[col], col, 'bar'))

        new_chart.add_series(InstanciatedSeries(years, total_subsector_invest_df[GlossaryCore.InvestmentsValue], 'Total', 'lines'))

        new_chart.post_processing_section_name = "Investments and capital"
        return new_chart





