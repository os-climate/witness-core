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
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class SectorRedistributionInvestsModel:
    """model for energy and investment redistribution between economy sectors"""
    def __init__(self):
        self.inputs = dict()

    def compute_invest_redistribution(self) -> tuple[dict, pd.DataFrame]:
        """distrubute total energy production between sectors"""
        economics_df: pd.DataFrame = self.inputs[GlossaryCore.EconomicsDfValue]
        net_output = economics_df[GlossaryCore.OutputNetOfDamage].values

        sectors_invests = {}
        all_sectors_invests_df = {GlossaryCore.Years: economics_df[GlossaryCore.Years].values}
        for sector in GlossaryCore.SectorsPossibleValues:
            sector_invests_values = self.inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'][
                                       GlossaryCore.ShareInvestment].values / 100. * net_output
            sector_invests_df = pd.DataFrame(
                {GlossaryCore.Years: economics_df[GlossaryCore.Years].values,
                 GlossaryCore.InvestmentsValue: sector_invests_values}
            )

            all_sectors_invests_df[sector] = sector_invests_values

            sectors_invests[sector] = sector_invests_df

        all_sectors_invests_df = pd.DataFrame(all_sectors_invests_df)
        all_sectors_invests_df[GlossaryCore.InvestmentsValue] = all_sectors_invests_df[GlossaryCore.SectorsPossibleValues].sum(axis=1)

        return sectors_invests, all_sectors_invests_df

    def compute(self, inputs: dict):
        self.inputs = inputs

        sectors_invests, all_sectors_invests_df = self.compute_invest_redistribution()

        return sectors_invests, all_sectors_invests_df
