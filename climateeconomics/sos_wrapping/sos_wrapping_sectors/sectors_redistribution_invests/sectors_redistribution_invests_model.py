import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class SectorRedistributionInvestsModel:
    """model for energy and investment redistribution between economy sectors"""
    def __init__(self):
        self.inputs = dict()
        self.sectors = list()

    def compute_invest_redistribution(self) -> tuple[dict, pd.DataFrame]:
        """distrubute total energy production between sectors"""
        economics_df: pd.DataFrame = self.inputs[GlossaryCore.EconomicsDfValue]
        net_output = economics_df[GlossaryCore.OutputNetOfDamage].values

        sectors_invests = {}
        all_sectors_invests_df = {GlossaryCore.Years: economics_df[GlossaryCore.Years].values}
        for sector in self.sectors:
            sector_invests_values = self.inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'][
                                       GlossaryCore.ShareInvestment].values / 100. * net_output
            sector_invests_df = pd.DataFrame(
                {GlossaryCore.Years: economics_df[GlossaryCore.Years].values,
                 GlossaryCore.InvestmentsValue: sector_invests_values}
            )

            all_sectors_invests_df[sector] = sector_invests_values

            sectors_invests[sector] = sector_invests_df

        all_sectors_invests_df = pd.DataFrame(all_sectors_invests_df)
        all_sectors_invests_df[GlossaryCore.InvestmentsValue] = all_sectors_invests_df[self.sectors].sum(axis=1)

        return sectors_invests, all_sectors_invests_df

    def compute(self, inputs: dict):
        self.inputs = inputs
        self.sectors = inputs[GlossaryCore.SectorListValue]

        sectors_invests, all_sectors_invests_df = self.compute_invest_redistribution()

        return sectors_invests, all_sectors_invests_df
