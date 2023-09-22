import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class SectorRedistributionModel:
    """model for energy and investment redistribution between economy sectors"""
    def __init__(self):
        self.inputs = dict()
        self.sectors = list()

    def compute_energy_redistribution(self) -> tuple[dict, pd.DataFrame]:
        """distrubute total energy production between sectors"""
        total_energy_production: pd.DataFrame = self.inputs[GlossaryCore.EnergyProductionValue]
        total_energy_production_values = total_energy_production[GlossaryCore.TotalProductionValue].values
        all_sectors_energy_df = {GlossaryCore.Years: total_energy_production[GlossaryCore.Years]}
        sectors_energy = {}
        for sector in self.sectors:
            sector_energy_values = self.inputs[f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'][
                                           GlossaryCore.ShareSectorEnergy].values /100. * total_energy_production_values
            sector_energy_df = pd.DataFrame(
                {GlossaryCore.Years: total_energy_production[GlossaryCore.Years].values,
                 GlossaryCore.TotalProductionValue: sector_energy_values}
            )

            sectors_energy[sector] = sector_energy_df
            all_sectors_energy_df[sector] = sector_energy_values

        all_sectors_energy_df = pd.DataFrame(all_sectors_energy_df)

        return sectors_energy, all_sectors_energy_df

    def compute_investment_redistribution(self) -> tuple[dict, pd.DataFrame]:
        """compute investment distribution between sectors"""
        total_investments: pd.DataFrame = self.inputs[GlossaryCore.InvestmentDfValue]
        total_investments_values = total_investments[GlossaryCore.InvestmentsValue].values

        all_sectors_investments_df = {}
        sectors_invesmtents = {}
        for sector in self.sectors:
            sector_investment_values = self.inputs[f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'][
                                           GlossaryCore.ShareInvestment].values /100. * total_investments_values
            sector_investment_df = pd.DataFrame(
                {GlossaryCore.Years: total_investments[GlossaryCore.Years].values,
                 GlossaryCore.InvestmentsValue: sector_investment_values}
            )

            sectors_invesmtents[sector] = sector_investment_df
            all_sectors_investments_df[sector] = sector_investment_values

        all_sectors_investments_df[GlossaryCore.Years] = total_investments[GlossaryCore.Years]
        all_sectors_investments_df = pd.DataFrame(all_sectors_investments_df)

        return sectors_invesmtents, all_sectors_investments_df

    def compute(self, inputs: dict) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame]:
        self.inputs = inputs
        self.sectors = inputs[GlossaryCore.SectorListValue]

        sectors_energy, all_sectors_energy_df = self.compute_energy_redistribution()
        sectors_invesmtents, all_sectors_investments_df = self.compute_investment_redistribution()

        return sectors_energy, all_sectors_energy_df, sectors_invesmtents, all_sectors_investments_df

    # Derivatives

    def dsector_invest_dsector_output(self, sector, grad_netoutput):
        """
        Compute gradient for sector invest wrt sectors outputs
        """
        # Sector invest = net output * sector_share_invest (share invest in%)
        #grad_sector_invest = grad_netoutput * sectors_invest_share[f'{sector}'].values / 100
        #return grad_sector_invest
