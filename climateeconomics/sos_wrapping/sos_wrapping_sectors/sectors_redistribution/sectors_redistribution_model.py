import numpy as np
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

    def compute_total_investments(self) -> pd.DataFrame:
        """compute investment distribution between sectors"""

        sector_investment_values = [self.inputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'][GlossaryCore.InvestmentsValue].values for sector in self.sectors]
        total_invests = np.sum(sector_investment_values, axis=0)

        total_invests_df = pd.DataFrame({
            GlossaryCore.Years: self.inputs[f'{self.sectors[0]}.{GlossaryCore.InvestmentDfValue}'][GlossaryCore.Years].values,
            GlossaryCore.InvestmentsValue: total_invests
        })

        return total_invests_df

    def compute(self, inputs: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        self.inputs = inputs
        self.sectors = inputs[GlossaryCore.SectorListValue]

        sectors_energy, all_sectors_energy_df = self.compute_energy_redistribution()
        total_investments = self.compute_total_investments()

        return sectors_energy, all_sectors_energy_df, total_investments
