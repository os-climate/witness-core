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
from typing import Any, Union

import pandas as pd
from pandas import DataFrame

from climateeconomics.glossarycore import GlossaryCore


class SectorRedistributionEnergyModel:
    """model for energy and investment redistribution between economy sectors"""

    def __init__(self):
        self.inputs = dict()
        self.sectors = list()
        self.deduced_sector = ""
        self.missing_sector_share = None

    def compute_energy_redistribution(self) -> tuple[
        dict[Union[str, Any], DataFrame], DataFrame, DataFrame, DataFrame]:
        """
        Distribute total energy production between sectors using sector list and share per sector input
        In addition to sectors list energy is distributed for residential and "other" category
        """
        total_energy_production: pd.DataFrame = self.inputs[
            GlossaryCore.EnergyProductionValue
        ]
        total_energy_production_values = total_energy_production[
            GlossaryCore.TotalProductionValue
        ].values
        all_sectors_energy_df = DataFrame({
            GlossaryCore.Years: total_energy_production[GlossaryCore.Years]
        })

        all_sectors_share_df = DataFrame({
            GlossaryCore.Years: total_energy_production[GlossaryCore.Years]
        })

        sectors_energy = {}
        computed_sectors = list(
            filter(lambda x: x != self.deduced_sector, self.sectors)
        )
        for sector in computed_sectors:
            # Add sector share to dataframe
            all_sectors_share_df[sector] = self.inputs[
                f"{sector}.{GlossaryCore.ShareSectorEnergyDfValue}"
            ][GlossaryCore.ShareSectorEnergy].values

            # Compute energy for the sector
            sector_energy_values = (
                self.inputs[f"{sector}.{GlossaryCore.ShareSectorEnergyDfValue}"][
                    GlossaryCore.ShareSectorEnergy
                ].values
                / 100.0
                * total_energy_production_values
            )

            all_sectors_energy_df[sector] = sector_energy_values

            sector_energy_df = pd.DataFrame(
                {
                    GlossaryCore.Years: total_energy_production[
                        GlossaryCore.Years
                    ].values,
                    GlossaryCore.TotalProductionValue: sector_energy_values,
                }
            )

            sectors_energy[sector] = sector_energy_df

        # Residential energy
        residential_energy_values = (
            self.inputs[GlossaryCore.ShareResidentialEnergyDfValue][
                GlossaryCore.ShareSectorEnergy
            ].values
            / 100
            * total_energy_production_values
        )

        residential_energy_df = pd.DataFrame(
            {
                GlossaryCore.Years: total_energy_production[GlossaryCore.Years].values,
                GlossaryCore.TotalProductionValue: residential_energy_values,
            }
        )
        all_sectors_energy_df[GlossaryCore.ResidentialCategory] = (
            residential_energy_values
        )

        # Other category
        other_energy_values = (
            self.inputs[GlossaryCore.ShareOtherEnergyDfValue][
                GlossaryCore.ShareSectorEnergy
            ].values
            / 100
            * total_energy_production_values
        )
        all_sectors_energy_df[GlossaryCore.OtherEnergyCategory] = other_energy_values

        # Compute leftover energy for last sector
        missing_sector_energy = (
            total_energy_production_values
            - all_sectors_energy_df.loc[
                :, all_sectors_energy_df.columns != GlossaryCore.Years
            ]
            .sum(axis=1)
            .values
        )

        all_sectors_energy_df[self.deduced_sector] = missing_sector_energy
        sectors_energy[self.deduced_sector] = pd.DataFrame(
            {
                GlossaryCore.Years: total_energy_production[GlossaryCore.Years].values,
                GlossaryCore.TotalProductionValue: missing_sector_energy,
            }
        )

        # Compute leftover share as the ratio of sector energy and total energy production x 100
        all_sectors_share_df[self.deduced_sector] = (
            missing_sector_energy / total_energy_production_values
        ) * 100.0

        return (
            sectors_energy,
            all_sectors_energy_df,
            residential_energy_df,
            all_sectors_share_df,
        )

    def compute(
        self, inputs: dict
    ) -> tuple[dict[Union[str, Any], DataFrame], DataFrame, DataFrame, DataFrame]:
        self.inputs = inputs
        self.sectors = inputs[GlossaryCore.SectorListValue]
        self.deduced_sector = inputs[GlossaryCore.MissingSectorNameValue]

        (
            sectors_energy,
            all_sectors_energy_df,
            residential_energy_df,
            all_sectors_share_df,
        ) = self.compute_energy_redistribution()

        return (
            sectors_energy,
            all_sectors_energy_df,
            residential_energy_df,
            all_sectors_share_df,
        )
