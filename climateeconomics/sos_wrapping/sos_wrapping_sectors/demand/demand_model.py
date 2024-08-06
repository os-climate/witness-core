'''
Copyright 2024 Capgemini

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


class DemandModel:
    """model for energy and investment redistribution between economy sectors"""

    def __init__(self):
        self.inputs = dict()
        self.sectors = list()
        self.years = None
        self.population = None

    def compute_sectors_demand(self) -> tuple[dict, pd.DataFrame]:
        """distrubute total energy production between sectors"""
        sectors_demand_dict = {}
        all_sectors_demand_df = {
            GlossaryCore.Years: self.inputs[
                f"{self.sectors[0]}.{GlossaryCore.SectorDemandPerCapitaDfValue}"
            ][GlossaryCore.Years].values
        }
        for sector in self.sectors:
            sector_demand_per_person_values = self.inputs[
                f"{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}"
            ][GlossaryCore.SectorDemandPerCapitaDfValue].values

            # $ /person x million person -> million $ -> 1e-6 T$
            sectors_demand_values = (
                sector_demand_per_person_values * self.population * 1e-6
            )

            sector_demand_df = pd.DataFrame(
                {
                    GlossaryCore.Years: self.years,
                    GlossaryCore.SectorGDPDemandDfValue: sectors_demand_values,
                }
            )
            all_sectors_demand_df[sector] = sectors_demand_values

            sectors_demand_dict[sector] = sector_demand_df

        all_sectors_demand_df = pd.DataFrame(all_sectors_demand_df)

        return sectors_demand_dict, all_sectors_demand_df

    def compute_total_demand(self, all_sectors_demand_df: pd.DataFrame) -> pd.DataFrame:
        """Computes total demand by summing the demand of all sectors"""
        return pd.DataFrame(
            data={
                GlossaryCore.Years: self.years,
                GlossaryCore.Consumption: all_sectors_demand_df[self.sectors].sum(axis=1)
            }
        )

    def compute(self, inputs: dict):
        self.inputs = inputs
        self.sectors = inputs[GlossaryCore.SectorListValue]
        self.years = inputs[GlossaryCore.PopulationDfValue][GlossaryCore.Years].values
        self.population = inputs[GlossaryCore.PopulationDfValue][
            GlossaryCore.PopulationValue
        ].values
        sectors_demand, all_sectors_demand_df = self.compute_sectors_demand()
        total_demand_df = self.compute_total_demand(all_sectors_demand_df)

        return sectors_demand, all_sectors_demand_df, total_demand_df
