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


class SectorizedConsumptionModel:
    """model for energy and investment redistribution between economy sectors"""

    def __init__(self):
        self.inputs = {}
        self.output_dict = {}

    def compute_sectors_consumption(self):
        """Sector Consumption S = Net output of sector S - invest in sector S - total energy invest X share energy prod attributed to sector S"""
        years = self.inputs[f"{self.inputs[GlossaryCore.SectorListValue][0]}.{GlossaryCore.ProductionDfValue}"][GlossaryCore.OutputNetOfDamage].values

        sectors_consumption_df = pd.DataFrame({GlossaryCore.Years: years})

        for sector in self.inputs[GlossaryCore.SectorListValue]:

            sector_breakdown_df = pd.DataFrame({GlossaryCore.Years: years})
            net_output_sector = self.inputs[f"{sector}.{GlossaryCore.ProductionDfValue}"][GlossaryCore.OutputNetOfDamage].values
            invest_sector =  - self.inputs[f"{sector}.{GlossaryCore.InvestmentDfValue}"][GlossaryCore.InvestmentsValue].values

            consumption_sector = net_output_sector + invest_sector
            sectors_consumption_df[sector] = consumption_sector

            sector_breakdown_df["Output net of damage"] = net_output_sector
            sector_breakdown_df["Investment in sector"] = invest_sector
            #sector_breakdown_df["Attributed investment in energy"] = invest_in_energy_attributed_to_sector
            sector_breakdown_df["Consumption"] = consumption_sector

            self.output_dict[f"{sector}_consumption_breakdown"] = sector_breakdown_df

        self.output_dict['consumption_detail_df'] = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.Consumption: sectors_consumption_df[self.inputs[GlossaryCore.SectorListValue]].values.sum(axis=1)
        })
        self.output_dict[GlossaryCore.SectorizedConsumptionDfValue] = sectors_consumption_df

    def compute(self, inputs: dict):
        self.inputs = inputs
        self.compute_sectors_consumption()
