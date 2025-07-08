'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/03 Copyright 2023 Capgemini

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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsModel():
    """
    Sector pyworld3
    General implementation of sector pyworld3
    """

    SECTORS_LIST = GlossaryCore.SectorsPossibleValues

    def __init__(self):
        """Constructor"""
        self.inputs = None
        self.economics_df = None
        self.outputs = {}
        self.economics_detail_df = None
        self.years_range = None
        self.sectors_list = None
        self.sum_invests_df = None
        self.damage_df = None
        self.gdp_percentage_per_section_df = None
        self.dict_sectors_detailed = None
        self.max_invest_constraint = None
        self.share_max_invest = None
        self.max_invest_constraint_ref = None

    def configure_parameters(self, inputs_dict):
        """Configure with inputs_dict from the discipline"""

        self.sectors_list = inputs_dict[GlossaryCore.SectorListValue]
        self.years_range = np.arange(inputs_dict[GlossaryCore.YearStart], inputs_dict[GlossaryCore.YearEnd] + 1)
        self.max_invest_constraint_ref = inputs_dict[GlossaryCore.MaxInvestConstraintRefName]
        # get share max invest and convert percentage
        self.share_max_invest = inputs_dict[GlossaryCore.ShareMaxInvestName] / 100 # input is in %

    def compute_economics(self):
        """Compute economics dataframes"""

        capital_to_sum = []
        u_capital_to_sum = []
        output_to_sum = []
        net_output_to_sum = []
        invest_to_sum = []
        for sector in self.sectors_list:
            capital_df_sector = self.inputs[f'{sector}.{GlossaryCore.CapitalDfValue}']
            production_df_sector = self.inputs[f'{sector}.{GlossaryCore.ProductionDfValue}']
            # get investment for each sector
            invest_sector = self.inputs[f'{sector}.{GlossaryCore.InvestmentDfValue}']
            capital_to_sum.append(capital_df_sector[GlossaryCore.Capital].values)
            u_capital_to_sum.append(capital_df_sector[GlossaryCore.UsableCapital].values)
            output_to_sum.append(production_df_sector[GlossaryCore.GrossOutput].values)
            net_output_to_sum.append(production_df_sector[GlossaryCore.OutputNetOfDamage].values)
            # add investment to a list
            invest_to_sum.append(invest_sector)

        self.sum_capital = np.sum(capital_to_sum, axis=0)
        self.sum_u_capital = np.sum(u_capital_to_sum, axis=0)
        self.sum_gross_output = np.sum(output_to_sum, axis=0)
        self.sum_net_output = np.sum(net_output_to_sum, axis=0)

        gross_output = pd.Series(self.sum_gross_output)
        output_growth = (gross_output.diff() / gross_output.shift(1)).fillna(0.)

        damages = self.sum_gross_output - self.sum_net_output
        economics_detail_df = pd.DataFrame({GlossaryCore.Years: self.years_range,
                                            GlossaryCore.Capital: self.sum_capital,
                                            GlossaryCore.UsableCapital: self.sum_u_capital,
                                            GlossaryCore.GrossOutput: self.sum_gross_output,
                                            GlossaryCore.OutputNetOfDamage: self.sum_net_output,
                                            GlossaryCore.OutputGrowth: output_growth,
                                            GlossaryCore.Damages: damages})

        self.outputs[GlossaryCore.CapitalDfValue] = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.Capital: self.sum_capital,
            GlossaryCore.UsableCapital: self.sum_u_capital,
           })

        economics_detail_df.index = self.years_range
        self.economics_detail_df = economics_detail_df
        self.economics_df = economics_detail_df[GlossaryCore.SectorizedEconomicsDf['dataframe_descriptor'].keys()]

        # compute total sum of all invests
        self.sum_invests_df = pd.DataFrame(columns= [GlossaryCore.Years, GlossaryCore.InvestmentsValue])
        self.sum_invests_df[GlossaryCore.Years] = self.years_range
        self.sum_invests_df[GlossaryCore.InvestmentsValue] = sum(df_invest[GlossaryCore.InvestmentsValue].values for df_invest in invest_to_sum)


    def compute_total_damages(self):
        self.damage_df = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
        })

        column_to_sum = list(GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys())
        column_to_sum.remove(GlossaryCore.Years)
        for col in column_to_sum:
            damage_to_sum = []
            for sector in self.sectors_list:
                damages_detailed_sector = self.inputs[f'{sector}.{GlossaryCore.DamageDetailedDfValue}']
                damages_sector = self.inputs[f'{sector}.{GlossaryCore.DamageDfValue}']
                if col in damages_sector.columns:
                    damage_to_sum.append(damages_sector[col].values)
                else:
                    damage_to_sum.append(damages_detailed_sector[col].values)

            self.damage_df[col] = np.sum(damage_to_sum, axis=0)

    def compute(self, inputs):
        """Compute all models for year range"""
        self.inputs = inputs
        self.configure_parameters(inputs)
        self.compute_economics()
        self.compute_max_invest_constraint()
        self.compute_total_damages()

    def compute_max_invest_constraint(self):
        """
        Method to compute maximum investment constraint in all sectors
        Used formula is : max_investment_constraint = (share_max_invest/100 * output_net_of_damage - total_invest) / max_invest_ref
        """
        self.max_invest_constraint = (self.share_max_invest * self.sum_net_output - self.sum_invests_df[GlossaryCore.InvestmentsValue].values)/self.max_invest_constraint_ref

    # GRADIENTS
    def get_derivative_sectors(self):
        """Compute gradient for netoutput and invest wrt net output from each sector"""
        grad_netoutput = np.identity(len(self.years_range))

        return grad_netoutput
