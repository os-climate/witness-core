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
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import AgricultureDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import ServicesDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import IndustrialDiscipline
from gemseo.third_party.prettytable.prettytable import NONE


class MacroeconomicsModel():
    """
    Sector pyworld3
    General implementation of sector pyworld3
    """

    SECTORS_DISC_LIST = [AgricultureDiscipline, ServicesDiscipline, IndustrialDiscipline]
    SECTORS_LIST = GlossaryCore.SectorsPossibleValues
    SECTORS_OUT_UNIT = {disc.sector_name: disc.prod_cap_unit for disc in SECTORS_DISC_LIST}

    def __init__(self, inputs_dict):
        """Constructor"""
        self.inputs = None
        self.economics_df = None
        self.economics_detail_df = None
        self.investments_df = None
        self.years_range = None
        self.configure_parameters(inputs_dict)

    def configure_parameters(self, inputs_dict):
        """Configure with inputs_dict from the discipline"""

        self.investments_df = inputs_dict[GlossaryCore.InvestmentDfValue]
        self.years_range = self.investments_df[GlossaryCore.Years].values

    def compute_economics(self):
        """Compute economics dataframes"""

        capital_to_sum = []
        u_capital_to_sum = []
        output_to_sum = []
        net_output_to_sum = []

        for sector in self.SECTORS_LIST:
            capital_df_sector = self.inputs[f'{sector}.{GlossaryCore.CapitalDfValue}']
            production_df_sector = self.inputs[f'{sector}.{GlossaryCore.ProductionDfValue}']
            capital_to_sum.append(capital_df_sector[GlossaryCore.Capital].values)
            u_capital_to_sum.append(capital_df_sector[GlossaryCore.UsableCapital].values)
            output_to_sum.append(production_df_sector[GlossaryCore.GrossOutput].values)
            net_output_to_sum.append(production_df_sector[GlossaryCore.OutputNetOfDamage].values)

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
        economics_detail_df.index = self.years_range
        self.economics_detail_df = economics_detail_df
        self.economics_df = economics_detail_df[GlossaryCore.SectorizedEconomicsDf['dataframe_descriptor'].keys()]

    def compute_consumption(self):
        """Consumption = Net output - Invests"""
        self.economics_detail_df[GlossaryCore.Consumption] = self.economics_detail_df[GlossaryCore.OutputNetOfDamage].values -\
                                                             self.investments_df[GlossaryCore.InvestmentsValue].values
    def compute(self, inputs):
        """Compute all models for year range"""
        self.inputs = inputs
        self.configure_parameters(inputs)
        self.compute_economics()
        self.compute_consumption()


    # GRADIENTS
    def get_derivative_sectors(self):
        """Compute gradient for netoutput and invest wrt net output from each sector"""
        grad_netoutput = np.identity(len(self.years_range))

        return grad_netoutput
