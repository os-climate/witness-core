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


from sostrades_optimization_plugins.models.differentiable_model import (
    DifferentiableModel,
)

from climateeconomics.glossarycore import GlossaryCore


class SubSectorModel(DifferentiableModel):
    """Generic sector discipline"""
    subsector_name = None
    sector_name = None

    def configure_years(self):
        self.years = self.np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        self.zeros_arrays = 0. * self.years

    def compute(self):
        self.configure_years()
        self.compute_investments()

    def compute_investments(self):
        if self.inputs["mdo_sectors_invest_level"] == 2:

            self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.Years}"] = self.years
            self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"] = \
                self.sum_cols(self.get_cols_input_dataframe(df_name=f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}", expect_years=True))
            for col in self.get_colnames_input_dataframe(df_name=f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}"):
                self.temp_variables[f'invest_details:{col}'] = self.inputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}:{col}"]
        elif self.inputs["mdo_sectors_invest_level"] == 0:
            net_output = self.inputs[f"{GlossaryCore.EconomicsDfValue}:{GlossaryCore.OutputNetOfDamage}"]
            share_invest_sector = self.inputs[f'{GlossaryCore.ShareSectorInvestmentDfValue}:{self.sector_name}']
            share_invest_sub_sectors = self.inputs[f'{self.sector_name}.{GlossaryCore.ShareSectorInvestmentDfValue}:{self.subsector_name}']
            share_invests_inside_sub_sectors_columns = self.get_colnames_input_dataframe(f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.SubShareSectorInvestDfValue}', expect_years=True)

            invest_sub_sector = net_output * GlossaryCore.conversion_dict[GlossaryCore.SectorizedEconomicsDf['unit']][GlossaryCore.SubSectorInvestDf['unit']] * share_invest_sector /100. * share_invest_sub_sectors / 100.
            self.temp_variables[f"invest_details:{GlossaryCore.Years}"] = self.years
            self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}:{GlossaryCore.Years}"] = self.years
            for col in share_invests_inside_sub_sectors_columns:
                self.temp_variables[f"invest_details:{col}"] = invest_sub_sector * self.inputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.SubShareSectorInvestDfValue}:{col}'] / 100.
                self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}:{col}"] = self.temp_variables[f'invest_details:{col}']

            self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.Years}"] = self.years
            self.outputs[f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"] = \
                self.sum_cols(self.get_cols_output_dataframe(df_name=f"{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}", expect_years=True))
        else:
            raise NotImplementedError("")