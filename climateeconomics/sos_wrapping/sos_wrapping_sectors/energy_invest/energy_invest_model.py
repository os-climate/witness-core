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


class EnergyInvestModel:
    """Model energy invest"""
    def __init__(self):
        self.co2_emissions_Gt = None
        self.co2_taxes = None
        self.co2_tax_efficiency = None
        self.added_renewables_investments = None
        self.raw_energy_investments = None
        self.energy_investments = None

    def compute_added_energy_investments_in_renewables(self):
        """added investments in renewables = emission * co2 taxes * co2 tax efficiency"""
        emissions = self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions] * 1e9  # t CO2
        co2_taxes = self.co2_taxes[GlossaryCore.CO2Tax]  # $/t
        co2_tax_eff = self.co2_tax_efficiency[GlossaryCore.CO2TaxEfficiencyValue] / 100.  # %
        renewables_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$

        self.added_renewables_investments = pd.DataFrame({
            GlossaryCore.Years: self.co2_emissions_Gt[GlossaryCore.Years],
            GlossaryCore.InvestmentsValue: renewables_investments * 10  # 100 G$
        })

    def compute_energy_investments(self):
        """energy investment = energy invest wo tax + energy invest from tax"""
        raw_invests = self.raw_energy_investments[GlossaryCore.EnergyInvestmentsWoTaxValue].values  # T$
        renewables_investments = self.added_renewables_investments[GlossaryCore.InvestmentsValue].values / 10  # T$

        energy_invests = raw_invests + renewables_investments  # T$

        self.energy_investments = pd.DataFrame({
            GlossaryCore.Years: self.raw_energy_investments[GlossaryCore.Years],
            GlossaryCore.EnergyInvestmentsValue: energy_invests * 10  # 100G$
        })

    def set_params(self, inputs):
        self.raw_energy_investments = inputs[GlossaryCore.EnergyInvestmentsWoTaxValue]
        self.co2_taxes = inputs[GlossaryCore.CO2TaxesValue]
        self.co2_tax_efficiency = inputs[GlossaryCore.CO2TaxEfficiencyValue]
        self.co2_emissions_Gt = inputs[GlossaryCore.CO2EmissionsGtValue]

    def compute(self, inputs):
        self.set_params(inputs)

        self.compute_added_energy_investments_in_renewables()
        self.compute_energy_investments()
