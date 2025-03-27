'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/06 Copyright 2023 Capgemini

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


class SectorModel(DifferentiableModel):
    """Sector pyworld3 General implementation of sector pyworld3"""

    def configure(self):
        """Configure with self.inputs from the discipline"""
        self.year_start = self.inputs[GlossaryCore.YearStart]  # year start
        self.year_end = self.inputs[GlossaryCore.YearEnd]  # year end
        self.years_range = self.np.arange(self.year_start,self.year_end + 1)
        self.sector_name = self.inputs["sector_name"]
        self.section_list = GlossaryCore.SectionDictSectors[self.sector_name]

        if not self.inputs['assumptions_dict']['compute_climate_impact_on_gdp']:
            self.inputs[GlossaryCore.DamageToProductivity] = False

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityGrowthRate}"] = \
            self.inputs['productivity_gr_start'] * self.np.exp(- self.inputs['decline_rate_tfp'] * (self.years_range - self.year_start))

    def compute_productivity(self):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        prod_wo_d = self.inputs['productivity_start']
        productivity_wo_damage_list = [self.inputs['productivity_start']]

        damage_fraction_output = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]
        productivity_growth = self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityGrowthRate}"]

        for prod_growth, damage_frac_year in zip(productivity_growth[:-1], damage_fraction_output[1:]):
            prod_wo_d = prod_wo_d / (1 - prod_growth / 5)
            productivity_wo_damage_list.append(prod_wo_d)

        productivity_w_damage_list = self.np.array(productivity_wo_damage_list) * (1 - damage_fraction_output)

        self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityWithDamage}"] = productivity_w_damage_list
        self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityWithoutDamage}"] = self.np.array(productivity_wo_damage_list)
        if self.inputs[GlossaryCore.DamageToProductivity]:
            self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.Productivity}"] = productivity_w_damage_list
        else:
            self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.Productivity}"] = self.np.array(productivity_wo_damage_list)

    def compute_capital(self):
        """
        K(t), Capital for time period, trillions $USD
        """
        capital = self.inputs['capital_start']
        capital_list = [capital]

        investments = self.inputs[f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"]
        for invest in investments[:-1]:
            capital = (1 - self.inputs['depreciation_capital']) * capital + invest
            capital_list.append(capital)

        self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.Capital}"] = self.np.array(capital_list)

    def compute_usable_capital(self):
        """Usable capital = capital utilisation ratio * energy efficiency * energy production"""
        self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.UsableCapital}"] =\
            self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.EnergyEfficiency}"] *\
            self.outputs[f"{self.sector_name}.{GlossaryCore.EnergyConsumptionValue}:Total"]

    def compute_gross_output(self):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.Years}"] = self.years_range
        alpha = self.inputs['output_alpha']
        gamma = self.inputs['output_gamma']
        productivity = self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.Productivity}"]
        working_pop = self.inputs[f"{GlossaryCore.WorkforceDfValue}:{self.sector_name}"]
        capital_u = self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.UsableCapital}"]
        output = productivity * (alpha * capital_u**gamma + (1 - alpha)* (working_pop)**gamma) **(1 / gamma)
        self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.GrossOutput}"] = output

    def compute_output_net_of_damage(self):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.inputs[GlossaryCore.DamageToProductivity]
        damefrac = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]
        gross_output = self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.GrossOutput}"]

        if not self.inputs['assumptions_dict']['compute_climate_impact_on_gdp']:
            output_net_of_d = gross_output
        else:
            if damage_to_productivity:
                damage = 1 - ((1 - damefrac) / (1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue] * damefrac))
                output_net_of_d = (1 - damage) * gross_output
            else:
                output_net_of_d = gross_output * (1 - damefrac)
        self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.OutputNetOfDamage}"] = output_net_of_d

    def compute_output_net_of_damage_per_section(self):
        """
        Splitting output net of damages between sections of the sector
        """
        self.outputs[f"{self.sector_name}.{GlossaryCore.SectionGdpDfValue}:{GlossaryCore.Years}"] = self.years_range
        for section in self.section_list:
            self.outputs[f"{self.sector_name}.{GlossaryCore.SectionGdpDfValue}:{section}"] = \
                self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.OutputNetOfDamage}"] /100. * \
                self.inputs[f"{self.sector_name}.{GlossaryCore.SectionGdpPercentageDfValue}:{section}"]

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        energy_efficiency = self.inputs['energy_eff_cst'] + \
                            self.inputs['energy_eff_max'] / \
                            (1 + self.np.exp(-self.inputs['energy_eff_k'] * (self.years_range - self.inputs['energy_eff_xzero'])))
        self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.EnergyEfficiency}"] = energy_efficiency

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor), and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityWithDamage}"]
        productivity_wo_damage = self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.ProductivityWithoutDamage}"]
        gross_output = self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.GrossOutput}"]
        applied_productivity = self.outputs[f"{GlossaryCore.ProductivityDfValue}:{GlossaryCore.Productivity}"]

        estimated_damage_from_productivity_loss = (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output
        if self.inputs[GlossaryCore.DamageToProductivity]:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = self.np.zeros_like(estimated_damage_from_productivity_loss)

        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromProductivityLoss}"] = damage_from_productivity_loss
        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromProductivityLoss}"] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]
        gross_output = self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.GrossOutput}"]
        net_output = self.outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.OutputNetOfDamage}"]

        damage_from_climate = self.np.zeros_like(gross_output)
        if self.inputs['assumptions_dict']['compute_climate_impact_on_gdp']:
            damage_from_climate = gross_output - net_output
            estimated_damage_from_climate = damage_from_climate
        else:
            if self.inputs[GlossaryCore.DamageToProductivity]:
                estimated_damage_from_climate = gross_output * damefrac * (1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue]) / (
                        1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue] * damefrac)
            else:
                estimated_damage_from_climate = gross_output * damefrac

        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromClimate}"] = damage_from_climate
        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromClimate}"] = estimated_damage_from_climate

    def compute_total_damages(self):
        """Damages are the sum of damages from climate + damges from loss of productivity"""

        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Years}"] = self.years_range

        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamages}"] = \
            self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromClimate}"] +\
            self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromProductivityLoss}"]

        self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Damages}"] = \
            self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromClimate}"] + \
            self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromProductivityLoss}"]

    def compute_energy_consumption_per_section(self):
        """
        Computing the energy consumption for each section of the sector

        section_energy_consumption (PWh) = sector_energy_production (Pwh) x section_energy_consumption_percentage (%)
        """
        conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.EnergyConsumptionDf['unit']][GlossaryCore.SectionEnergyConsumptionDf['unit']]
        self.outputs[f"{self.sector_name}.{GlossaryCore.SectionEnergyConsumptionDfValue}:{GlossaryCore.Years}"] = self.years_range
        for section in self.section_list:
            self.outputs[f"{self.sector_name}.{GlossaryCore.SectionEnergyConsumptionDfValue}:{section}"] = \
                self.outputs[f"{self.sector_name}.{GlossaryCore.EnergyConsumptionValue}:Total"] * \
                self.inputs[f"{self.sector_name}.{GlossaryCore.SectionEnergyConsumptionPercentageDfValue}:{section}"] / 100.\
                * conversion_factor

    def compute_energy_demand_and_consumption(self):
        """Compute energy demand and consumption"""
        self.outputs[f"{self.sector_name}_{GlossaryCore.EnergyDemandValue}:Total"] = \
            self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.Capital}"] * \
            self.inputs['capital_utilisation_ratio'] / \
            self.outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}:{GlossaryCore.EnergyEfficiency}"]  # TWh = T$ * / (T$/TWh)

        self.outputs[f"{self.sector_name}.{GlossaryCore.EnergyConsumptionValue}:Total"] = \
            self.outputs[f"{self.sector_name}_{GlossaryCore.EnergyDemandValue}:Total"] * \
            self.inputs[f"{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}:Total"] / 100.

    # RUN
    def compute(self):
        """Compute all models for year range"""
        self.configure()
        self.compute_productivity_growthrate()
        self.compute_productivity()
        self.compute_energy_efficiency()

        self.compute_capital()
        self.compute_energy_demand_and_consumption()
        self.compute_usable_capital()
        self.compute_gross_output()
        self.compute_output_net_of_damage()

        self.compute_output_net_of_damage_per_section()
        self.compute_energy_consumption_per_section()

        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()

        self.prepare_outputs()

    def prepare_outputs(self):
        for col in GlossaryCore.DamageDf['dataframe_descriptor'].keys():
            self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDfValue}:{col}"] = \
                self.outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}:{col}"]


