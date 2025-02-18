'''
Copyright 2024 Capgemini
Modifications on 2023/06/21-2024/06/24 Copyright 2023 Capgemini

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


class AgricultureModel(DifferentiableModel):
    """Agriculture model class"""
    name = 'Agriculture'
    sub_sectors = [GlossaryCore.Forestry, GlossaryCore.Crop]
    sub_sector_commun_variables = [
        (GlossaryCore.DamageDfValue, GlossaryCore.SubsectorDamagesDf, GlossaryCore.DamageDf, True),
        (GlossaryCore.ProductionDfValue, GlossaryCore.SubsectorProductionDf, GlossaryCore.SectorProductionDf, True),
        (GlossaryCore.ProdForStreamName.format('biomass_dry'), GlossaryCore.ProdForStreamVar, GlossaryCore.ProdForStreamVar, True),
        (GlossaryCore.CapitalDfValue, GlossaryCore.SubsectorCapitalDf, GlossaryCore.SectorCapitalDf, True),
        ("biomass_dry_price", GlossaryCore.PriceDf, GlossaryCore.PriceDf, False),
    ]

    def configure_years(self):
        self.years = self.np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        self.zeros_array = 0. * self.years

    def compute(self):
        self.configure_years()
        self.compute_sector_investments()
        self.sum_commun_variables()
        self.compute_sector_emissions()
        self.compute_damages_detailed()
        self.compute_biomass_dry_price()

    def sum_commun_variables(self):
        for varname, var_descr_input, var_descr_output, do_sum in self.sub_sector_commun_variables:
            if do_sum:
                self.outputs[f'{self.name}.{varname}:{GlossaryCore.Years}'] = self.years
                conversion_factor = GlossaryCore.conversion_dict[var_descr_input['unit']][var_descr_output['unit']]
                for column in var_descr_input['dataframe_descriptor'].keys():
                    if column != GlossaryCore.Years:
                        self.outputs[f'{self.name}.{varname}:{column}'] = self.zeros_array
                        for subsector in self.sub_sectors:
                            self.outputs[f'{self.name}.{varname}:{column}'] = self.outputs[f'{self.name}.{varname}:{column}'] + self.inputs[f'{subsector}.{varname}:{column}'] * conversion_factor


    def compute_sector_investments(self):
        if self.inputs["mdo_sectors_invest_level"] == 0:
            self.outputs[f"{self.name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.Years}"] = self.years
            self.outputs[f"{self.name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"] = self.zeros_array
            for subsector in self.sub_sectors:
                self.outputs[f"{self.name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"] = \
                    self.outputs[f"{self.name}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"] + \
                    self.inputs[f"{self.name}.{subsector}.{GlossaryCore.InvestmentDfValue}:{GlossaryCore.InvestmentsValue}"]

        else:
            raise NotImplementedError("")

    def compute_sector_emissions(self):
        """compute agriculture emissions"""
        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}:{GlossaryCore.Forestry}"] = self.inputs["Forestry.CO2_land_emission_df:emitted_CO2_evol_cumulative"]
        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}:{GlossaryCore.Crop}"] = self.inputs[f'{GlossaryCore.FoodEmissionsName}:{GlossaryCore.CO2}']

        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}:{GlossaryCore.Crop}"] = self.inputs[f'{GlossaryCore.FoodEmissionsName}:{GlossaryCore.CH4}']

        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}:{GlossaryCore.Crop}"] = self.inputs[f'{GlossaryCore.FoodEmissionsName}:{GlossaryCore.N2O}']

    def compute_damages_detailed(self):
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Damages}"] = self.outputs[f"{self.name}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Damages}"]
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamages}"] = self.outputs[f"{self.name}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Damages}"]
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromProductivityLoss}"] = self.zeros_array
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromProductivityLoss}"] = self.zeros_array
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromClimate}"] = self.outputs[f"{self.name}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Damages}"]
        self.outputs[f"{self.name}.{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromClimate}"] = self.outputs[f"{self.name}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Damages}"]

    def compute_biomass_dry_price(self):
        """Compute the biomass dry price of agriculture sector"""
        biomass_dry_price = self.zeros_array
        total_biomass_dry_prod = self.outputs[f"{self.name}.{GlossaryCore.ProdForStreamName.format(GlossaryCore.biomass_dry)}:Total"]
        for subsector in AgricultureModel.sub_sectors:
            share_of_prod = self.inputs[f"{subsector}.{GlossaryCore.ProdForStreamName.format(GlossaryCore.biomass_dry)}:Total"] / total_biomass_dry_prod
            subsector_biomassy_dry_price = self.inputs[f"{subsector}.biomass_dry_price:{subsector}"]
            biomass_dry_price = biomass_dry_price + subsector_biomassy_dry_price * share_of_prod

        self.outputs[f"{self.name}.biomass_dry_price:{GlossaryCore.biomass_dry}"] = biomass_dry_price
