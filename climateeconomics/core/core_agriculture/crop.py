'''
Copyright 2022 Airbus SAS
Modifications on 27/03/2025 Copyright 2025 Capgemini

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
from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.subsector_model import (
    SubSectorModel,
)


class Crop(SubSectorModel):
    """
    Crop model class 
    """
    subsector_name = GlossaryCore.Crop
    sector_name = GlossaryCore.SectorAgriculture
    streams_energy_prod = [GlossaryEnergy.biomass_dry, GlossaryEnergy.wet_biomass]

    def compute(self):
        SubSectorModel.compute(self)
        for ft in self.inputs[GlossaryCore.FoodTypesName]:
            self.ft = ft
            self.compute_capital()
            self.compute_energy_demand()
            self.compute_energy_consumption()
            self.compute_usable_capital()
            self.compute_productions()
            self.split_production_between_energy_and_consumers()
            self.compute_emissions()
            self.compute_supply_chain_waste_and_its_reusage()
            self.compute_land_use()
            self.compute_consumers_waste_and_its_reusage()

            self.compute_kcal_per_capita()
            self.compute_price()
            self.compute_economic_damages()
            self.compute_economics()

        self.compute_totals()
        self.compute_crop_capital()
        self.compute_biomass_dry_price()

        self.compute_kcal_infos()
        self.compute_kg_infos()
        self.add_years_to_dfs()

    def compute_kcal_infos(self):
        self.outputs['kcal_dict_infos'] = {}
        self.outputs['kcal_dict_infos']['Land use (m²/kcal)'] = { # m2 / kcal =  (m2 / kg) / (kcal / kg)
            key: value1 / value2 for (key, value1), value2 in zip(
                self.inputs[GlossaryCore.FoodTypeLandUseByProdUnitName].items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )
        }

        self.outputs['kcal_dict_infos']['Green house gases emissions (kgCO2eq/kcal)'] = {}
        for food_type in self.inputs[GlossaryCore.FoodTypesName]:
            co2_eq_per_kg_prod = sum([self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)][food_type] * ClimateEcoDiscipline.GWP_100_default[ghg]
                                      for ghg in GlossaryCore.GreenHouseGases])
            kcal_per_kg = self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName][food_type]

            # co2 eq / kcal = (co2eq/kg) / (kcal/kg)
            self.outputs['kcal_dict_infos']['Green house gases emissions (kgCO2eq/kcal)'][food_type] = co2_eq_per_kg_prod / kcal_per_kg

        self.outputs['kcal_dict_infos']['Capital intensity ($/kcal)'] = {
            # ($ / kg ) / (kcal / kg) = $ / kcal
            key: 1 / value1 / value2 for (key, value1), value2 in zip(
                self.inputs[GlossaryCore.FoodTypeCapitalIntensityName].items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )
        }

        for dict_name, dict_values in self.outputs['kcal_dict_infos'].items():
            if isinstance(list(dict_values.values())[0], float):
                self.outputs['kcal_dict_infos'][dict_name] = dict(sorted(dict_values.items(), key=lambda item: item[1], reverse=True))

    def compute_kg_infos(self):
        self.outputs['kg_dict_infos'] = {}
        for info_name, dict_values in self.outputs['kcal_dict_infos'].items():
            # m2 / kg = (m2 / kcal) * kcal/kg
            self.outputs['kg_dict_infos'][info_name.replace('kcal', 'kg')] = {ft: value1 * self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName][self.ft] for ft, value1 in dict_values.items()}
        self.outputs['kg_dict_infos']['Capital intensity ($/kg)'] = {key: 1 / value1 for key, value1 in self.inputs[GlossaryCore.FoodTypeCapitalIntensityName].items()}
        for dict_name, dict_values in self.outputs['kg_dict_infos'].items():
            if isinstance(list(dict_values.values())[0], float):
                self.outputs['kg_dict_infos'][dict_name] = dict(sorted(dict_values.items(), key=lambda item: item[1], reverse=True))

    def compute_capital(self):
        # forecasting capital of food type
        capital_food_type = [self.inputs[GlossaryCore.FoodTypeCapitalStartName][self.ft]]  # G$
        invest_ft = self.temp_variables[f'invest_details:{self.ft}']
        depreciation_rate_ft = self.inputs[GlossaryCore.FoodTypeCapitalDepreciationRateName][self.ft]
        for invest in invest_ft[:-1]:
            capital_food_type.append(capital_food_type[-1] * (1 - depreciation_rate_ft / 100) + invest)
        self.outputs[f"{GlossaryCore.FoodTypeCapitalName}:{self.ft}"] = self.np.array(capital_food_type) # G$

    def compute_usable_capital(self):
        # limiting capital to usable capital, depending on the variation of ratios of energy and workforce per capital, relative to year start
        workforce_agri = self.inputs[f'{GlossaryCore.WorkforceDfValue}:{GlossaryCore.SectorAgriculture}']
        capital_food_type = self.outputs[f"{GlossaryCore.FoodTypeCapitalName}:{self.ft}"]
        year_start_workforce_per_capital = workforce_agri[0] / self.inputs[GlossaryCore.FoodTypeCapitalStartName][self.ft]

        workforce_per_capital = workforce_agri / capital_food_type

        usable_capital_food_type = capital_food_type * self.np.minimum(1, workforce_per_capital / year_start_workforce_per_capital) * \
                                   self.inputs[f"{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}:Total"] / 100.

        self.outputs[f"usable_capital_breakdown:{self.ft}"] = usable_capital_food_type
        self.outputs[f"non_used_capital_breakdown:{self.ft}"] = capital_food_type - usable_capital_food_type

    def compute_productions(self):
        # computing production : usable capital * capital intensity
        # G$ * t/k$ = 10^9 $ * t/k$ = 10^6 t = Mt
        production_raw = self.outputs[f"usable_capital_breakdown:{self.ft}"] *\
                         self.inputs[GlossaryCore.FoodTypeCapitalIntensityName][self.ft]

        if self.inputs["assumptions_dict"]["compute_climate_impact_on_gdp"]:
            production_wasted_by_productivity_loss = production_raw * (- self.inputs[f"{GlossaryCore.CropProductivityReductionName}:{GlossaryCore.CropProductivityReductionName}"]) / 100.  # Mt
            production_wasted_by_climate_damages = (production_raw - production_wasted_by_productivity_loss) *\
                                               self.inputs[f'{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}']  # Mt
        else:
            production_wasted_by_productivity_loss = self.zeros_arrays
            production_wasted_by_climate_damages = self.zeros_arrays

        self.outputs[f"{GlossaryCore.FoodTypeNotProducedDueToClimateChangeName}:{self.ft}"] = production_wasted_by_productivity_loss
        self.outputs[f"{GlossaryCore.FoodTypeWasteByClimateDamagesName}:{self.ft}"] = production_wasted_by_climate_damages
        self.temp_variables[f"production_before_waste:{self.ft}"] = production_raw - production_wasted_by_productivity_loss  # Mt

        # split energy and food production land use and emissions
        self.temp_variables[f"share_dedicated_to_food:{self.ft}"] = self.zeros_arrays + 1.
        for stream in Crop.streams_energy_prod:
            self.temp_variables[f"share_dedicated_to_food:{self.ft}"] = self.temp_variables[f"share_dedicated_to_food:{self.ft}"] * (1 - self.inputs[f"{GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(stream)}:{self.ft}"] / 100.)

        self.temp_variables[f"production_raw:{self.ft}"] = 0. + production_raw

    def compute_emissions(self):
        # emissions
        self.outputs[f"{GlossaryCore.FoodTypeFoodGWPEmissionsName}:{self.ft}"] = self.zeros_arrays
        for ghg in GlossaryCore.GreenHouseGases:
            self.outputs[f"{GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)}:{self.ft}"] = \
                self.temp_variables[f"production_before_waste:{self.ft}"] * \
                self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)][self.ft] * self.temp_variables[f"share_dedicated_to_food:{self.ft}"] / 1e3  # Mt * kgCO2eq / kg = 10^9 kg * kgCO2eq / kg / 10^3 = G kgCO2eq = 10-3 Gt
            self.outputs[f"{GlossaryCore.FoodTypeEnergyEmissionsName.format(ghg)}:{self.ft}"] = \
                self.temp_variables[f"production_before_waste:{self.ft}"] * \
                self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)][self.ft] * (1 - self.temp_variables[f"share_dedicated_to_food:{self.ft}"]) / 1e3  # Mt * kgCO2eq / kg = 10^9 kg * kgCO2eq / kg / 10^3 = G kgCO2eq = 10-3 Gt
            self.outputs[f"{GlossaryCore.FoodTypeFoodGWPEmissionsName}:{self.ft}"] = \
                self.outputs[f"{GlossaryCore.FoodTypeFoodGWPEmissionsName}:{self.ft}"] + \
                self.outputs[f"{GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)}:{self.ft}"] * ClimateEcoDiscipline.GWP_100_default[ghg]

    def compute_land_use(self):
        # land use

        # Mt * m² / kg = 10^9 kg * m² / kg / 10^4= G m² / 10^4 = G ha
        self.outputs[f"{GlossaryCore.CropFoodLandUseName}_breakdown:{self.ft}"] = \
            self.temp_variables[f"share_dedicated_to_food:{self.ft}"] * self.temp_variables[f"production_raw:{self.ft}"] * \
            self.inputs[GlossaryCore.FoodTypeLandUseByProdUnitName][self.ft] * GlossaryCore.conversion_dict['m²']['ha']

        # Mt * m² / kg = 10^9 kg * m² / kg / 10^4= G m² / 10^4 = G ha
        self.outputs[f"{GlossaryCore.CropEnergyLandUseName}_breakdown:{self.ft}"] = \
            (1 - self.temp_variables[f"share_dedicated_to_food:{self.ft}"]) * self.temp_variables[f"production_raw:{self.ft}"] * \
            self.inputs[GlossaryCore.FoodTypeLandUseByProdUnitName][self.ft] * GlossaryCore.conversion_dict['m²']['ha']

    def split_production_between_energy_and_consumers(self):
        net_production = self.temp_variables[f"production_before_waste:{self.ft}"] - self.outputs[f"{GlossaryCore.FoodTypeWasteByClimateDamagesName}:{self.ft}"]  # Mt
        production_for_consumers = 0. + net_production
        for stream in Crop.streams_energy_prod:
            self.outputs[f"{GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream)}_breakdown:{self.ft}"] =\
                net_production * self.inputs[f"{GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(stream)}:{self.ft}"] / 100.
            production_for_consumers = production_for_consumers - self.outputs[f"{GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream)}_breakdown:{self.ft}"]
        self.outputs[f"{GlossaryCore.FoodTypeProductionName}:{self.ft}"] = production_for_consumers

    def compute_supply_chain_waste_and_its_reusage(self):
        self.outputs[f'{GlossaryCore.FoodTypeWasteAtSupplyChainName}:{self.ft}'] = self.outputs[f"{GlossaryCore.FoodTypeProductionName}:{self.ft}"] * \
                                         self.inputs[GlossaryCore.FoodTypeWasteSupplyChainShareName][
                                             self.ft] / 100.  # Mt
        for stream in Crop.streams_energy_prod:
            self.outputs[f"{GlossaryCore.WasteSupplyChainReusedForEnergyProdName.format(stream)}_breakdown:{self.ft}"] = self.outputs[f'{GlossaryCore.FoodTypeWasteAtSupplyChainName}:{self.ft}'] * self.inputs[
                f"{GlossaryCore.FoodTypeShareWasteSupplyChainUsedToStreamProdName.format(stream)}:{self.ft}"] / 100.  # Mt

        self.outputs[f"{GlossaryCore.FoodTypeDeliveredToConsumersName}:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeProductionName}:{self.ft}"] -\
            self.outputs[f'{GlossaryCore.FoodTypeWasteAtSupplyChainName}:{self.ft}']  # Mt



    def compute_kcal_per_capita(self):
        population = self.inputs[f'{GlossaryCore.PopulationDfValue}:{GlossaryCore.PopulationValue}']

        # Mt / (M person) * 1000 = kg / person
        self.outputs[f"food_per_capita_per_year:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeDeliveredToConsumersName}:{self.ft}"] / population * 1000

        # Mt * kcal/ kg = 10^9 kg * kcal/kg = 10^9 kcal  = G kcal
        kcal_produced_for_consumers = self.outputs[f"{GlossaryCore.FoodTypeDeliveredToConsumersName}:{self.ft}"] * \
                                      self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName][self.ft]

        # Gkcal / (10^6 person) / (day) * 1000 = k kcal / person / day * 1000 = kcal / person / day
        self.outputs[f"{GlossaryCore.CaloriesPerCapitaBreakdownValue}:{self.ft}"] = kcal_produced_for_consumers / population / 365. * 1000

    def compute_consumers_waste_and_its_reusage(self):
        self.outputs[f"{GlossaryCore.FoodTypeWasteByConsumersName}:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeDeliveredToConsumersName}:{self.ft}"] * \
            self.inputs[GlossaryCore.FoodTypeWasteByConsumersShareName][self.ft] / 100.  # Mt

        self.outputs[f"{GlossaryCore.CropProdForAllStreamName}:{self.ft}"] = 0.
        for stream in Crop.streams_energy_prod:
            self.outputs[f"{GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream)}_breakdown:{self.ft}"] = \
                self.outputs[f"{GlossaryCore.FoodTypeWasteByConsumersName}:{self.ft}"] *\
                self.inputs[f"{GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(stream)}:{self.ft}"] / 100.  # Mt

            self.outputs[f"Crop.{GlossaryCore.ProdForStreamName.format(stream)}_breakdown:{self.ft}"] = \
                self.outputs[f"{GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream)}_breakdown:{self.ft}"] + \
                self.outputs[f"{GlossaryCore.WasteSupplyChainReusedForEnergyProdName.format(stream)}_breakdown:{self.ft}"] + \
                self.outputs[f"{GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream)}_breakdown:{self.ft}"]  # Mt

            self.outputs[f"{GlossaryCore.CropProdForAllStreamName}:{self.ft}"] = self.outputs[f"{GlossaryCore.CropProdForAllStreamName}:{self.ft}"] +\
                                                                                 self.outputs[f"Crop.{GlossaryCore.ProdForStreamName.format(stream)}_breakdown:{self.ft}"]

    def compute_totals(self):

        # name of the breakdown df : (total_df, column_name_for_total)
        dataframes_to_totalize = {
            GlossaryCore.CropFoodLandUseName + "_breakdown": (GlossaryCore.CropFoodLandUseName, "Total"),
            GlossaryCore.CropEnergyLandUseName + "_breakdown": (GlossaryCore.CropEnergyLandUseName, "Total"),
            GlossaryCore.CaloriesPerCapitaBreakdownValue: (GlossaryCore.CaloriesPerCapitaValue, "kcal_pc"),
            "non_used_capital" + "_breakdown": ("non_used_capital", "Total"),
            GlossaryCore.Damages + "_breakdown": (f"Crop.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            GlossaryCore.GrossOutput + "_breakdown": (f"Crop.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            GlossaryCore.OutputNetOfDamage + "_breakdown": (f"Crop.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
            GlossaryCore.CropFoodNetGdpName + "_breakdown": (GlossaryCore.CropFoodNetGdpName, "Total"),
            GlossaryCore.CropEnergyNetGdpName + "_breakdown": (GlossaryCore.CropEnergyNetGdpName, "Total"),
            GlossaryCore.DamagesFromProductivityLoss + "_breakdown": (f"Crop.{GlossaryCore.DamageDetailedDfValue}", GlossaryCore.DamagesFromProductivityLoss),
            GlossaryCore.DamagesFromClimate + "_breakdown": (f"Crop.{GlossaryCore.DamageDetailedDfValue}", GlossaryCore.DamagesFromClimate),

        }
        for ghg in GlossaryCore.GreenHouseGases:
            dataframes_to_totalize[GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)] = (GlossaryCore.FoodEmissionsName, ghg)
            dataframes_to_totalize[GlossaryCore.FoodTypeEnergyEmissionsName.format(ghg)] = (GlossaryCore.CropEnergyEmissionsName, ghg)

        for stream in self.streams_energy_prod:
            dataframes_to_totalize["Crop." + GlossaryCore.ProdForStreamName.format(stream) + "_breakdown"] = ("Crop." + GlossaryCore.ProdForStreamName.format(stream), "Total")

        for stream in self.streams_energy_prod:
            dataframes_to_totalize[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + "_breakdown"] = (GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream), "Total")
            dataframes_to_totalize[GlossaryCore.WasteSupplyChainReusedForEnergyProdName.format(stream) + "_breakdown"] = (GlossaryCore.WasteSupplyChainReusedForEnergyProdName.format(stream), "Total")
            dataframes_to_totalize[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream) + "_breakdown"] = (GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream), "Total")

        for breakdown_df_varname, (total_df_varname, total_colname) in dataframes_to_totalize.items():
            cols_to_sum = self.get_cols_output_dataframe(df_name=breakdown_df_varname)
            self.outputs[f"{total_df_varname}:{total_colname}"] = self.sum_cols(cols_to_sum)

    def add_years_to_dfs(self):
        for df_outputname in self.get_dataframes():
            self.outputs[f"{df_outputname}:{GlossaryCore.Years}"] = self.years

    def compute_price(self):

        # compute unitary price
        energy_price = self.inputs[f"{GlossaryCore.EnergyMeanPriceValue}:{GlossaryCore.EnergyPriceValue}"]

        self.outputs[f'{self.ft}_price_details:{GlossaryCore.Years}'] = self.years
        self.outputs[f'{self.ft}_price_details:Labor'] = self.zeros_arrays + self.inputs[GlossaryCore.FoodTypeLaborCostByProdUnitName][self.ft] / 1000  # $/ton to $/kg
        self.outputs[f'{self.ft}_price_details:Energy'] = self.inputs[GlossaryCore.FoodTypeEnergyIntensityByProdUnitName][self.ft] * energy_price / 1e6  # kWh/ton * $/MWh = kWh/ton * $/(kkWh) = $/(k ton) =$/(Mkg) then /1e6 to $/kg
        self.outputs[f'{self.ft}_price_details:Capital maintenance'] = self.zeros_arrays + self.inputs[GlossaryCore.FoodTypeCapitalMaintenanceCostName][self.ft] * energy_price / 1e6  # kWh/ton * $/MWh = kWh/ton * $/(kkWh) = $/(k ton) =$/(Mkg) then /1e6 to $/kg
        self.outputs[f"{self.ft}_price_details:Capex amortization"] = self.zeros_arrays + self.inputs[GlossaryCore.FoodTypeCapitalAmortizationCostName][self.ft] / 1000  # $/ton to $/kg
        self.outputs[f"{self.ft}_price_details:Feeding"] = self.zeros_arrays + self.inputs[GlossaryCore.FoodTypeFeedingCostsName][self.ft] / 1000  # $/ton to $/kg
        self.outputs[f"{self.ft}_price_details:Fertilization and pesticides"] = self.zeros_arrays + self.inputs[GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName][self.ft] / 1000  # $/ton to $/kg

        price_wo_margin = self.sum_cols(self.get_cols_output_dataframe(f"{self.ft}_price_details", expect_years=True))

        margin_share_of_final_price = self.inputs[GlossaryCore.FoodTypesPriceMarginShareName][self.ft]
        margin = margin_share_of_final_price / 100 * price_wo_margin / (1 - margin_share_of_final_price / 100)

        self.outputs[f"{self.ft}_price_details:Margin"] = margin
        final_price = price_wo_margin + margin  # $/kg

        self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{self.ft}"] = final_price

    def compute_economic_damages(self):
        # compute gross output for crop:
        self.outputs[f"{GlossaryCore.DamagesFromProductivityLoss}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeNotProducedDueToClimateChangeName}:{self.ft}"] *\
            self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{self.ft}"]# Mt * $ / kg = G $

        self.outputs[f"{GlossaryCore.DamagesFromClimate}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeWasteByClimateDamagesName}:{self.ft}"] *\
            self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{self.ft}"]


        self.outputs[f"{GlossaryCore.Damages}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.DamagesFromProductivityLoss}_breakdown:{self.ft}"] + \
            self.outputs[f"{GlossaryCore.DamagesFromClimate}_breakdown:{self.ft}"]

    def compute_economics(self):
        """Gdp in G$"""
        self.outputs[f"{GlossaryCore.CropFoodNetGdpName}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeDeliveredToConsumersName}:{self.ft}"] * \
            self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{self.ft}"]

        self.outputs[f"{GlossaryCore.CropEnergyNetGdpName}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.CropProdForAllStreamName}:{self.ft}"] *\
            self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{self.ft}"]

        self.outputs[f"{GlossaryCore.OutputNetOfDamage}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.CropFoodNetGdpName}_breakdown:{self.ft}"] + \
            self.outputs[f"{GlossaryCore.CropEnergyNetGdpName}_breakdown:{self.ft}"]

        self.outputs[f"{GlossaryCore.GrossOutput}_breakdown:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.OutputNetOfDamage}_breakdown:{self.ft}"] + \
            self.outputs[f"{GlossaryCore.Damages}_breakdown:{self.ft}"]

    def compute_crop_capital(self):
        self.outputs[f'Crop.{GlossaryCore.CapitalDfValue}:{GlossaryCore.Years}'] = self.years
        self.outputs[f'Crop.{GlossaryCore.CapitalDfValue}:{GlossaryCore.Capital}'] = \
            self.sum_cols(self.get_cols_output_dataframe(f"{GlossaryCore.FoodTypeCapitalName}"))
        self.outputs[f'Crop.{GlossaryCore.CapitalDfValue}:{GlossaryCore.UsableCapital}'] = self.outputs[f'Crop.{GlossaryCore.CapitalDfValue}:{GlossaryCore.Capital}']

    def compute_biomass_dry_price(self):
        total_biomass_dry_prod = self.outputs[f"{GlossaryCore.Crop}.{GlossaryCore.ProdForStreamName.format(GlossaryEnergy.biomass_dry)}:Total"]
        biomass_dry_price = self.zeros_arrays
        for ft in self.inputs[GlossaryCore.FoodTypesName]:
            share_of_biomass_dry_prod_ft = self.outputs[f"{GlossaryCore.Crop}.{GlossaryCore.ProdForStreamName.format(GlossaryEnergy.biomass_dry)}_breakdown:{ft}"] / total_biomass_dry_prod
            ft_price = self.outputs[f"{GlossaryCore.FoodTypesPriceName}:{ft}"]
            biomass_dry_price = biomass_dry_price + share_of_biomass_dry_prod_ft * ft_price

        self.outputs[f"{GlossaryCore.Crop}.biomass_dry_price:{GlossaryCore.Crop}"] = biomass_dry_price

    def compute_energy_demand(self):
        """Energy demand = Capital * Capital energy intensity"""
        # G$ * kWh / k$  = M * kWh = GWh, then converted to energy demand df unit.
        conversion_factor = GlossaryCore.conversion_dict['GWh'][GlossaryCore.EnergyDemandDf['unit']]
        self.outputs[f"{self.subsector_name}_{GlossaryCore.EnergyDemandValue}:{self.ft}"] = \
            self.outputs[f"{GlossaryCore.FoodTypeCapitalName}:{self.ft}"] *\
            self.inputs[f"{GlossaryCore.FoodTypeCapitalEnergyIntensityName}"][self.ft] * conversion_factor

    def compute_energy_consumption(self):
        """Energy consumed = Energy demand *c ratio of availability of energy"""
        conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.EnergyDemandDf['unit']][GlossaryCore.EnergyConsumptionDf['unit']]
        self.outputs[f"{self.subsector_name}.{GlossaryCore.EnergyConsumptionValue}:{self.ft}"] = \
            self.outputs[f"{self.subsector_name}_{GlossaryCore.EnergyDemandValue}:{self.ft}"] *\
            self.inputs[f"{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}:Total"] / 100. * conversion_factor
