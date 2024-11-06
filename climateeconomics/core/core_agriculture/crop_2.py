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
import autograd.numpy as np
import pandas as pd

from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy


class Crop:
    """
    Crop model class 
    """

    def __init__(self, param):
        '''
        Constructor
        '''
        self.inputs = {}
        self.outputs = {}

        self.dataframes_to_totalize_by_food_type = {
            GlossaryCore.FoodTypeLandUseName: (GlossaryCore.FoodLandUseName, "Total"),
            GlossaryCore.CaloriesPerCapitaBreakdownValue: (GlossaryCore.CaloriesPerCapitaValue, "kcal_pc"),
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry): (GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry), "Total"),
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass): (GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass), "Total"),
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CO2): (GlossaryCore.CropFoodEmissionsName, GlossaryCore.CO2),
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CH4): (GlossaryCore.CropFoodEmissionsName, GlossaryCore.CH4),
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.N2O): (GlossaryCore.CropFoodEmissionsName, GlossaryCore.N2O),

            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry) + "_breakdown": (GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry), "Total"),
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass) + "_breakdown": (GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass), "Total"),
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.biomass_dry) + "_breakdown": (GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.biomass_dry), "Total"),
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.wet_biomass) + "_breakdown": (GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.wet_biomass), "Total"),
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown": (GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.biomass_dry), "Total"),
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown": (GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.wet_biomass), "Total"),
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown": (GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry), "Total"),
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown": (GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass), "Total"),
        }

    def init_dataframes(self):
        years = np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        dataframe_to_init = [
            GlossaryCore.FoodTypeWasteAtProductionDistributionName,
            GlossaryCore.FoodTypeWasteByConsumersName,
            GlossaryCore.FoodTypeNotProducedDueToClimateChangeName,
            GlossaryCore.FoodTypeWasteByClimateDamagesName,
            GlossaryCore.FoodTypeDeliveredToConsumersName,
            GlossaryCore.CaloriesPerCapitaBreakdownValue,
            GlossaryCore.FoodTypeLandUseName,
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CO2),
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CH4),
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.N2O),

            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry) + "_breakdown",
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass) + "_breakdown",
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.biomass_dry) + "_breakdown",
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.wet_biomass) + "_breakdown",
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown",
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown",
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown",
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown",

            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry),
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass),
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.biomass_dry),
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.wet_biomass),
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.biomass_dry),
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.wet_biomass),
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry),
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass),

            "unused_energy" + "_breakdown",
            "unused_workforce" + "_breakdown"
        ]
        for df_name in dataframe_to_init:
            self.outputs[df_name] = pd.DataFrame({GlossaryCore.Years: years})

        for (df_name, _) in self.dataframes_to_totalize_by_food_type.values():
            self.outputs[df_name] = pd.DataFrame({GlossaryCore.Years: years})

        for (df_name, column) in self.dataframes_to_totalize_by_food_type.values():
            self.outputs[df_name][column] = 0.

    def compute(self, inputs: dict):
        self.inputs = inputs
        self.init_dataframes()

        for food_type in inputs[GlossaryCore.FoodTypesName]:
            output_food_type = self.compute_food_type(
                invest_food=self.inputs[f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}'][GlossaryCore.InvestmentsValue].values,
                energy_consumption_food=self.inputs[f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}'][GlossaryCore.TotalProductionValue].values,
                workforce_food=self.inputs[GlossaryCore.WorkforceDfValue][GlossaryCore.SectorAgriculture].values,
                crop_productivity_reduction=self.inputs[GlossaryCore.CropProductivityReductionName][GlossaryCore.CropProductivityReductionName].values,
                damage_fraction=self.inputs[GlossaryCore.DamageFractionDfValue][GlossaryCore.DamageFractionOutput].values,
                population=self.inputs[GlossaryCore.PopulationDfValue][GlossaryCore.PopulationValue].values,
                share_invest_food_type=self.inputs[GlossaryCore.ShareInvestFoodTypesName][food_type].values,
                share_energy_consumption_food_type=self.inputs[GlossaryCore.ShareEnergyUsageFoodTypesName][food_type].values,
                share_workforce_food_type=self.inputs[GlossaryCore.ShareWorkforceFoodTypesName][food_type].values,
                food_type_capex=self.inputs[GlossaryCore.FoodTypeCapexName][food_type],
                food_type_energy_need=self.inputs[GlossaryCore.FoodTypeEnergyNeedName][food_type],
                food_type_workforce_need=self.inputs[GlossaryCore.FoodTypeWorkforceNeedName][food_type],
                share_food_waste_before_distribution=self.inputs[GlossaryCore.FoodTypeWasteAtProductionShareName][food_type].values,
                share_food_waste_by_consumers=self.inputs[GlossaryCore.FoodTypeWasteByConsumersShareName][food_type].values,
                co2_emissions_per_prod_unit=self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CO2)][food_type],
                ch4_emissions_per_prod_unit=self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CH4)][food_type],
                n2o_emissions_per_prod_unit=self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.N2O)][food_type],
                share_dedicated_to_biomass_dry_prod=self.inputs[GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.biomass_dry)][food_type].values,
                share_dedicated_to_biomass_wet_prod=self.inputs[GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.wet_biomass)][food_type].values,
                kcal_per_prod_unit=self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName][food_type],
                land_use_by_prod_unit=self.inputs[GlossaryCore.FoodTypeLandUseByProdUnitName][food_type],
                share_user_waste_reused_for_energy_prod_biomass_dry=self.inputs[GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.biomass_dry)][food_type].values,
                share_user_waste_reused_for_energy_prod_biomass_wet=self.inputs[GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.wet_biomass)][food_type].values,
                share_waste_before_distrib_reused_for_energy_prod_biomass_dry=self.inputs[GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.biomass_dry)][food_type].values,
                share_waste_before_distrib_reused_for_energy_prod_biomass_wet=self.inputs[GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.wet_biomass)][food_type].values,
            )
            for varname, value in output_food_type.items():
                self.outputs[varname][food_type] = value
                if varname in self.dataframes_to_totalize_by_food_type:
                    varname_total_df, column_total_df = self.dataframes_to_totalize_by_food_type[varname]
                    self.outputs[varname_total_df][column_total_df] += value

            self.compute_kcal_infos()
            self.compute_kg_infos()

    @staticmethod
    def compute_food_type(
            # coupling inputs first
            invest_food: np.ndarray,
            energy_consumption_food: np.ndarray,
            workforce_food: np.ndarray,
            damage_fraction: np.ndarray,
            crop_productivity_reduction: np.ndarray,

            # the rest
            share_invest_food_type: np.ndarray,
            share_energy_consumption_food_type: np.ndarray,
            share_workforce_food_type: np.ndarray,
            food_type_energy_need: np.ndarray,
            food_type_workforce_need: np.ndarray,
            food_type_capex: np.ndarray,
            share_dedicated_to_biomass_dry_prod: np.ndarray,
            share_dedicated_to_biomass_wet_prod: np.ndarray,
            share_user_waste_reused_for_energy_prod_biomass_dry: np.ndarray,
            share_user_waste_reused_for_energy_prod_biomass_wet: np.ndarray,
            share_waste_before_distrib_reused_for_energy_prod_biomass_dry: np.ndarray,
            share_waste_before_distrib_reused_for_energy_prod_biomass_wet: np.ndarray,
            land_use_by_prod_unit: np.ndarray,
            kcal_per_prod_unit: np.ndarray,
            co2_emissions_per_prod_unit: np.ndarray,
            ch4_emissions_per_prod_unit: np.ndarray,
            n2o_emissions_per_prod_unit: np.ndarray,
            share_food_waste_before_distribution: np.ndarray,
            share_food_waste_by_consumers: np.ndarray,
            population: np.ndarray,
    ):
        invest_food_type = invest_food * share_invest_food_type / 100. # T$
        energy_allocated_to_food_type = energy_consumption_food * share_energy_consumption_food_type / 100.  # Pwh
        workforce_allocated_to_food_type = workforce_food * share_workforce_food_type / 100.  # million people

        production_wo_ratio = invest_food_type / food_type_capex * 10**6  # T$ / ($/ton) / 1O^6 = T$ / ($/ton) = T ton, so need to multiply by 10^6. Prod in Mt

        # convert energy need from kWh/ton to PWh/Mt : PWh/Mt = (10^12 kWh/ 10^6 ton) = 10^6 kWh/ton
        food_type_energy_need = food_type_energy_need / 10**12 * 10**6  # PWh/Mt

        # convert workforce need from person/ton to million people/Mt : million people/Mt = (10^6 person/ 10^6 ton) = person/ton
        food_type_workforce_need = food_type_workforce_need

        applied_ratio, applied_ratio_df, df_ratios = Crop.compute_ratio(production_wo_ratio=production_wo_ratio,
                                                                        needs={'energy': food_type_energy_need,
                                                                               'workforce': food_type_workforce_need},
                                                                        availability={
                                                                            'energy': energy_allocated_to_food_type,
                                                                            'workforce': workforce_allocated_to_food_type})

        unused_energy = energy_allocated_to_food_type - applied_ratio * food_type_energy_need * production_wo_ratio # Pwh
        unused_workforce = workforce_allocated_to_food_type - applied_ratio * food_type_workforce_need * production_wo_ratio # million people

        production_raw = production_wo_ratio * applied_ratio # Mt * unitless
        production_wasted_by_productivity_loss = production_raw * crop_productivity_reduction / 100. # Mt
        production_wasted_by_immediate_damages = production_wasted_by_productivity_loss * damage_fraction # Mt
        production_before_waste = production_raw - production_wasted_by_productivity_loss # Mt
        co2_emissions_food = production_before_waste * co2_emissions_per_prod_unit * (1 - share_dedicated_to_biomass_dry_prod / 100. - share_dedicated_to_biomass_wet_prod / 100.)  # Mt_food * (kg{ghg}/kg_food) = 10^9 kg_food * kg_ghg / kg_food = 10^9 kg_ghg = Gt kg_ghg
        ch4_emissions_food = production_before_waste * ch4_emissions_per_prod_unit * (1 - share_dedicated_to_biomass_dry_prod / 100. - share_dedicated_to_biomass_wet_prod / 100.)  # Mt_food * (kg{ghg}/kg_food) = 10^9 kg_food * kg_ghg / kg_food = 10^9 kg_ghg = Gt kg_ghg
        n2o_emissions_food = production_before_waste * n2o_emissions_per_prod_unit * (1 - share_dedicated_to_biomass_dry_prod / 100. - share_dedicated_to_biomass_wet_prod / 100.)  # Mt_food * (kg{ghg}/kg_food) = 10^9 kg_food * kg_ghg / kg_food = 10^9 kg_ghg = Gt kg_ghg

        net_production = production_before_waste - production_wasted_by_immediate_damages # Mt
        production_dedicated_to_biomass_dry = net_production * share_dedicated_to_biomass_dry_prod / 100. # Mt
        production_dedicated_to_biomass_wet = net_production * share_dedicated_to_biomass_wet_prod / 100. # Mt
        production_dedicated_to_energy = production_dedicated_to_biomass_dry + production_dedicated_to_biomass_wet  # Mt

        production_for_consumers = net_production - production_dedicated_to_energy  # Mt
        food_waste_before_distribution = production_for_consumers * share_food_waste_before_distribution / 100.  # Mt

        food_waste_before_distribution_reused_for_energy_prod_biomass_dry = food_waste_before_distribution * share_waste_before_distrib_reused_for_energy_prod_biomass_dry / 100.  # Mt
        food_waste_before_distribution_reused_for_energy_prod_biomass_wet = food_waste_before_distribution * share_waste_before_distrib_reused_for_energy_prod_biomass_wet / 100.  # Mt

        production_delivered_to_consumers = production_for_consumers - food_waste_before_distribution  # Mt
        food_waste_by_consumers = production_delivered_to_consumers * share_food_waste_by_consumers / 100. # Mt
        consumers_waste_reused_for_energy_prod_biomass_dry = food_waste_by_consumers * share_user_waste_reused_for_energy_prod_biomass_dry / 100. # Mt
        consumers_waste_reused_for_energy_prod_biomass_wet = food_waste_by_consumers * share_user_waste_reused_for_energy_prod_biomass_wet / 100. # Mt

        total_biomass_dry_prod_available = production_dedicated_to_biomass_dry + food_waste_before_distribution_reused_for_energy_prod_biomass_dry + consumers_waste_reused_for_energy_prod_biomass_dry # Mt
        total_biomass_wet_prod_available = production_dedicated_to_biomass_wet + food_waste_before_distribution_reused_for_energy_prod_biomass_wet + consumers_waste_reused_for_energy_prod_biomass_wet # Mt

        kcal_produced_for_consumers = production_delivered_to_consumers * kcal_per_prod_unit  # Mt * kcal/ kg = 10^9 kg * kcal/kg = 10^9 kcal  = G kcal
        kcal_per_pers_per_day = kcal_produced_for_consumers / population / 365. * 1000  # Gkcal / (10^6 person) / (day) * 1000 = k kcal / person / day * 1000 = kcal / person / day
        land_use_food = production_raw * (1 - share_dedicated_to_biomass_dry_prod / 100. + share_dedicated_to_biomass_wet_prod / 100.) * land_use_by_prod_unit / (10**5)  # Mt * m² / kg = 10^9 kg * m² / kg / 10^5= G m² / 10^5 = G ha
        return {
            GlossaryCore.FoodTypeLandUseName: land_use_food,
            GlossaryCore.FoodTypeWasteAtProductionDistributionName: food_waste_before_distribution,
            GlossaryCore.FoodTypeWasteByConsumersName: food_waste_by_consumers,
            GlossaryCore.FoodTypeDeliveredToConsumersName: production_delivered_to_consumers,
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CO2): co2_emissions_food,
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.CH4): ch4_emissions_food,
            GlossaryCore.FoodTypeEmissionsName.format(GlossaryCore.N2O): n2o_emissions_food,
            GlossaryCore.FoodTypeNotProducedDueToClimateChangeName: production_wasted_by_productivity_loss,
            GlossaryCore.FoodTypeWasteByClimateDamagesName: production_wasted_by_immediate_damages,

            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry) + "_breakdown": production_dedicated_to_biomass_dry,
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass) + "_breakdown": production_dedicated_to_biomass_wet,

            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.biomass_dry) + "_breakdown": food_waste_before_distribution_reused_for_energy_prod_biomass_dry,
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(GlossaryEnergy.wet_biomass) + "_breakdown": food_waste_before_distribution_reused_for_energy_prod_biomass_wet,

            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown": consumers_waste_reused_for_energy_prod_biomass_dry,
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown": consumers_waste_reused_for_energy_prod_biomass_wet,

            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.biomass_dry) + "_breakdown": total_biomass_dry_prod_available,
            GlossaryCore.CropProdForEnergyName.format(GlossaryEnergy.wet_biomass) + "_breakdown": total_biomass_wet_prod_available,

            GlossaryCore.CaloriesPerCapitaBreakdownValue: kcal_per_pers_per_day,
            "unused_energy" + "_breakdown": unused_energy,
            "unused_workforce" + "_breakdown": unused_workforce,
        }
    @staticmethod
    def compute_ratio(production_wo_ratio: np.ndarray,
                      needs: dict[str: np.ndarray],
                      availability: dict[str: np.ndarray]):
        """compute ratio to apply to limit ratio"""

        if set(needs.keys()) != set(availability.keys()):
            raise ValueError("'needs' and 'available' dict inputs should have same keys")
        ratios_dict = {}
        for resource in needs.keys():
            consumption_wo_ratio = production_wo_ratio * needs[resource]
            availability_resource = availability[resource]
            limiting_ratio_resource = np.minimum(availability_resource / consumption_wo_ratio, 1)
            ratios_dict[resource] = limiting_ratio_resource

        df_ratios = pd.DataFrame(ratios_dict)
        limiting_input = [df_ratios.columns[i] for i in df_ratios.values.argmin(axis=1)]
        applied_ratio = np.array(list(ratios_dict.values())).min(axis=0)
        applied_ratio_df = pd.DataFrame({
            'applied_ratio': applied_ratio,
            'limiting_input': limiting_input
        })
        return applied_ratio, applied_ratio_df, df_ratios

    def compute_kcal_infos(self):
        self.outputs['kcal_dict_infos'] = {}
        self.outputs['kcal_dict_infos']['Land use (m²/kcal)'] = { # m2 / kcal =  (m2 / kg) / (kcal / kg)
            key: value1 / value2 for (key, value1), value2 in zip(
                self.inputs[GlossaryCore.FoodTypeLandUseByProdUnitName].items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )
        }

        self.outputs['kcal_dict_infos']['Green house gases emissions (CO2eq/kcal)'] = {}
        for food_type in self.inputs[GlossaryCore.FoodTypesName]:
            co2_eq_per_kg_prod = sum([self.inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)][food_type] * ClimateEcoDiscipline.GWP_100_default[ghg]
                                      for ghg in GlossaryCore.GreenHouseGases])
            kcal_per_kg = self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName][food_type]

            # co2 eq / kcal = (co2eq/kg) / (kcal/kg)
            self.outputs['kcal_dict_infos']['Green house gases emissions (CO2eq/kcal)'][food_type] = co2_eq_per_kg_prod / kcal_per_kg

        self.outputs['kcal_dict_infos']['Capex ($/kcal)'] = { # $ / kcal * 1000 =  ($/ton) / (kcal / kg) * 1000  = ($ / 1000 ) / kcal * 1000 = $ / kcal
            key: value1 / value2 * 1000 for (key, value1), value2 in zip(
                self.inputs[GlossaryCore.FoodTypeCapexName].items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )
        }

    def compute_kg_infos(self):
        self.outputs['kg_dict_infos'] = {}
        for info_name, dict_values in self.outputs['kcal_dict_infos'].items():
            # m2 / kg = (m2 / kcal) * kcal/kg
            self.outputs['kg_dict_infos'][info_name.replace('kcal', 'kg')] = {key: value1 * value2 for (key, value1), value2 in zip(
                dict_values.items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )}
        self.outputs['kg_dict_infos']['Capex ($/kg)'] = {
            # $ / kg = ($ / ton) * 1000
        key: value1 / 1000 for key, value1 in self.inputs[GlossaryCore.FoodTypeCapexName].items()
        }