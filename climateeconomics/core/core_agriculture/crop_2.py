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
from autograd import jacobian
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class Crop:
    """
    Crop model class 
    """

    streams_energy_prod = [GlossaryEnergy.biomass_dry, GlossaryEnergy.wet_biomass]

    def __init__(self):
        """Constructor"""
        self.inputs = {}
        self.outputs = {}

        # couplings
        self.dataframes_to_totalize_by_food_type_couplings = {
            GlossaryCore.CropFoodLandUseName + "_breakdown": (GlossaryCore.CropFoodLandUseName, "Total"),
            GlossaryCore.CropEnergyLandUseName + "_breakdown": (GlossaryCore.CropEnergyLandUseName, "Total"),
            GlossaryCore.CaloriesPerCapitaBreakdownValue: (GlossaryCore.CaloriesPerCapitaValue, "kcal_pc"),
            "non_used_capital" + "_breakdown": ("non_used_capital", "Total"),
        }
        for ghg in GlossaryCore.GreenHouseGases:
            self.dataframes_to_totalize_by_food_type_couplings[GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)] = (GlossaryCore.CropFoodEmissionsName, ghg)
            self.dataframes_to_totalize_by_food_type_couplings[GlossaryCore.FoodTypeEnergyEmissionsName.format(ghg)] = (GlossaryCore.CropEnergyEmissionsName, ghg)

        for stream in self.streams_energy_prod:
            self.dataframes_to_totalize_by_food_type_couplings[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + "_breakdown"] = (GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream), "Total")

        # non couplings + couplings
        self.dataframes_to_totalize_by_food_type = self.dataframes_to_totalize_by_food_type_couplings

        for stream in self.streams_energy_prod:
            self.dataframes_to_totalize_by_food_type[GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream) + "_breakdown"] = (GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream), "Total")
            self.dataframes_to_totalize_by_food_type[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream) + "_breakdown"] = (GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream), "Total")

        # these are the parameters that are required to compute each food type
        self.params_for_food_types = [
            GlossaryCore.FoodTypeCapitalStartName,
            GlossaryCore.FoodTypeCapitalDepreciationRateName,
            GlossaryCore.FoodTypeCapitalIntensityName,
            GlossaryCore.FoodTypeKcalByProdUnitName,
            GlossaryCore.FoodTypeWasteByConsumersShareName,
            GlossaryCore.FoodTypeLandUseByProdUnitName,
            GlossaryCore.FoodTypeWasteAtProdAndDistribShareName,
        ]
        for ghg in GlossaryCore.GreenHouseGases:
            self.params_for_food_types.append(GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg))
        for stream_ouput in [GlossaryEnergy.biomass_dry, GlossaryEnergy.wet_biomass]:
            self.params_for_food_types.append(GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(stream_ouput))
            self.params_for_food_types.append(GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(stream_ouput))
            self.params_for_food_types.append(GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(stream_ouput))

        # mapping of the coupling inputs to the compute function, used for the gradients with autograd
        self.mapping_coupling_inputs_argument_number = {
            0: (f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}", GlossaryCore.TotalProductionValue),
            1: (GlossaryCore.WorkforceDfValue, GlossaryCore.SectorAgriculture),
            2: (GlossaryCore.CropProductivityReductionName, GlossaryCore.CropProductivityReductionName),
            3: (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            4: (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
        }

    def get_params_food_type(self, food_type: str):
        params_food_type= {}
        for param in self.params_for_food_types:
            params_food_type[param] = self.inputs[param][food_type] if isinstance(self.inputs[param], dict) else self.inputs[param][food_type].values
        return params_food_type

    def init_dataframes(self):
        years = np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        dataframe_to_init = [
            GlossaryCore.FoodTypeProductionName,
            GlossaryCore.FoodTypeWasteAtProductionDistributionName,
            GlossaryCore.FoodTypeWasteByConsumersName,
            GlossaryCore.FoodTypeNotProducedDueToClimateChangeName,
            GlossaryCore.FoodTypeWasteByClimateDamagesName,
            GlossaryCore.FoodTypeDeliveredToConsumersName,
        ]

        for stream in self.streams_energy_prod:
            for var in [
                GlossaryCore.FoodTypeDedicatedToProductionForStreamName,
                GlossaryCore.WasteBeforeDistribReusedForEnergyProdName,
                GlossaryCore.CropProdForEnergyName
            ]:

                dataframe_to_init.append(var.format(stream))
                dataframe_to_init.append(var.format(stream) + "_breakdown")

        for df_name in dataframe_to_init:
            self.outputs[df_name] = pd.DataFrame({GlossaryCore.Years: years})

        for df_name1, (df_name2, _) in self.dataframes_to_totalize_by_food_type.items():
            self.outputs[df_name1] = pd.DataFrame({GlossaryCore.Years: years})
            self.outputs[df_name2] = pd.DataFrame({GlossaryCore.Years: years})

        for (df_name, column) in self.dataframes_to_totalize_by_food_type.values():
            self.outputs[df_name][column] = 0.

    def compute(self, inputs: dict):
        self.inputs = inputs
        self.init_dataframes()
        for food_type in inputs[GlossaryCore.FoodTypesName]:
            output_food_type = self.compute_food_type(*self.get_args(food_type))
            for varname, value in output_food_type.items():
                self.outputs[varname][food_type] = value
                if varname in self.dataframes_to_totalize_by_food_type:
                    varname_total_df, column_total_df = self.dataframes_to_totalize_by_food_type[varname]
                    self.outputs[varname_total_df][column_total_df] += value

        self.compute_kcal_infos()
        self.compute_kg_infos()

    def get_coupling_inputs_arrays(self) -> tuple:
        """returns the tuple of all the coupling inputs arrays for the compute_food_type function"""
        return tuple([self.inputs[varname][colname].values for varname, colname in self.mapping_coupling_inputs_argument_number.values()])

    def _null_derivative(self):
        nb_years = self.inputs[GlossaryCore.YearEnd] - self.inputs[GlossaryCore.YearStart] + 1
        return np.zeros((nb_years, nb_years))

    def get_args(self, food_type: str):
        return self.get_coupling_inputs_arrays() + (self.inputs[GlossaryCore.FoodTypesInvestName][food_type].values, self.get_params_food_type(food_type))
    def jacobians(self):
        """Compute the gradients using autograd"""
        # gradients dict structure: [input_varname][input_columnname][output_varname][output_colomnname] = value

        gradients = {}

        # jacobians to sum on all food types
        for index, (ci_varname, ci_colomn_name) in enumerate(self.mapping_coupling_inputs_argument_number.values()):
            gradients[ci_varname] = {ci_colomn_name: {}}
            for food_type in self.inputs[GlossaryCore.FoodTypesName]:
                args = self.get_args(food_type)
                jac_coupling_input_food_type = jacobian(lambda *args: self.wrap_outputs_to_arrays(self.compute_food_type(*args)), index)
                gradient_food_type = jac_coupling_input_food_type(*args)
                dict_jacobians_of_food_type = self.unwrap_arrays_to_outputs(gradient_food_type)
                for varname, value in dict_jacobians_of_food_type.items():
                    co_varname, co_colname = self.dataframes_to_totalize_by_food_type_couplings[varname]
                    if co_varname not in gradients[ci_varname][ci_colomn_name]:
                        gradients[ci_varname][ci_colomn_name][co_varname] = {}
                    if co_colname not in gradients[ci_varname][ci_colomn_name][co_varname]:
                        gradients[ci_varname][ci_colomn_name][co_varname][co_colname] = self._null_derivative()
                    gradients[ci_varname][ci_colomn_name][co_varname][co_colname] += value

        # gradients wrt invest food type
        ci_varname = GlossaryCore.FoodTypesInvestName
        gradients[ci_varname] = {}
        for food_type in self.inputs[GlossaryCore.FoodTypesName]:
            ci_colomn_name = food_type
            gradients[ci_varname][ci_colomn_name] = {}
            args = self.get_args(food_type)
            jac_coupling_input_food_type = jacobian(lambda *args: self.wrap_outputs_to_arrays(self.compute_food_type(*args)), 5)
            gradient_food_type = jac_coupling_input_food_type(*args)
            dict_jacobians_of_food_type = self.unwrap_arrays_to_outputs(gradient_food_type)
            for varname, value in dict_jacobians_of_food_type.items():
                co_varname, co_colname = self.dataframes_to_totalize_by_food_type_couplings[varname]
                if co_varname not in gradients[ci_varname][ci_colomn_name]:
                    gradients[ci_varname][ci_colomn_name][co_varname] = {}
                gradients[ci_varname][ci_colomn_name][co_varname][co_colname] = value

        return gradients

    def wrap_outputs_to_arrays(self, outputs: dict):
        """
        gathers the dictionnary outputs of the compute food type function and flattens it in an array
        helps for the using autograd jacobian which only deals with arrays
        """
        return np.array([outputs[varname] for varname in self.dataframes_to_totalize_by_food_type_couplings.keys()])

    def unwrap_arrays_to_outputs(self, array: dict):
        """converts the array output of autograd back to dictionnary to store derivatives values"""
        return {varname: value for varname, value in zip(self.dataframes_to_totalize_by_food_type_couplings.keys(), array)}

    @staticmethod
    def compute_food_type(
            energy_allocated_to_agri: np.ndarray,  # 0
            workforce_agri: np.ndarray,  # 1
            damage_fraction: np.ndarray,  # 2
            crop_productivity_reduction: np.ndarray,  # 3
            population: np.ndarray,  # 4
            invest_food_type: np.ndarray,  # 5
            params: dict,
    ):
        outputs = {}
        # forecasting capital of food type
        capital_food_type = [params[GlossaryCore.FoodTypeCapitalStartName]]  # G$
        for invest in invest_food_type[:-1]:
            capital_food_type.append(capital_food_type[-1] * (1 - params[GlossaryCore.FoodTypeCapitalDepreciationRateName] / 100) + invest)
        capital_food_type = np.array(capital_food_type)  # G$

        # limiting capital to usable capital, depending on the variation of ratios of energy and workforce per capital, relative to year start
        year_start_energy_per_capital = energy_allocated_to_agri[0] / params[GlossaryCore.FoodTypeCapitalStartName]
        year_start_workforce_per_capital = workforce_agri[0] / params[GlossaryCore.FoodTypeCapitalStartName]

        energy_per_capital = energy_allocated_to_agri / capital_food_type
        workforce_per_capital = workforce_agri / capital_food_type

        usable_capital_food_type = capital_food_type * np.minimum(1, np.minimum(energy_per_capital / year_start_energy_per_capital, workforce_per_capital / year_start_workforce_per_capital))
        outputs["non_used_capital_breakdown"] = capital_food_type - usable_capital_food_type

        # computing production : usable capital * capital intensity
        production_raw = usable_capital_food_type * params[GlossaryCore.FoodTypeCapitalIntensityName]  # G$ * t/k$ = 10^9 $ * t/k$ = 10^6 t = Mt

        production_wasted_by_productivity_loss = production_raw * crop_productivity_reduction / 100. # Mt
        outputs[GlossaryCore.FoodTypeWasteByClimateDamagesName] = production_wasted_by_productivity_loss * damage_fraction # Mt
        production_before_waste = production_raw - production_wasted_by_productivity_loss  # Mt

        # split energy and food production land use and emissions
        share_dedicated_to_food = 1
        for stream in Crop.streams_energy_prod:
            share_dedicated_to_food *= (1 - params[GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(stream)] / 100.)
        # emissions
        for ghg in GlossaryCore.GreenHouseGases:
            outputs[GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)] = production_before_waste * params[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)] * share_dedicated_to_food  # Mt_food * (kg{ghg}/kg_food) = 10^9 kg_food * kg_ghg / kg_food = 10^9 kg_ghg = Gt kg_ghg
            outputs[GlossaryCore.FoodTypeEnergyEmissionsName.format(ghg)] = production_before_waste * params[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)] * (1 - share_dedicated_to_food)  # Mt_food * (kg{ghg}/kg_food) = 10^9 kg_food * kg_ghg / kg_food = 10^9 kg_ghg = Gt kg_ghg

        # land use
        outputs[GlossaryCore.CropFoodLandUseName + "_breakdown"] = share_dedicated_to_food * production_raw * params[GlossaryCore.FoodTypeLandUseByProdUnitName] / (10 ** 4)   # Mt * m² / kg = 10^9 kg * m² / kg / 10^4= G m² / 10^4 = G ha
        outputs[GlossaryCore.CropEnergyLandUseName + "_breakdown"] = (1 - share_dedicated_to_food) * production_raw * params[GlossaryCore.FoodTypeLandUseByProdUnitName] / (10 ** 4)   # Mt * m² / kg = 10^9 kg * m² / kg / 10^4= G m² / 10^4 = G ha

        net_production = production_before_waste - outputs[GlossaryCore.FoodTypeWasteByClimateDamagesName]  # Mt
        production_for_consumers = net_production
        for stream in Crop.streams_energy_prod:
            outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + "_breakdown"] = net_production * params[GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(stream)] / 100.
            production_for_consumers -= outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + "_breakdown"]

        food_waste_at_prod_and_distrib = production_for_consumers * params[GlossaryCore.FoodTypeWasteAtProdAndDistribShareName] / 100.  # Mt
        for stream in Crop.streams_energy_prod:
            outputs[GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream) + "_breakdown"] = food_waste_at_prod_and_distrib * params[GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(stream)] / 100.  # Mt

        production_delivered_to_consumers = production_for_consumers - food_waste_at_prod_and_distrib  # Mt
        outputs[GlossaryCore.FoodTypeWasteByConsumersName] = production_delivered_to_consumers * params[GlossaryCore.FoodTypeWasteByConsumersShareName] / 100. # Mt

        for stream in Crop.streams_energy_prod:
            outputs[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream) + "_breakdown"] = outputs[GlossaryCore.FoodTypeWasteByConsumersName] * params[GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(stream)] / 100. # Mt
            outputs[GlossaryCore.CropProdForEnergyName.format(stream) + "_breakdown"] = \
                outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + "_breakdown"] +\
                outputs[GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream) + "_breakdown"] +\
                outputs[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream) + "_breakdown"] # Mt

        kcal_produced_for_consumers = production_delivered_to_consumers * params[GlossaryCore.FoodTypeKcalByProdUnitName]  # Mt * kcal/ kg = 10^9 kg * kcal/kg = 10^9 kcal  = G kcal

        outputs[GlossaryCore.CaloriesPerCapitaBreakdownValue] = kcal_produced_for_consumers / population / 365. * 1000  # Gkcal / (10^6 person) / (day) * 1000 = k kcal / person / day * 1000 = kcal / person / day


        outputs.update({
            GlossaryCore.FoodTypeProductionName: production_for_consumers,
            GlossaryCore.FoodTypeWasteAtProductionDistributionName: food_waste_at_prod_and_distrib,
            GlossaryCore.FoodTypeDeliveredToConsumersName: production_delivered_to_consumers,
            GlossaryCore.FoodTypeNotProducedDueToClimateChangeName: production_wasted_by_productivity_loss,
        })

        return outputs

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

        self.outputs['kcal_dict_infos']['Capital intensity (kcal/$)'] = {
            # (kg / $) * (kcal / kg) = kcal / $
            key: value1 * value2 for (key, value1), value2 in zip(
                self.inputs[GlossaryCore.FoodTypeCapitalIntensityName].items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )
        }

        for dict_name, dict_values in self.outputs['kcal_dict_infos'].items():
            self.outputs['kcal_dict_infos'][dict_name] = dict(sorted(dict_values.items(), key=lambda item: item[1], reverse=True))

    def compute_kg_infos(self):
        self.outputs['kg_dict_infos'] = {}
        for info_name, dict_values in self.outputs['kcal_dict_infos'].items():
            # m2 / kg = (m2 / kcal) * kcal/kg
            self.outputs['kg_dict_infos'][info_name.replace('kcal', 'kg')] = {key: value1 * value2 for (key, value1), value2 in zip(
                dict_values.items(), self.inputs[GlossaryCore.FoodTypeKcalByProdUnitName].values()
            )}
        self.outputs['kg_dict_infos']['Capital intensity (kg/$)'] = {key: value1 for key, value1 in
                                                         self.inputs[GlossaryCore.FoodTypeCapitalIntensityName].items()}
        for dict_name, dict_values in self.outputs['kg_dict_infos'].items():
            self.outputs['kg_dict_infos'][dict_name] = dict(sorted(dict_values.items(), key=lambda item: item[1], reverse=True))