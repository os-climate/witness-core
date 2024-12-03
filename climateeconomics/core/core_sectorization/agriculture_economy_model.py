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
from autograd import jacobian
from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class AgricultureEconomyModel:
    """
    Agriculture economy model
    """

    streams_energy_prod = [GlossaryEnergy.biomass_dry, GlossaryEnergy.wet_biomass]

    def __init__(self):
        """Constructor"""
        self.inputs = {}
        self.outputs = {}

        # couplings
        self.coupling_dataframes_not_totalized = [
            GlossaryCore.FoodTypeDeliveredToConsumersName,
            GlossaryCore.FoodTypeCapitalName,
        ]
        self.dataframes_to_totalize_by_food_type_couplings = {
            GlossaryCore.Damages + "_breakdown": (f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            GlossaryCore.EstimatedDamages + "_breakdown": (f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}", GlossaryCore.EstimatedDamages),

            GlossaryCore.GrossOutput + "_breakdown": (f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            GlossaryCore.OutputNetOfDamage + "_breakdown": (f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
        }

        # non couplings + couplings
        self.dataframes_to_totalize_by_food_type = self.dataframes_to_totalize_by_food_type_couplings
        self.dataframes_to_totalize_by_food_type.update({
            GlossaryCore.CropFoodNetGdpName + "_breakdown": (GlossaryCore.CropFoodNetGdpName, "Total"),
            GlossaryCore.CropEnergyNetGdpName + "_breakdown": (GlossaryCore.CropEnergyNetGdpName, "Total"),
        })

        # these are the parameters that are required to compute each food type
        self.params_for_food_types = [
            GlossaryCore.FoodTypeEnergyIntensityByProdUnitName,
            GlossaryCore.FoodTypeLaborIntensityByProdUnitName,
            GlossaryCore.FoodTypeCapitalMaintenanceCostName,
            GlossaryCore.FoodTypesPriceMarginShareName,
        ]
        # mapping of the coupling inputs to the compute function, used for the gradients with autograd
        self.mapping_coupling_inputs_argument_number_food_types = {
            0: GlossaryCore.FoodTypeCapitalName,
            1: GlossaryCore.FoodTypeNotProducedDueToClimateChangeName,
            2: GlossaryCore.FoodTypeWasteByClimateDamagesName,
            3: GlossaryCore.FoodTypeDeliveredToConsumersName,
            4: GlossaryCore.CropProdForAllStreamName,
        }

        self.mapping_coupling_inputs_argument_number = {
            5: (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue)
        }

    def get_params_food_type(self, food_type: str) -> dict:
        params_food_type = {}
        for param in self.params_for_food_types:
            params_food_type[param] = self.inputs[param][food_type] if isinstance(self.inputs[param], dict) else self.inputs[param][food_type].values

        for var in [GlossaryCore.YearStart, GlossaryCore.YearEnd]:
            params_food_type[var] = self.inputs[var]

        return params_food_type

    def get_couplings_food_type(self, food_type: str):
        params_food_type= {}
        for param in self.params_for_food_types:
            params_food_type[param] = self.inputs[param][food_type] if isinstance(self.inputs[param], dict) else self.inputs[param][food_type].values

        return params_food_type

    def init_dataframes(self):
        years = np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        dataframe_to_init = [
            GlossaryCore.FoodTypesPriceName
        ]

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
            output_food_type, price_breakdown_df = self.compute_food_type(*self.get_args(food_type))
            for varname, value in output_food_type.items():
                self.outputs[varname][food_type] = value
                if varname in self.dataframes_to_totalize_by_food_type:
                    varname_total_df, column_total_df = self.dataframes_to_totalize_by_food_type[varname]
                    self.outputs[varname_total_df][column_total_df] += value
            self.outputs[f"{food_type}_price_breakdown"] = price_breakdown_df

    def get_coupling_inputs_arrays_food_types(self, food_type: str) -> tuple:
        """returns the tuple of all the coupling inputs arrays for the compute_food_type function"""
        return tuple([self.inputs[varname][food_type].values for varname in self.mapping_coupling_inputs_argument_number_food_types.values()])

    def get_coupling_inputs_arrays(self) -> tuple:
        """returns the tuple of all the coupling inputs arrays for the compute_food_type function"""
        return tuple([self.inputs[varname][colname].values for varname, colname in self.mapping_coupling_inputs_argument_number.values()])

    def _null_derivative(self):
        nb_years = self.inputs[GlossaryCore.YearEnd] - self.inputs[GlossaryCore.YearStart] + 1
        return np.zeros((nb_years, nb_years))

    def get_args(self, food_type: str):
        return self.get_coupling_inputs_arrays_food_types(food_type) + \
               self.get_coupling_inputs_arrays() + \
               (self.get_params_food_type(food_type), )

    def jacobians(self):
        """Compute the gradients using autograd"""
        # gradients dict structure: [input_varname][input_columnname][output_varname][output_colomnname] = value

        gradients = {}

        # jacobians to sum on all food types
        for index, (ci_varname, ci_colomn_name) in enumerate(self.mapping_coupling_inputs_argument_number_food_types.values()):
            gradients[ci_varname] = {ci_colomn_name: {}}
            for food_type in self.inputs[GlossaryCore.FoodTypesName]:
                args = self.get_args(food_type)
                jac_coupling_input_food_type = jacobian(lambda *args: self.wrap_outputs_to_arrays(self.compute_food_type(*args)[0]), index)
                gradient_food_type = jac_coupling_input_food_type(*args)
                dict_jacobians_of_food_type = self.unwrap_arrays_to_outputs(gradient_food_type)
                for varname, value in dict_jacobians_of_food_type.items():
                    co_varname, co_colname = self.dataframes_to_totalize_by_food_type_couplings[varname] if varname in self.dataframes_to_totalize_by_food_type_couplings else (varname, food_type)
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
            jac_coupling_input_food_type = jacobian(lambda *args: self.wrap_outputs_to_arrays(self.compute_food_type(*args))[0], 5)
            gradient_food_type = jac_coupling_input_food_type(*args)
            dict_jacobians_of_food_type = self.unwrap_arrays_to_outputs(gradient_food_type)
            for varname, value in dict_jacobians_of_food_type.items():
                co_varname, co_colname = self.dataframes_to_totalize_by_food_type_couplings[varname] if varname in self.dataframes_to_totalize_by_food_type_couplings else (varname, food_type)
                if co_varname not in gradients[ci_varname][ci_colomn_name]:
                    gradients[ci_varname][ci_colomn_name][co_varname] = {}
                gradients[ci_varname][ci_colomn_name][co_varname][co_colname] = value

        return gradients

    def wrap_outputs_to_arrays(self, outputs: dict):
        """
        gathers the dictionnary outputs of the compute food type function and flattens it in an array
        helps for the using autograd jacobian which only deals with arrays
        """
        return np.array([outputs[varname] for varname in list(self.dataframes_to_totalize_by_food_type_couplings.keys()) + self.coupling_dataframes_not_totalized])

    def unwrap_arrays_to_outputs(self, array: dict):
        """converts the array output of autograd back to dictionnary to store derivatives values"""
        return {varname: value for varname, value in zip(list(self.dataframes_to_totalize_by_food_type_couplings.keys()) + self.coupling_dataframes_not_totalized,
                                                         array)}

    @staticmethod
    def compute_food_type(
            capital_food_type: np.ndarray,  # 0
            production_loss_from_prod_loss: np.ndarray,  # 1
            production_loss_from_immediate_climate_damages: np.ndarray,  # 2
            production_delivered_to_consumers: np.ndarray,  # 3
            production_for_all_streams: np.ndarray,  # 4
            energy_price: np.ndarray, # 5

            params: dict,
    ):
        outputs = {}

        # compute unitary price
        labor_cost = params[GlossaryCore.FoodTypeLaborIntensityByProdUnitName] / 1000 # $/ton to $/kg
        energy_cost = params[GlossaryCore.FoodTypeEnergyIntensityByProdUnitName] * energy_price / 1e6 # kWh/ton * $/MWh = kWh/ton * $/kkWh = $/(k ton) =$/(Mkg) then /1e6 to $/kg
        capital_maintenance_cost = capital_food_type * params[GlossaryCore.FoodTypeCapitalMaintenanceCostName]

        capex_amortization_cost = 0.

        price_wo_margin = labor_cost + energy_cost + capital_maintenance_cost + capex_amortization_cost

        margin_share_of_final_price = params[GlossaryCore.FoodTypesPriceMarginShareName]
        margin = margin_share_of_final_price / 100 * price_wo_margin / (1 - margin_share_of_final_price / 100)
        final_price = price_wo_margin + margin # $/kg


        # compute gross output for crop:
        damages_prod_loss = production_loss_from_prod_loss * final_price / 1e3 # Mt * $ / kg = G kg * $ / kg so divide by 1e3 to get T$
        damages_immediate_climate_damages = production_loss_from_immediate_climate_damages * final_price / 1e3
        damages = damages_prod_loss + damages_immediate_climate_damages / 1e3

        # gdp for energy
        net_gdp_energy = production_for_all_streams * final_price / 1e3

        # gdp food
        net_gdp_food = production_delivered_to_consumers * final_price / 1e3

        net_gdp = net_gdp_food + net_gdp_energy

        # price breakdown df
        price_breakdown_df = pd.DataFrame({
            GlossaryCore.Years: np.arange(params[GlossaryCore.YearStart], params[GlossaryCore.YearEnd] + 1),
            "Labor": labor_cost,
            "Energy": energy_cost,
            "Capital maintenance": capital_maintenance_cost,
            "Capital amortization": capex_amortization_cost,
            "Margin": margin
        })


        gross_output = net_gdp + damages
        outputs.update({
            GlossaryCore.Damages + "_breakdown": damages,
            GlossaryCore.EstimatedDamages + "_breakdown": damages,

            GlossaryCore.GrossOutput + "_breakdown": gross_output,
            GlossaryCore.OutputNetOfDamage + "_breakdown": net_gdp,

            GlossaryCore.CropFoodNetGdpName + "_breakdown": net_gdp_food,
            GlossaryCore.CropEnergyNetGdpName + "_breakdown": net_gdp_energy,

            GlossaryCore.FoodTypesPriceName: final_price
        })


        return outputs, price_breakdown_df
