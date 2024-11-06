'''
Copyright 2024 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
from typing import Union
import copy
import logging

import numpy as np
import pandas as pd

from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from climateeconomics.core.core_agriculture.crop_2 import Crop
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class CropDiscipline(ClimateEcoDiscipline):
    """Crop discipline for computing food production"""
    _ontology_data = {
        'label': 'Crop food Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'compute food production',
        'icon': 'fas fa-seedling fa-fw',
        'version': '',
    }
    streams_energy_prod = [GlossaryEnergy.biomass_dry, GlossaryEnergy.wet_biomass]

    invest_df = copy.deepcopy(GlossaryCore.InvestDf)
    invest_df['namespace'] = GlossaryCore.NS_FOOD

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        GlossaryCore.WorkforceDfValue: GlossaryCore.WorkforceDf,
        GlossaryCore.CropProductivityReductionName: GlossaryCore.CropProductivityReductionDf,
        GlossaryCore.FoodTypesName: GlossaryCore.FoodTypesVar,
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}': GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf),
        f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}': GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyProductionDfSectors),
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        'constraint_calories_ref': {'type': 'float', 'default': 4000.},
        'constraint_calories_limit': {'type': 'float', 'default': 2000.},
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
    }

    DESC_OUT = {
        GlossaryCore.FoodLandUseName: GlossaryCore.FoodLandUseVar,
        GlossaryCore.CropFoodEmissionsName: GlossaryCore.CropFoodEmissionsVar,
        GlossaryCore.CropFoodKcalForConsumersName: GlossaryCore.CropFoodKcalForConsumersVar,
        GlossaryCore.CaloriesPerCapitaValue: GlossaryCore.CaloriesPerCapita,
        "kcal_dict_infos": {'type': 'dict',},
        "kg_dict_infos": {'type': 'dict',},
    }
    df_output_streams = {
        GlossaryCore.FoodTypeDedicatedToProductionForStreamName: GlossaryCore.FoodTypeDedicatedToProductionForStreamVar,
        GlossaryCore.WasteBeforeDistribReusedForEnergyProdName: GlossaryCore.WasteBeforeDistribReusedForEnergyProdVar,
        GlossaryCore.ConsumerWasteUsedForEnergyName: GlossaryCore.ConsumerWasteUsedForEnergyVar,
        GlossaryCore.CropProdForEnergyName: GlossaryCore.CropProdForEnergyVar,
    }
    for stream in streams_energy_prod:
        for df_name, df_var_descr in df_output_streams.items():
            df_stream_crop_prod = copy.deepcopy(df_var_descr)
            df_stream_crop_prod["description"] = df_stream_crop_prod["description"].format(stream)
            DESC_OUT.update({df_name.format(stream): df_stream_crop_prod})

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.crop_model = None

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.FoodTypesName in self.get_data_in():
            food_types = self.get_sosdisc_inputs(GlossaryCore.FoodTypesName)
            if food_types is not None:
                dataframes_descriptors = {
                    GlossaryCore.Years: ("int", [1900, GlossaryCore.YearEndDefault], False),
                    **{ft: ("float", None, False) for ft in food_types}
                }
                # inputs
                dataframes_inputs = {
                    GlossaryCore.ShareInvestFoodTypesName: GlossaryCore.ShareInvestFoodTypesVar,
                    GlossaryCore.ShareEnergyUsageFoodTypesName: GlossaryCore.ShareEnergyUsageFoodTypesVar,
                    GlossaryCore.ShareWorkforceFoodTypesName: GlossaryCore.ShareWorkforceFoodTypesVar,
                    GlossaryCore.FoodTypeEnergyNeedName: GlossaryCore.FoodTypeEnergyNeedVar,
                    GlossaryCore.FoodTypeWorkforceNeedName: GlossaryCore.FoodTypeWorkforceNeedVar,
                    GlossaryCore.FoodTypeCapexName: GlossaryCore.FoodTypeCapexVar,
                    GlossaryCore.FoodTypeWasteAtProductionShareName: GlossaryCore.FoodTypeWasteAtProductionShareVar,
                    GlossaryCore.FoodTypeWasteByConsumersShareName: GlossaryCore.FoodTypeWasteByConsumersShareVar,
                    GlossaryCore.FoodTypeLandUseByProdUnitName: GlossaryCore.FoodTypeLandUseByProdUnitVar,
                    GlossaryCore.FoodTypeKcalByProdUnitName: GlossaryCore.FoodTypeKcalByProdUnitVar,
                }
                for stream in self.streams_energy_prod:
                    df_shares = {
                        GlossaryCore.FoodTypeShareDedicatedToStreamProdName: GlossaryCore.FoodTypeShareDedicatedToStreamProdVar,
                        GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName: GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdVar,
                        GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName: GlossaryCore.FoodTypeShareWasteBeforeDistrbUsedToStreamProdVar,
                    }
                    for df_name, df_var in df_shares.items():
                        df_share = copy.deepcopy(df_var)
                        df_share["description"] = df_share["description"].format(stream)
                        dataframes_inputs[df_name.format(stream)] = df_share

                for ghg in GlossaryCore.GreenHouseGases:
                    df_ghg_by_prod_unit = copy.deepcopy(GlossaryCore.FoodTypeEmissionsByProdUnitVar)
                    df_ghg_by_prod_unit["unit"] = df_ghg_by_prod_unit["unit"].format(ghg)
                    df_ghg_by_prod_unit["description"] = df_ghg_by_prod_unit["description"].format(ghg)
                    dataframes_inputs[GlossaryCore.FoodTypeEmissionsByProdUnitName.format(ghg)] = df_ghg_by_prod_unit

                for varname, df_input in dataframes_inputs.items():
                    df_input["dataframe_descriptor"] = dataframes_descriptors
                    dynamic_inputs[varname] = df_input

                # outputs
                dataframes_outputs = {
                    # coupling with breakdown
                    GlossaryCore.FoodTypeWasteAtProductionDistributionName: GlossaryCore.FoodTypeWasteAtProductionDistributionVar,
                    GlossaryCore.FoodTypeWasteByConsumersName: GlossaryCore.FoodTypeWasteByConsumersVar,
                    GlossaryCore.FoodTypeNotProducedDueToClimateChangeName: GlossaryCore.FoodTypeNotProducedDueToClimateChangeVar,
                    GlossaryCore.FoodTypeWasteByClimateDamagesName: GlossaryCore.FoodTypeWasteByClimateDamagesVar,
                    GlossaryCore.FoodTypeDeliveredToConsumersName: GlossaryCore.FoodTypeDeliveredToConsumersVar,
                    GlossaryCore.FoodTypeLandUseName: GlossaryCore.FoodTypeLandUseVar,
                    GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.biomass_dry): GlossaryCore.FoodTypeDedicatedToProductionForStreamVar,
                    GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(GlossaryEnergy.wet_biomass): GlossaryCore.FoodTypeDedicatedToProductionForStreamVar,
                    GlossaryCore.CaloriesPerCapitaBreakdownValue: GlossaryCore.CaloriesPerCapitaBreakdown,
                    "unused_energy" + "_breakdown": {"type": "dataframe", "unit": "PWh",},
                    "unused_workforce" + "_breakdown": {"type": "dataframe", "unit": "million people",},
                }

                for stream in self.streams_energy_prod:
                    df_output_streams = {
                        GlossaryCore.FoodTypeDedicatedToProductionForStreamName: GlossaryCore.FoodTypeDedicatedToProductionForStreamVar,
                        GlossaryCore.WasteBeforeDistribReusedForEnergyProdName: GlossaryCore.WasteBeforeDistribReusedForEnergyProdVar,
                        GlossaryCore.ConsumerWasteUsedForEnergyName: GlossaryCore.ConsumerWasteUsedForEnergyVar,
                        GlossaryCore.CropProdForEnergyName: GlossaryCore.CropProdForEnergyVar,
                    }
                    for df_name, df_var in df_output_streams.items():
                        df_out = copy.deepcopy(df_var)
                        df_out["description"] = df_out["description"].format(stream)
                        dataframes_outputs[df_name.format(stream) + "_breakdown"] = df_out

                for ghg in GlossaryCore.GreenHouseGases:
                    df_ghg = copy.deepcopy(GlossaryCore.FoodTypeEmissionsVar)
                    df_ghg["description"] = df_ghg["description"].format(ghg)
                    dataframes_outputs[GlossaryCore.FoodTypeEmissionsName.format(ghg)] = df_ghg

                for varname, df_output in dataframes_outputs.items():
                    df_output["dataframe_descriptor"] = dataframes_descriptors
                    dynamic_outputs[varname] = df_output

        self.add_inputs(dynamic_inputs)
        #self.update_default_values()
        self.add_outputs(dynamic_outputs)

    def update_default_values(self):
        if GlossaryCore.YearEnd in self.get_data_in() and GlossaryCore.YearStart in self.get_data_in() and GlossaryCore.FoodTypesName in self.get_data_in():
            year_start, year_end, food_types = self.get_sosdisc_inputs([GlossaryCore.YearStart, GlossaryCore.YearEnd, GlossaryCore.FoodTypesName])
            if year_start is not None and food_types is not None:
                years = np.arange(year_start, year_end + 1)
                default_dict_values = {
                    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CO2): {
                        GlossaryCore.RedMeat: 0.0,
                        GlossaryCore.WhiteMeat: 3.95,
                        GlossaryCore.Milk: 0.0,
                        GlossaryCore.Eggs: 1.88,
                        GlossaryCore.RiceAndMaize: 0.84,
                        GlossaryCore.Cereals: 0.12,
                        GlossaryCore.FruitsAndVegetables: 0.44,
                        GlossaryCore.Fish: 2.37,
                        GlossaryCore.OtherFood: 0.48
                    },
                    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CH4): {
                        GlossaryCore.RedMeat: 6.823e-1,
                        GlossaryCore.WhiteMeat: 1.25e-2,
                        GlossaryCore.Milk: 3.58e-2,
                        GlossaryCore.Eggs: 0.0,
                        GlossaryCore.RiceAndMaize: 3.17e-2,
                        # negligible methane in this category
                        GlossaryCore.Cereals: 0.0,
                        GlossaryCore.FruitsAndVegetables: 0.0,
                        # consider fish farm only
                        GlossaryCore.Fish: 3.39e-2,
                        GlossaryCore.OtherFood: 0.,
                        },
                    GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.N2O): {
                        GlossaryCore.RedMeat: 9.268e-3,
                        GlossaryCore.WhiteMeat: 3.90e-4,
                        GlossaryCore.Milk: 2.40e-4,
                        GlossaryCore.Eggs: 1.68e-4,
                        GlossaryCore.RiceAndMaize: 9.486e-4,
                        GlossaryCore.Cereals: 1.477e-3,
                        GlossaryCore.FruitsAndVegetables: 2.63e-4,
                        GlossaryCore.Fish: 0.,  # no crop or livestock related
                        GlossaryCore.OtherFood: 1.68e-3,
                        },
                    GlossaryCore.FoodTypeKcalByProdUnitName: {
                        GlossaryCore.RedMeat: 1551.05,
                        GlossaryCore.WhiteMeat: 2131.99,
                        GlossaryCore.Milk: 921.76,
                        GlossaryCore.Eggs: 1425.07,
                        GlossaryCore.RiceAndMaize: 2572.46,
                        GlossaryCore.Cereals: 2964.99,
                        GlossaryCore.FruitsAndVegetables: 559.65,
                        GlossaryCore.Fish: 609.17,
                        GlossaryCore.OtherFood: 3061.06,
                    },
                    GlossaryCore.FoodTypeLandUseByProdUnitName: {
                        GlossaryCore.RedMeat: 345.,
                        GlossaryCore.WhiteMeat: 14.5,
                        GlossaryCore.Milk: 8.95,
                        GlossaryCore.Eggs: 6.27,
                        GlossaryCore.RiceAndMaize: 2.89,
                        GlossaryCore.Cereals: 4.5,
                        GlossaryCore.FruitsAndVegetables: 0.8,
                        GlossaryCore.Fish: 0.,
                        GlossaryCore.OtherFood: 5.1041,
                    },
                    GlossaryCore.FoodTypeWasteAtProductionShareName: {
                        GlossaryCore.RedMeat: 3,
                        GlossaryCore.WhiteMeat: 3,
                        GlossaryCore.Milk: 8,
                        GlossaryCore.Eggs: 7,
                        GlossaryCore.RiceAndMaize: 8,
                        GlossaryCore.Cereals: 10,
                        GlossaryCore.FruitsAndVegetables: 15,
                        GlossaryCore.Fish: 10,
                        GlossaryCore.OtherFood: 5,
                    },
                    GlossaryCore.FoodTypeWasteByConsumersName: {
                        GlossaryCore.RedMeat: 3,
                        GlossaryCore.WhiteMeat: 3,
                        GlossaryCore.Milk: 8,
                        GlossaryCore.Eggs: 7,
                        GlossaryCore.RiceAndMaize: 8,
                        GlossaryCore.Cereals: 10,
                        GlossaryCore.FruitsAndVegetables: 15,
                        GlossaryCore.Fish: 10,
                        GlossaryCore.OtherFood: 5,
                    },
                    GlossaryCore.FoodTypeEnergyNeedName: {
                        GlossaryCore.RedMeat: 1,
                        GlossaryCore.WhiteMeat: 1,
                        GlossaryCore.Milk: 1,
                        GlossaryCore.Eggs: 1,
                        GlossaryCore.RiceAndMaize: 1,
                        GlossaryCore.Cereals: 1,
                        GlossaryCore.FruitsAndVegetables: 1,
                        GlossaryCore.Fish: 1,
                        GlossaryCore.OtherFood: 1,
                    },
                    GlossaryCore.FoodTypeWorkforceNeedName: {
                        GlossaryCore.RedMeat: 3,
                        GlossaryCore.WhiteMeat: 3,
                        GlossaryCore.Milk: 8,
                        GlossaryCore.Eggs: 7,
                        GlossaryCore.RiceAndMaize: 8,
                        GlossaryCore.Cereals: 10,
                        GlossaryCore.FruitsAndVegetables: 15,
                        GlossaryCore.Fish: 10,
                        GlossaryCore.OtherFood: 5,
                    },
                    GlossaryCore.ShareInvestFoodTypesName: {
                        GlossaryCore.RedMeat: 1/9 * 100.,
                        GlossaryCore.WhiteMeat: 1/9 * 100.,
                        GlossaryCore.Milk: 1/9 * 100.,
                        GlossaryCore.Eggs: 1/9 * 100.,
                        GlossaryCore.RiceAndMaize: 1/9 * 100.,
                        GlossaryCore.Cereals: 1/9 * 100.,
                        GlossaryCore.FruitsAndVegetables: 1/9 * 100.,
                        GlossaryCore.Fish: 1/9 * 100.,
                        GlossaryCore.OtherFood: 1/9 * 100.,
                    },
                    GlossaryCore.ShareEnergyUsageFoodTypesName: {
                        GlossaryCore.RedMeat: 1/9 * 100.,
                        GlossaryCore.WhiteMeat: 1/9 * 100.,
                        GlossaryCore.Milk: 1/9 * 100.,
                        GlossaryCore.Eggs: 1/9 * 100.,
                        GlossaryCore.RiceAndMaize: 1/9 * 100.,
                        GlossaryCore.Cereals: 1/9 * 100.,
                        GlossaryCore.FruitsAndVegetables: 1/9 * 100.,
                        GlossaryCore.Fish: 1/9 * 100.,
                        GlossaryCore.OtherFood: 1/9 * 100.,
                    },
                }

                out = {}
                for varname, default_dict_values_var in default_dict_values.items():
                    df = pd.DataFrame({
                        GlossaryCore.Years: years,
                        **default_dict_values_var
                    })
                    #self.update_default_value(varname, 'in', df)
                    out.update({varname: df})
                self.set_dynamic_default_values(out)

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.crop_model = Crop(param)

    def run(self):
        # -- get inputs
        input_dict = self.get_sosdisc_inputs()

        # -- configure class with inputs
        self.crop_model.compute(input_dict)

        outputs = self.crop_model.outputs
        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        pass

    def get_chart_filter_list(self):
        chart_filters = []
        # chart_list = ["assets emissions", "assets emissions per category",
        #               "transportation subsector assets emissions", "agriculture subsector assets emissions",
        #               "energy subsector assets emissions", "Oil and Gas subsector assets emissions"]
        chart_list = [
            "Production",
            "Damages",
            "Waste",
            'Land use',
            'Emissions',
            'Crop for energy production',
            "Food data (kcal)",
            "Food data (kg)",
            "Calibration",
        ]

        chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        instanciated_charts = []
        charts = []

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values

        inputs = self.get_sosdisc_inputs()
        outputs = self.get_sosdisc_outputs()
        if "Production" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.CaloriesPerCapitaBreakdownValue],
                charts_name="Calories per capita",
                unit=GlossaryCore.CaloriesPerCapitaBreakdown['unit'],
                df_total=outputs[GlossaryCore.CaloriesPerCapitaValue],
                column_total="kcal_pc",
                post_proc_category="Production",
            )
            instanciated_charts.append(new_chart)

        if "Production" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeDeliveredToConsumersName],
                charts_name="Production delivered to consumers",
                unit=GlossaryCore.FoodTypeDeliveredToConsumersVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Production",
                lines=True,
            )
            instanciated_charts.append(new_chart)

        if "Damages" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeNotProducedDueToClimateChangeName],
                charts_name="Food not produced due to loss of productivity (climate change)",
                unit=GlossaryCore.FoodTypeNotProducedDueToClimateChangeVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Damages",
                lines=True,
                note={"Loss of productivity": "climate change lowers crop yields"}
            )
            instanciated_charts.append(new_chart)

        if "Damages" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeWasteByClimateDamagesName],
                charts_name="Food produced and waste due to climate damages",
                unit=GlossaryCore.FoodTypeWasteByClimateDamagesVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Damages",
                lines=True,
                note={"Climate damages": "Extreme events such as floodings, droughts, fires destroy production."}
            )
            instanciated_charts.append(new_chart)

        if "Waste" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeWasteAtProductionDistributionName],
                charts_name="Food wasted at production and distribution level",
                unit=GlossaryCore.FoodTypeWasteAtProductionDistributionVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Waste",
                lines=True,
                note={"Waste at production and distribution": "some of it is reused for energy production"},
            )
            instanciated_charts.append(new_chart)

        if "Waste" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeWasteByConsumersName],
                charts_name="Food wasted at consumers level",
                unit=GlossaryCore.FoodTypeWasteByConsumersVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Waste",
                lines=True,
                note={"Food waste by consumers": "some of it is reused for energy production"}
            )
            instanciated_charts.append(new_chart)

        if "Land use" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeLandUseName],
                charts_name="Land use",
                unit=GlossaryCore.FoodLandUseVar['unit'],
                df_total=outputs[GlossaryCore.FoodLandUseName],
                column_total="Total",
                post_proc_category="Land use"
            )
            instanciated_charts.append(new_chart)

        if "Emissions" in charts:
            for ghg in GlossaryCore.GreenHouseGases:
                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.FoodTypeEmissionsName.format(ghg)],
                    charts_name=f"{ghg} Emissions of food production",
                    unit=GlossaryCore.CropFoodEmissionsVar['unit'],
                    df_total=outputs[GlossaryCore.CropFoodEmissionsName],
                    column_total=ghg,
                    post_proc_category="Emissions"
                )
                instanciated_charts.append(new_chart)

        if "Energy production" in charts:
            for stream in self.streams_energy_prod:
                stream_nicer = stream.replace('_', ' ')
                stream_nicer = stream_nicer.capitalize()
                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream) + '_breakdown'],
                    charts_name=f"Dedicated crop production for {stream_nicer} stream",
                    unit=GlossaryCore.FoodTypeDedicatedToProductionForStreamVar['unit'],
                    df_total=outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream)],
                    column_total="Total",
                    post_proc_category="Energy production",
                )
                instanciated_charts.append(new_chart)

                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream) + '_breakdown'],
                    charts_name=f"Waste before distribution reused for {stream_nicer} production",
                    unit=GlossaryCore.WasteBeforeDistribReusedForEnergyProdVar['unit'],
                    df_total=outputs[GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream)],
                    column_total="Total",
                    post_proc_category="Energy production",
                )
                instanciated_charts.append(new_chart)

                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream) + '_breakdown'],
                    charts_name=f"Consumers waste reused for {stream_nicer} production",
                    unit=GlossaryCore.ConsumerWasteUsedForEnergyVar['unit'],
                    df_total=outputs[GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream)],
                    column_total="Total",
                    post_proc_category="Energy production",
                )
                instanciated_charts.append(new_chart)

                new_chart = self.graph_total_prod_for_energy_chart(stream=stream)
                instanciated_charts.append(new_chart)

        if "Food data (kcal)" in charts:
            for output_name, dict_values in outputs['kcal_dict_infos'].items():
                unit = output_name.split(" (")[1]
                unit = unit.replace(')','')
                new_chart = self.get_dict_bar_plot(
                    dict_values=dict_values,
                    charts_name=output_name.split(' (')[0],
                    unit=unit,
                    post_proc_category="Food data (kcal)",
                    note={"CO2 equivalent": "100-years basis"} if "emissions" in output_name.lower() else {}
                )
                instanciated_charts.append(new_chart)

        if "Food data (kg)" in charts:
            for output_name, dict_values in outputs['kg_dict_infos'].items():
                unit = output_name.split(" (")[1]
                unit = unit.replace(')', '')
                new_chart = self.get_dict_bar_plot(
                    dict_values=dict_values,
                    charts_name=output_name.split(' (')[0],
                    unit=unit,
                    post_proc_category="Food data (kg)",
                    note={"CO2 equivalent": "100-years basis"} if "emissions" in output_name.lower() else {}
                )
                instanciated_charts.append(new_chart)

        if "Calibration" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs["unused_energy" + "_breakdown"],
                charts_name="Unused energy",
                unit="PWh",
                df_total=inputs[f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}'],
                column_total=GlossaryCore.TotalProductionValue,
                post_proc_category="Calibration",
            )
            instanciated_charts.append(new_chart)

            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs["unused_workforce" + "_breakdown"],
                charts_name="Unused workforce",
                unit="milllion people",
                df_total=inputs[GlossaryCore.WorkforceDfValue],
                column_total=GlossaryCore.SectorAgriculture,
                post_proc_category="Calibration",
            )
            instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_breakdown_charts_on_food_type(self,
                                          df_all_food_types: pd.DataFrame,
                                          charts_name: str,
                                          unit: str,
                                          df_total: Union[None, pd.DataFrame],
                                          column_total: Union[None, str],
                                          post_proc_category: Union[None, str],
                                          lines: bool = False,
                                          note: Union[dict, None] = None):

        years = df_all_food_types[GlossaryCore.Years]
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, unit, stacked_bar=True, chart_name=charts_name, y_min_zero=lines)

        for col in df_all_food_types.columns:
            if col != GlossaryCore.Years:
                new_series = InstanciatedSeries(years, df_all_food_types[col], str(col).capitalize(), 'bar' if not lines else "lines", True)
                new_chart.add_series(new_series)

        if df_total is not None and column_total is not None:
            new_series = InstanciatedSeries(years, df_total[column_total], 'Total', 'lines', True)
            new_chart.add_series(new_series)

        if post_proc_category is not None:
            new_chart.post_processing_section_name = post_proc_category

        if note is not None:
            new_chart.annotation_upper_left = note
        return new_chart

    def graph_total_prod_for_energy_chart(self, stream: str):
        stream_nicer = stream.replace('_', ' ')
        stream_nicer = stream_nicer.capitalize()

        dfs_to_sum = {
            GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream): GlossaryCore.FoodTypeDedicatedToProductionForStreamVar,
            GlossaryCore.WasteBeforeDistribReusedForEnergyProdName.format(stream): GlossaryCore.WasteBeforeDistribReusedForEnergyProdVar,
            GlossaryCore.ConsumerWasteUsedForEnergyName.format(stream): GlossaryCore.ConsumerWasteUsedForEnergyVar,
        }
        df_total = self.get_sosdisc_outputs(GlossaryCore.CropProdForEnergyName.format(stream))

        years = df_total[GlossaryCore.Years]
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.CropProdForEnergyVar['unit'], stacked_bar=True,
                                             chart_name=f"{stream_nicer} for energy production: breakdown of production")

        for df_name, df_var_descr in dfs_to_sum.items():
            df_value = self.get_sosdisc_outputs(df_name)
            description = df_var_descr["description"].format(stream_nicer)
            new_series = InstanciatedSeries(years, df_value["Total"], description, 'bar', True)
            new_chart.add_series(new_series)

        new_series = InstanciatedSeries(years, df_total["Total"], 'Total', 'lines', True)
        new_chart.add_series(new_series)

        new_chart.post_processing_section_name = "Energy production"

        return new_chart

    def get_dict_bar_plot(self,
                          dict_values: dict,
                          charts_name: str,
                          unit: str,
                          post_proc_category: Union[None, str],
                          note: Union[dict, None] = None):

        new_chart = TwoAxesInstanciatedChart('', unit, stacked_bar=True, chart_name=charts_name)

        for key, value in dict_values.items():
            if key != GlossaryCore.Years:
                new_series = InstanciatedSeries([str(key).capitalize()], [value], '', 'bar', True)
                new_chart.add_series(new_series)

        if post_proc_category is not None:
            new_chart.post_processing_section_name = post_proc_category

        if note is not None:
            new_chart.annotation_upper_left = note
        return new_chart
