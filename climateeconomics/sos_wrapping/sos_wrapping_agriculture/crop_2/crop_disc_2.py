'''
Copyright 2024 Capgemini
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
import copy
import logging
from typing import Union

import pandas as pd
import plotly.colors
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
    streams_energy_prod = Crop.streams_energy_prod

    invest_df = copy.deepcopy(GlossaryCore.InvestDf)
    invest_df['namespace'] = GlossaryCore.NS_CROP

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        GlossaryCore.WorkforceDfValue: GlossaryCore.WorkforceDf,
        GlossaryCore.CropProductivityReductionName: GlossaryCore.CropProductivityReductionDf,
        GlossaryCore.FoodTypesName: GlossaryCore.FoodTypesVar,
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}': GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyProductionDfSectors),
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
    }

    DESC_OUT = {
        GlossaryCore.CropFoodLandUseName: GlossaryCore.CropFoodLandUseVar,
        GlossaryCore.CropEnergyLandUseName: GlossaryCore.CropEnergyLandUseVar,
        GlossaryCore.CropFoodEmissionsName: GlossaryCore.CropFoodEmissionsVar,
        GlossaryCore.CropEnergyEmissionsName: GlossaryCore.CropEnergyEmissionsVar,
        GlossaryCore.CaloriesPerCapitaValue: GlossaryCore.CaloriesPerCapita,
        "non_used_capital": {"type": "dataframe", "unit": "G$", "description": "Lost capital due to missing workforce or energy attribution to agriculture sector"},
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
        self.food_types_colors = {}
        self.crop_model = Crop()

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
                dynamic_inputs.update({
                    GlossaryCore.FoodTypeCapitalStartName: GlossaryCore.FoodTypeCapitalStartVar,
                    GlossaryCore.FoodTypeCapitalIntensityName: GlossaryCore.FoodTypeCapitalIntensityVar,
                    GlossaryCore.FoodTypeCapitalDepreciationRateName: GlossaryCore.FoodTypeCapitalDepreciationRateVar,
                    GlossaryCore.FoodTypeLandUseByProdUnitName: GlossaryCore.FoodTypeLandUseByProdUnitVar,
                    GlossaryCore.FoodTypeKcalByProdUnitName: GlossaryCore.FoodTypeKcalByProdUnitVar,
                })

                dataframes_inputs = {
                    GlossaryCore.FoodTypeWasteAtProdAndDistribShareName: GlossaryCore.FoodTypeWasteAtProductionShareVar,
                    GlossaryCore.FoodTypeWasteByConsumersShareName: GlossaryCore.FoodTypeWasteByConsumersShareVar,
                    GlossaryCore.FoodTypesInvestName: GlossaryCore.FoodTypesInvestVar}

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
                    GlossaryCore.FoodTypeCapitalName: GlossaryCore.FoodTypeCapitalVar,
                    GlossaryCore.FoodTypeProductionName: GlossaryCore.FoodTypeProductionVar,
                    GlossaryCore.FoodTypeWasteAtProductionDistributionName: GlossaryCore.FoodTypeWasteAtProductionDistributionVar,
                    GlossaryCore.FoodTypeWasteByConsumersName: GlossaryCore.FoodTypeWasteByConsumersVar,
                    GlossaryCore.FoodTypeNotProducedDueToClimateChangeName: GlossaryCore.FoodTypeNotProducedDueToClimateChangeVar,
                    GlossaryCore.FoodTypeWasteByClimateDamagesName: GlossaryCore.FoodTypeWasteByClimateDamagesVar,
                    GlossaryCore.FoodTypeDeliveredToConsumersName: GlossaryCore.FoodTypeDeliveredToConsumersVar,
                    GlossaryCore.CropFoodLandUseName + "_breakdown": {"type": "dataframe", "unit": "(Gha)", "description": "Land used by each food type for food production"},
                    GlossaryCore.CropEnergyLandUseName + "_breakdown": {"type": "dataframe", "unit": "(Gha)", "description": "Land used by each food type for energy production in first intention. That is "},
                    GlossaryCore.CaloriesPerCapitaBreakdownValue: GlossaryCore.CaloriesPerCapitaBreakdown,
                    "non_used_capital_breakdown": {"type": "dataframe", "unit": "G$", "description": "Lost capital due to missing workforce or energy attribution to agriculture sector"}
                }

                for stream in self.streams_energy_prod:
                    dataframes_outputs[GlossaryCore.FoodTypeDedicatedToProductionForStreamName.format(stream)] = GlossaryCore.FoodTypeDedicatedToProductionForStreamVar

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
                    for varname, variable in [
                        (GlossaryCore.FoodTypeFoodEmissionsName, GlossaryCore.FoodTypeFoodEmissionsVar),
                        (GlossaryCore.FoodTypeEnergyEmissionsName, GlossaryCore.FoodTypeEnergyEmissionsVar),
                    ]:
                        df_ghg = copy.deepcopy(variable)
                        df_ghg["description"] = df_ghg["description"].format(ghg)
                        dataframes_outputs[varname.format(ghg)] = df_ghg

                for varname, df_output in dataframes_outputs.items():
                    df_output["dataframe_descriptor"] = dataframes_descriptors
                    dynamic_outputs[varname] = df_output

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        # -- get inputs
        input_dict = self.get_sosdisc_inputs()
        self.crop_model.compute(input_dict)

        self.store_sos_outputs_values(self.crop_model.outputs)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        gradients = self.crop_model.jacobians()

        for input_varname in gradients:
            for input_colname in gradients[input_varname]:
                for output_varname in gradients[input_varname][input_colname]:
                    for output_colname, value in gradients[input_varname][input_colname][output_varname].items():
                        self.set_partial_derivative_for_other_types(
                            (output_varname, output_colname),
                            (input_varname, input_colname),
                            value)

    def get_chart_filter_list(self):
        chart_list = [
            "Production",
            "Damages",
            "Waste",
            'Land use',
            'Emissions',
            'Crop for energy production',
            "Food data (kcal)",
            "Food data (kg)",
        ]

        return [ChartFilter("Charts", chart_list, chart_list, "charts")]

    def get_post_processing_list(self, filters=None):
        instanciated_charts = []
        charts = []

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values

        outputs = self.get_sosdisc_outputs()
        self.food_types_colors = {
            GlossaryCore.RedMeat: 'crimson',
            GlossaryCore.WhiteMeat: 'burlywood',
            GlossaryCore.Milk: 'ivory',
            GlossaryCore.Eggs: 'yellow',
            GlossaryCore.RiceAndMaize: 'gold',
            GlossaryCore.Cereals: 'olive',
            GlossaryCore.FruitsAndVegetables: 'green',
            GlossaryCore.Fish: 'cornflowerblue',
            GlossaryCore.OtherFood: 'lightslategrey'
        }
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

        if "Production" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypeProductionName],
                charts_name="Net production",
                unit=GlossaryCore.FoodTypeProductionVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Production",
                lines=True,
                note={"Net production": "waste is not applied"}
            )
            instanciated_charts.append(new_chart)

        if "Production" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs["non_used_capital_breakdown"],
                charts_name="Non used capital",
                unit=GlossaryCore.FoodTypeCapitalStartVar['unit'],
                df_total=outputs["non_used_capital"],
                column_total="Total",
                post_proc_category="Production",
                note={"Non used capital": "due to missing workforce or energy attribution to agriculture sector"}
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
                charts_name="Food produced and wasted due to climate damages",
                unit=GlossaryCore.FoodTypeWasteByClimateDamagesVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Damages",
                lines=True,
                note={"Climate damages": "Extreme events such as floodings, droughts, fires destroy production."}
            )
            instanciated_charts.append(new_chart)

        if "Damages" in charts:
            df_sum = outputs[GlossaryCore.FoodTypeNotProducedDueToClimateChangeName] + outputs[GlossaryCore.FoodTypeWasteByClimateDamagesName]
            df_sum[GlossaryCore.Years] = outputs[GlossaryCore.FoodTypeNotProducedDueToClimateChangeName][GlossaryCore.Years]
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=df_sum,
                charts_name="Total damages due to climate (productivity loss + extreme events impacts)",
                unit=GlossaryCore.FoodTypeWasteByClimateDamagesVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Damages",
                lines=True,
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

        if "Waste" in charts:
            df_sum = outputs[GlossaryCore.FoodTypeWasteAtProductionDistributionName] + outputs[GlossaryCore.FoodTypeWasteByConsumersName]
            df_sum[GlossaryCore.Years] = outputs[GlossaryCore.FoodTypeWasteAtProductionDistributionName][GlossaryCore.Years]
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=df_sum,
                charts_name="Total wasted food (production + distribution + consumers)",
                unit=GlossaryCore.FoodTypeWasteByConsumersVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Waste",
                lines=True,
            )
            instanciated_charts.append(new_chart)

        if "Land use" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.CropFoodLandUseName + "_breakdown"],
                charts_name="Land use for food production",
                unit=GlossaryCore.CropFoodLandUseVar['unit'],
                df_total=outputs[GlossaryCore.CropFoodLandUseName],
                column_total="Total",
                post_proc_category="Land use"
            )
            instanciated_charts.append(new_chart)

            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.CropEnergyLandUseName + "_breakdown"],
                charts_name="Land use for energy production",
                unit=GlossaryCore.CropFoodLandUseVar['unit'],
                df_total=outputs[GlossaryCore.CropEnergyLandUseName],
                column_total="Total",
                post_proc_category="Land use"
            )
            instanciated_charts.append(new_chart)

        if "Emissions" in charts:
            for ghg in GlossaryCore.GreenHouseGases:
                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.FoodTypeFoodEmissionsName.format(ghg)],
                    charts_name=f"{ghg} Emissions of food production",
                    unit=GlossaryCore.CropFoodEmissionsVar['unit'],
                    df_total=outputs[GlossaryCore.CropFoodEmissionsName],
                    column_total=ghg,
                    post_proc_category="Emissions"
                )
                instanciated_charts.append(new_chart)

            for ghg in GlossaryCore.GreenHouseGases:
                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[GlossaryCore.FoodTypeEnergyEmissionsName.format(ghg)],
                    charts_name=f"{ghg} Emissions of energy production",
                    unit=GlossaryCore.CropEnergyEmissionsVar['unit'],
                    df_total=outputs[GlossaryCore.CropEnergyEmissionsName],
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

        list_food_types = list(df_all_food_types.columns)
        list_food_types.remove(GlossaryCore.Years)
        if not lines:
            dict_mean_columns = {col: df_all_food_types[col].mean() for col in df_all_food_types.columns if col != GlossaryCore.Years}
            sorted_dict_columns = dict(sorted(dict_mean_columns.items(), key=lambda item: item[1], reverse=True))
            list_food_types = list(sorted_dict_columns.keys())
        for col in list_food_types:
            dict_color = {'color': self.food_types_colors[col]} if col in self.food_types_colors else None
            kwargs = {'line': dict_color} if lines else {'marker': dict_color}
            new_series = InstanciatedSeries(years, df_all_food_types[col], str(col).capitalize(), 'bar' if not lines else "lines", True, **kwargs)
            new_chart.add_series(new_series)

        if df_total is not None and column_total is not None:
            new_series = InstanciatedSeries(years, df_total[column_total], 'Total', 'lines', True, line={'color': 'gray'})
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
                dict_color = {'color': self.food_types_colors[key]} if key in self.food_types_colors else None
                new_series = InstanciatedSeries([str(key).capitalize()], [value], '', 'bar', True, marker=dict_color)
                new_chart.add_series(new_series)

        if post_proc_category is not None:
            new_chart.post_processing_section_name = post_proc_category

        if note is not None:
            new_chart.annotation_upper_left = note
        return new_chart


def generate_distinct_colors(labels: list[str]):
    """
    Generate a dictionary of distinct plotly colors for a given list of labels.

    :param labels: List of string labels
    :return: Dictionary mapping each label to a distinct plotly color
    """
    # Get a list of plotly colors
    plotly_colors = plotly.colors.qualitative.Plotly

    # Ensure we have enough colors for the labels
    if len(labels) > len(plotly_colors):
        raise ValueError("Not enough distinct colors available for the given labels")

    # Create a dictionary mapping each label to a distinct color
    color_dict = {label: plotly_colors[i % len(plotly_colors)] for i, label in enumerate(labels)}

    return color_dict
