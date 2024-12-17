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
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_sectorization.agriculture_economy_model import (
    AgricultureEconomyModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop_2.crop_disc_2 import (
    CropDiscipline,
)


class AgricultureEconomyDiscipline(ClimateEcoDiscipline):
    """Crop discipline for computing food production"""
    _ontology_data = {
        'label': 'Agriculture economy model',
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

    invest_df = copy.deepcopy(GlossaryCore.InvestDf)
    invest_df['namespace'] = GlossaryCore.NS_CROP

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        GlossaryCore.FoodTypesName: GlossaryCore.FoodTypesVar,
        GlossaryCore.EnergyMeanPriceValue: GlossaryCore.EnergyMeanPrice,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.CropFoodNetGdpName: GlossaryCore.CropFoodGdpVar,
        GlossaryCore.CropEnergyNetGdpName: GlossaryCore.CropEnergyGdpVar,
        f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}": GlossaryCore.get_dynamic_variable(GlossaryCore.ProductionDf),
    }

    df_sectors_dict = {
        f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}": GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDf),
        f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDetailedDfValue}": GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf),
    }
    for df_name, df_var_descr in df_sectors_dict.items():
        df_sectors = copy.deepcopy(df_var_descr)
        df_sectors["namespace"] = GlossaryCore.NS_SECTORS
        DESC_OUT.update({df_name: df_sectors})

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.food_types_colors = {}
        self.model = AgricultureEconomyModel()

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.FoodTypesName in self.get_data_in():
            year_start, year_end, food_types = self.get_sosdisc_inputs([GlossaryCore.YearStart, GlossaryCore.YearEnd, GlossaryCore.FoodTypesName])
            if year_start is not None and year_end is not None and food_types is not None:
                dataframes_descriptors = {
                    GlossaryCore.Years: ("int", [year_start, year_end], False),
                    **{ft: ("float", [0., 1e30], False) for ft in food_types}
                }
                # inputs

                dynamic_inputs = {
                    # data used for price computation
                    GlossaryCore.FoodTypeEnergyIntensityByProdUnitName: GlossaryCore.FoodTypeEnergyIntensityByProdUnitVar,
                    GlossaryCore.FoodTypeLaborCostByProdUnitName: GlossaryCore.FoodTypeLaborCostByProdUnitVar,
                    GlossaryCore.FoodTypeCapitalMaintenanceCostName: GlossaryCore.FoodTypeCapitalMaintenanceCostVar,
                    GlossaryCore.FoodTypeCapitalAmortizationCostName: GlossaryCore.FoodTypeCapitalAmortizationCostVar,
                    GlossaryCore.FoodTypesPriceMarginShareName: GlossaryCore.FoodTypesPriceMarginShareVar,
                    GlossaryCore.FoodTypeFeedingCostsName: GlossaryCore.FoodTypeFeedingCostsVar,
                    GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: GlossaryCore.FoodTypeFertilizationAndPesticidesCostsVar,
                }
                dataframes_inputs = {
                    # economic data
                    GlossaryCore.FoodTypeCapitalName: GlossaryCore.FoodTypeCapitalVar,
                    GlossaryCore.FoodTypesInvestName: GlossaryCore.FoodTypesInvestVar,

                    # damages
                    GlossaryCore.FoodTypeNotProducedDueToClimateChangeName: GlossaryCore.FoodTypeNotProducedDueToClimateChangeVar,
                    GlossaryCore.FoodTypeWasteByClimateDamagesName: GlossaryCore.FoodTypeWasteByClimateDamagesVar,

                    # data used for price computation
                    GlossaryCore.FoodTypeDeliveredToConsumersName: GlossaryCore.FoodTypeDeliveredToConsumersVar,
                    GlossaryCore.CropProdForAllStreamName: GlossaryCore.CropProdForAllStreamVar,
                }


                for varname, df_input in dataframes_inputs.items():
                    df_input["dataframe_descriptor"] = dataframes_descriptors
                    dynamic_inputs[varname] = df_input

                # outputs
                dataframes_outputs = {
                    # non coupling
                    GlossaryCore.Damages + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Applied damages for each food types",},
                    GlossaryCore.EstimatedDamages + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Estimated damages for each food types, not necessarily applied",},
                    GlossaryCore.GrossOutput + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Gross output for each food type",},
                    GlossaryCore.OutputNetOfDamage + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Output net of damage for each food type",},
                    GlossaryCore.CropFoodNetGdpName + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Output net of damage for each food type (from food production)", },
                    GlossaryCore.CropEnergyNetGdpName + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Output net of damage for each food type (from food production)", },
                    GlossaryCore.DamagesFromClimate + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Damages due to extreme climate events breakdown", },
                    GlossaryCore.DamagesFromProductivityLoss + "_breakdown" : {"type": "dataframe", "unit": "T$", "description": "Damages due to productivity loss breakdown", },
                    GlossaryCore.FoodTypesPriceName:GlossaryCore.FoodTypesPriceVar,
                }

                for varname, df_output in dataframes_outputs.items():
                    df_output["dataframe_descriptor"] = dataframes_descriptors
                    dynamic_outputs[varname] = df_output

                for ft in food_types:
                    dynamic_outputs[f"{ft}_price_breakdown"] = {"type": "dataframe", "unit": "$/kg", "description": f"details of price composition for the food type {ft}"}

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        # -- get inputs
        input_dict = self.get_sosdisc_inputs()
        self.model.compute(input_dict)

        self.store_sos_outputs_values(self.model.outputs)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        gradients = self.model.jacobians()

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
            "Capital",
            "Investments",
            "Prices",
            "Output",
            "Damages",
            "Food products costs data"
            # Gdp net of damage
            # Capital
            # Invests
            # Waste in dollars
        ]
        food_types_list = self.get_sosdisc_inputs(GlossaryCore.FoodTypesName)
        selected_food_types = food_types_list[:3]
        return [
            ChartFilter("Charts", chart_list, chart_list, "charts"),
            ChartFilter("Food types", filter_values=food_types_list, selected_values=selected_food_types, filter_key="food_types_selected"),
        ]

    def get_post_processing_list(self, filters=None):
        instanciated_charts = []
        charts = []
        selected_food_types = []
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values
                if chart_filter.filter_key == 'food_types_selected':
                    selected_food_types = chart_filter.selected_values

        inputs = self.get_sosdisc_inputs()
        outputs = self.get_sosdisc_outputs()
        self.food_types_colors = CropDiscipline.food_types_colors
        if "Output" in charts:
            new_chart = self.chart_gross_and_net_output(outputs)
            instanciated_charts.append(new_chart)

            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.OutputNetOfDamage + "_breakdown"],
                charts_name="Agriculture sector net output breakdown",
                unit=GlossaryCore.ProductionDf['unit'],
                df_total=outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}"],
                column_total=GlossaryCore.OutputNetOfDamage,
                post_proc_category="Output"
            )
            instanciated_charts.append(new_chart)

            new_chart = self.chart_net_gdp_food_energy_breakdown(outputs)
            instanciated_charts.append(new_chart)

        if "Capital" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=inputs[GlossaryCore.FoodTypeCapitalName],
                charts_name="Capital",
                unit=GlossaryCore.FoodTypeCapitalVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Capital & Investments"
            )
            instanciated_charts.append(new_chart)

        if "Food products costs data" in charts:
            plots = {
                "Labor": (GlossaryCore.FoodTypeLaborCostByProdUnitName, GlossaryCore.FoodTypeLaborCostByProdUnitVar),
                "Energy intensity": (GlossaryCore.FoodTypeEnergyIntensityByProdUnitName, GlossaryCore.FoodTypeEnergyIntensityByProdUnitVar),
                "Fertilization and pesticides": (GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName, GlossaryCore.FoodTypeFertilizationAndPesticidesCostsVar),
                "Feeding": (GlossaryCore.FoodTypeFeedingCostsName, GlossaryCore.FoodTypeFeedingCostsVar),
                "Capital maintenance": (GlossaryCore.FoodTypeCapitalMaintenanceCostName, GlossaryCore.FoodTypeCapitalMaintenanceCostVar),
                "Capital amortization": (GlossaryCore.FoodTypeCapitalAmortizationCostName, GlossaryCore.FoodTypeCapitalAmortizationCostVar),
            }
            for chart_name, (inputname, input_var_descr) in plots.items():
                dict_values = dict(sorted(inputs[inputname].items(), key=lambda item: item[1], reverse=True))
                new_chart = self.get_dict_bar_plot(
                        dict_values=dict_values,
                        charts_name=chart_name,
                        unit=input_var_descr['unit'],
                        post_proc_category="Food products data",
                    )
                instanciated_charts.append(new_chart)

        if "Investments" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=inputs[GlossaryCore.FoodTypesInvestName],
                charts_name="Investments",
                unit=GlossaryCore.FoodTypeCapitalVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Capital & Investments"
            )
            food_types = inputs[GlossaryCore.FoodTypesName]
            years = inputs[GlossaryCore.FoodTypesInvestName][GlossaryCore.Years]
            total_invests = inputs[GlossaryCore.FoodTypesInvestName][food_types].sum(axis=1)
            new_chart.add_series(InstanciatedSeries(years, total_invests, 'Total', 'lines', True, line={'color': 'gray'}))
            instanciated_charts.append(new_chart)

        if "Damages" in charts:
            new_chart = self.get_chart_damages(outputs)
            instanciated_charts.append(new_chart)

        if "Prices" in charts:
            new_chart = self.get_breakdown_charts_on_food_type(
                df_all_food_types=outputs[GlossaryCore.FoodTypesPriceName],
                charts_name="Prices",
                unit=GlossaryCore.FoodTypesPriceVar['unit'],
                df_total=None,
                column_total=None,
                post_proc_category="Prices",
                lines=True
            )
            instanciated_charts.append(new_chart)
            for food_type in selected_food_types:
                new_chart = self.get_breakdown_charts_on_food_type(
                    df_all_food_types=outputs[f"{food_type}_price_breakdown"],
                    charts_name=f"{food_type.capitalize()} price details",
                    unit=GlossaryCore.FoodTypesPriceVar['unit'],
                    df_total=outputs[GlossaryCore.FoodTypesPriceName],
                    column_total=food_type,
                    post_proc_category="Prices",
                    note={"Price": "Price of selling to distributors."}
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
            values = df_all_food_types[col].values
            if not min(values) == max(values) == 0:
                new_series = InstanciatedSeries(years, values, str(col).capitalize(), 'bar' if not lines else "lines", True, **kwargs)
                new_chart.add_series(new_series)

        if df_total is not None and column_total is not None:
            new_series = InstanciatedSeries(years, df_total[column_total], 'Total', 'lines', True, line={'color': 'gray'})
            new_chart.add_series(new_series)

        if post_proc_category is not None:
            new_chart.post_processing_section_name = post_proc_category

        if note is not None:
            new_chart.annotation_upper_left = note
        return new_chart

    def chart_net_gdp_food_energy_breakdown(self, outputs):
        variables = {
            "Energy": GlossaryCore.CropEnergyNetGdpName,
            "Food": GlossaryCore.CropFoodNetGdpName
        }
        total_df = outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}"]
        years = total_df[GlossaryCore.Years]
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ProductionDf["unit"], stacked_bar=True, chart_name="Net output breakdown (food & energy)", y_min_zero=True)
        for key, varname  in variables.items():
            df = outputs[varname]
            new_series = InstanciatedSeries(years, df["Total"], key, 'bar', True)
            new_chart.add_series(new_series)

        new_series = InstanciatedSeries(years, total_df[GlossaryCore.OutputNetOfDamage], "Agriculture sector output net of damages", 'lines', True)
        new_chart.add_series(new_series)
        new_chart.post_processing_section_name = "Output"

        return new_chart

    def get_chart_damages(self, outputs):
        damage_detailed_df = outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDetailedDfValue}"]
        damage_df = outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}"]
        years = damage_detailed_df[GlossaryCore.Years]
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.DamageDetailedDf["unit"], stacked_bar=True, chart_name="Damages", y_min_zero=True)
        variables_bar_plot = {
            "Extreme climate events": GlossaryCore.DamagesFromClimate,
            "Productivity loss": GlossaryCore.DamagesFromProductivityLoss,
        }
        for key, colname  in variables_bar_plot.items():
            new_series = InstanciatedSeries(years, damage_detailed_df[colname], key, 'bar', True)
            new_chart.add_series(new_series)

        new_series = InstanciatedSeries(years, damage_df[GlossaryCore.Damages], "Damages", 'lines', True)
        new_chart.add_series(new_series)
        new_chart.post_processing_section_name = "Output"
        return new_chart

    def chart_gross_and_net_output(self, outputs):
        production_df = outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}"]
        damage_df = outputs[f"{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}"]
        years = production_df[GlossaryCore.Years]
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.DamageDetailedDf["unit"], stacked_bar=True, chart_name="Gross and net output of Agriculture sector", y_min_zero=True)
        variables_bar_plot = {
            "Gross output": GlossaryCore.GrossOutput,
            "Output net of damages": GlossaryCore.OutputNetOfDamage,
        }
        for key, colname  in variables_bar_plot.items():
            new_series = InstanciatedSeries(years, production_df[colname], key, 'lines', True)
            new_chart.add_series(new_series)

        new_series = InstanciatedSeries(years, -damage_df[GlossaryCore.Damages], "Damages", 'bar', True)
        new_chart.add_series(new_series)
        new_chart.post_processing_section_name = "Output"
        new_chart.annotation_upper_left = {"Note": "does not include Forestry activities output."}
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
                if value != 0.0:
                    new_series = InstanciatedSeries([str(key).capitalize()], [value], '', 'bar', True, marker=dict_color)
                    new_chart.add_series(new_series)

        if post_proc_category is not None:
            new_chart.post_processing_section_name = post_proc_category

        if note is not None:
            new_chart.annotation_upper_left = note
        return new_chart
