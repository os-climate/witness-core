'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2025 Capgemini

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
import logging
from copy import deepcopy

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.core.core_sectorization.agriculture_model import AgricultureModel
from climateeconomics.glossarycore import GlossaryCore


class AgricultureSectorDiscipline(AutodifferentiedDisc):
    """Agriculture sector for witness sectorized version"""
    _ontology_data = {
        'label': 'Agriculture sector model for WITNESS Sectorized version',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'compute food production',
        'icon': "fa-solid fa-tractor",
        'version': '',
    }

    sector_name = AgricultureModel.name

    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'default': GlossaryCore.YearStartDefault, 'structuring': True,'unit': GlossaryCore.Years, 'visibility': 'Shared', 'namespace': 'ns_public', 'range': [1950, 2080]},
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        "mdo_sectors_invest_level": GlossaryCore.MDOSectorsLevel,

        # Emissions inputs
        GlossaryCore.FoodEmissionsName: GlossaryCore.FoodEmissionsVar,
        GlossaryCore.CropEnergyEmissionsName: GlossaryCore.CropEnergyEmissionsVar,

        f'{GlossaryCore.Forestry}.CO2_land_emission_df': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': "Shared", 'namespace': GlossaryCore.NS_AGRI, AutodifferentiedDisc.GRADIENTS: True},

    }

    for sub_sector in AgricultureModel.sub_sectors:
        for commun_variable_name, commun_variable_descr, _, _ in AgricultureModel.sub_sector_commun_variables:
            DESC_IN.update({
                f"{sub_sector}.{commun_variable_name}": GlossaryCore.get_subsector_variable(
                    subsector_name=sub_sector, sector_namespace=GlossaryCore.NS_AGRI, var_descr=commun_variable_descr),
            })

    DESC_OUT = {
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2): {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               GlossaryCore.Forestry: ('float', None, False),
                                                                               'Crop': ('float', None, False)}},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4): {'type': 'dataframe', 'unit': 'GtCH4', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               'Crop': ('float', None, False), }},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O): {'type': 'dataframe', 'unit': 'GtN2O', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               },}
    }
    for commun_variable_name, commun_variable_descr, _ , _ in AgricultureModel.sub_sector_commun_variables:
        var_descr = deepcopy(commun_variable_descr)
        var_descr["namespace"] = GlossaryCore.NS_SECTORS
        DESC_OUT.update({f"{sector_name}.{commun_variable_name}": var_descr})

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.model = AgricultureModel()

    def add_additionnal_dynamic_variables(self):
        return {}, {}
    def setup_sos_disciplines(self):  # type: (...) -> None
        dynamic_inputs = {}
        dynamic_outputs = {}

        damage_detailed = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf)
        damage_detailed.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}"] = damage_detailed

        if "mdo_sectors_invest_level" in self.get_data_in():
            mdo_sectors_invest_level = self.get_sosdisc_inputs("mdo_sectors_invest_level")
            if mdo_sectors_invest_level is not None:
                if mdo_sectors_invest_level == 0:
                    # then we compute invest in subs sector as sub-sector-invest = Net outpput * Share invest sector * Share invest sub sector * Shares invests inside Sub sector
                    # and we go bottom-up until Macroeconomics
                    for sub_sector in AgricultureModel.sub_sectors:
                        dynamic_inputs[f'{self.sector_name}.{sub_sector}.{GlossaryCore.InvestmentDfValue}'] =\
                            GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                    dynamic_outputs[f'{self.sector_name}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                elif mdo_sectors_invest_level == 2:
                    # then invests in subsectors are in G$
                    dynamic_inputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}'] = GlossaryCore.get_dynamic_variable(
                        GlossaryCore.SubSectorInvestDf)
                else:
                    raise NotImplementedError('')
        di, do = self.add_additionnal_dynamic_variables()
        di.update(dynamic_inputs)
        do.update(dynamic_outputs)
        self.add_inputs(di)
        self.add_outputs(do)

    def get_chart_filter_list(self):
        chart_list = ['Emissions',
                      'Economical output',
                      'Capital',
                      "Economical damages",
                      "Biomass dry price",
                      'Biomass dry production']
        return [ChartFilter("Charts", chart_list, chart_list, "charts"),]

    def get_post_processing_list(self, filters=None):
        instanciated_charts = []
        charts = []
        selected_food_types = []
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values
        food_emissions_df = self.get_sosdisc_inputs(GlossaryCore.FoodEmissionsName)
        years = food_emissions_df[GlossaryCore.Years]
        if "Emissions" in charts:
            agri_emissions = self.get_sosdisc_outputs(GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2))
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.FoodEmissionsVar["unit"],
                                                 stacked_bar=True, chart_name="CO2 Emissions of Agriculture sector",
                                                 y_min_zero=True)
            new_series = InstanciatedSeries(years, agri_emissions[GlossaryCore.Forestry], "Forestry", 'bar', True)
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(years, agri_emissions[GlossaryCore.Crop], "Crop", 'bar', True)
            new_chart.add_series(new_series)


            new_chart.post_processing_section_name = "GHG Emissions"
            new_chart.annotation_upper_left = {"Note": "does not include Crop for energy emissions (they are associated to energy sector emissions)"}
            instanciated_charts.append(new_chart)

        if "Emissions" in charts:
            for ghg in [GlossaryCore.CH4, GlossaryCore.N2O]:
                # CH4 chart:
                agri_emissions = self.get_sosdisc_outputs(
                    GlossaryCore.insertGHGAgriLandEmissions.format(ghg))
                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.FoodEmissionsVar["unit"],
                                                     stacked_bar=True, chart_name=f"{ghg} Emissions of Agriculture sector",
                                                     y_min_zero=True)

                for subsector in AgricultureModel.sub_sectors:
                    if subsector in agri_emissions:
                        new_series = InstanciatedSeries(years, agri_emissions[subsector], subsector.capitalize(), 'bar', True)
                        new_chart.add_series(new_series)

                new_chart.post_processing_section_name = "GHG Emissions"
                new_chart.annotation_upper_left = {
                    "Note": "does not include Crop for energy emissions (they are associated to energy sector emissions)"}
                instanciated_charts.append(new_chart)

        if "Economical output" in charts:
            # Output net of damages
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorProductionDf["unit"],
                                                 stacked_bar=True, chart_name="Output net of damage for Agriculture sector",)
            for subsector in AgricultureModel.sub_sectors:
                conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.SubsectorProductionDf["unit"]][GlossaryCore.SectorProductionDf["unit"]]
                subsector_data = self.get_sosdisc_inputs(f"{subsector}.{GlossaryCore.ProductionDfValue}")[GlossaryCore.OutputNetOfDamage] * conversion_factor
                new_series = InstanciatedSeries(years, subsector_data, subsector, 'bar', True)
                new_chart.add_series(new_series)

            sector_data = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.ProductionDfValue}")[GlossaryCore.OutputNetOfDamage]

            new_series = InstanciatedSeries(years, sector_data, "Total", 'lines', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Economical data"
            instanciated_charts.append(new_chart)

        if "Economical damages" in charts:
            # Damages
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorProductionDf["unit"],
                                                 stacked_bar=True, chart_name="Economical damages for Agriculture sector",)
            for subsector in AgricultureModel.sub_sectors:
                conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.SubsectorDamagesDf["unit"]][GlossaryCore.DamageDf["unit"]]
                subsector_data = self.get_sosdisc_inputs(f"{subsector}.{GlossaryCore.DamageDfValue}")[GlossaryCore.Damages] * conversion_factor
                new_series = InstanciatedSeries(years, subsector_data, subsector, 'bar', True)
                new_chart.add_series(new_series)

            sector_data = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.DamageDfValue}")[GlossaryCore.Damages]

            new_series = InstanciatedSeries(years, sector_data, "Total", 'lines', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Economical data"
            instanciated_charts.append(new_chart)

        if "Biomass dry production" in charts:
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.FoodEmissionsVar["unit"],
                                                 stacked_bar=True,
                                                 chart_name="Biomass dry production of Agriculture sector")
            for subsector in AgricultureModel.sub_sectors:
                conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.ProdForStreamVar["unit"]][GlossaryCore.ProdForStreamVar["unit"]]
                subsector_data = self.get_sosdisc_inputs(f"{subsector}.{GlossaryCore.ProdForStreamName.format('biomass_dry')}")["Total"] * conversion_factor
                new_series = InstanciatedSeries(years, subsector_data, subsector, 'bar', True)
                new_chart.add_series(new_series)

            sector_data = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.ProdForStreamName.format('biomass_dry')}")["Total"]

            new_series = InstanciatedSeries(years, sector_data, "Total", 'lines', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Biomass dry"
            instanciated_charts.append(new_chart)

        if "Biomass dry price" in charts:
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.PriceDf["unit"],
                                                 stacked_bar=True,
                                                 chart_name="Biomass dry price", y_min_zero=True)
            for subsector in AgricultureModel.sub_sectors:
                subsector_data = self.get_sosdisc_inputs(f"{subsector}.biomass_dry_price")[subsector]
                new_series = InstanciatedSeries(years, subsector_data, subsector, 'lines', True)
                new_chart.add_series(new_series)

            sector_data = self.get_sosdisc_outputs(f"{self.sector_name}.biomass_dry_price")[GlossaryCore.biomass_dry]

            new_series = InstanciatedSeries(years, sector_data, "Total", 'lines', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Biomass dry"
            instanciated_charts.append(new_chart)

        if "Capital" in charts:
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorCapitalDf["unit"],
                                                 stacked_bar=True, chart_name="Capital stock of Agriculture sector")
            for subsector in AgricultureModel.sub_sectors:
                conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.SubsectorCapitalDf["unit"]][GlossaryCore.SectorCapitalDf["unit"]]
                subsector_data = self.get_sosdisc_inputs(f"{subsector}.{GlossaryCore.CapitalDfValue}")[GlossaryCore.Capital] * conversion_factor
                new_series = InstanciatedSeries(years, subsector_data, subsector, 'bar', True)
                new_chart.add_series(new_series)

            sector_data = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.CapitalDfValue}")[GlossaryCore.Capital]

            new_series = InstanciatedSeries(years, sector_data, "Total", 'lines', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)


        return instanciated_charts


