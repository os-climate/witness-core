"""
Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.demand.demand_model import (
    DemandModel,
)


class DemandDiscipline(SoSWrapp):
    """Discipline demand"""

    # ontology information
    _ontology_data = {
        "label": "Demand WITNESS Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fa-solid fa-chart-pie",
        "version": "",
    }
    _maturity = "Research"

    DESC_IN = {
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
    }

    DESC_OUT = {GlossaryCore.AllSectorsDemandDfValue: GlossaryCore.AllSectorsDemandDf}

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                dynamic_inputs[f"{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}"] = (
                    GlossaryCore.get_dynamic_variable(GlossaryCore.SectorDemandPerCapitaDf)
                )
                dynamic_outputs[f"{sector}.{GlossaryCore.SectorGDPDemandDfValue}"] = GlossaryCore.get_dynamic_variable(
                    GlossaryCore.SectorGDPDemandDf
                )

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        """run method"""
        inputs = self.get_sosdisc_inputs()

        model = DemandModel()

        sectors_demand, all_sectors_demand_df = model.compute(inputs)

        sector_list = inputs[GlossaryCore.SectorListValue]

        outputs = {
            GlossaryCore.AllSectorsDemandDfValue: all_sectors_demand_df,
        }

        for sector in sector_list:
            outputs[f"{sector}.{GlossaryCore.SectorGDPDemandDfValue}"] = sectors_demand[sector]

        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()

        sectors_list = inputs[GlossaryCore.SectorListValue]
        population = inputs[GlossaryCore.PopulationDfValue][GlossaryCore.PopulationValue].values

        for sector in sectors_list:
            sector_demand_per_capita = inputs[f"{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}"][
                GlossaryCore.SectorDemandPerCapitaDfValue
            ].values
            self.set_partial_derivative_for_other_types(
                (f"{sector}.{GlossaryCore.SectorGDPDemandDfValue}", GlossaryCore.SectorGDPDemandDfValue),
                (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
                np.diag(sector_demand_per_capita * 1e-6),
            )

            self.set_partial_derivative_for_other_types(
                (f"{sector}.{GlossaryCore.SectorGDPDemandDfValue}", GlossaryCore.SectorGDPDemandDfValue),
                (f"{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}", GlossaryCore.SectorDemandPerCapitaDfValue),
                np.diag(population) * 1e-6,
            )

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = [
            GlossaryCore.SectorGDPDemandDf,
            GlossaryCore.SectorDemandPerCapitaDfValue,
        ]

        chart_filters.append(ChartFilter("Charts filter", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        all_filters = True
        charts = []

        if filters is not None:
            charts = filters

        instanciated_charts = []

        if all_filters or GlossaryCore.SectorGDPDemandDf:
            # first graph
            all_sectors_demand_df = self.get_sosdisc_outputs(GlossaryCore.AllSectorsDemandDfValue)
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            chart_name = f"Demand by sectors [{GlossaryCore.AllSectorsDemandDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, GlossaryCore.AllSectorsDemandDf["unit"], stacked_bar=True, chart_name=chart_name
            )

            years = list(all_sectors_demand_df[GlossaryCore.Years])

            total_demand = []
            for sector in sector_list:
                demand_sector_per_capita = all_sectors_demand_df[sector].values
                total_demand.append(demand_sector_per_capita)
                new_series = InstanciatedSeries(years, list(demand_sector_per_capita), sector, "bar", True)
                new_chart.series.append(new_series)
            total_demand = np.sum(total_demand, axis=0)
            new_series = InstanciatedSeries(years, list(total_demand), "Total", "lines", True)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

            # second graph
            chart_name = f"Share of demand by sector [%]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "%", stacked_bar=True, chart_name=chart_name)

            for sector in sector_list:
                demand_sector_per_capita = all_sectors_demand_df[sector].values
                share_sector = demand_sector_per_capita / total_demand * 100.0
                new_series = InstanciatedSeries(years, list(share_sector), sector, "bar", True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if all_filters or GlossaryCore.SectorGDPDemandDf:
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            chart_name = f"Demand per person by sectors [{GlossaryCore.SectorDemandPerCapitaDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                GlossaryCore.SectorDemandPerCapitaDf["unit"],
                stacked_bar=False,
                chart_name=chart_name,
            )

            for sector in sector_list:
                demand_sector_per_capita_df = self.get_sosdisc_inputs(
                    f"{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}"
                )
                years = list(demand_sector_per_capita_df[GlossaryCore.Years].values)
                demand_sector_per_capita = demand_sector_per_capita_df[GlossaryCore.SectorDemandPerCapitaDfValue].values
                new_series = InstanciatedSeries(years, list(demand_sector_per_capita), sector, "lines", True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        return instanciated_charts
