'''
Copyright 2024 Capgemini

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

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.consumption.consumption_model import (
    SectorizedConsumptionModel,
)


class ConsumptionDiscipline(SoSWrapp):
    """Discipline demand"""

    # ontology information
    _ontology_data = {
        "label": "Consumption WITNESS Model",
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
        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.AllSectorsShareEnergyDfValue: GlossaryCore.AllSectorsShareEnergyDf,
    }

    DESC_OUT = {
        GlossaryCore.SectorizedConsumptionDfValue: GlossaryCore.SectorizedConsumptionDf,
        "consumption_detail_df": {"type": "dataframe", "unit": "-"},
    }

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                dynamic_inputs[f"{sector}.{GlossaryCore.InvestmentDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                dynamic_inputs[f"{sector}.{GlossaryCore.ProductionDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.ProductionDf)
                dynamic_outputs[f"{sector}_consumption_breakdown"] = GlossaryCore.get_dynamic_variable(GlossaryCore.ConsumptionSectorBreakdown)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        """run method"""
        inputs = self.get_sosdisc_inputs()

        model = SectorizedConsumptionModel()

        model.compute(inputs)

        self.store_sos_outputs_values(model.output_dict)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()
        total_energy_invest = inputs[GlossaryCore.EnergyInvestmentsWoTaxValue][GlossaryCore.EnergyInvestmentsWoTaxValue].values
        share_energy_consumption_sector_df = inputs[GlossaryCore.AllSectorsShareEnergyDfValue]
        years = share_energy_consumption_sector_df[GlossaryCore.Years].values
        identity = np.eye(len(years))
        for sector in inputs[GlossaryCore.SectorListValue]:
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.SectorizedConsumptionDfValue, sector),
                (f"{sector}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
                identity)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.SectorizedConsumptionDfValue, sector),
                (f"{sector}.{GlossaryCore.InvestmentDfValue}", GlossaryCore.InvestmentsValue),
                -identity)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.SectorizedConsumptionDfValue, sector),
                (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue),
                np.diag(- share_energy_consumption_sector_df[sector].values / 100))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.SectorizedConsumptionDfValue, sector),
                (GlossaryCore.AllSectorsShareEnergyDfValue, sector),
                np.diag(- total_energy_invest / 100.))

    def get_chart_filter_list(self):
        chart_filters = []
        chart_list = ["Sectorized Consumption"]
        chart_filters.append(ChartFilter("Charts filter", chart_list, chart_list, "charts"))
        return chart_filters

    def get_post_processing_list(self, filters=None):
        charts = []

        if filters is not None:
            for filter in filters:
                charts.extend(filter.selected_values)

        instanciated_charts = []

        inputs = self.get_sosdisc_inputs()
        outputs = self.get_sosdisc_outputs()
        sectorized_consumption_df = outputs[GlossaryCore.SectorizedConsumptionDfValue]
        years = sectorized_consumption_df[GlossaryCore.Years]
        if "Sectorized Consumption" in charts:
            chart_name = 'Breakdown of consumption between sectors'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorizedConsumptionDf['unit'], chart_name=chart_name, stacked_bar=True)

            for sector in sectorized_consumption_df.columns:
                if sector != GlossaryCore.Years:
                    new_series = InstanciatedSeries(years, sectorized_consumption_df[sector], sector, 'bar', True)
                    new_chart.add_series(new_series)
            instanciated_charts.append(new_chart)

        if "Sectorized Consumption" in charts:
            for sector in inputs[GlossaryCore.SectorListValue]:
                chart_name = f'Sector {sector} consumption breakdown'
                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ConsumptionSectorBreakdown['unit'], chart_name=chart_name, stacked_bar=True)
                bar_col = [
                    "Output net of damage",
                    "Investment in sector",
                    "Attributed investment in energy",
                ]
                breakdown_df_sector = outputs[f"{sector}_consumption_breakdown"]
                for col in bar_col:
                    new_series = InstanciatedSeries(years, breakdown_df_sector[col], col, 'bar', True)
                    new_chart.add_series(new_series)

                new_series = InstanciatedSeries(years, breakdown_df_sector["Consumption"], "Consumption", 'lines', True)
                new_chart.add_series(new_series)
                instanciated_charts.append(new_chart)

        return instanciated_charts
