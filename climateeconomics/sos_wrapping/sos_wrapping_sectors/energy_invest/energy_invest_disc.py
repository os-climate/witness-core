'''
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
'''
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.energy_invest.energy_invest_model import (
    EnergyInvestModel,
)


class EnergyInvestDiscipline(ClimateEcoDiscipline):
    "UtilityModel discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Energy invest WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-child fa-fw',
        'version': '',
    }
    DESC_IN = {
        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
        GlossaryCore.CO2TaxEfficiencyValue: GlossaryCore.CO2TaxEfficiency,
        GlossaryCore.CO2TaxesValue: GlossaryCore.CO2Taxes,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }
    DESC_OUT = {
        GlossaryCore.RenewablesEnergyInvestmentsValue: GlossaryCore.RenewablesEnergyInvestments,
        GlossaryCore.EnergyInvestmentsValue: GlossaryCore.EnergyInvestments,
    }

    def run(self):
        """run"""
        inputs = self.get_sosdisc_inputs()
        
        self.model = EnergyInvestModel()

        self.model.compute(inputs)

        dict_values = {
            GlossaryCore.RenewablesEnergyInvestmentsValue: self.model.added_renewables_investments,
            GlossaryCore.EnergyInvestmentsValue: self.model.energy_investments,
        }
        
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """sos jacobian"""
        pass

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.EnergyInvestmentsValue]
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        raw_invests = self.get_sosdisc_inputs(GlossaryCore.EnergyInvestmentsWoTaxValue)[GlossaryCore.EnergyInvestmentsWoTaxValue].values * 1000
        added_invests_renawables = self.get_sosdisc_outputs(GlossaryCore.RenewablesEnergyInvestmentsValue)[GlossaryCore.InvestmentsValue].values * 100
        total_energy_invests = self.get_sosdisc_outputs(GlossaryCore.EnergyInvestmentsValue)[GlossaryCore.EnergyInvestmentsValue].values * 100
        years = list(self.get_sosdisc_inputs(GlossaryCore.EnergyInvestmentsWoTaxValue)[GlossaryCore.Years].values)

        if GlossaryCore.EnergyInvestmentsValue in chart_list:

            chart_name = 'Breakdown of energy investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Investment [G$]',
                                                 chart_name=chart_name, stacked_bar=True)

            new_series = InstanciatedSeries(
                years, list(added_invests_renawables),
                'Invest in renewables from CO2 tax', InstanciatedSeries.BAR_DISPLAY, True)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, list(raw_invests),
                'Raw investments in energy', InstanciatedSeries.BAR_DISPLAY, True)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(years, list(total_energy_invests),
                                            'Total energy investments', 'lines', True)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyInvestmentsValue in chart_list:

            chart_name = 'Composition of energy investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '%',
                                                 chart_name=chart_name, stacked_bar=True)

            added_invests_share = added_invests_renawables / total_energy_invests * 100.
            new_series = InstanciatedSeries(years, list(added_invests_share),
                'Invest in renewables from CO2 tax', InstanciatedSeries.BAR_DISPLAY, True)

            new_chart.series.append(new_series)

            raw_invests_share = raw_invests / total_energy_invests * 100.
            new_series = InstanciatedSeries(years, list(raw_invests_share),
                                            'Raw investments in energy', InstanciatedSeries.BAR_DISPLAY, True)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
