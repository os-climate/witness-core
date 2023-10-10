import numpy as np
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution.sectors_redistribution_model import \
    SectorRedistributionModel
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, \
    InstanciatedSeries


class SectorsRedistributionDiscipline(SoSWrapp):
    """Discipline redistributing energy production and global investment into sectors"""

    DESC_IN = {
        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProductionDf,
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
    }

    DESC_OUT = {
        GlossaryCore.RedistributionEnergyProductionDfValue: GlossaryCore.RedistributionEnergyProductionDf,
        GlossaryCore.InvestmentDfValue: GlossaryCore.InvestmentDf,
    }

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                dynamic_inputs[f'{sector}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                dynamic_inputs[f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.ShareSectorEnergyDf)

                dynamic_outputs[f'{sector}.{GlossaryCore.EnergyProductionValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyProductionDf)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        """run method"""
        inputs = self.get_sosdisc_inputs()

        model = SectorRedistributionModel()

        sectors_energy, all_sectors_energy_df, total_invests_df = model.compute(inputs)

        sector_list = inputs[GlossaryCore.SectorListValue]

        outputs = {
            GlossaryCore.RedistributionEnergyProductionDfValue: all_sectors_energy_df,
            GlossaryCore.InvestmentDfValue: total_invests_df,
        }

        for sector in sector_list:
            outputs[f'{sector}.{GlossaryCore.EnergyProductionValue}'] = sectors_energy[sector]

        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()
        outputs = self.get_sosdisc_outputs()

        sectors_list = inputs[GlossaryCore.SectorListValue]
        total_energy_production = inputs[GlossaryCore.EnergyProductionValue][GlossaryCore.TotalProductionValue].values
        total_invests = outputs[GlossaryCore.InvestmentDfValue][GlossaryCore.InvestmentsValue].values
        identity = np.eye(len(total_invests))

        for sector in sectors_list:
            sector_share_energy = inputs[f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'][GlossaryCore.ShareSectorEnergy].values
            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
                (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
                np.diag(sector_share_energy/ 100.)
            )

            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
                (f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}', GlossaryCore.ShareSectorEnergy),
                np.diag(total_energy_production / 100.)
            )
            
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.InvestmentDfValue, GlossaryCore.InvestmentsValue),
                (f'{sector}.{GlossaryCore.InvestmentDfValue}', GlossaryCore.InvestmentsValue),
                identity
            )

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = [GlossaryCore.RedistributionEnergyProductionDfValue,
                      GlossaryCore.ShareSectorEnergyDfValue,]

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):
        all_filters = True
        charts = []

        if filters is not None:
            charts = filters

        instanciated_charts = []
        if all_filters or GlossaryCore.RedistributionEnergyProductionDf:
            # first graph
            redistribution_energy_production_df = self.get_sosdisc_outputs(GlossaryCore.RedistributionEnergyProductionDfValue)
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            chart_name = f"Energy allocated to sectors [{GlossaryCore.RedistributionEnergyProductionDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.RedistributionEnergyProductionDf['unit'],
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            years = list(redistribution_energy_production_df[GlossaryCore.Years])
            for sector in sector_list:
                new_series = InstanciatedSeries(years,
                                                list(redistribution_energy_production_df[sector]),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # second graph
            chart_name = f"Share of total energy production allocated to sectors [%]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 '%',
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                share_sector = self.get_sosdisc_inputs(f"{sector}.{GlossaryCore.ShareSectorEnergyDfValue}")[GlossaryCore.ShareSectorEnergy]
                new_series = InstanciatedSeries(years,
                                                list(share_sector),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if all_filters or GlossaryCore.InvestmentsValue:
            # first graph
            total_investments_df = self.get_sosdisc_outputs(
                GlossaryCore.InvestmentDfValue)
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            chart_name = f"Investments breakdown by sectors [{GlossaryCore.InvestmentDf['unit']}]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 GlossaryCore.InvestmentDf['unit'],
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            years = list(total_investments_df[GlossaryCore.Years])
            for sector in sector_list:
                sector_invest = self.get_sosdisc_inputs(f"{sector}.{GlossaryCore.InvestmentDfValue}")[GlossaryCore.InvestmentsValue].values
                new_series = InstanciatedSeries(years,
                                                list(sector_invest),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # second graph
            chart_name = f"Share of total investments production allocated to sectors [%]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 '%',
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                sector_invest = self.get_sosdisc_inputs(f"{sector}.{GlossaryCore.InvestmentDfValue}")[
                    GlossaryCore.InvestmentsValue].values
                share_sector = sector_invest / total_investments_df[GlossaryCore.InvestmentsValue].values
                new_series = InstanciatedSeries(years,
                                                list(share_sector),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts

