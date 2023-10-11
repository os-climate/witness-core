import numpy as np
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_energy.sectors_redistribution_energy_model import \
    SectorRedistributionEnergyModel
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, \
    InstanciatedSeries


class SectorsRedistributionEnergyDiscipline(SoSWrapp):
    """Discipline redistributing energy production and global investment into sectors"""

    DESC_IN = {
        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProductionDf,
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.MissingSectorNameValue: GlossaryCore.MissingSectorName,
    }

    DESC_OUT = {
        GlossaryCore.RedistributionEnergyProductionDfValue: GlossaryCore.RedistributionEnergyProductionDf,
    }

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            deduced_sector = self.get_sosdisc_inputs(GlossaryCore.MissingSectorNameValue)

            # share percentage for last sector is determined as 100 % - share other sector
            for sector in sector_list:
                if sector != deduced_sector:
                    dynamic_inputs[f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.ShareSectorEnergyDf)

            for sector in sector_list:
                dynamic_outputs[f'{sector}.{GlossaryCore.EnergyProductionValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyProductionDf)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        """run method"""
        inputs = self.get_sosdisc_inputs()

        model = SectorRedistributionEnergyModel()

        sectors_energy, all_sectors_energy_df= model.compute(inputs)

        sector_list = inputs[GlossaryCore.SectorListValue]

        outputs = {
            GlossaryCore.RedistributionEnergyProductionDfValue: all_sectors_energy_df,
        }

        for sector in sector_list:
            outputs[f'{sector}.{GlossaryCore.EnergyProductionValue}'] = sectors_energy[sector]

        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """compute gradients"""
        inputs = self.get_sosdisc_inputs()
        sectors_list = inputs[GlossaryCore.SectorListValue]
        deduced_sector = inputs[GlossaryCore.MissingSectorNameValue]

        computed_sectors = list(filter(lambda x: x != deduced_sector, sectors_list))

        total_energy_production = inputs[GlossaryCore.EnergyProductionValue][GlossaryCore.TotalProductionValue].values

        sum_share_other_sectors = []
        for sector in computed_sectors:
            sector_share_energy = inputs[f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'][GlossaryCore.ShareSectorEnergy].values

            sum_share_other_sectors.append(sector_share_energy)
            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
                (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
                np.diag(sector_share_energy / 100.)
            )

            self.set_partial_derivative_for_other_types(
                (f'{sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
                (f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}', GlossaryCore.ShareSectorEnergy),
                np.diag(total_energy_production / 100.)
            )

            self.set_partial_derivative_for_other_types(
                (f'{deduced_sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
                (f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}', GlossaryCore.ShareSectorEnergy),
                np.diag(-total_energy_production / 100.)
            )
        sum_share_other_sectors = np.sum(sum_share_other_sectors, axis=0)

        self.set_partial_derivative_for_other_types(
            (f'{deduced_sector}.{GlossaryCore.EnergyProductionValue}', GlossaryCore.TotalProductionValue),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            np.diag(1 - sum_share_other_sectors / 100.)
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
            total_production_values = self.get_sosdisc_inputs(GlossaryCore.EnergyProductionValue)[GlossaryCore.TotalProductionValue].values
            redistribution_energy_production_df = self.get_sosdisc_outputs(GlossaryCore.RedistributionEnergyProductionDfValue)
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            chart_name = f"Energy allocated to sectors [TWh]"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[TWh]',
                                                 stacked_bar=True,
                                                 chart_name=chart_name)

            years = list(redistribution_energy_production_df[GlossaryCore.Years])
            for sector in sector_list:
                new_series = InstanciatedSeries(years,
                                                list(redistribution_energy_production_df[sector] * 1000),
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
                share_sector = redistribution_energy_production_df[sector].values / total_production_values * 100

                new_series = InstanciatedSeries(years,
                                                list(share_sector),
                                                sector, 'bar', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
