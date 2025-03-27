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
from copy import deepcopy
from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.charts_tools import graph_gross_and_net_output
from climateeconomics.core.core_sectorization.sector_model import SectorModel
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class SectorDiscipline(AutodifferentiedDisc):
    """Generic sector discipline"""
    prod_cap_unit = 'T$' # to overwrite if necessary
    NS_SECTORS = GlossaryCore.NS_SECTORS
    DESC_IN = {
        GlossaryCore.SectionListValue: GlossaryCore.SectionList,
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.EnergyMarketRatioAvailabilitiesValue: GlossaryCore.EnergyMarketRatioAvailabilities,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        'productivity_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'capital_start': {'type': 'float', 'unit': 'T$', 'user_level': 2},
        GlossaryCore.WorkforceDfValue: GlossaryCore.WorkforceDf,
        'productivity_gr_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'decline_rate_tfp': {'type': 'float', 'user_level': 3, 'unit': '-'},
        # Usable capital
        'capital_utilisation_ratio': {'type': 'float', 'default': 0.8, 'user_level': 3, 'unit': '-'},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.85, 'user_level': 3, 'unit': '-'},
        'energy_eff_k': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'energy_eff_cst': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'energy_eff_xzero': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'sector_name': {'type': 'string', 'user_level': 3, 'unit': '-', 'structuring': True},
        'energy_eff_max': {'type': 'float', 'user_level': 3, 'unit': '-'},
        # Production function param
        'output_alpha': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'depreciation_capital': {'type': 'float', 'user_level': 2, 'unit': '-'},
        GlossaryCore.DamageToProductivity: {'type': 'bool', 'default': True,
                                   'visibility': 'Shared',
                                   'unit': '-', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.FractionDamageToProductivityValue: GlossaryCore.FractionDamageToProductivity,
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS,
                  'user_level': 1, 'unit': '-'},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }
    DESC_OUT = {
        GlossaryCore.ProductivityDfValue: GlossaryCore.ProductivityDf,
        #'growth_rate_df': {'type': 'dataframe', 'unit': '-'},
    }


    def setup_sos_disciplines(self):
        """setup sos disciplines"""
        dynamic_outputs = {}
        dynamic_inputs = {}
        values_dict, go = self.collect_var_for_dynamic_setup(["sector_name", GlossaryCore.YearStart, GlossaryCore.YearEnd])
        if go:
            sector_name = values_dict["sector_name"]
            df_descriptor = {
                GlossaryCore.Years: ('int', [1900, 2100], True),
                **{section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector_name]}
            }
            section_gdp_percentage_var = deepcopy(GlossaryCore.SectionGdpPercentageDf)
            section_gdp_percentage_var["dataframe_descriptor"] = df_descriptor
            section_gdp_percentage_var.update({'namespace': GlossaryCore.NS_SECTORS})
            global_data_dir = join(dirname(dirname(dirname(__file__))), 'data')
            section_gdp_percentage_df_default = pd.read_csv(
                join(global_data_dir,
                     f'weighted_average_percentage_{sector_name.lower()}_sections.csv'))
            section_gdp_percentage_dict = {
                **{GlossaryCore.Years: np.arange(values_dict[GlossaryCore.YearStart], values_dict[GlossaryCore.YearEnd] + 1), },
                **dict(zip(section_gdp_percentage_df_default.columns[1:],
                           section_gdp_percentage_df_default.values[0, 1:]))
            }
            section_gdp_percentage_var['default'] = pd.DataFrame(section_gdp_percentage_dict)

            dynamic_inputs[f"{sector_name}.{GlossaryCore.SectionGdpPercentageDfValue}"] = section_gdp_percentage_var

            section_energy_consumption_percentage_var = deepcopy(GlossaryCore.SectionEnergyConsumptionPercentageDf)
            section_energy_consumption_percentage_var.update({'namespace': GlossaryCore.NS_SECTORS})
            section_energy_consumption_percentage_var["dataframe_descriptor"] = df_descriptor
            section_energy_consumption_percentage_df_default = pd.read_csv(
                join(global_data_dir,
                     f'energy_consumption_percentage_{sector_name.lower()}_sections.csv'))
            section_energy_consumption_percentage_dict = {
                **{GlossaryCore.Years: np.arange(values_dict[GlossaryCore.YearStart], values_dict[GlossaryCore.YearEnd] + 1), },
                **dict(zip(section_energy_consumption_percentage_df_default.columns[1:],
                           section_energy_consumption_percentage_df_default.values[0, 1:]))
            }
            section_energy_consumption_percentage_var['default'] = pd.DataFrame(section_energy_consumption_percentage_dict)

            dynamic_inputs[f"{sector_name}.{GlossaryCore.SectionEnergyConsumptionPercentageDfValue}"] = section_energy_consumption_percentage_var

            dynamic_inputs[f"{sector_name}.{GlossaryCore.InvestmentDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
            dynamic_outputs[f"{sector_name}.{GlossaryCore.ProductionDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.SectorProductionDf)

            dynamic_outputs[f"{sector_name}_{GlossaryCore.EnergyDemandValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyDemandDf)
            dynamic_outputs[f"{sector_name}.{GlossaryCore.EnergyConsumptionValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyConsumptionDf)

            capital_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.CapitalDf)
            capital_df_disc[self.NAMESPACE] = self.NS_SECTORS
            dynamic_outputs[f"{sector_name}.{GlossaryCore.CapitalDfValue}"] = capital_df_disc


            damage_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDf)
            damage_df_disc.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
            dynamic_outputs[f"{sector_name}.{GlossaryCore.DamageDfValue}"] = damage_df_disc
            damage_detailed = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf)
            damage_detailed.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
            dynamic_outputs[f"{sector_name}.{GlossaryCore.DamageDetailedDfValue}"] = damage_detailed

            # section energy consumption df variable
            section_energy_consumption_df_variable = deepcopy(GlossaryCore.SectionEnergyConsumptionDf)
            section_energy_consumption_df_variable["dataframe_descriptor"].update(
                {section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector_name]}
            )
            dynamic_outputs[f"{sector_name}.{GlossaryCore.SectionEnergyConsumptionDfValue}"] = section_energy_consumption_df_variable

            # section gdp value df variable
            section_gdf_df_variable = deepcopy(GlossaryCore.SectionGdpDf)
            section_gdf_df_variable["dataframe_descriptor"].update(
                {section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector_name]}
            )
            dynamic_outputs[f"{sector_name}.{GlossaryCore.SectionGdpDfValue}"] = section_gdf_df_variable


            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def init_execution(self):
        self.model = SectorModel()

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['sector output',
                      GlossaryCore.InvestmentsValue,
                      'output growth',
                      'Energy demand',
                      GlossaryCore.Damages,
                      GlossaryCore.UsableCapital,
                      GlossaryCore.Capital,
                      GlossaryCore.EmploymentRate,
                      GlossaryCore.Workforce,
                      GlossaryCore.Productivity,
                      GlossaryCore.EnergyEfficiency,
                      GlossaryCore.SectionGdpPart,
                      GlossaryCore.SectionEnergyConsumptionPart,
                      ]

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        sector_name = self.get_sosdisc_inputs("sector_name")
        production_df = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.ProductionDfValue}")
        detailed_capital_df = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.CapitalDfValue}")
        workforce_df = self.get_sosdisc_inputs(GlossaryCore.WorkforceDfValue)
        #growth_rate_df = self.get_sosdisc_outputs('growth_rate_df')
        max_capital_utilisation_ratio = self.get_sosdisc_inputs('max_capital_utilisation_ratio')
        compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
        damages_to_productivity = self.get_sosdisc_inputs(GlossaryCore.DamageToProductivity) and compute_climate_impact_on_gdp
        damage_detailed_df = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.DamageDetailedDfValue}")
        years = list(production_df[GlossaryCore.Years].values)


        if 'sector output' in chart_list:
            chart_name = f'{sector_name} sector economics output'
            new_chart = graph_gross_and_net_output(chart_name=chart_name,
                                                   compute_climate_impact_on_gdp=compute_climate_impact_on_gdp,
                                                   damages_to_productivity=damages_to_productivity,
                                                   economics_detail_df=production_df,
                                                   damage_detailed_df=damage_detailed_df)
            instanciated_charts.append(new_chart)

        if GlossaryCore.UsableCapital in chart_list:
            capital_df = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.CapitalDfValue}")
            first_serie = capital_df[GlossaryCore.Capital]
            second_serie = capital_df[GlossaryCore.UsableCapital]

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital stock [trillion dollars]',
                                                 chart_name=chart_name, y_min_zero=True)
            note = {'Productive Capital': ' Non energy capital'}
            new_chart.annotation_upper_left = note

            visible_line = True
            ordonate_data = list(first_serie)
            percentage_productive_capital_stock = list(
                first_serie * max_capital_utilisation_ratio)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Usable capital', 'lines', visible_line)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, percentage_productive_capital_stock,
                f'{max_capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:

            to_plot = {}
            if compute_climate_impact_on_gdp:
                to_plot.update({GlossaryCore.DamagesFromClimate: 'Immediate climate damage (applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (not damages_to_productivity) +'applied to gross output)',})
            else:
                to_plot.update({GlossaryCore.EstimatedDamagesFromClimate: 'Immediate climate damage (estimation not applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (not damages_to_productivity) +'applied to gross output)',})
            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            all_damages = damage_detailed_df[GlossaryCore.EstimatedDamages].values

            chart_name = 'Breakdown of damages' + ' (not applied)' * (not compute_climate_impact_on_gdp)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Trillion$2020',
                                                 chart_name=chart_name, stacked_bar=True)

            for key, legend in to_plot.items():
                ordonate_data = list(damage_detailed_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend, 'bar', True)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(all_damages), 'All damages', 'lines', True)

            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(applied_damages), 'Total applied', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Workforce in chart_list:

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 chart_name=chart_name, y_min_zero=True)

            visible_line = True
            ordonate_data = list(workforce_df[sector_name])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = {
                GlossaryCore.ProductivityWithoutDamage: 'Without damages',
                GlossaryCore.ProductivityWithDamage: 'With damages'}
            productivity_df = self.get_sosdisc_outputs(GlossaryCore.ProductivityDfValue)
            extra_name = 'damages applied' if damages_to_productivity else 'damages not applied'
            chart_name = f'Total Factor Productivity ({extra_name})'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Factor Productivity [no unit]',
                                                 chart_name=chart_name, y_min_zero=True)

            for key, legend in to_plot.items():
                visible_line = True

                ordonate_data = list(productivity_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend, 'lines', visible_line)

                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if "Energy demand" in chart_list:

            chart_name = 'Energy demand and consumption'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.EnergyDemandDf['unit'],
                                                 chart_name=chart_name, y_min_zero=True)

            df_demand = self.get_sosdisc_outputs(f"{sector_name}_{GlossaryCore.EnergyDemandValue}")
            df_conso = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.EnergyConsumptionValue}")

            new_series = InstanciatedSeries(years, df_demand['Total'], "Demand", 'lines', True)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(years, df_conso['Total'], "Consumption", 'bar', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Energy"
            instanciated_charts.append(new_chart)


        if GlossaryCore.EnergyEfficiency in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]
            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital energy efficiency [-]',
                                                 chart_name=chart_name, y_min_zero=True)

            for key in to_plot:
                visible_line = True
                ordonate_data = list(detailed_capital_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)
                new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)

        """
        if 'output growth' in chart_list:

            to_plot = ['net_output_growth_rate']
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' growth rate [-]',
                                                 chart_name=chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(growth_rate_df[key])
                new_series = InstanciatedSeries(years, ordonate_data, key, 'lines', visible_line)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)
        """

        if GlossaryCore.SectionGdpPart in chart_list:
            sections_gdp = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.SectionGdpDfValue}")
            sections_gdp = sections_gdp.drop('years', axis=1)

            chart_name = f'Breakdown of GDP per section for {sector_name} sector [T$]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectionGdpPart,
                                                     chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section, section_value in sections_gdp.items():
                new_series = InstanciatedSeries(
                    years, list(section_value),f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_gdp.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))

        if GlossaryCore.SectionEnergyConsumptionPart in chart_list:
            sections_energy_consumption = self.get_sosdisc_outputs(f"{sector_name}.{GlossaryCore.SectionEnergyConsumptionDfValue}")
            sections_energy_consumption = sections_energy_consumption.drop('years', axis=1)

            chart_name = f'Breakdown of energy consumption per section for {sector_name} sector'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'PWh',
                                                     chart_name=chart_name, stacked_bar=True)

            # loop on all sections of the sector
            for section, section_value in sections_energy_consumption.items():
                new_series = InstanciatedSeries(
                    years, list(section_value),f'{section}', display_type=InstanciatedSeries.BAR_DISPLAY)
                new_chart.add_series(new_series)

            # have a full label on chart (for long names)
            fig = new_chart.to_plotly()
            fig.update_traces(hoverlabel=dict(namelength=-1))
            # if dictionaries has big size, do not show legend, otherwise show it
            if len(list(sections_energy_consumption.keys())) > 5:
                fig.update_layout(showlegend=False)
            else:
                fig.update_layout(showlegend=True)
            instanciated_charts.append(InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name,
                default_title=True, default_legend=False))
            new_chart.post_processing_section_name = "Energy"

        for chart in instanciated_charts:
            if chart.post_processing_section_name is None:
                chart.post_processing_section_name = "Key charts"

        return instanciated_charts
