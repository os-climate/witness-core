'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/30-2023/11/09 Copyright 2023 Capgemini

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
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.macroeconomics_model_v1 import MacroEconomics
from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsDiscipline(AutodifferentiedDisc):
    """Macroeconomics discipline for WITNESS"""

    # ontology information
    _ontology_data = {
        'label': 'Macroeconomics WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': "fa-solid fa-money-bill-trend-up",
        'version': '',
    }
    _maturity = 'Research'
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        'productivity_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'capital_start_non_energy': {'type': 'float', 'unit': 'G$', 'user_level': 2},
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,

        GlossaryCore.WorkingAgePopulationDfValue: {'type': 'dataframe', 'unit': 'millions of people',
                                                   'visibility': 'Shared',
                                                   'namespace': GlossaryCore.NS_WITNESS,
                                                   AutodifferentiedDisc.GRADIENTS: True,
                                                   'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                            GlossaryCore.Population1570: (
                                                                                'float', None, False),
                                                                            }
                                                   },
        'productivity_gr_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02387787, 'user_level': 3, 'unit': '-'},
        # Usable capital
        'capital_utilisation_ratio': {'type': 'float', 'default': 0.8, 'user_level': 3, 'unit': '-'},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.85, 'user_level': 3, 'unit': '-'},
        'energy_eff_k': {'type': 'float', 'default': 0.05085, 'user_level': 3, 'unit': '-'},
        'energy_eff_cst': {'type': 'float', 'default': 0.9835, 'user_level': 3, 'unit': '-'},
        'energy_eff_xzero': {'type': 'float', 'default': 2012.8327, 'user_level': 3, 'unit': '-'},
        'energy_eff_max': {'type': 'float', 'default': 3.5165, 'user_level': 3, 'unit': '-'},
        # Production function param
        'output_alpha': {'type': 'float', 'default': 0.86537, 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'depreciation_capital': {'type': 'float', 'default': 0.07, 'user_level': 2, 'unit': '-'},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'unit': '-', 'visibility': 'Shared',
                                'namespace': GlossaryCore.NS_WITNESS},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared',
                             'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        # sectorisation
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,

        GlossaryCore.DamageToProductivity: {'type': 'bool'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'visibility': 'Shared',
                                                         'namespace': GlossaryCore.NS_WITNESS, 'default': 0.3,
                                                         'unit': '-', 'user_level': 2},

        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.ShareNonEnergyInvestmentsValue: GlossaryCore.ShareNonEnergyInvestment,
        GlossaryCore.EnergyMixNetProductionsDfValue: GlossaryCore.EnergyMixNetProductionsDf,
        GlossaryCore.EnergyMarketRatioAvailabilitiesValue : GlossaryCore.EnergyMarketRatioAvailabilities,

        'init_output_growth': {'type': 'float', 'default': -0.046154, 'unit': '-', 'user_level': 2},
        # Employment rate param
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3, 'unit': '-'},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3, 'unit': '-'},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3, 'unit': '-'},
        GlossaryCore.EnergyCapitalDfValue: {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared',
                                            'namespace': GlossaryCore.NS_WITNESS,
                                            'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                     GlossaryCore.Capital: ('float', None, False), }
                                            },
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        GlossaryCore.SectionListValue: GlossaryCore.SectionList,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.SectorEnergyConsumptionPercentageDfName: GlossaryCore.SectorEnergyConsumptionPercentageDf,
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDetailDfValue: GlossaryCore.EconomicsDetailDf,
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        GlossaryCore.DamageDfValue: GlossaryCore.DamageDf,
        GlossaryCore.DamageDetailedDfValue: GlossaryCore.DamageDetailedDf,
        GlossaryCore.WorkforceDfValue: {'type': GlossaryCore.WorkforceDf['type'],
                                        'unit': GlossaryCore.WorkforceDf['unit']},
        GlossaryCore.CapitalDfValue: {'type': 'dataframe', 'unit': '-',
                                      'dataframe_descriptor': GlossaryCore.CapitalDf['dataframe_descriptor']},
        GlossaryCore.TotalGDPGroupDFName: GlossaryCore.TotalGDPGroupDF,
        GlossaryCore.PercentageGDPGroupDFName: GlossaryCore.PercentageGDPGroupDF,
        GlossaryCore.GDPCountryDFName: GlossaryCore.GDPCountryDF,
        GlossaryCore.ResidentialEnergyConsumptionDfValue: GlossaryCore.ResidentialEnergyConsumptionDf,
        f"{GlossaryCore.Macroeconomics}_{GlossaryCore.EnergyDemandValue}": GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyDemandDf),
        f"{GlossaryCore.Macroeconomics}.{GlossaryCore.EnergyConsumptionValue}": GlossaryCore.get_dynamic_variable(GlossaryCore.EnergyConsumptionDf),
    }

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}
        sectorlist = None
        sectionlist = None
        if self.get_data_in() is not None:
            year_end = None
            if GlossaryCore.YearEnd in self.get_data_in() and GlossaryCore.YearStart in self.get_data_in():
                year_start, year_end = self.get_sosdisc_inputs([GlossaryCore.YearStart, GlossaryCore.YearEnd])

            if 'assumptions_dict' in self.get_data_in():
                assumptions_dict = self.get_sosdisc_inputs('assumptions_dict')
                compute_gdp: bool = assumptions_dict['compute_gdp']
                # if compute gdp is not activated, we add gdp input
                if not compute_gdp:
                    gross_output_df = self.get_default_gross_output_in()
                    dynamic_inputs.update({'gross_output_in': {'type': 'dataframe', 'unit': 'G$',
                                                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                                               'dataframe_descriptor': {
                                                                   GlossaryCore.Years: ('float', None, False),
                                                                   GlossaryCore.GrossOutput: (
                                                                       'float', None, True)},
                                                               'default': gross_output_df,
                                                               'dataframe_edition_locked': False,
                                                               'namespace': GlossaryCore.NS_WITNESS}})

            if GlossaryCore.SectorListValue in self.get_data_in():
                sectorlist = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

            if GlossaryCore.SectionListValue in self.get_data_in():
                sectionlist = self.get_sosdisc_inputs(GlossaryCore.SectionListValue)

            if sectorlist is not None and year_end is not None:
                sector_gdg_desc = GlossaryCore.get_dynamic_variable(GlossaryCore.SectorGdpDf)
                default_value_energy_consumption_dict = DatabaseWitnessCore.EnergyConsumptionPercentageSectionsDict.value
                default_non_energy_emissions_dict = DatabaseWitnessCore.SectionsNonEnergyEmissionsDict.value
                for sector in sectorlist:
                    sector_gdg_desc['dataframe_descriptor'].update({sector: ('float', [1.e-8, 1e30], True)})
                    # change default value for each sector for energy consumption and non energy emissions
                    sector_energy_consumption_percentage_dict = GlossaryCore.get_dynamic_variable(GlossaryCore.SectorEnergyConsumptionPercentageDf)
                    df_default_val = default_value_energy_consumption_dict[sector]
                    df_default_val = df_default_val.loc[df_default_val[GlossaryCore.Years] <= year_end]
                    sector_energy_consumption_percentage_dict.update({"default": df_default_val})
                    non_energy_emissions_sections_dict = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionNonEnergyEmissionGdpDf)
                    non_energy_emissions_sections_dict.update({"default": default_non_energy_emissions_dict[sector]})
                    # add to dynamic inputs
                    dynamic_inputs.update({f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}': sector_energy_consumption_percentage_dict})

                    # section energy consumption
                    section_energy_consumption_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionEnergyConsumptionDf)
                    section_energy_consumption_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]}                    )
                    dynamic_outputs.update({f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}": section_energy_consumption_df_variable})

                    # sections gdp value df
                    section_gdp_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionGdpDf)
                    section_gdp_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    dynamic_outputs.update({f"{sector}.{GlossaryCore.SectionGdpDfValue}": section_gdp_df_variable})

                # make sure the namespaces references are good in case shared namespaces were reassociated
                dynamic_outputs.update({GlossaryCore.SectorGdpDfValue: sector_gdg_desc,})


            # add section gdp percentage variable
            if sectionlist is not None:
                section_gdp_percentage = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionGdpPercentageDf)
                # update dataframe descriptor
                for section in sectionlist:
                    section_gdp_percentage['dataframe_descriptor'].update({section: ('float', [1.e-8, 1e30], True)})
                dynamic_inputs.update({GlossaryCore.SectionGdpPercentageDfValue: section_gdp_percentage})

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

        self.update_default_values()

    def get_default_gross_output_in(self):
        '''
        Get default values for gross_output_in into GDP PPP economics_df_ssp3.csv file
        '''
        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefault
        if GlossaryCore.YearStart in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                [GlossaryCore.YearStart, GlossaryCore.YearEnd])
        years = np.arange(year_start, year_end + 1)
        global_data_dir = join(Path(__file__).parents[3], 'data')
        gross_output_ssp3_file = join(global_data_dir, 'economics_df_ssp3.csv')
        if isfile(gross_output_ssp3_file):
            gross_output_df = pd.read_csv(gross_output_ssp3_file)[[GlossaryCore.Years, GlossaryCore.GrossOutput]]

            if gross_output_df.iloc[0][GlossaryCore.Years] > year_start:
                df_list = [gross_output_df]

                df_list.extend([pd.DataFrame({GlossaryCore.Years: year,
                                              GlossaryCore.GrossOutput: gross_output_df.iloc[0][
                                                  GlossaryCore.GrossOutput]})
                                for year in np.arange(year_start, gross_output_df.iloc[0][GlossaryCore.Years])])

                gross_output_df = pd.concat(df_list, ignore_index=True)

                gross_output_df = gross_output_df.sort_values(by=GlossaryCore.Years)
                gross_output_df = gross_output_df.reset_index()
                gross_output_df = gross_output_df.drop(columns=['index'])

            elif gross_output_df.iloc[0][GlossaryCore.Years] < year_start:
                gross_output_df = gross_output_df[gross_output_df[GlossaryCore.Years] > year_start - 1]
            if gross_output_df.iloc[-1][GlossaryCore.Years] > year_end:
                gross_output_df = gross_output_df[gross_output_df[GlossaryCore.Years] < year_end + 1]
            elif gross_output_df.iloc[-1][GlossaryCore.Years] < year_end:
                df_list = [gross_output_df]

                df_list.extend([pd.DataFrame({GlossaryCore.Years: year,
                                              GlossaryCore.GrossOutput: gross_output_df.iloc[-1][
                                                  GlossaryCore.GrossOutput]}) for year in
                                np.arange(gross_output_df.iloc[-1][GlossaryCore.Years] + 1,
                                          year_end + 1)])

                gross_output_df = pd.concat(df_list, ignore_index=True)


        else:
            gross_output_df = pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.GrossOutput: np.linspace(130., 255., len(years))})
        gross_output_df = gross_output_df.reset_index()
        gross_output_df = gross_output_df.drop(columns=['index'])

        return gross_output_df

    def update_default_values(self):
        """
        Update all default dataframes with years
        """
        if self.get_data_in() is not None:
            if GlossaryCore.YearEnd in self.get_data_in() and GlossaryCore.YearStart in self.get_data_in():
                year_start, year_end = self.get_sosdisc_inputs([GlossaryCore.YearStart, GlossaryCore.YearEnd])
                if year_start is not None:
                    self.update_default_value('productivity_start', 'in', DatabaseWitnessCore.MacroProductivityStart.get_value_at_year(year_start))
                    self.update_default_value('productivity_gr_start', 'in', DatabaseWitnessCore.MacroProductivityGrowthStart.get_value_at_year(year_start))
                    self.update_default_value('capital_start_non_energy', 'in', DatabaseWitnessCore.MacroNonEnergyCapitalStart.get_value_at_year(year_start))
                if year_start is not None and year_end is not None:
                    default_val = DatabaseWitnessCore.EnergyConsumptionPercentageSectorDict.get_all_cols_between_years(year_start, year_end)
                    self.update_default_value(GlossaryCore.SectorEnergyConsumptionPercentageDfName, 'in', default_val)

                    years = np.arange(year_start, year_end + 1)
                    share_non_energy_investment = pd.DataFrame(
                        {GlossaryCore.Years: years,
                         GlossaryCore.ShareNonEnergyInvestmentsValue: [DatabaseWitnessCore.ShareInvestNonEnergy.value] * len(years)})

                    self.set_dynamic_default_values(
                        {GlossaryCore.ShareNonEnergyInvestmentsValue: share_non_energy_investment, })

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.model = MacroEconomics(param)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.GrossOutput,
                      GlossaryCore.OutputNetOfDamage,
                      GlossaryCore.Damages,
                      GlossaryCore.InvestmentsValue,
                      GlossaryCore.UsableCapital,
                      GlossaryCore.Capital,
                      "Energy demand",
                      GlossaryCore.EmploymentRate,
                      GlossaryCore.Workforce,
                      GlossaryCore.Productivity,
                      GlossaryCore.EnergyEfficiency,
                      ]
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_detail_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue))
        capital_df = self.get_sosdisc_outputs(GlossaryCore.CapitalDfValue)
        capital_utilisation_ratio, max_capital_utilisation_ratio = deepcopy(
            self.get_sosdisc_inputs(['capital_utilisation_ratio', 'max_capital_utilisation_ratio']))
        workforce_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.WorkforceDfValue))
        sector_gdp_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.SectorGdpDfValue))
        economics_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDfValue))
        sectors_list = deepcopy(
            self.get_sosdisc_inputs(GlossaryCore.SectorListValue))
        years = list(economics_detail_df[GlossaryCore.Years].values)
        compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
        damages_to_productivity = self.get_sosdisc_inputs(
            GlossaryCore.DamageToProductivity) and compute_climate_impact_on_gdp
        damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)

        if GlossaryCore.GrossOutput in chart_list:
            chart_name = 'Gross and net of damage output per year'
            new_chart = graph_gross_and_net_output(chart_name=chart_name,
                                                   compute_climate_impact_on_gdp=compute_climate_impact_on_gdp,
                                                   damages_to_productivity=damages_to_productivity,
                                                   economics_detail_df=economics_detail_df,
                                                   damage_detailed_df=damage_detailed_df)
            instanciated_charts.append(new_chart)

        if GlossaryCore.OutputNetOfDamage in chart_list:

            to_plot = [GlossaryCore.InvestmentsValue, GlossaryCore.Consumption]

            legend = {GlossaryCore.InvestmentsValue: 'Investments',
                      GlossaryCore.Consumption: 'Consumption', }

            chart_name = 'Breakdown of net output'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'bar', True)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_detail_df[GlossaryCore.OutputNetOfDamage].values), 'Net output', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:

            damage_detailed_df = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
            damage_to_productivity = self.get_sosdisc_inputs(
                GlossaryCore.DamageToProductivity) and compute_climate_impact_on_gdp
            to_plot = {}
            if compute_climate_impact_on_gdp:
                to_plot.update({GlossaryCore.DamagesFromClimate: 'Immediate climate damage (applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (
                                    not damage_to_productivity) + 'applied to gross output)', })
            else:
                to_plot.update({
                    GlossaryCore.EstimatedDamagesFromClimate: 'Immediate climate damage (estimation not applied to net output)',
                    GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (
                        not damage_to_productivity) + 'applied to gross output)', })

            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            all_damages = damage_detailed_df[GlossaryCore.EstimatedDamages].values
            chart_name = 'Breakdown of damages' + ' (not applied)' * (not compute_climate_impact_on_gdp)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key, legend in to_plot.items():
                ordonate_data = list(damage_detailed_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend, 'bar', True)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(all_damages), 'Total all damages', 'lines', True)

            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(applied_damages), 'Total applied', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.InvestmentsValue in chart_list:

            to_plot = [GlossaryCore.EnergyInvestmentsValue,
                       GlossaryCore.NonEnergyInvestmentsValue]

            legend = {GlossaryCore.InvestmentsValue: 'Total investments',
                      GlossaryCore.EnergyInvestmentsValue: 'Energy',
                      GlossaryCore.NonEnergyInvestmentsValue: 'Non-energy sectors', }


            chart_name = 'Breakdown of Investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'investment [trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, list(economics_detail_df[key]), legend[key], InstanciatedSeries.BAR_DISPLAY, visible_line)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_detail_df[GlossaryCore.InvestmentsValue]),
                legend[GlossaryCore.InvestmentsValue],
                'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.UsableCapital in chart_list:
            capital_df = self.get_sosdisc_outputs(GlossaryCore.CapitalDfValue)
            first_serie = capital_df[GlossaryCore.NonEnergyCapital]
            second_serie = capital_df[GlossaryCore.UsableCapital]

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, y_min_zero=True)
            note = {'Productive Capital': ' Non energy capital'}
            new_chart.annotation_upper_left = note

            visible_line = True
            ordonate_data = list(first_serie)
            percentage_productive_capital_stock = list(
                first_serie * capital_utilisation_ratio)
            percentage_max_productive_capital_stock = list(
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
                f'{capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, percentage_max_productive_capital_stock,
                f'{max_capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)


        if GlossaryCore.Capital in chart_list:
            energy_capital_df = self.get_sosdisc_inputs(GlossaryCore.EnergyCapitalDfValue)

            first_serie = capital_df[GlossaryCore.NonEnergyCapital]
            second_serie = energy_capital_df[GlossaryCore.Capital]
            third_serie = capital_df[GlossaryCore.Capital]

            chart_name = 'Capital stock per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)
            visible_line = True
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

            ordonate_data = list(first_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Non energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)

            ordonate_data_ter = list(third_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_ter, 'Total capital stock', 'lines', visible_line)
            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)

        if GlossaryCore.EmploymentRate in chart_list:
            chart_name = 'Employment rate'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'employment rate',
                                                 chart_name=chart_name, y_min_zero=True)

            visible_line = True
            ordonate_data = list(workforce_df[GlossaryCore.EmploymentRate])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.EmploymentRate, 'lines', visible_line)

            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Workforce"

            instanciated_charts.append(new_chart)

        if GlossaryCore.Workforce in chart_list:
            working_age_pop_df = self.get_sosdisc_inputs(
                GlossaryCore.WorkingAgePopulationDfValue)

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 chart_name=chart_name, y_min_zero=True)

            visible_line = True
            ordonate_data = list(workforce_df[GlossaryCore.Workforce])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)
            ordonate_data_bis = list(working_age_pop_df[GlossaryCore.Population1570])
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Working-age population', 'lines', visible_line)
            new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Workforce"
            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = {
                GlossaryCore.ProductivityWithoutDamage: 'Without damages',
                GlossaryCore.ProductivityWithDamage: 'With damages'}
            extra_name = 'damages applied' if damages_to_productivity else 'damages not applied'
            chart_name = f'Total Factor Productivity ({extra_name})'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Factor Productivity [no unit]',
                                                 chart_name=chart_name, stacked_bar=True, y_min_zero=True)

            for key, legend in to_plot.items():
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend, 'lines', visible_line)

                new_chart.add_series(new_series)
            new_chart.post_processing_section_name = "Workforce"
            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyEfficiency in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]


            chart_name = 'Capital energy efficiency'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'no unit',
                                                 chart_name=chart_name, y_min_zero=True)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(capital_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)

        if "Energy demand" in chart_list:

            chart_name = 'Energy demand and consumption'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.EnergyDemandDf['unit'],
                                                 chart_name=chart_name, y_min_zero=True)

            df_demand = self.get_sosdisc_outputs(f"{GlossaryCore.Macroeconomics}_{GlossaryCore.EnergyDemandValue}")
            df_conso = self.get_sosdisc_outputs(f"{GlossaryCore.Macroeconomics}.{GlossaryCore.EnergyConsumptionValue}")

            new_series = InstanciatedSeries(years, df_demand['Total'], "Demand", 'lines', True)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(years, df_conso['Total'], "Consumption", 'bar', True)
            new_chart.add_series(new_series)

            new_chart.post_processing_section_name = "Capital"
            instanciated_charts.append(new_chart)

        if GlossaryCore.SectorGdpPart in chart_list:
            to_plot = sectors_list
            legend = {sector: sector for sector in sectors_list}
            # Graph with distribution per sector in absolute value
            legend[GlossaryCore.OutputNetOfDamage] = 'Total GDP net of damage'

            chart_name = 'Breakdown of GDP per sector [T$]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorGdpPart,
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, list(sector_gdp_df[key]), legend[key], InstanciatedSeries.BAR_DISPLAY, visible_line)

                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_df[GlossaryCore.OutputNetOfDamage]),
                legend[GlossaryCore.OutputNetOfDamage],
                'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            # graph in percentage of GDP
            total_gdp = economics_df[GlossaryCore.OutputNetOfDamage].values
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "Contribution [%]",
                                                 chart_name=GlossaryCore.ChartSectorGDPPercentage, stacked_bar=True)

            for sector in sectors_list:
                sector_gdp_part = sector_gdp_df[sector] / total_gdp * 100.
                sector_gdp_part = np.nan_to_num(sector_gdp_part, nan=0.)
                serie = InstanciatedSeries(list(sector_gdp_df[GlossaryCore.Years]), list(sector_gdp_part), sector,
                                           'bar', True)
                new_chart.add_series(serie)

            instanciated_charts.append(new_chart)

        for chart in instanciated_charts:
            if chart.post_processing_section_name is None:
                chart.post_processing_section_name = "Key charts"

        return instanciated_charts

def breakdown_gdp(economics_detail_df, damage_detailed_df, compute_climate_impact_on_gdp, damages_to_productivity):
    """ returns dashboard graph for output """
    to_plot_line = [GlossaryCore.OutputNetOfDamage]

    to_plot_bar = [GlossaryCore.EnergyInvestmentsValue,
                   GlossaryCore.NonEnergyInvestmentsValue,
                   GlossaryCore.Consumption]

    legend = {GlossaryCore.OutputNetOfDamage: 'Net GDP',
              GlossaryCore.InvestmentsValue: 'Total investments',
              GlossaryCore.EnergyInvestmentsValue: 'Energy investments',
              GlossaryCore.NonEnergyInvestmentsValue: 'Non-energy investments',
              GlossaryCore.Consumption: 'Consumption',
              }

    years = list(economics_detail_df[GlossaryCore.Years].values)

    chart_name = 'Breakdown of GDP per year'

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                         chart_name=chart_name, stacked_bar=True,
                                         y_min_zero=False)

    new_chart = new_chart.to_plotly()

    gross_output = economics_detail_df[GlossaryCore.GrossOutput].values

    for key in to_plot_bar:
        ordonate_data = list(economics_detail_df[key])
        new_chart.add_trace(go.Scatter(
            x=years,
            y=ordonate_data,
            opacity=0.7,
            line=dict(width=1.25),
            name=legend[key],
            stackgroup='one',
        ))

    for key in to_plot_line:
        ordonate_data = list(economics_detail_df[key])
        new_chart.add_trace(go.Scatter(
            x=years,
            y=ordonate_data,
            mode='lines',
            name=legend[key],
        ))

    new_chart.add_trace(go.Scatter(
        x=years,
        y=list(economics_detail_df[GlossaryCore.InvestmentsValue]),
        mode='lines',
        name=legend[GlossaryCore.InvestmentsValue],
    ))

    new_chart.add_trace(go.Scatter(x=years, y=list(gross_output),
                                   mode='lines',
                                   name="Gross GDP"
                                   ))

    if compute_climate_impact_on_gdp:
        ordonate_data = list(-damage_detailed_df[GlossaryCore.DamagesFromClimate])

        new_chart.add_trace(go.Scatter(
            x=years,
            y=ordonate_data,
            opacity=0.7,
            line=dict(width=1.25),
            name='Immediate damages from climate',
            stackgroup='two',
        ))

        if damages_to_productivity:
            gdp_without_damage_to_prod = gross_output + damage_detailed_df[
                GlossaryCore.EstimatedDamagesFromProductivityLoss].values

            new_chart.add_trace(go.Scatter(
                x=years,
                y=list(gdp_without_damage_to_prod),
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines',
                fillcolor='rgba(200, 200, 200, 0.3)',
                line={'dash': 'dash', 'color': 'rgb(200, 200, 200)'},
                opacity=0.2,
                name='Estimation of GDP without damages', ))

    new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)

    return new_chart
