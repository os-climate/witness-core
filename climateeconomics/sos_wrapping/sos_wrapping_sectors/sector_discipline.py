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
import copy
from copy import deepcopy
import pandas as pd

from climateeconomics.core.core_sectorization.sector_model import SectorModel
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, \
    InstanciatedSeries
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


class SectorDiscipline(ClimateEcoDiscipline):
    """Generic sector discipline"""
    sector_name = 'UndefinedSector'  # to overwrite
    prod_cap_unit = 'T$' # to overwrite if necessary
    NS_SECTORS = GlossaryCore.NS_SECTORS
    DESC_IN = {
        GlossaryCore.SectionGdpPercentageDfValue: GlossaryCore.SectionGdpPercentageDf,
        GlossaryCore.SectionListValue: GlossaryCore.SectionList,
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: ClimateEcoDiscipline.YEAR_END_DESC_IN,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'productivity_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'capital_start': {'type': 'float', 'unit': 'T$', 'user_level': 2},
        GlossaryCore.WorkforceDfValue: GlossaryCore.WorkforceDf,
        'productivity_gr_start': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'decline_rate_tfp': {'type': 'float', 'user_level': 3, 'unit': '-'},
        # Usable capital
        'capital_utilisation_ratio': {'type': 'float', 'default': 0.8, 'user_level': 3, 'unit': '-'},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.95, 'user_level': 3, 'unit': '-'},
        'energy_eff_k': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'energy_eff_cst': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'energy_eff_xzero': {'type': 'float', 'user_level': 3, 'unit': '-'},
        'energy_eff_max': {'type': 'float', 'user_level': 3, 'unit': '-'},
        # Production function param
        'output_alpha': {'type': 'float', 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'depreciation_capital': {'type': 'float', 'user_level': 2, 'unit': '-'},
        GlossaryCore.DamageToProductivity: {'type': 'bool', 'default': True,
                                   'visibility': 'Shared',
                                   'unit': '-', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.FractionDamageToProductivityValue: GlossaryCore.FractionDamageToProductivity,
        GlossaryCore.EnergyProductionValue: {'type': 'dataframe', 'unit': GlossaryCore.EnergyProductionDf['unit'],
                                             'dataframe_descriptor': GlossaryCore.EnergyProductionDf['dataframe_descriptor'],
                                             'dataframe_edition_locked': False},
        'scaling_factor_energy_production': {'type': 'float', 'default': 1e3, 'user_level': 2, 'visibility': 'Shared',
                                             'namespace': GlossaryCore.NS_WITNESS, 'unit': '-'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS,
                  'user_level': 1, 'unit': '-'},
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'user_level': 2, 'unit': '-'},
        'ref_emax_enet_constraint': {'type': 'float', 'default': 60e3, 'user_level': 3,
                                     'visibility': 'Shared', 'namespace': 'ns_ref',
                                     'unit': '-'},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        'prod_function_fitting': {'type': 'bool', 'default': False,
                                  'visibility': 'Shared',
                                  'unit': '-', 'namespace': GlossaryCore.NS_MACRO, 'structuring': True}
    }
    DESC_OUT = {
        GlossaryCore.SectionGdpDfValue: GlossaryCore.SectionGdpDf,
        GlossaryCore.ProductivityDfValue: GlossaryCore.ProductivityDf,
        'growth_rate_df': {'type': 'dataframe', 'unit': '-'},
        GlossaryCore.EnergyWastedObjective: {'type': 'array',
                                             'unit': '-',
                                             'namespace': 'ns_functions'}
    }

    def setup_sos_disciplines(self):
        """setup sos disciplines"""
        dynamic_outputs = {}
        dynamic_inputs = {}

        if GlossaryCore.WorkforceDfValue in self.get_sosdisc_inputs():
            workforce_df: pd.DataFrame = self.get_sosdisc_inputs(GlossaryCore.WorkforceDfValue)
            if workforce_df is not None and self.sector_name not in workforce_df.columns:
                raise Exception(f"Data integrity : workforce_df does should have a column for sector {self.sector_name}")
        if 'prod_function_fitting' in self.get_sosdisc_inputs():
            prod_function_fitting = self.get_sosdisc_inputs('prod_function_fitting')
            if prod_function_fitting:
                dynamic_inputs['energy_eff_max_range_ref'] = {'type': 'float', 'unit': '-', 'default': 5}
                dynamic_inputs['hist_sector_investment'] = {'type': 'dataframe', 'unit': '-',
                                                            'dataframe_descriptor': {},
                                                            'dynamic_dataframe_columns': True}
                dynamic_outputs['longterm_energy_efficiency'] = {'type': 'dataframe', 'unit': '-'}
                dynamic_outputs['range_energy_eff_constraint'] = {'type': 'array', 'unit': '-',
                                                                  'dataframe_descriptor': {},
                                                                  'dynamic_dataframe_columns': True}
        if GlossaryCore.SectionListValue in self.get_data_in():
            sectionlist = self.get_sosdisc_inputs(GlossaryCore.SectionListValue)

        dynamic_inputs[f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.ProductionDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.ProductionDf)
        capital_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.CapitalDf)
        capital_df_disc[self.NAMESPACE] = self.NS_SECTORS
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.CapitalDfValue}"] = capital_df_disc

        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.DetailedCapitalDfValue}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.DetailedCapitalDf)

        damage_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDf)
        damage_df_disc.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.DamageDfValue}"] = damage_df_disc
        damage_detailed = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf)
        damage_detailed.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}"] = damage_detailed

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def init_execution(self):
        param = self.get_sosdisc_inputs(in_dict=True)
        self.model = SectorModel()
        self.model.configure_parameters(param, self.sector_name)

    def run(self):
        # Get inputs
        param = self.get_sosdisc_inputs(in_dict=True)
        # configure param
        self.model.configure_parameters(param, self.sector_name)
        # coupling df
        damage_fraction_df = param[GlossaryCore.DamageFractionDfValue]
        energy_production = param[GlossaryCore.EnergyProductionValue]
        sector_investment = param[f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}"]
        workforce_df = param[GlossaryCore.WorkforceDfValue]
        prod_function_fitting = param['prod_function_fitting']
        section_gdp_percentage_df = param[GlossaryCore.SectionGdpPercentageDfValue]

        model_inputs = {
            GlossaryCore.SectionGdpPercentageDfValue: section_gdp_percentage_df,
            GlossaryCore.DamageFractionDfValue: damage_fraction_df[[GlossaryCore.Years, GlossaryCore.DamageFractionOutput]],
            GlossaryCore.EnergyProductionValue: energy_production,
            GlossaryCore.InvestmentDfValue: sector_investment,
            GlossaryCore.WorkforceDfValue: workforce_df}
        # Model execution
        production_df, detailed_capital_df, productivity_df, damage_df, growth_rate_df, emax_enet_constraint, lt_energy_eff, range_energy_eff_cstrt, section_gdp_df = self.model.compute(
            model_inputs)

        # Store output data
        dict_values = {GlossaryCore.ProductivityDfValue: productivity_df,
                       f"{self.sector_name}.{GlossaryCore.DetailedCapitalDfValue}": detailed_capital_df,
                       'growth_rate_df': growth_rate_df,
                       f"{self.sector_name}.{GlossaryCore.DamageDfValue}": damage_df[GlossaryCore.DamageDf['dataframe_descriptor'].keys()],
                       f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}": damage_df[GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys()],
                       f"{self.sector_name}.{GlossaryCore.ProductionDfValue}": production_df[GlossaryCore.ProductionDf['dataframe_descriptor'].keys()],
                       f"{self.sector_name}.{GlossaryCore.CapitalDfValue}": detailed_capital_df[[GlossaryCore.Years, GlossaryCore.Capital, GlossaryCore.UsableCapital, GlossaryCore.UsableCapitalUnbounded]],
                       GlossaryCore.EnergyWastedObjective: self.model.energy_wasted_objective,
                       GlossaryCore.SectionGdpDfValue: self.model.section_gdp_df,
                       }

        if prod_function_fitting:
            dict_values['longterm_energy_efficiency'] = lt_energy_eff
            dict_values['range_energy_eff_constraint'] = range_energy_eff_cstrt

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradiant of coupling variable
        inputs: - energy
                - investment
                - damage
                - workforce
        outputs: - capital
                - usable capital
                - output
        """

        # gradients wrt workforce
        d_gross_output_d_workforce = self.model.compute_doutput_dworkforce()
        d_net_output_d_workforce = self.model.dnetoutput(d_gross_output_d_workforce)
        d_damage_from_climate_d_workforce = self.model.d_damages_from_climate_d_user_input(
            d_gross_output_d_workforce, d_net_output_d_workforce)
        d_damage_from_productivity_loss_d_workforce = self.model.d_damages_from_productivity_loss_d_user_input(
            d_gross_output_d_workforce)
        d_damages_d_workforce = self.model.d_damages_d_user_input(
            d_damage_from_productivity_loss_d_workforce, d_damage_from_climate_d_workforce)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            (GlossaryCore.WorkforceDfValue, self.sector_name),
            d_gross_output_d_workforce)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.WorkforceDfValue, self.sector_name),
            d_net_output_d_workforce)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            (GlossaryCore.WorkforceDfValue, self.sector_name),
            d_damages_d_workforce)

        # gradients wrt damage:
        dproductivity_ddamage = self.model.d_productivity_d_damage_frac_output()
        d_gross_output_d_damage_frac_output = self.model.doutput_ddamage(
            dproductivity_ddamage)
        d_net_output_d_damage_frac_output = self.model.dnetoutput_ddamage(
            d_gross_output_d_damage_frac_output)

        d_damages_from_climate_d_damage_frac_output = self.model.d_damages_from_climate_d_user_input(
            d_gross_output_d_damage_frac_output, d_net_output_d_damage_frac_output)
        d_damages_from_productivity_loss_d_damage_frac_output = self.model.d_damages_from_productivity_loss_d_damage_fraction_output(
            d_gross_output_d_damage_frac_output)
        d_damages_d_damage_frac_output = self.model.d_damages_d_user_input(
            d_damages_from_climate_d_damage_frac_output, d_damages_from_productivity_loss_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            d_gross_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            d_net_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            d_damages_d_damage_frac_output)


        # gradients wrt invest
        # If production fitting = true we use the investment from another input
        prod_function_fitting = self.get_sosdisc_inputs('prod_function_fitting')
        if prod_function_fitting:
            invest_df = 'hist_sector_investment'
        else:
            invest_df = f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}"
        dcapital_dinvest, d_Ku_d_invests = self.model.dcapital_dinvest()
        d_gross_output_d_invests = self.model.doutput_dinvest(d_Ku_d_invests)
        d_net_output_d_invests = self.model.dnetoutput(d_gross_output_d_invests)
        d_enegy_wasted_obj_d_invest, d_EWO_d_invests = self.model.d_enegy_wasted_obj_d_invest(dcapital_dinvest)
        d_damage_from_climate_d_invests = self.model.d_damages_from_climate_d_user_input(d_gross_output_d_invests, d_net_output_d_invests)
        d_damage_from_productivity_loss_d_invests = self.model.d_damages_from_productivity_loss_d_user_input(
            d_gross_output_d_invests)
        d_damages_d_invests = self.model.d_damages_d_user_input(
            d_damage_from_productivity_loss_d_invests, d_damage_from_climate_d_invests)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyWastedObjective,),
            (invest_df, GlossaryCore.InvestmentsValue),
            d_EWO_d_invests)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.CapitalDfValue}", GlossaryCore.Capital),
            (invest_df, GlossaryCore.InvestmentsValue),
            dcapital_dinvest)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.CapitalDfValue}", GlossaryCore.UsableCapital),
            (invest_df, GlossaryCore.InvestmentsValue),
            d_Ku_d_invests)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            (invest_df, GlossaryCore.InvestmentsValue),
            d_gross_output_d_invests)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
            (invest_df, GlossaryCore.InvestmentsValue),
            d_net_output_d_invests)
        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            (invest_df, GlossaryCore.InvestmentsValue),
            d_damages_d_invests)

        # gradients wrt energy production
        d_gross_output_d_energy_production, d_UKu_d_E, d_Ku_d_E, d_Ew_dE = self.model.d_Y_Ku_Ew_Constraint_d_energy()
        d_net_output_d_energy_production = self.model.dnetoutput(d_gross_output_d_energy_production)
        d_damage_from_climate_d_energy_production = self.model.d_damages_from_climate_d_user_input(
            d_gross_output_d_energy_production, d_net_output_d_energy_production)
        d_damage_from_productivity_loss_d_energy_production = self.model.d_damages_from_productivity_loss_d_user_input(
            d_gross_output_d_energy_production)
        d_damages_d_energy_production = self.model.d_damages_d_user_input(
            d_damage_from_productivity_loss_d_energy_production, d_damage_from_climate_d_energy_production)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.CapitalDfValue}", GlossaryCore.UsableCapital),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_Ku_d_E)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.CapitalDfValue}", GlossaryCore.UsableCapitalUnbounded),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_UKu_d_E)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.GrossOutput),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_gross_output_d_energy_production)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.ProductionDfValue}", GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_net_output_d_energy_production)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyWastedObjective,),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_Ew_dE)

        self.set_partial_derivative_for_other_types(
            (f"{self.sector_name}.{GlossaryCore.DamageDfValue}", GlossaryCore.Damages),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_damages_d_energy_production)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['sector output',
                      GlossaryCore.InvestmentsValue,
                      'output growth',
                      GlossaryCore.Damages,
                      'energy supply',
                      GlossaryCore.UsableCapital,
                      GlossaryCore.Capital,
                      GlossaryCore.EmploymentRate,
                      GlossaryCore.Workforce,
                      GlossaryCore.Productivity,
                      GlossaryCore.EnergyEfficiency,
                      GlossaryCore.EnergyUsage,
                      GlossaryCore.SectionGdpPart,
                      ]

        prod_func_fit = self.get_sosdisc_inputs('prod_function_fitting')
        if prod_func_fit:
            chart_list.append('long term energy efficiency')
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        production_df = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.ProductionDfValue}")
        detailed_capital_df = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.DetailedCapitalDfValue}")
        productivity_df = self.get_sosdisc_outputs(GlossaryCore.ProductivityDfValue)
        section_gdp_df = self.get_sosdisc_outputs(GlossaryCore.SectionGdpDfValue)
        workforce_df = self.get_sosdisc_inputs(GlossaryCore.WorkforceDfValue)
        growth_rate_df = self.get_sosdisc_outputs('growth_rate_df')
        capital_utilisation_ratio = self.get_sosdisc_inputs('capital_utilisation_ratio')
        prod_func_fit = self.get_sosdisc_inputs('prod_function_fitting')
        compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
        damages_to_productivity = self.get_sosdisc_inputs(GlossaryCore.DamageToProductivity) and compute_climate_impact_on_gdp
        if prod_func_fit:
            lt_energy_eff = self.get_sosdisc_outputs('longterm_energy_efficiency')

        if 'sector output' in chart_list:

            to_plot = [GlossaryCore.GrossOutput, GlossaryCore.OutputNetOfDamage]

            legend = {GlossaryCore.GrossOutput: 'Sector gross output',
                      GlossaryCore.OutputNetOfDamage: 'Sector output net of damage'}

            years = list(production_df.index)

            chart_name = f'{self.sector_name} sector economics output'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion dollars]',
                                                 chart_name=chart_name, stacked_bar=True, y_min_zero=not compute_climate_impact_on_gdp)

            for key in to_plot:
                visible_line = True
                ordonate_data = list(production_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.add_series(new_series)

            if compute_climate_impact_on_gdp:
                damage_detailed_df = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}")
                ordonate_data = list(-damage_detailed_df[GlossaryCore.DamagesFromClimate].values)
                new_series = InstanciatedSeries(years, ordonate_data, 'Immediate damages from climate', 'bar')
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.UsableCapital in chart_list:
            capital_df = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.CapitalDfValue}")
            first_serie = capital_df[GlossaryCore.Capital]
            second_serie = capital_df[GlossaryCore.UsableCapital]
            years = list(capital_df.index)

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital stock [trillion dollars]',
                                                 chart_name=chart_name, y_min_zero=True)
            note = {'Productive Capital': ' Non energy capital'}
            new_chart.annotation_upper_left = note

            visible_line = True
            ordonate_data = list(first_serie)
            percentage_productive_capital_stock = list(
                first_serie * capital_utilisation_ratio)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Usable capital', 'lines', visible_line)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, list(capital_df[GlossaryCore.UsableCapitalUnbounded]), 'Unbounded Usable capital', 'lines',
                visible_line)

            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                years, percentage_productive_capital_stock,
                f'{capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Damages in chart_list:

            damage_detailed_df = self.get_sosdisc_outputs(f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}")

            to_plot = {}
            if compute_climate_impact_on_gdp:
                to_plot.update({GlossaryCore.DamagesFromClimate: f'Immediate climate damage (applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (not damages_to_productivity) +'applied to gross output)',})
            else:
                to_plot.update({GlossaryCore.EstimatedDamagesFromClimate: f'Immediate climate damage (estimation not applied to net output)',
                                GlossaryCore.EstimatedDamagesFromProductivityLoss: 'Damages due to loss of productivity (estimation ' + 'not ' * (not damages_to_productivity) +'applied to gross output)',})
            applied_damages = damage_detailed_df[GlossaryCore.Damages].values
            all_damages = damage_detailed_df[GlossaryCore.EstimatedDamagesFromClimate].values + damage_detailed_df[GlossaryCore.EstimatedDamagesFromProductivityLoss].values

            years = list(damage_detailed_df.index)
            chart_name = f'Breakdown of damages' + ' (not applied)' * (not compute_climate_impact_on_gdp)

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

        if GlossaryCore.EnergyUsage in chart_list:
            economics_df = self.get_sosdisc_outputs(GlossaryCore.ProductivityDfValue)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'TWh',
                                                 chart_name=GlossaryCore.EnergyUsage,
                                                 stacked_bar=True)

            to_plot = [GlossaryCore.UsedEnergy, GlossaryCore.UnusedEnergy]
            for p in to_plot:
                new_series = InstanciatedSeries(
                    list(economics_df[GlossaryCore.Years]),
                    list(economics_df[p]),
                    p, 'bar', True)
                new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                list(economics_df[GlossaryCore.Years]),
                list(economics_df[GlossaryCore.OptimalEnergyProduction]),
                GlossaryCore.OptimalEnergyProduction, 'lines', True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Capital in chart_list:
            serie = detailed_capital_df[GlossaryCore.Capital]
            years = list(detailed_capital_df.index)

            chart_name = f'{self.sector_name} capital stock per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital stock [Trillion dollars]',
                                                 chart_name=chart_name, stacked_bar=True)
            ordonate_data = list(serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Industrial capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.add_series(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Workforce in chart_list:
            years = list(workforce_df[GlossaryCore.Years])

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 chart_name=chart_name, y_min_zero=True)

            visible_line = True
            ordonate_data = list(workforce_df[self.sector_name])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = {
                GlossaryCore.ProductivityWithoutDamage: 'Without damages',
                GlossaryCore.ProductivityWithDamage: 'With damages'}
            productivity_df = self.get_sosdisc_outputs(GlossaryCore.ProductivityDfValue)
            years = list(productivity_df.index)
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

        if GlossaryCore.EnergyEfficiency in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]
            years = list(detailed_capital_df.index)
            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital energy efficiency [-]',
                                                 chart_name=chart_name, y_min_zero=True)

            for key in to_plot:
                visible_line = True
                ordonate_data = list(detailed_capital_df[key])
                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'Energy_supply' in chart_list:
            to_plot = [GlossaryCore.TotalProductionValue]

            legend = {
                GlossaryCore.TotalProductionValue: 'energy supply with oil production from energy pyworld3'}

            energy_production = deepcopy(
                self.get_sosdisc_inputs(GlossaryCore.EnergyProductionValue))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production[GlossaryCore.TotalProductionValue] * \
                               scaling_factor_energy_production

            data_to_plot_dict = {
                GlossaryCore.TotalProductionValue: total_production}

            chart_name = 'Energy supply'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $]',
                                                 chart_name=chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(data_to_plot_dict[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'output growth' in chart_list:

            to_plot = ['net_output_growth_rate']
            years = list(growth_rate_df.index)
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' growth rate [-]',
                                                 chart_name=chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(growth_rate_df[key])
                new_series = InstanciatedSeries(years, ordonate_data, key, 'lines', visible_line)
                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if 'long term energy efficiency' in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]

            years = list(lt_energy_eff[GlossaryCore.Years])

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Capital energy efficiency [-]',
                                                 chart_name=chart_name, y_min_zero=True)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(lt_energy_eff[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.SectionGdpPart in chart_list:
            sections_gdp = self.get_sosdisc_outputs(GlossaryCore.SectionGdpDfValue)
            years = list(section_gdp_df.index)

            chart_name = f'Breakdown of GDP per section for {self.sector_name} sector [T$]'

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

        return instanciated_charts
