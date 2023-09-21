"""
Copyright 2022 Airbus SAS

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
from pathlib import Path
import copy
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.macroeconomics_model_v1 import MacroEconomics
from sostrades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from os.path import join, isfile
from copy import deepcopy
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min
from sostrades_core.tools.cst_manager.constraint_manager import compute_ddelta_constraint
from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsDiscipline(ClimateEcoDiscipline):
    "Macroeconomics discipline for WITNESS"

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
        'icon': 'fas fa-industry fa-fw',
        'version': '',
    }
    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: ClimateEcoDiscipline.YEAR_END_DESC_IN,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'productivity_start': {'type': 'float', 'default': 0.27357, 'user_level': 2, 'unit': '-'},
        GlossaryCore.InitialGrossOutput['var_name']: GlossaryCore.InitialGrossOutput,
        'capital_start_non_energy': {'type': 'float', 'unit': 'T$', 'default': 360.5487346, 'user_level': 2},
        GlossaryCore.DamageDf['var_name']: GlossaryCore.DamageDf,
        GlossaryCore.PopulationDf['var_name']: GlossaryCore.PopulationDf,

        'working_age_population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared',
                                      'namespace': 'ns_witness',
                                      'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                               'population_1570': ('float', None, False),
                                                               }
                                      },
        'productivity_gr_start': {'type': 'float', 'default': 0.004781, 'user_level': 2, 'unit': '-'},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02387787, 'user_level': 3, 'unit': '-'},
        # Usable capital
        'capital_utilisation_ratio': {'type': 'float', 'default': 0.8, 'user_level': 3, 'unit': '-'},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.95, 'user_level': 3, 'unit': '-'},
        'energy_eff_k': {'type': 'float', 'default': 0.05085, 'user_level': 3, 'unit': '-'},
        'energy_eff_cst': {'type': 'float', 'default': 0.9835, 'user_level': 3, 'unit': '-'},
        'energy_eff_xzero': {'type': 'float', 'default': 2012.8327, 'user_level': 3, 'unit': '-'},
        'energy_eff_max': {'type': 'float', 'default': 3.5165, 'user_level': 3, 'unit': '-'},
        # Production function param
        'output_alpha': {'type': 'float', 'default': 0.86537, 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'depreciation_capital': {'type': 'float', 'default': 0.07, 'user_level': 2, 'unit': '-'},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'unit': '-', 'visibility': 'Shared',
                                'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared',
                             'namespace': 'ns_witness', 'user_level': 2},
        # sectorisation
        GlossaryCore.SectorsList['var_name'] : GlossaryCore.SectorsList,
        # Lower and upper bounds
        'lo_capital': {'type': 'float', 'unit': 'T$', 'default': 1.0, 'user_level': 3},
        'lo_conso': {'type': 'float', 'unit': 'T$', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        'hi_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 70, 'user_level': 3},
        'ref_pc_consumption_constraint': {'type': 'float', 'unit': 'k$', 'default': 1, 'user_level': 3,
                                          'namespace': 'ns_ref'},
        'damage_to_productivity': {'type': 'bool'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'default': 0.3,
                             'unit': '-', 'user_level': 2},

        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.ShareNonEnergyInvestment['var_name']: GlossaryCore.ShareNonEnergyInvestment,
        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProduction,

        'scaling_factor_energy_production': {'type': 'float', 'default': 1e3, 'unit': '-', 'user_level': 2,
                                             'visibility': 'Shared', 'namespace': 'ns_witness'},
        'alpha': ClimateEcoDiscipline.ALPHA_DESC_IN,
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'unit': '-', 'user_level': 2},
        GlossaryCore.CO2EmissionsGt['var_name']: GlossaryCore.CO2EmissionsGt,
        'CO2_tax_efficiency': {'type': 'dataframe', 'unit': '%',
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                        'CO2_tax_efficiency': ('float', None, False), }
                               },
        'co2_invest_limit': {'type': 'float', 'default': 2.0, 'unit': 'factor of energy investment'},
        GlossaryCore.CO2Taxes['var_name']: GlossaryCore.CO2Taxes,
        # Employment rate param
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3, 'unit': '-'},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3, 'unit': '-'},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3, 'unit': '-'},
        'ref_emax_enet_constraint': {'type': 'float', 'default': 60e3, 'unit': '-', 'user_level': 3,
                                     'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'usable_capital_ref': {'type': 'float', 'unit': 'T$', 'default': 0.3, 'user_level': 3,
                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'energy_capital': {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared', 'namespace': 'ns_witness',
                           'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                    'energy_capital': ('float', None, False), }
                           },
        'delta_capital_cons_limit': {'type': 'float', 'unit': 'G$', 'default': 50, 'user_level': 3,
                                     'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDetail_df['var_name']: GlossaryCore.EconomicsDetail_df,
        GlossaryCore.EconomicsDf['var_name']: GlossaryCore.EconomicsDf,
        GlossaryCore.EnergyInvestments["var_name"]: GlossaryCore.EnergyInvestments,
        GlossaryCore.EnergyInvestmentsWoRenewable['var_name']: GlossaryCore.EnergyInvestmentsWoRenewable,
        'pc_consumption_constraint': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                      'namespace': 'ns_functions', 'unit': 'k$'},
        GlossaryCore.WorkforceDfValue: {'type': GlossaryCore.WorkforceDf['type'], 'unit': GlossaryCore.WorkforceDf['unit']},
        'emax_enet_constraint': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                 'namespace': 'ns_functions'},
        'delta_capital_objective': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_functions'},
        'delta_capital_objective_weighted': {'type': 'array', 'unit': '-',
                                             'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                             'namespace': 'ns_functions'},
        'delta_capital_objective_wo_exp_min': {'type': 'array', 'unit': '-'},
        GlossaryCore.CapitalDfValue: {'type': 'dataframe', 'unit': '-'},
        'delta_capital_constraint': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_functions'},
        'delta_capital_constraint_dc': {'type': 'array', 'unit': '-',
                                        'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                        'namespace': 'ns_functions'},
        'delta_capital_lintoquad': {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_functions'}
    }

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}
        sectorlist = None
        if self.get_data_in() is not None:
            if 'assumptions_dict' in self.get_data_in():
                assumptions_dict = self.get_sosdisc_inputs('assumptions_dict')
                compute_gdp: bool = assumptions_dict['compute_gdp']
                # if compute gdp is not activated, we add gdp input
                if not compute_gdp:
                    gross_output_df = self.get_default_gross_output_in()
                    dynamic_inputs.update({'gross_output_in': {'type': 'dataframe', 'unit': 'G$',
                                                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                                        GlossaryCore.GrossOutput: (
                                                                                            'float', None, True)},
                                                               'default': gross_output_df,
                                                               'dataframe_edition_locked': False,
                                                               'namespace': 'ns_witness'}})

            if GlossaryCore.SectorsList['var_name'] in self.get_data_in():
                sectorlist = self.get_sosdisc_inputs(GlossaryCore.SectorsList['var_name'])

        if sectorlist is not None:
            sector_gdg_desc = copy.deepcopy(GlossaryCore.SectorGdpDf)  # deepcopy not to modify dataframe_descriptor in Glossary
            for sector in sectorlist:
                sector_gdg_desc['dataframe_descriptor'].update({sector: ('float', [1.e-8, 1e30], True)})
            # make sure the namespaces references are good in case shared namespaces were reassociated
            sector_gdg_desc[SoSWrapp.NS_REFERENCE] = self.get_shared_ns_dict()[sector_gdg_desc[SoSWrapp.NAMESPACE]]
            dynamic_outputs.update({GlossaryCore.SectorGdpDf['var_name']: sector_gdg_desc})

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

        self.update_default_with_years()

    def get_default_gross_output_in(self):
        '''
        Get default values for gross_output_in into GDP PPP economics_df_ssp3.csv file
        '''
        year_start = 2020
        year_end = 2100
        if GlossaryCore.YearStart in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                [GlossaryCore.YearStart, GlossaryCore.YearEnd])
        years = np.arange(year_start, year_end + 1)
        global_data_dir = join(Path(__file__).parents[3], 'data')
        gross_output_ssp3_file = join(global_data_dir, 'economics_df_ssp3.csv')
        gross_output_df = None
        if isfile(gross_output_ssp3_file):
            gross_output_df = pd.read_csv(gross_output_ssp3_file)[[GlossaryCore.Years,GlossaryCore.GrossOutput]]

            if gross_output_df.iloc[0][GlossaryCore.Years] > year_start:
                gross_output_df = gross_output_df.append([{GlossaryCore.Years:year,GlossaryCore.GrossOutput:gross_output_df.iloc[0][GlossaryCore.GrossOutput]} for year in np.arange(year_start,gross_output_df.iloc[0][GlossaryCore.Years])], ignore_index=True)
                gross_output_df = gross_output_df.sort_values(by=GlossaryCore.Years)
                gross_output_df = gross_output_df.reset_index()
                gross_output_df = gross_output_df.drop(columns=['index'])

            elif gross_output_df.iloc[0][GlossaryCore.Years] < year_start:
                gross_output_df = gross_output_df[gross_output_df[GlossaryCore.Years]>year_start-1]
            if gross_output_df.iloc[-1][GlossaryCore.Years] > year_end:
                gross_output_df = gross_output_df[gross_output_df[GlossaryCore.Years]<year_end+1]
            elif gross_output_df.iloc[-1][GlossaryCore.Years] < year_end:
                gross_output_df = gross_output_df.append([{GlossaryCore.Years:year,GlossaryCore.GrossOutput:gross_output_df.iloc[-1][GlossaryCore.GrossOutput]} for year in np.arange(gross_output_df.iloc[-1][GlossaryCore.Years]+1, year_end+1)], ignore_index=True)

        else:
            gross_output_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.GrossOutput: np.linspace(130., 255., len(years))})
        gross_output_df = gross_output_df.reset_index()
        gross_output_df = gross_output_df.drop(columns=['index'])

        return gross_output_df

    def update_default_with_years(self):
        """
        Update all default dataframes with years
        """
        if GlossaryCore.YearStart in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                [GlossaryCore.YearStart, GlossaryCore.YearEnd])
            years = np.arange(year_start, year_end + 1)
            intermediate_point = 30
            CO2_tax_efficiency = np.concatenate(
                (np.linspace(30, intermediate_point, 15), np.asarray([intermediate_point] * (len(years) - 15))))

            co2_tax_efficiency_default = pd.DataFrame({GlossaryCore.Years: years,
                                                       'CO2_tax_efficiency': CO2_tax_efficiency})

            share_non_energy_investment = pd.DataFrame(
                {GlossaryCore.Years: years,
                 GlossaryCore.ShareNonEnergyInvestmentsValue: [27.0 - 2.6] * len(years)})

            self.set_dynamic_default_values(
                {'CO2_tax_efficiency': co2_tax_efficiency_default,
                 GlossaryCore.ShareNonEnergyInvestmentsValue: share_non_energy_investment ,})

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.macro_model = MacroEconomics(param)

    def run(self):
        param = self.get_sosdisc_inputs()
        damage_df = param.pop(GlossaryCore.DamageDfValue)
        energy_production = param.pop('energy_production')
        co2_emissions_Gt = param.pop(GlossaryCore.CO2EmissionsGtValue)
        co2_taxes = param.pop(GlossaryCore.CO2TaxesValue)
        co2_tax_efficiency = param.pop('CO2_tax_efficiency')
        co2_invest_limit = param.pop('co2_invest_limit')
        population_df = param.pop(GlossaryCore.PopulationDfValue)
        working_age_population_df = param.pop('working_age_population_df')
        energy_capital_df = param['energy_capital']
        compute_gdp: bool = param['assumptions_dict']['compute_gdp']
        sector_list = param[GlossaryCore.SectorsList['var_name']]
        macro_inputs = {GlossaryCore.DamageDfValue: damage_df[[GlossaryCore.Years, GlossaryCore.DamageFractionOutput]],
                        'energy_production': energy_production,
                        'scaling_factor_energy_production': param['scaling_factor_energy_production'],
                        GlossaryCore.EnergyInvestmentsWoTaxValue: param[GlossaryCore.EnergyInvestmentsWoTaxValue],
                        GlossaryCore.ShareNonEnergyInvestmentsValue: param[GlossaryCore.ShareNonEnergyInvestmentsValue],
                        GlossaryCore.CO2EmissionsGtValue: co2_emissions_Gt,
                        GlossaryCore.CO2TaxesValue: co2_taxes,
                        'CO2_tax_efficiency': co2_tax_efficiency,
                        'co2_invest_limit': co2_invest_limit,
                        GlossaryCore.PopulationDfValue: population_df[[GlossaryCore.Years, GlossaryCore.PopulationValue]],
                        'working_age_population_df': working_age_population_df[[GlossaryCore.Years, 'population_1570']],
                        'energy_capital_df': energy_capital_df,
                        'compute_gdp': compute_gdp,
                        GlossaryCore.SectorsList['var_name']: sector_list
                        }

        if not compute_gdp:
            macro_inputs.update({'gross_output_in': param['gross_output_in']})

        # Model execution
        economics_detail_df, economics_df, energy_investment, energy_investment_wo_renewable, \
            pc_consumption_constraint, workforce_df, capital_df, emax_enet_constraint, sector_gdp_df = \
            self.macro_model.compute(macro_inputs)

        # Store output data
        dict_values = {GlossaryCore.EconomicsDetail_df['var_name']: economics_detail_df,
                       GlossaryCore.EconomicsDfValue: economics_df,
                       GlossaryCore.EnergyInvestmentsValue: energy_investment,
                       GlossaryCore.EnergyInvestmentsWoRenewableValue: energy_investment_wo_renewable,
                       GlossaryCore.SectorGdpDf['var_name']: sector_gdp_df,
                       'pc_consumption_constraint': pc_consumption_constraint,
                       GlossaryCore.WorkforceDfValue: workforce_df,
                       'delta_capital_objective': self.macro_model.delta_capital_objective,
                       'delta_capital_objective_wo_exp_min': self.macro_model.delta_capital_objective_wo_exp_min,
                       GlossaryCore.CapitalDfValue: capital_df,
                       'emax_enet_constraint': emax_enet_constraint,
                       'delta_capital_objective_weighted': self.macro_model.delta_capital_objective_with_alpha,
                       'delta_capital_constraint': self.macro_model.delta_capital_cons,
                       'delta_capital_constraint_dc': self.macro_model.delta_capital_cons_dc, #todo: useless, to remove
                       'delta_capital_lintoquad': self.macro_model.delta_capital_lintoquad,
                       }

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable

        """

        inputs_dict = deepcopy(self.get_sosdisc_inputs())
        outputs_dict = deepcopy(self.get_sosdisc_outputs())

        scaling_factor_energy_production, ref_pc_consumption_constraint, ref_emax_enet_constraint, usable_capital_ref_raw, capital_ratio, alpha = [
            inputs_dict[key] for key in [
                'scaling_factor_energy_production', 'ref_pc_consumption_constraint',
                'ref_emax_enet_constraint', 'usable_capital_ref', 'capital_utilisation_ratio', 'alpha']]

        year_start = inputs_dict[GlossaryCore.YearStart]
        year_end = inputs_dict[GlossaryCore.YearEnd]
        time_step = inputs_dict[GlossaryCore.TimeStep]
        nb_years = len(np.arange(year_start, year_end + 1, time_step))
        usable_capital_ref = usable_capital_ref_raw * nb_years
        capital_df, delta_capital_objective_wo_exp_min = [outputs_dict[key] for key in [
            GlossaryCore.CapitalDfValue, 'delta_capital_objective_wo_exp_min']]
        npzeros = np.zeros(
            (self.macro_model.nb_years, self.macro_model.nb_years))

        # Compute gradient for coupling variable co2_emissions_Gt
        d_energy_invest_d_co2_emissions, d_investment_d_co2_emissions = self.macro_model.d_investment_d_co2emissions()
        d_consumption_d_co2_emissions = self.macro_model.d_consumption_d_user_input(
            npzeros, d_investment_d_co2_emissions)
        d_consumption_pc_d_co2_emissions = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_co2_emissions)
        d_capital_d_co2_emissions = self.macro_model.d_capital_d_user_input(npzeros)
        d_emax_constraint_d_co2_emissions = self.macro_model.d_emax_constraint_d_user_input(d_capital_d_co2_emissions)
        ddelta_capital_objective_dco2_emissions = (
                                                          capital_ratio * d_capital_d_co2_emissions / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)

        ne_capital = self.macro_model.capital_df['non_energy_capital'].values
        usable_capital = self.macro_model.capital_df[GlossaryCore.UsableCapital].values
        ref_usable_capital = self.macro_model.usable_capital_ref * self.macro_model.nb_years
        delta_capital_cons_limit = self.macro_model.delta_capital_cons_limit
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = ddelta_capital_cons_dc_dusable_capital * d_capital_d_co2_emissions
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            d_energy_invest_d_co2_emissions * 10.)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), d_consumption_pc_d_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            - d_consumption_pc_d_co2_emissions / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), d_emax_constraint_d_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            ddelta_capital_objective_dco2_emissions)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            alpha * ddelta_capital_objective_dco2_emissions)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), ddelta_capital_cons_dc)

        ddelta_capital_objective_dco2_emissions = (
                                                          capital_ratio * d_capital_d_co2_emissions / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            ddelta_capital_objective_dco2_emissions * usable_capital_ref,
            delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            -ddelta_capital_cons / usable_capital_ref)

        # Compute gradient for coupling variable Total production
        d_usable_capital_d_energy = self.macro_model.d_usable_capital_d_energy()
        d_gross_output_d_energy = self.macro_model.d_gross_output_d_energy()
        d_net_output_d_energy = self.macro_model.d_net_output_d_user_input(d_gross_output_d_energy)
        d_energy_investment_d_energy, d_investment_d_energy, d_non_energy_investment_d_energy = self.macro_model.d_investment_d_user_input(d_net_output_d_energy)
        d_consumption_d_energy = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_energy, d_investment_d_energy)
        d_consumption_pc_d_energy = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_energy)
        d_capital_d_energy = self.macro_model.d_capital_d_user_input(d_non_energy_investment_d_energy)
        d_emax_constraint_d_energy = self.macro_model.d_emax_constraint_d_user_input(d_capital_d_energy)
        ddelta_capital_objective_denergy_production = (scaling_factor_energy_production * (
                capital_ratio * d_capital_d_energy - np.identity(
            nb_years) * capital_ratio * capital_df[
                    GlossaryCore.EnergyEfficiency].values / 1000) / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)

        ddelta_capital_cons = self.compute_ddelta_capital_cons(scaling_factor_energy_production * (
                capital_ratio * d_capital_d_energy - np.identity(nb_years) * capital_ratio * capital_df[
            GlossaryCore.EnergyEfficiency].values / 1000),
                                                               delta_capital_objective_wo_exp_min * usable_capital_ref)  # if compute_gdp else npzeros
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(ddelta_capital_cons_dc_dusable_capital, (
                d_usable_capital_d_energy - d_capital_d_energy * capital_ratio)) * scaling_factor_energy_production
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
                d_usable_capital_d_energy - d_capital_d_energy * capital_ratio)) +
                                    np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * d_capital_d_energy)) * \
                                   scaling_factor_energy_production

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            ('energy_production', GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * d_gross_output_d_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), ('energy_production', GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * d_net_output_d_energy) # todo : false when damage ?
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), ('energy_production', GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * d_consumption_pc_d_energy) # todo : false when damage ?
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('energy_production',
                                             GlossaryCore.TotalProductionValue), - scaling_factor_energy_production
                                                                  * d_consumption_pc_d_energy / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), ('energy_production', GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * d_energy_investment_d_energy)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('energy_production', GlossaryCore.TotalProductionValue),
            - scaling_factor_energy_production * (np.identity(nb_years) / ref_emax_enet_constraint - d_emax_constraint_d_energy))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('energy_production', GlossaryCore.TotalProductionValue),
            ddelta_capital_objective_denergy_production)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',
             ), ('energy_production', GlossaryCore.TotalProductionValue),
            alpha * ddelta_capital_objective_denergy_production)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',), ('energy_production', GlossaryCore.TotalProductionValue),
            - ddelta_capital_cons / usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('energy_production', GlossaryCore.TotalProductionValue), ddelta_capital_cons_dc)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('energy_production', GlossaryCore.TotalProductionValue), ddelta_capital_lintoquad)

        # Compute gradient for coupling variable damage (column damage frac output)
        d_gross_output_d_damage_frac_output = self.macro_model.d_gross_output_d_damage_frac_output()
        d_net_output_d_damage_frac_output = self.macro_model.d_net_output_d_damage_frac_output(d_gross_output_d_damage_frac_output)
        (denergy_investment_d_damage_frac_output,
         d_investment_d_damage_frac_output,
         d_non_energy_investment_d_damage_frac_output) = self.macro_model.d_investment_d_user_input(
            d_net_output_d_damage_frac_output)

        d_consumption_d_damage_frac_output = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_damage_frac_output, d_investment_d_damage_frac_output)
        d_consumption_pc_d_damage_frac_output = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_damage_frac_output)
        d_capital_d_damage_frac_output = self.macro_model.d_capital_d_user_input(d_non_energy_investment_d_damage_frac_output)
        d_emax_constraint_d_damage_frac_output = self.macro_model.d_emax_constraint_d_user_input(d_capital_d_damage_frac_output)
        ddelta_capital_objective_ddamage_df = (
                                                      capital_ratio * d_capital_d_damage_frac_output / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min,
            1e-15)

        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            capital_ratio * d_capital_d_damage_frac_output, delta_capital_objective_wo_exp_min * usable_capital_ref)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), d_gross_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), d_net_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), d_consumption_pc_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            - d_consumption_pc_d_damage_frac_output / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            denergy_investment_d_damage_frac_output / 1e3)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), d_emax_constraint_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), ddelta_capital_objective_ddamage_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            alpha * ddelta_capital_objective_ddamage_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -d_capital_d_damage_frac_output * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
                - d_capital_d_damage_frac_output * capital_ratio)) +
                                    np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * d_capital_d_damage_frac_output))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), ddelta_capital_lintoquad)

        # Compute gradients wrt population_df
        d_consumption_pc_d_population = self.macro_model.d_consumption_pc_d_population()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue), d_consumption_pc_d_population)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            - d_consumption_pc_d_population / ref_pc_consumption_constraint)

        # Compute gradients with respect to working age population
        d_workforce_d_working_age_population = self.macro_model.d_workforce_d_workagepop()
        d_gross_output_d_working_age_population = self.macro_model.d_gross_output_d_working_pop()
        d_net_output_d_work_age_population = self.macro_model.d_net_output_d_user_input(d_gross_output_d_working_age_population)
        d_energy_investment_d_working_age_population, \
        d_investment_d_working_age_population, \
        d_non_energy_investment_d_working_age_population = self.macro_model.d_investment_d_user_input(
            d_net_output_d_work_age_population)
        d_consumption_d_working_age_population = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_work_age_population, d_investment_d_working_age_population)
        d_consumption_pc_d_working_age_population = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_working_age_population)
        dcapital = self.macro_model.d_capital_d_user_input(d_non_energy_investment_d_working_age_population)
        demaxconstraint = self.macro_model.d_emax_constraint_d_user_input(dcapital)

        ddelta_capital_objective_dworking_age_pop_df = (np.dot(
            capital_ratio * dcapital, d_workforce_d_working_age_population) / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)

        ddelta_capital_cons = self.compute_ddelta_capital_cons(np.dot(
            capital_ratio * dcapital, d_workforce_d_working_age_population),
            delta_capital_objective_wo_exp_min * usable_capital_ref)

        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(ddelta_capital_cons_dc_dusable_capital, np.dot(
            dcapital * capital_ratio, d_workforce_d_working_age_population))
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = \
            np.dot(ddelta_capital_lintoquad_dusable_capital,
                   np.dot(dcapital * capital_ratio, d_workforce_d_working_age_population)) - \
            np.dot(ddelta_capital_lintoquad_dtolerable_delta,
                   np.dot(0.15 * dcapital, d_workforce_d_working_age_population))

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkforceDfValue, 'workforce'), ('working_age_population_df', 'population_1570'), d_workforce_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), ('working_age_population_df', 'population_1570'),
            d_workforce_d_working_age_population * d_gross_output_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), ('working_age_population_df', 'population_1570'),
            d_workforce_d_working_age_population * d_net_output_d_work_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), ('working_age_population_df', 'population_1570'),
            d_workforce_d_working_age_population * d_energy_investment_d_working_age_population / 1e3)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), ('working_age_population_df', 'population_1570'),
            d_workforce_d_working_age_population * d_consumption_pc_d_working_age_population)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('working_age_population_df', 'population_1570'),
            - d_consumption_pc_d_working_age_population / ref_pc_consumption_constraint * d_workforce_d_working_age_population)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('working_age_population_df', 'population_1570'),
            np.dot(demaxconstraint, d_workforce_d_working_age_population))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('working_age_population_df', 'population_1570'),
            ddelta_capital_objective_dworking_age_pop_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('working_age_population_df', 'population_1570'),
            alpha * ddelta_capital_objective_dworking_age_pop_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',
             ), ('working_age_population_df', 'population_1570'),
            - ddelta_capital_cons / usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('working_age_population_df', 'population_1570'),
            - ddelta_capital_cons_dc)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('working_age_population_df', 'population_1570'), -ddelta_capital_lintoquad)

        # Compute gradients with respect to energy_investment
        d_investment_d_energy_investment_wo_tax, d_energy_investment_d_energy_investment_wo_tax,\
        _, d_energy_investment_wo_renewable_d_energy_investment_wo_tax = \
            self.macro_model.d_investment_d_energy_investment_wo_tax()

        d_net_output_d_energy_invest = self.macro_model.d_net_output_d_energy_invest()
        d_consumption_d_energy_invest = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_energy_invest, d_investment_d_energy_investment_wo_tax)
        dconsumption_pc = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_energy_invest)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue),
            d_energy_investment_d_energy_investment_wo_tax * 10)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsWoRenewableValue, GlossaryCore.EnergyInvestmentsWoRenewableValue),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue),
            d_energy_investment_wo_renewable_d_energy_investment_wo_tax)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue),
            - dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)

        # Compute gradient CO2 Taxes
        d_energy_investment_d_co2_tax = self.macro_model.d_energy_investment_d_co2_tax()
        d_investment_d_co2_tax = d_energy_investment_d_co2_tax
        d_net_output_d_co2_tax = np.zeros((nb_years, nb_years))
        d_consumption_d_co2_tax = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_co2_tax, d_investment_d_co2_tax)
        d_consumption_pc_d_co2_tax = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_co2_tax)
        d_capital_d_co2_tax = self.macro_model.d_capital_d_user_input(npzeros)
        d_emax_constraint_d_co2_tax = self.macro_model.d_emax_constraint_d_user_input(d_capital_d_co2_tax)
        d_delta_capital_objective_dco2_tax = (
                                                     capital_ratio * d_capital_d_co2_tax / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)
        d_delta_capital_cons_d_co2_tax = self.compute_ddelta_capital_cons(
            d_delta_capital_objective_dco2_tax * usable_capital_ref,
            delta_capital_objective_wo_exp_min * usable_capital_ref)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax),
            d_energy_investment_d_co2_tax * 10.)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), d_consumption_pc_d_co2_tax)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), - d_consumption_pc_d_co2_tax / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), d_emax_constraint_d_co2_tax)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), d_delta_capital_objective_dco2_tax)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), alpha * d_delta_capital_objective_dco2_tax)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), - d_delta_capital_cons_d_co2_tax / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -d_capital_d_co2_tax * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), ddelta_capital_cons_dc)

        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
                - d_capital_d_co2_tax * capital_ratio)) +
                                    np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * d_capital_d_co2_tax))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), ddelta_capital_lintoquad)

        # Compute gradient WRT share investment non energy
        d_investment_d_share_investment_non_energy, d_non_energy_invest_d_share_investment_non_energy =\
            self.macro_model.d_investment_d_share_investment_non_energy()
        d_net_output_d_share_investment_non_energy = np.zeros((nb_years, nb_years))
        d_consumption_d_share_investment_non_energy = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_share_investment_non_energy, d_investment_d_share_investment_non_energy)
        dconsumption_pc_d_share_investment_non_energy = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_share_investment_non_energy)
        d_capital_d_share_investment_non_energy = self.macro_model.d_capital_d_user_input(d_non_energy_invest_d_share_investment_non_energy)
        demaxconstraint = self.macro_model.d_emax_constraint_d_user_input(d_capital_d_share_investment_non_energy)
        ddelta_capital_objective_d_share_investment_non_energy = (
                                                         capital_ratio * d_capital_d_share_investment_non_energy / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)

        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            capital_ratio * d_capital_d_share_investment_non_energy, delta_capital_objective_wo_exp_min * usable_capital_ref)

        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -d_capital_d_share_investment_non_energy * capital_ratio)

        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
                - d_capital_d_share_investment_non_energy * capital_ratio)) +
                                    np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * d_capital_d_share_investment_non_energy))

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue), dconsumption_pc_d_share_investment_non_energy)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            - dconsumption_pc_d_share_investment_non_energy / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue), demaxconstraint)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            ddelta_capital_objective_d_share_investment_non_energy)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            alpha * ddelta_capital_objective_d_share_investment_non_energy)

        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',
             ),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            - ddelta_capital_cons / usable_capital_ref)

        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
             ddelta_capital_cons_dc)

        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            ddelta_capital_cons_dc)

        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            ddelta_capital_lintoquad)

    def compute_ddelta_capital_cons(self, ddelta, delta_wo_exp_min):
        """
        Compute ddelta capital constraint
        """

        return ddelta * delta_wo_exp_min * np.sign(delta_wo_exp_min) * compute_dfunc_with_exp_min(delta_wo_exp_min ** 2,
                                                                                                  1e-15) / np.sqrt(
            compute_func_with_exp_min(
                delta_wo_exp_min ** 2, 1e-15))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['output of damage',
                      'gross output and gross output bis',
                      GlossaryCore.EnergyInvestmentsValue,
                      GlossaryCore.InvestmentsValue,
                      GlossaryCore.EnergyInvestmentsWoTaxValue,
                      GlossaryCore.Consumption,
                      'Output growth rate',
                      'energy supply',
                      'usable capital',
                      # 'energy to sustain capital', # TODO: wip on post-pro
                      GlossaryCore.Capital,
                      'employment_rate',
                      'workforce',
                      GlossaryCore.Productivity,
                      'energy efficiency',
                      GlossaryCore.Emax,
                      GlossaryCore.SectorGdpPart,
                      ]
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        economics_detail_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDetail_df['var_name']))
        co2_invest_limit, capital_utilisation_ratio = deepcopy(
            self.get_sosdisc_inputs(['co2_invest_limit', 'capital_utilisation_ratio']))
        workforce_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.WorkforceDfValue))
        sector_gdp_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.SectorGdpDf['var_name']))
        economics_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDf['var_name']))
        sectors_list = deepcopy(
            self.get_sosdisc_inputs(GlossaryCore.SectorsList['var_name']))

        if 'output of damage' in chart_list:

            to_plot = [GlossaryCore.GrossOutput, GlossaryCore.OutputNetOfDamage]

            legend = {GlossaryCore.GrossOutput: 'world gross output',
                      GlossaryCore.OutputNetOfDamage: 'world output net of damage'}

            years = list(economics_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    economics_detail_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Economics output (Power Purchase Parity)'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.InvestmentsValue in chart_list:

            to_plot = [GlossaryCore.EnergyInvestmentsValue,
                       GlossaryCore.NonEnergyInvestmentsValue]

            legend = {GlossaryCore.InvestmentsValue: 'Total investments',
                      GlossaryCore.EnergyInvestmentsValue: 'Energy',
                      GlossaryCore.NonEnergyInvestmentsValue: 'Non-energy sectors',}

            years = list(economics_detail_df.index)

            chart_name = 'Breakdown of Investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'investment [trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, list(economics_detail_df[key]), legend[key], InstanciatedSeries.BAR_DISPLAY, visible_line)

                new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_detail_df[GlossaryCore.InvestmentsValue]),
                legend[GlossaryCore.InvestmentsValue],
                'lines', True)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyInvestmentsValue in chart_list:

            to_plot = [GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsFromTaxValue]

            legend = {GlossaryCore.EnergyInvestmentsWoTaxValue: 'Base invest from macroeconomics (without taxes)',
                      GlossaryCore.EnergyInvestmentsFromTaxValue: 'Added invests from CO2 taxes'}

            years = list(economics_detail_df.index)

            chart_name = 'Breakdown of energy investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'investment [trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, list(economics_detail_df[key]), legend[key], InstanciatedSeries.BAR_DISPLAY, visible_line)

                new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_detail_df[GlossaryCore.EnergyInvestmentsValue]),
                'Energy investments',
                'lines', True)

            new_chart.series.append(new_series)

            """
            # CO2 invest Limit
            visible_line = True
            ordonate_data = list(
                economics_detail_df[GlossaryCore.EnergyInvestmentsWoTaxValue] * co2_invest_limit)
            abscisse_data = np.linspace(
                year_start, year_end, len(years))
            new_series = InstanciatedSeries(
                abscisse_data.tolist(), ordonate_data, 'CO2 invest limit: co2_invest_limit * energy_investment_wo_tax',
                'scatter', visible_line)
            
            new_chart.series.append(new_series)
            """
            instanciated_charts.append(new_chart)

        if 'usable capital' in chart_list:
            capital_df = self.get_sosdisc_outputs(GlossaryCore.CapitalDfValue)
            first_serie = capital_df['non_energy_capital']
            second_serie = capital_df[GlossaryCore.UsableCapital]
            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values[GlossaryCore.UsableCapital], max_values[GlossaryCore.UsableCapital] = self.get_greataxisrange(
                first_serie)
            min_values[GlossaryCore.Capital], max_values[GlossaryCore.Capital] = self.get_greataxisrange(
                second_serie)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Trillion $2020 PPP',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            note = {'Productive Capital': ' Non energy capital'}
            new_chart.annotation_upper_left = note

            visible_line = True
            ordonate_data = list(first_serie)
            percentage_productive_capital_stock = list(
                first_serie * capital_utilisation_ratio)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Productive Capital Stock', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Usable capital', 'lines', visible_line)
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, percentage_productive_capital_stock,
                f'{capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Capital in chart_list:
            energy_capital_df = self.get_sosdisc_inputs('energy_capital')
            first_serie = capital_df['non_energy_capital']
            second_serie = energy_capital_df['energy_capital']
            third_serie = capital_df[GlossaryCore.Capital]
            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values[GlossaryCore.UsableCapital], max_values[GlossaryCore.UsableCapital] = self.get_greataxisrange(
                first_serie)
            min_values[GlossaryCore.Capital], max_values[GlossaryCore.Capital] = self.get_greataxisrange(
                second_serie)
            min_values['energy_capital'], max_values['energy_capital'] = self.get_greataxisrange(
                third_serie)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Capital stock per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Trillion $2020 PPP',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name, stacked_bar=True)
            visible_line = True
            ordonate_data = list(first_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Non energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.series.append(new_series)
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.series.append(new_series)
            ordonate_data_ter = list(third_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_ter, 'Total capital stock', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Consumption in chart_list:

            to_plot = [GlossaryCore.Consumption]

            years = list(economics_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_detail_df[to_plot])

            chart_name = 'Global consumption over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' global consumption [trillion $2020]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'employment_rate' in chart_list:
            years = list(workforce_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = 0, 1

            chart_name = 'Employment rate'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'employment rate',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(workforce_df['employment_rate'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'employment_rate', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'workforce' in chart_list:
            working_age_pop_df = self.get_sosdisc_inputs(
                'working_age_population_df')
            years = list(workforce_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                working_age_pop_df['population_1570'])

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(workforce_df['workforce'])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)
            ordonate_data_bis = list(working_age_pop_df['population_1570'])
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Working-age population', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = [GlossaryCore.Productivity]

            years = list(economics_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_detail_df[to_plot])

            chart_name = 'Total Factor Productivity'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Factor Productivity [no unit]',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'energy efficiency' in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]

            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                capital_df[to_plot])

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'no unit',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(capital_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Emax in chart_list:
            to_plot = GlossaryCore.Emax
            energy_production = deepcopy(
                self.get_sosdisc_inputs('energy_production'))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production[GlossaryCore.TotalProductionValue] * \
                               scaling_factor_energy_production

            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values[GlossaryCore.Emax], max_values[GlossaryCore.Emax] = self.get_greataxisrange(
                capital_df[to_plot])
            min_values['energy'], max_values['energy'] = self.get_greataxisrange(
                total_production)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'E_max value and Net Energy'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Twh',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value], chart_name)
            visible_line = True

            ordonate_data = list(capital_df[to_plot])
            ordonate_data_enet = list(total_production)

            new_series = InstanciatedSeries(
                years, ordonate_data, 'E_max', 'lines', visible_line)
            note = {
                'E_max': ' maximum energy that capital stock can absorb for production'}
            new_chart.annotation_upper_left = note
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, ordonate_data_enet, 'Net energy', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Energy_supply' in chart_list:
            to_plot = [GlossaryCore.TotalProductionValue]

            legend = {
                GlossaryCore.TotalProductionValue: 'energy supply with oil production from energy pyworld3'}

            # inputs = discipline.get_sosdisc_inputs()
            # energy_production = inputs.pop('energy_production')
            energy_production = deepcopy(
                self.get_sosdisc_inputs('energy_production'))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production[GlossaryCore.TotalProductionValue] * \
                               scaling_factor_energy_production

            data_to_plot_dict = {
                GlossaryCore.TotalProductionValue: total_production}

            years = list(economics_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_detail_df[to_plot])

            chart_name = 'Energy supply'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(data_to_plot_dict[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Output growth rate' in chart_list:

            to_plot = ['output_growth']

            legend = {'output_growth': 'output growth rate from WITNESS'}

            years = list(economics_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_detail_df['output_growth'])

            chart_name = 'Output growth rate over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Output  growth rate',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)


        if GlossaryCore.SectorGdpPart in chart_list:
            to_plot = sectors_list
            legend = {sector: sector for sector in sectors_list}
            # Graph with distribution per sector in absolute value
            legend[GlossaryCore.OutputNetOfDamage] = 'Total GDP net of damage'

            years = list(sector_gdp_df[GlossaryCore.Years])

            chart_name = 'Breakdown of GDP per sector [G$]'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SectorGdpPart,
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                new_series = InstanciatedSeries(
                    years, list(sector_gdp_df[key]), legend[key], InstanciatedSeries.BAR_DISPLAY, visible_line)

                new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, list(economics_df[GlossaryCore.OutputNetOfDamage]),
                legend[GlossaryCore.OutputNetOfDamage],
                'lines', True)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # graph in percentage of GDP
            chart_name = 'Breakdown of GDP per sector [%]'
            total_gdp = economics_df[GlossaryCore.OutputNetOfDamage].values
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, "Contribution [%]",
                                                 chart_name=GlossaryCore.ChartSectorGDPPercentage, stacked_bar=True)

            for sector in sectors_list:
                sector_gdp_part = sector_gdp_df[sector] / total_gdp * 100.
                sector_gdp_part = np.nan_to_num(sector_gdp_part, nan=0.)
                serie = InstanciatedSeries(list(sector_gdp_df[GlossaryCore.Years]), list(sector_gdp_part), sector, 'bar', True)
                new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

        return instanciated_charts
