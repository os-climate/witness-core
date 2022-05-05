'''
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
'''
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.macroeconomics_model_v1 import MacroEconomics
from sos_trades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from copy import deepcopy
from sos_trades_core.tools.base_functions.exp_min import compute_func_with_exp_min
from sos_trades_core.tools.cst_manager.constraint_manager import compute_delta_constraint, compute_ddelta_constraint


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
        'damage_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'G$'},
        'year_start': {'type': 'int', 'default': 2020,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'productivity_start': {'type': 'float', 'default': 0.27357, 'user_level': 2},
        'init_gross_output': {'type': 'float', 'unit': 'trillions $', 'visibility': 'Shared', 'default': 130.187,
                              'namespace': 'ns_witness', 'user_level': 2},
        'capital_start_non_energy': {'type': 'float', 'unit': 'trillions $', 'default': 360.5487346, 'user_level': 2},
        'population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'working_age_population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'productivity_gr_start': {'type': 'float', 'default': 0.004781, 'user_level': 2},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02387787, 'user_level': 3},
        # Usable capital
        'capital_utilisation_ratio':  {'type': 'float', 'default': 0.8, 'user_level': 3},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.95, 'user_level': 3},
        'energy_eff_k':  {'type': 'float', 'default': 0.05085, 'user_level': 3},
        'energy_eff_cst': {'type': 'float', 'default': 0.9835, 'user_level': 3},
        'energy_eff_xzero': {'type': 'float', 'default': 2012.8327, 'user_level': 3},
        'energy_eff_max': {'type': 'float', 'default': 3.5165, 'user_level': 3},
        # Production function param
        'output_alpha': {'type': 'float', 'default': 0.86537, 'user_level': 2},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2},
        'depreciation_capital': {'type': 'float', 'default': 0.07, 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'default': 0.0, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        # Lower and upper bounds
        'lo_capital': {'type': 'float', 'unit': 'trillions $', 'default': 1.0, 'user_level': 3},
        'lo_conso': {'type': 'float', 'unit': 'trillions $', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        'hi_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 70, 'user_level': 3},
        'ref_pc_consumption_constraint': {'type': 'float', 'unit': 'k$', 'default': 1, 'user_level': 3, 'namespace': 'ns_ref'},
        'damage_to_productivity': {'type': 'bool'},
        'frac_damage_prod': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'default': 0.3, 'user_level': 2},
        'total_investment_share_of_gdp': {'type': 'dataframe', 'unit': '%', 'dataframe_descriptor': {'years': ('float', None, False),
                                                                                                     'share_investment': ('float', None, True)}, 'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'share_energy_investment': {'type': 'dataframe', 'unit': '%', 'dataframe_descriptor': {'years': ('float', None, False),
                                                                                               'share_investment': ('float', None, True)}, 'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        # energy_production stored in PetaWh for coupling variables scaling
        'energy_production': {'type': 'dataframe', 'visibility': 'Shared', 'unit': 'PWh', 'namespace': 'ns_energy_mix'},
        'scaling_factor_energy_production': {'type': 'float', 'default': 1e3, 'user_level': 2, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'scaling_factor_energy_investment': {'type': 'float', 'default': 1e2, 'user_level': 2, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness',
                  'user_level': 1},
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'user_level': 2},
        'co2_emissions_Gt': {'type': 'dataframe', 'visibility': 'Shared',
                             'namespace': 'ns_energy_mix', 'unit': 'Gt'},
        'CO2_tax_efficiency': {'type': 'dataframe', 'unit': '%'},
        'co2_invest_limit': {'type': 'float', 'default': 2.0},
        'CO2_taxes': {'type': 'dataframe', 'unit': '$/tCO2', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        # Employment rate param
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3},
        'ref_emax_enet_constraint': {'type': 'float', 'default': 60e3, 'user_level': 3, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'usable_capital_ref': {'type': 'float', 'unit': 'G$', 'default': 0.3, 'user_level': 3, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'energy_capital': {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'delta_capital_cons_limit': {'type': 'float', 'unit': 'G$', 'default': 50, 'user_level': 3, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
    }

    DESC_OUT = {
        'economics_detail_df': {'type': 'dataframe'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        # since scaling_factor_energy_investment is 100
        'energy_investment': {'type': 'dataframe', 'visibility': 'Shared', 'unit': '100G$', 'namespace': 'ns_witness'},
        'energy_investment_wo_renewable': {'type': 'dataframe', 'unit': '100G$'},
        'global_investment_constraint': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'pc_consumption_constraint': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        'workforce_df':  {'type': 'dataframe'},
        'emax_enet_constraint':  {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        'delta_capital_objective': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_functions'},
        'delta_capital_objective_weighted': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                             'namespace': 'ns_functions'},
        'delta_capital_objective_wo_exp_min': {'type': 'array'},
        'capital_df':  {'type': 'dataframe'},
        'delta_capital_constraint': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_functions'},
        'delta_capital_constraint_dc': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                        'namespace': 'ns_functions'},
        'delta_capital_lintoquad': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'namespace': 'ns_functions'}
    }

    def setup_sos_disciplines(self):

        self.update_default_with_years()

    def update_default_with_years(self):
        '''
        Update all default dataframes with years 
        '''
        if 'year_start' in self._data_in:
            year_start, year_end = self.get_sosdisc_inputs(
                ['year_start', 'year_end'])
            years = np.arange(year_start, year_end + 1)
            intermediate_point = 30
            CO2_tax_efficiency = np.concatenate(
                (np.linspace(30, intermediate_point, 15), np.asarray([intermediate_point] * (len(years) - 15))))

            co2_tax_efficiency_default = pd.DataFrame({'years': years,
                                                       'CO2_tax_efficiency': CO2_tax_efficiency})

            share_energy_investment = pd.DataFrame(
                {'years': years, 'share_investment': np.ones(len(years)) * 1.65}, index=years)
            total_investment_share_of_gdp = pd.DataFrame(
                {'years': years, 'share_investment': np.ones(len(years)) * 27.0}, index=years)

            self.set_dynamic_default_values(
                {'CO2_tax_efficiency': co2_tax_efficiency_default,
                 'share_energy_investment': share_energy_investment,
                 'total_investment_share_of_gdp': total_investment_share_of_gdp})

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.macro_model = MacroEconomics(param)

    def run(self):
        # Get inputs
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        damage_df = param.pop('damage_df')
        energy_production = param.pop('energy_production')
        share_energy_investment = param.pop('share_energy_investment')
        total_investment_share_of_gdp = param.pop(
            'total_investment_share_of_gdp')
        co2_emissions_Gt = param.pop('co2_emissions_Gt')
        co2_taxes = param.pop('CO2_taxes')
        co2_tax_efficiency = param.pop('CO2_tax_efficiency')
        co2_invest_limit = param.pop('co2_invest_limit')
        population_df = param.pop('population_df')
        working_age_population_df = param.pop('working_age_population_df')
        energy_capital_df = param['energy_capital']

        macro_inputs = {'damage_df': damage_df[['years', 'damage_frac_output']],
                        'energy_production': energy_production,
                        'scaling_factor_energy_production': param['scaling_factor_energy_production'],
                        'scaling_factor_energy_investment': param['scaling_factor_energy_investment'],
                        # share energy investment is in %
                        'share_energy_investment': share_energy_investment,
                        'total_investment_share_of_gdp': total_investment_share_of_gdp,
                        'co2_emissions_Gt': co2_emissions_Gt,
                        'CO2_taxes': co2_taxes,
                        'CO2_tax_efficiency': co2_tax_efficiency,
                        'co2_invest_limit': co2_invest_limit,
                        'population_df': population_df[['years', 'population']],
                        'working_age_population_df': working_age_population_df[['years', 'population_1570']],
                        'energy_capital_df': energy_capital_df
                        }
        # Check inputs
        count = len(
            [i for i in list(share_energy_investment['share_investment']) if np.real(i) > 100.0])
        if count > 0:
            print(
                'For at least one year, the share of energy investment is above 100% of total investment')
        # Model execution
        economics_df, energy_investment, global_investment_constraint, energy_investment_wo_renewable, \
            pc_consumption_constraint, workforce_df, capital_df, emax_enet_constraint = \
            self.macro_model.compute(macro_inputs)

        # Store output data
        dict_values = {'economics_detail_df': economics_df,
                       'economics_df': economics_df[['years', 'gross_output', 'pc_consumption', 'output_net_of_d']],
                       'energy_investment': energy_investment,
                       'global_investment_constraint': global_investment_constraint,
                       'energy_investment_wo_renewable': energy_investment_wo_renewable,
                       'pc_consumption_constraint': pc_consumption_constraint,
                       'workforce_df': workforce_df,
                       'emax_enet_constraint': emax_enet_constraint,
                       'delta_capital_objective': self.macro_model.delta_capital_objective,
                       'delta_capital_objective_wo_exp_min': self.macro_model.delta_capital_objective_wo_exp_min,
                       'capital_df': capital_df,
                       'emax_enet_constraint': emax_enet_constraint,
                       'delta_capital_objective_weighted': self.macro_model.delta_capital_objective_with_alpha,
                       'delta_capital_constraint': self.macro_model.delta_capital_cons,
                       'delta_capital_constraint_dc': self.macro_model.delta_capital_cons_dc,
                       'delta_capital_lintoquad': self.macro_model.delta_capital_lintoquad
                       }

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable 

        """

        scaling_factor_energy_production, scaling_factor_energy_investment, ref_pc_consumption_constraint, ref_emax_enet_constraint, usable_capital_ref_raw, capital_ratio, alpha = self.get_sosdisc_inputs(
            ['scaling_factor_energy_production', 'scaling_factor_energy_investment', 'ref_pc_consumption_constraint', 'ref_emax_enet_constraint', 'usable_capital_ref', 'capital_utilisation_ratio', 'alpha'])

        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        time_step = self.get_sosdisc_inputs('time_step')
        nb_years = len(np.arange(year_start, year_end + 1, time_step))
        usable_capital_ref = usable_capital_ref_raw * nb_years
        capital_df, delta_capital_objective_wo_exp_min = self.get_sosdisc_outputs(
            ['capital_df', 'delta_capital_objective_wo_exp_min'])
        npzeros = np.zeros(
            (self.macro_model.nb_years, self.macro_model.nb_years))
        # Compute gradient for coupling variable co2_emissions_Gt
        denergy_invest, dinvestment = self.macro_model.compute_dinvest_dco2emissions()
        dconsumption = self.macro_model.compute_dconsumption(
            npzeros, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        dcapital = self.macro_model.dcapital(npzeros)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        ddelta_capital_objective_dco2_emissions = (
            capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(delta_capital_objective_wo_exp_min, 1e-15)
        ne_capital = self.macro_model.capital_df['non_energy_capital'].values
        usable_capital = self.macro_model.capital_df['usable_capital'].values
        ref_usable_capital = self.macro_model.usable_capital_ref * self.macro_model.nb_years
        delta_capital_cons_limit = self.macro_model.delta_capital_cons_limit
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = ddelta_capital_cons_dc_dusable_capital * dcapital
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'),
            ('co2_emissions_Gt', 'Total CO2 emissions'), denergy_invest / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('co2_emissions_Gt', 'Total CO2 emissions'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('co2_emissions_Gt', 'Total CO2 emissions'), - dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('co2_emissions_Gt', 'Total CO2 emissions'), demaxconstraint)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('co2_emissions_Gt', 'Total CO2 emissions'), ddelta_capital_objective_dco2_emissions)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('co2_emissions_Gt', 'Total CO2 emissions'),  alpha * ddelta_capital_objective_dco2_emissions)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('co2_emissions_Gt', 'Total CO2 emissions'), ddelta_capital_cons_dc)

        ddelta_capital_objective_dco2_emissions = (capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            ddelta_capital_objective_dco2_emissions * usable_capital_ref,
            delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',), ('co2_emissions_Gt', 'Total CO2 emissions'),
            -ddelta_capital_cons / usable_capital_ref)

        # Compute gradient for coupling variable Total production
        dcapitalu_denergy = self.macro_model.dusablecapital_denergy()
        dgross_output = self.macro_model.dgrossoutput_denergy(
            dcapitalu_denergy)
        dnet_output = self.macro_model.dnet_output(dgross_output)
        denergy_investment, dinvestment, dne_investment = self.macro_model.dinvestment(
            dnet_output)
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        dcapital = self.macro_model.dcapital(dne_investment)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        ddelta_capital_objective_denergy_production = (scaling_factor_energy_production * (capital_ratio * dcapital - np.identity(
            nb_years) * capital_ratio * capital_df['energy_efficiency'].values / 1000) / usable_capital_ref) * compute_dfunc_with_exp_min(delta_capital_objective_wo_exp_min, 1e-15)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('energy_production', 'Total production'), scaling_factor_energy_production * dgross_output)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('energy_production', 'Total production'), scaling_factor_energy_production * dnet_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('energy_production', 'Total production'), scaling_factor_energy_production * dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('energy_production',
                                             'Total production'), - scaling_factor_energy_production
            * dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('energy_production', 'Total production'), scaling_factor_energy_production * denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('energy_production', 'Total production'), - scaling_factor_energy_production * (np.identity(nb_years) / ref_emax_enet_constraint - demaxconstraint))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('energy_production', 'Total production'), ddelta_capital_objective_denergy_production)  # e_max = capital*1e3/ (capital_utilisation_ratio * energy_efficiency)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',
             ), ('energy_production', 'Total production'),
            alpha * ddelta_capital_objective_denergy_production)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(scaling_factor_energy_production * (
            capital_ratio * dcapital - np.identity(nb_years) * capital_ratio * capital_df[
                'energy_efficiency'].values / 1000), delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',), ('energy_production', 'Total production'),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(ddelta_capital_cons_dc_dusable_capital, (
            dcapitalu_denergy - dcapital * capital_ratio)) * scaling_factor_energy_production
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('energy_production', 'Total production'), ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
            dcapitalu_denergy - dcapital * capital_ratio)) +
            np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * dcapital)) * \
            scaling_factor_energy_production
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('energy_production', 'Total production'), ddelta_capital_lintoquad)
#        Compute gradient for coupling variable damage
        dproductivity = self.macro_model.compute_dproductivity()
        dgross_output = self.macro_model.dgross_output_ddamage(dproductivity)
        dnet_output = self.macro_model.dnet_output_ddamage(dgross_output)
        denergy_investment, dinvestment, dne_investment = self.macro_model.dinvestment(
            dnet_output)
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        dcapital = self.macro_model.dcapital(dne_investment)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        ddelta_capital_objective_ddamage_df = (capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(delta_capital_objective_wo_exp_min,
                                                                                                                           1e-15)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('damage_df', 'damage_frac_output'), dgross_output)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('damage_df', 'damage_frac_output'), dnet_output)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('damage_df', 'damage_frac_output'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('damage_df', 'damage_frac_output'), - dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('damage_df', 'damage_frac_output'), denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('damage_df', 'damage_frac_output'), demaxconstraint)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('damage_df', 'damage_frac_output'), ddelta_capital_objective_ddamage_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('damage_df', 'damage_frac_output'),  alpha * ddelta_capital_objective_ddamage_df)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            capital_ratio * dcapital, delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',), ('damage_df', 'damage_frac_output'),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -dcapital * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('damage_df', 'damage_frac_output'), ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
            - dcapital * capital_ratio)) +
            np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * dcapital))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('damage_df', 'damage_frac_output'), ddelta_capital_lintoquad)
        # compute gradient for coupling variable population
        dconsumption_pc = self.macro_model.compute_dconsumption_pc_dpopulation()
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('population_df', 'population'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(('pc_consumption_constraint',), ('population_df', 'population'),
                                                    - dconsumption_pc / ref_pc_consumption_constraint)

        # compute gradient for coupling variable working age population
        dworkforce_dworkingagepop = self.macro_model.compute_dworkforce_dworkagepop()
        self.set_partial_derivative_for_other_types(
            ('workforce_df', 'workforce'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop)
        dgross_output = self.macro_model.dgrossoutput_dworkingpop()
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop * dgross_output)
        dnet_output = self.macro_model.dnet_output(dgross_output)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop * dnet_output)
        denergy_investment, dinvestment, dne_investment = self.macro_model.dinvestment(
            dnet_output)
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop * denergy_investment / scaling_factor_energy_investment * 1e3)
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop * dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('working_age_population_df', 'population_1570'), - dconsumption_pc / ref_pc_consumption_constraint * dworkforce_dworkingagepop)
        dcapital = self.macro_model.dcapital(dne_investment)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('working_age_population_df', 'population_1570'), np.dot(demaxconstraint, dworkforce_dworkingagepop))
        ddelta_capital_objective_dworking_age_pop_df = (np.dot(
            capital_ratio * dcapital, dworkforce_dworkingagepop) / usable_capital_ref) * compute_dfunc_with_exp_min(delta_capital_objective_wo_exp_min, 1e-15)

        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('working_age_population_df', 'population_1570'), ddelta_capital_objective_dworking_age_pop_df)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('working_age_population_df', 'population_1570'), alpha * ddelta_capital_objective_dworking_age_pop_df)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(np.dot(
            capital_ratio * dcapital, dworkforce_dworkingagepop), delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',
             ), ('working_age_population_df', 'population_1570'),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(ddelta_capital_cons_dc_dusable_capital, np.dot(
            dcapital * capital_ratio, dworkforce_dworkingagepop))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('working_age_population_df', 'population_1570'), - ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = \
            np.dot(ddelta_capital_lintoquad_dusable_capital, np.dot(dcapital * capital_ratio, dworkforce_dworkingagepop)) - \
            np.dot(ddelta_capital_lintoquad_dtolerable_delta,
                   np.dot(0.15 * dcapital, dworkforce_dworkingagepop))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('working_age_population_df', 'population_1570'), -ddelta_capital_lintoquad)
        # compute gradients for share_energy_investment
        denergy_investment, denergy_investment_wo_renewable = self.macro_model.compute_denergy_investment_dshare_energy_investement()
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('share_energy_investment', 'share_investment'), denergy_investment * 1e3 / scaling_factor_energy_investment)
        self.set_partial_derivative_for_other_types(
            ('energy_investment_wo_renewable', 'energy_investment_wo_renewable'), (
                'share_energy_investment', 'share_investment'),
            denergy_investment_wo_renewable)
        dinvestment, dne_investment = self.macro_model.compute_dinvestment_dshare_energy_investement(
            denergy_investment)
        dnet_output = np.zeros((nb_years, nb_years))
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        dcapital = self.macro_model.dcapital(dne_investment)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        ddelta_capital_objective_dshare_energy = (
            capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(delta_capital_objective_wo_exp_min, 1e-15)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('share_energy_investment', 'share_investment'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',
             ), ('share_energy_investment', 'share_investment'),
            - dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('share_energy_investment', 'share_investment'), demaxconstraint)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('share_energy_investment', 'share_investment'),  ddelta_capital_objective_dshare_energy)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('share_energy_investment', 'share_investment'),  alpha * ddelta_capital_objective_dshare_energy)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            capital_ratio * dcapital, delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',
             ), ('share_energy_investment', 'share_investment'),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -dcapital * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('share_energy_investment', 'share_investment'), ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
            - dcapital * capital_ratio)) +
            np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * dcapital))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('share_energy_investment', 'share_investment'), ddelta_capital_lintoquad)
        # compute gradient CO2 Taxes
        denergy_investment = self.macro_model.compute_denergy_investment_dco2_tax()
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('CO2_taxes', 'CO2_tax'), denergy_investment * 1e3 / scaling_factor_energy_investment)
        dinvestment = denergy_investment
        dnet_output = np.zeros((nb_years, nb_years))
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('CO2_taxes', 'CO2_tax'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('CO2_taxes', 'CO2_tax'), - dconsumption_pc / ref_pc_consumption_constraint)
        dcapital = self.macro_model.dcapital(npzeros)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('CO2_taxes', 'CO2_tax'),  demaxconstraint)
        ddelta_capital_objective_dco2_tax = (capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('CO2_taxes', 'CO2_tax'), ddelta_capital_objective_dco2_tax)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('CO2_taxes', 'CO2_tax'), alpha * ddelta_capital_objective_dco2_tax)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            ddelta_capital_objective_dco2_tax * usable_capital_ref, delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',), ('CO2_taxes', 'CO2_tax'), - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -dcapital * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('CO2_taxes', 'CO2_tax'), ddelta_capital_cons_dc)

        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
            - dcapital * capital_ratio)) +
            np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * dcapital))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('CO2_taxes', 'CO2_tax'), ddelta_capital_lintoquad)
        # compute gradient total_share_investment_gdp
        dinvestment, dne_invest = self.macro_model.compute_dinvestment_dtotal_share_of_gdp()
        dnet_output = np.zeros((nb_years, nb_years))
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        dcapital = self.macro_model.dcapital(dne_invest)
        demaxconstraint = self.macro_model.demaxconstraint(dcapital)
        ddelta_capital_objective_dtotal_invest = (capital_ratio * dcapital / usable_capital_ref) * compute_dfunc_with_exp_min(
            delta_capital_objective_wo_exp_min, 1e-15)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('total_investment_share_of_gdp', 'share_investment'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('total_investment_share_of_gdp', 'share_investment'), - dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('total_investment_share_of_gdp', 'share_investment'),  demaxconstraint)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective',), ('total_investment_share_of_gdp', 'share_investment'),  ddelta_capital_objective_dtotal_invest)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_objective_weighted',), ('total_investment_share_of_gdp', 'share_investment'),  alpha * ddelta_capital_objective_dtotal_invest)
        ddelta_capital_cons = self.compute_ddelta_capital_cons(
            capital_ratio * dcapital, delta_capital_objective_wo_exp_min * usable_capital_ref)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint',
             ), ('total_investment_share_of_gdp', 'share_investment'),
            - ddelta_capital_cons / usable_capital_ref)
        ddelta_capital_cons_dc_dusable_capital, _, _ = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=delta_capital_cons_limit, delta_type='hardmin', reference_value=ref_usable_capital)
        ddelta_capital_cons_dc = np.dot(
            ddelta_capital_cons_dc_dusable_capital, -dcapital * capital_ratio)
        self.set_partial_derivative_for_other_types(
            ('delta_capital_constraint_dc',), ('total_investment_share_of_gdp', 'share_investment'), ddelta_capital_cons_dc)
        ddelta_capital_lintoquad_dusable_capital, _, ddelta_capital_lintoquad_dtolerable_delta = compute_ddelta_constraint(
            value=usable_capital, goal=capital_ratio * ne_capital,
            tolerable_delta=0.15 * ne_capital, delta_type='normal', reference_value=ref_usable_capital)
        ddelta_capital_lintoquad = (np.dot(ddelta_capital_lintoquad_dusable_capital, (
            - dcapital * capital_ratio)) +
            np.dot(ddelta_capital_lintoquad_dtolerable_delta, 0.15 * dcapital))
        self.set_partial_derivative_for_other_types(
            ('delta_capital_lintoquad',), ('total_investment_share_of_gdp', 'share_investment'), ddelta_capital_lintoquad)

    def compute_ddelta_capital_cons(self, ddelta, delta_wo_exp_min):
        '''
        Compute ddelta capital constraint
        '''

        return ddelta * delta_wo_exp_min * np.sign(delta_wo_exp_min) * compute_dfunc_with_exp_min(delta_wo_exp_min ** 2, 1e-15) / np.sqrt(compute_func_with_exp_min(
            delta_wo_exp_min ** 2, 1e-15))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['output of damage', 'gross output and gross output bis',
                      'investment', 'energy_investment', 'consumption',
                      'Output growth rate', 'energy supply',
                      'usable capital', 'capital', 'employment_rate', 'workforce', 'productivity', 'energy efficiency', 'e_max']
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

        economics_df = deepcopy(
            self.get_sosdisc_outputs('economics_detail_df'))
        co2_invest_limit, capital_utilisation_ratio = deepcopy(
            self.get_sosdisc_inputs(['co2_invest_limit', 'capital_utilisation_ratio']))
        workforce_df = deepcopy(
            self.get_sosdisc_outputs('workforce_df'))

        if 'output of damage' in chart_list:

            to_plot = ['gross_output', 'output_net_of_d']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'gross_output': 'world gross output',
                      'output_net_of_d': 'world output net of damage'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Economics output'

            new_chart = TwoAxesInstanciatedChart('years', 'world output [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'investment' in chart_list:

            to_plot = ['investment', 'energy_investment']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'investment': 'total investment capacities',
                      'energy_investment': 'investment capacities in the energy sector'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Total investment capacities and energy investment capacities'

            new_chart = TwoAxesInstanciatedChart('years', 'investment [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'energy_investment' in chart_list:

            to_plot = ['energy_investment',
                       'energy_investment_wo_tax', 'energy_investment_from_tax']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'energy_investment': 'investment capacities in the energy sector',
                      'energy_investment_wo_tax': 'base invest from macroeconomics',
                      'energy_investment_from_tax': 'added invest from CO2 taxes'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values['energy_investment_wo_tax'], max_values['energy_investment_wo_tax'] = self.get_greataxisrange(
                economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())
            # Max value is energy_invest_wo_tax * co2_invest_limit (2 by
            # default)
            if co2_invest_limit >= 1:
                max_value *= co2_invest_limit
            chart_name = 'Breakdown of energy investments'

            new_chart = TwoAxesInstanciatedChart('years', 'investment [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            # CO2 invest Limit
            visible_line = True
            ordonate_data = list(
                economics_df['energy_investment_wo_tax'] * co2_invest_limit)
            abscisse_data = np.linspace(
                year_start, year_end, len(years))
            new_series = InstanciatedSeries(
                abscisse_data.tolist(), ordonate_data, 'CO2 invest limit: co2_invest_limit * energy_investment_wo_tax', 'scatter', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'usable capital' in chart_list:

            capital_df = self.get_sosdisc_outputs('capital_df')
            first_serie = capital_df['non_energy_capital']
            second_serie = capital_df['usable_capital']
            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values['usable_capital'], max_values['usable_capital'] = self.get_greataxisrange(
                first_serie)
            min_values['capital'], max_values['capital'] = self.get_greataxisrange(
                second_serie)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart('years', 'Trillion dollars',
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
                years, percentage_productive_capital_stock, f'{capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'capital' in chart_list:
            energy_capital_df = self.get_sosdisc_inputs('energy_capital')
            first_serie = capital_df['non_energy_capital']
            second_serie = energy_capital_df['energy_capital']
            third_serie = capital_df['capital']
            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values['usable_capital'], max_values['usable_capital'] = self.get_greataxisrange(
                first_serie)
            min_values['capital'], max_values['capital'] = self.get_greataxisrange(
                second_serie)
            min_values['energy_capital'], max_values['energy_capital'] = self.get_greataxisrange(
                third_serie)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Capital stock per year'

            new_chart = TwoAxesInstanciatedChart('years', 'Trillion dollars',
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

        if 'consumption' in chart_list:

            to_plot = ['consumption']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Global consumption over the years'

            new_chart = TwoAxesInstanciatedChart('years', ' global consumption [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

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

            new_chart = TwoAxesInstanciatedChart('years', 'employment rate',
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

            new_chart = TwoAxesInstanciatedChart('years', 'Number of people [million]',
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

        if 'productivity' in chart_list:

            to_plot = ['productivity']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Total Factor Productivity'

            new_chart = TwoAxesInstanciatedChart('years', 'Total Factor Productivity [no unit]',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'energy efficiency' in chart_list:

            to_plot = ['energy_efficiency']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                capital_df[to_plot])

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart('years', 'no unit',
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

        if 'e_max' in chart_list:

            to_plot = 'e_max'
            energy_production = deepcopy(
                self.get_sosdisc_inputs('energy_production'))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production['Total production'] * \
                scaling_factor_energy_production
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values['e_max'], max_values['e_max'] = self.get_greataxisrange(
                capital_df[to_plot])
            min_values['energy'], max_values['energy'] = self.get_greataxisrange(
                total_production)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'E_max value and Net Energy'

            new_chart = TwoAxesInstanciatedChart('years', 'Twh',
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
            to_plot = ['Total production']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {
                'Total production': 'energy supply with oil production from energy model'}

            #inputs = discipline.get_sosdisc_inputs()
            #energy_production = inputs.pop('energy_production')
            energy_production = deepcopy(
                self.get_sosdisc_inputs('energy_production'))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production['Total production'] * \
                scaling_factor_energy_production

            data_to_plot_dict = {
                'Total production': total_production}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Energy supply'

            new_chart = TwoAxesInstanciatedChart('years', 'world output [trillion $]',
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
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'output_growth': 'output growth rate from WITNESS'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df['output_growth'])

            chart_name = 'Output growth rate over the years'

            new_chart = TwoAxesInstanciatedChart('years', ' Output  growth rate',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
