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
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_thesis_narrative.macroeconomics_narrative_energy import MacroEconomics
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from copy import deepcopy
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
        GlossaryCore.DamageDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        GlossaryCore.YearStart: {'type': 'int', 'default': 2020, 'visibility': 'Shared', 'unit': 'year',
                       'namespace': 'ns_witness'},
        GlossaryCore.YearEnd: {'type': 'int', 'default': 2100, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        GlossaryCore.TimeStep: {'type': 'int', 'default': 1, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'productivity_start': {'type': 'float', 'default': 0.974422, 'user_level': 2},
        GlossaryCore.InitialGrossOutput['var_name']: GlossaryCore.InitialGrossOutput,
        'capital_start': {'type': 'float', 'unit': 'trillions $', 'default': 355.9210491, 'user_level': 2},
        GlossaryCore.PopulationDf['var_name']: GlossaryCore.PopulationDf,
        'productivity_gr_start': {'type': 'float', 'default': 0.042925, 'user_level': 2},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02351234, 'user_level': 3},
        'depreciation_capital': {'type': 'float', 'default': 0.08, 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'lo_capital': {'type': 'float', 'unit': 'trillions $', 'default': 1.0, 'user_level': 3},
        'lo_conso': {'type': 'float', 'unit': 'trillions $', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        'hi_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 70, 'user_level': 3},
        'ref_pc_consumption_constraint': {'type': 'float', 'unit': 'k$', 'default': 1, 'user_level': 3,
                                          'namespace': 'ns_ref'},
        'damage_to_productivity': {'type': 'bool'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'default': 0.3,
                             'user_level': 2},
        GlossaryCore.InvestmentShareGDPValue: {'type': 'dataframe', 'unit': '%',
                                          'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                   'share_investment': ('float', None, True)},
                                          'dataframe_edition_locked': False, 'visibility': 'Shared',
                                          'namespace': 'ns_witness'},
        'share_energy_investment': {'type': 'dataframe', 'unit': '%',
                                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                             'share_investment': ('float', None, True)},
                                    'dataframe_edition_locked': False, 'visibility': 'Shared',
                                    'namespace': 'ns_witness'},

        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProductionDf,
        'scaling_factor_energy_production': {'type': 'float', 'default': 1e3, 'user_level': 2, 'visibility': 'Shared',
                                             'namespace': 'ns_witness'},
        'scaling_factor_energy_investment': {'type': 'float', 'default': 1e2, 'user_level': 2, 'visibility': 'Shared',
                                             'namespace': 'ns_witness'},
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'user_level': 2},
        'output_k_exponent': {'type': 'float', 'default': 0.1924566, 'user_level': 3},
        'output_pop_exponent': {'type': 'float', 'default': 0.10810935, 'user_level': 3},
        'output_energy_exponent': {'type': 'float', 'default': 0.24858645, 'user_level': 3},
        'output_energy_share': {'type': 'float', 'default': 0.01, 'user_level': 3},
        'output_exponent': {'type': 'float', 'default': 2.75040992, 'user_level': 3},
        'output_pop_share': {'type': 'float', 'default': 0.29098974, 'user_level': 3},
        'output_alpha': {'type': 'float', 'default': 0.86537, 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        GlossaryCore.WorkingAgePopulationDfValue: {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared',
                                      'namespace': 'ns_witness'},

        'hassler': {'type': 'bool', 'default': False},
        'output_alpha_hassler': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'output_gamma_hassler': {'type': 'float', 'default': 0.046, 'user_level': 2, 'unit': '-'},
        'output_epsilon_hassler': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},

        'pop_factor': {'type': 'float', 'default': 1e-3, 'user_level': 3},
        'energy_factor': {'type': 'float', 'default': 1e-4, 'user_level': 3},
        'decline_rate_energy_productivity': {'type': 'float', 'default': 0.01345699, 'user_level': 3},
        'init_energy_productivity': {'type': 'float', 'default': 3.045177, 'user_level': 2},
        'init_energy_productivity_gr': {'type': 'float', 'default': 0.0065567, 'user_level': 2},
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
        'CO2_tax_efficiency': {'type': 'dataframe', 'unit': '%'},
        'co2_invest_limit': {'type': 'float', 'default': 2.0},
        GlossaryCore.CO2TaxesValue: {'type': 'dataframe', 'unit': '$/tCO2', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        # Employment rate param
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3, 'unit': '-'},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3, 'unit': '-'},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3, 'unit': '-'},

    }

    DESC_OUT = {
        GlossaryCore.EconomicsDetailDfValue: {'type': 'dataframe'},
        GlossaryCore.EconomicsDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        GlossaryCore.EnergyInvestmentsValue: {'type': 'dataframe', 'visibility': 'Shared', 'unit': 'G$', 'namespace': 'ns_witness'},
        GlossaryCore.EnergyInvestmentsWoRenewableValue: {'type': 'dataframe', 'unit': 'G$'},
        'global_investment_constraint': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                         'namespace': 'ns_witness'},
        GlossaryCore.WorkforceDfValue: {'type': GlossaryCore.WorkforceDf['type'], 'unit': GlossaryCore.WorkforceDf['unit']},
        'pc_consumption_constraint': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                      'namespace': 'ns_witness'}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.macro_model = MacroEconomics(param)

    def run(self):
        # Get inputs
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        damage_df = param.pop(GlossaryCore.DamageDfValue)
        energy_production = param.pop(GlossaryCore.EnergyProductionValue)
        share_energy_investment = param.pop('share_energy_investment')
        total_investment_share_of_gdp = param.pop(
            GlossaryCore.InvestmentShareGDPValue)
        co2_emissions_Gt = param.pop(GlossaryCore.CO2EmissionsGtValue)
        co2_taxes = param.pop(GlossaryCore.CO2TaxesValue)
        co2_tax_efficiency = param.pop('CO2_tax_efficiency')
        co2_invest_limit = param.pop('co2_invest_limit')
        population_df = param.pop(GlossaryCore.PopulationDfValue)
        working_age_population_df = param.pop(GlossaryCore.WorkingAgePopulationDfValue)

        macro_inputs = {GlossaryCore.DamageDfValue: damage_df[[GlossaryCore.Years, GlossaryCore.DamageFractionOutput]],
                        GlossaryCore.EnergyProductionValue: energy_production,
                        'scaling_factor_energy_production': param['scaling_factor_energy_production'],
                        'scaling_factor_energy_investment': param['scaling_factor_energy_investment'],
                        # share energy investment is in %
                        'share_energy_investment': share_energy_investment,
                        GlossaryCore.InvestmentShareGDPValue: total_investment_share_of_gdp,
                        GlossaryCore.CO2EmissionsGtValue: co2_emissions_Gt,
                        GlossaryCore.CO2TaxesValue: co2_taxes,
                        'CO2_tax_efficiency': co2_tax_efficiency,
                        'co2_invest_limit': co2_invest_limit,
                        GlossaryCore.PopulationDfValue: population_df[[GlossaryCore.Years, GlossaryCore.PopulationValue]],
                        GlossaryCore.WorkingAgePopulationDfValue: working_age_population_df[[GlossaryCore.Years, GlossaryCore.Population1570]]
                        }
        # Check inputs
        count = len(
            [i for i in list(share_energy_investment['share_investment']) if np.real(i) > 100.0])
        if count > 0:
            print(
                'For at least one year, the share of energy investment is above 100% of total investment')
        # Model execution
        economics_df, workforce_df, energy_investment, global_investment_constraint, energy_investment_wo_renewable, pc_consumption_constraint = \
            self.macro_model.compute(macro_inputs)

        # Store output data
        dict_values = {GlossaryCore.EconomicsDetailDfValue: economics_df,
                       GlossaryCore.EconomicsDfValue: economics_df[[GlossaryCore.Years, GlossaryCore.GrossOutput, GlossaryCore.PerCapitaConsumption, GlossaryCore.OutputNetOfDamage]],
                       GlossaryCore.EnergyInvestmentsValue: energy_investment,
                       GlossaryCore.WorkforceDfValue: workforce_df,
                       'global_investment_constraint': global_investment_constraint,
                       GlossaryCore.EnergyInvestmentsWoRenewableValue: energy_investment_wo_renewable,
                       'pc_consumption_constraint': pc_consumption_constraint}


        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradiant of coupling variable to compute:
        economics_df
          - GlossaryCore.GrossOutput,
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.OutputGrowth
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.OutputNetOfDamage,
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.NetOutput,
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.Consumption
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.PerCapitaConsumption
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'interest_rate'
              - damage_df, damage_frac_output
              - energy_production, Total production
          - GlossaryCore.Productivity,
              - damage_df, damage_frac_output
          - energy_productivity
              - damage_df, damage_frac_output
          - capital
              - damage_df, damage_frac_output
              - energy_production, Total production
          - investment
              - damage_df, damage_frac_output
              - energy_production, Total production
              - total_investment_share_of_gdp
          - energy_investment
              - damage_df, damage_frac_output
              - energy_production, Total production
              - share_energy_investment

        """
        scaling_factor_energy_production = self.get_sosdisc_inputs(
            'scaling_factor_energy_production')
        scaling_factor_energy_investment = self.get_sosdisc_inputs(
            'scaling_factor_energy_investment')
        ref_pc_consumption_constraint = self.get_sosdisc_inputs(
            'ref_pc_consumption_constraint')
        dproductivity = self.macro_model.compute_dproductivity()
        denergy_productivity = self.macro_model.compute_denergy_productivity()
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_damage(
            denergy_productivity, dproductivity)
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d_damage(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), dconsumption_pc)

        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            - dconsumption_pc / ref_pc_consumption_constraint)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput), doutput_net_of_d)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$

        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_denergy_supply()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            - scaling_factor_energy_production \
            * dconsumption_pc / ref_pc_consumption_constraint)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * doutput_net_of_d)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            scaling_factor_energy_production * denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$
        # compute gradient for design variable share_energy_investment
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dshare_energy_investment()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), ('share_energy_investment', 'share_investment'),
            denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), ('share_energy_investment', 'share_investment'), dgross_output / 100)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), ('share_energy_investment', 'share_investment'),
            doutput_net_of_d / scaling_factor_energy_investment)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), ('share_energy_investment', 'share_investment'), dconsumption_pc / 100)

        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('share_energy_investment', 'share_investment'), - dconsumption_pc \
                                                                                             / (
                                                                                                         ref_pc_consumption_constraint * 100))

        # compute gradient for design variable total_investment_share_of_gdp
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dtotal_investment_share_of_gdp()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.InvestmentShareGDPValue, 'share_investment'),
            dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.InvestmentShareGDPValue, 'share_investment'),
            dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), (GlossaryCore.InvestmentShareGDPValue, 'share_investment'), - dconsumption_pc \
                                                                                                   / (
                                                                                                               ref_pc_consumption_constraint * 100))

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.InvestmentShareGDPValue, 'share_investment'),
            doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.InvestmentShareGDPValue, 'share_investment'),
            denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for coupling variable co2_emissions_Gt
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dCO2_emission_gt()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), - dconsumption_pc \
                                                                                         / (
                                                                                                     ref_pc_consumption_constraint * 100))

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for design variable CO2_taxes
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dCO2_taxes()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), - dconsumption_pc \
                                                                      / (ref_pc_consumption_constraint * 100))

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax),
            denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for coupling variable population
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dworkforce()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc_dpopulation(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        # compute gradient for coupling variable working age population
        dworkforce_dworkingagepop = self.macro_model.compute_dworkforce_dworkagepop()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkforceDfValue, GlossaryCore.Workforce), (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570), dworkforce_dworkingagepop)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput), (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570), dgross_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage), (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            dnet_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue), (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            denergy_investment / scaling_factor_energy_investment * 1e3)

        self.set_partial_derivative_for_other_types(
             ('pc_consumption_constraint',), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
             - dconsumption_pc / ref_pc_consumption_constraint)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption), (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue), dconsumption_pc)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['output of damage', 'gross output and gross output bis',
                      GlossaryCore.InvestmentsValue, GlossaryCore.EnergyInvestmentsValue, GlossaryCore.PopulationValue, GlossaryCore.Productivity, GlossaryCore.Consumption,
                      'Output growth rate', 'energy supply', 'energy productivity']
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
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue))
        co2_invest_limit = deepcopy(
            self.get_sosdisc_inputs('co2_invest_limit'))

        if 'output of damage' in chart_list:

            to_plot = [GlossaryCore.GrossOutput, GlossaryCore.NetOutput]

            legend = {GlossaryCore.GrossOutput: 'world gross output',
                      GlossaryCore.NetOutput: 'world output net of damage'}

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

            chart_name = 'Economics output (Power Purchase Parity)'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
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

        if GlossaryCore.InvestmentsValue in chart_list:

            to_plot = [GlossaryCore.InvestmentsValue, GlossaryCore.EnergyInvestmentsValue]

            legend = {GlossaryCore.InvestmentsValue: 'total investment capacities',
                      GlossaryCore.EnergyInvestmentsValue: 'investment capacities in the energy sector'}

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

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'investment (trill $)',
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

        if GlossaryCore.EnergyInvestmentsValue in chart_list:

            to_plot = [GlossaryCore.EnergyInvestmentsValue,
                       GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsFromTaxValue]

            legend = {GlossaryCore.EnergyInvestmentsValue: 'investment capacities in the energy sector',
                      GlossaryCore.EnergyInvestmentsWoTaxValue: 'base invest from macroeconomic',
                      GlossaryCore.EnergyInvestmentsFromTaxValue: 'added invest from CO2 taxes'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values[GlossaryCore.EnergyInvestmentsWoTaxValue], max_values[GlossaryCore.EnergyInvestmentsWoTaxValue] = self.get_greataxisrange(
                economics_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())
            # Max value is energy_invest_wo_tax * co2_invest_limit (2 by
            # default)
            if co2_invest_limit >= 1:
                max_value *= co2_invest_limit
            chart_name = 'Breakdown of energy investments'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'investment (trill $)',
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
                economics_df[GlossaryCore.EnergyInvestmentsWoTaxValue] * co2_invest_limit)
            abscisse_data = np.linspace(
                year_start, year_end, len(years))
            new_series = InstanciatedSeries(
                abscisse_data.tolist(), ordonate_data, 'CO2 invest limit: co2_invest_limit * energy_investment_wo_tax',
                'scatter', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.PopulationValue in chart_list:
            population_df = self.get_sosdisc_inputs(GlossaryCore.PopulationDfValue)

            years = list(population_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                population_df[GlossaryCore.PopulationValue])

            chart_name = 'Population evolution over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' population (million)',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(population_df[GlossaryCore.PopulationValue])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.PopulationValue, 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = [GlossaryCore.Productivity]

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Total Factor Productivity'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Factor Productivity',
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

        if 'energy productivity' in chart_list:

            to_plot = ['energy_productivity']

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Energy Productivity'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'global productivity',
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
        if GlossaryCore.Consumption in chart_list:

            to_plot = [GlossaryCore.Consumption]

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Global consumption over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' global consumption (trill $)',
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

        if 'Energy_supply' in chart_list:
            to_plot = [GlossaryCore.TotalProductionValue]

            legend = {
                GlossaryCore.TotalProductionValue: 'energy supply with oil production from energy pyworld3'}

            # inputs = discipline.get_sosdisc_inputs()
            # energy_production = inputs.pop(GlossaryCore.EnergyProductionValue)
            energy_production = deepcopy(
                self.get_sosdisc_inputs(GlossaryCore.EnergyProductionValue))
            scaling_factor_energy_production = self.get_sosdisc_inputs(
                'scaling_factor_energy_production')
            total_production = energy_production[GlossaryCore.TotalProductionValue] * \
                               scaling_factor_energy_production

            data_to_plot_dict = {
                GlossaryCore.TotalProductionValue: total_production}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Energy supply'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output (trill $)',
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

            to_plot = [GlossaryCore.OutputGrowth]

            legend = {GlossaryCore.OutputGrowth: 'output growth rate from WITNESS'}

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[GlossaryCore.OutputGrowth])

            chart_name = 'Output growth rate over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Output  growth rate',
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
