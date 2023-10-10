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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from os.path import join, isfile
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
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: ClimateEcoDiscipline.YEAR_END_DESC_IN,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'productivity_start': {'type': 'float', 'default': 0.27357, 'user_level': 2, 'unit': '-'},
        GlossaryCore.InitialGrossOutput['var_name']: GlossaryCore.InitialGrossOutput,
        'capital_start_non_energy': {'type': 'float', 'unit': 'T$', 'default': 360.5487346, 'user_level': 2},
        GlossaryCore.DamageDf['var_name']: GlossaryCore.DamageDf,
        GlossaryCore.PopulationDf['var_name']: GlossaryCore.PopulationDf,

        GlossaryCore.WorkingAgePopulationDfValue: {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared',
                                      'namespace': 'ns_witness',
                                      'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                               GlossaryCore.Population1570: ('float', None, False),
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
        GlossaryCore.SectorListValue : GlossaryCore.SectorsList,
        # Lower and upper bounds
        'lo_capital': {'type': 'float', 'unit': 'T$', 'default': 1.0, 'user_level': 3},
        'lo_conso': {'type': 'float', 'unit': 'T$', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        'hi_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 70, 'user_level': 3},

        'damage_to_productivity': {'type': 'bool'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'default': 0.3,
                             'unit': '-', 'user_level': 2},

        GlossaryCore.EnergyInvestmentsWoTaxValue: GlossaryCore.EnergyInvestmentsWoTax,
        GlossaryCore.ShareNonEnergyInvestment['var_name']: GlossaryCore.ShareNonEnergyInvestment,
        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProductionDf,
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'unit': '-', 'user_level': 2},
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
        GlossaryCore.CO2TaxEfficiencyValue: GlossaryCore.CO2TaxEfficiency,
        'co2_invest_limit': {'type': 'float', 'default': 2.0, 'unit': 'factor of energy investment'},
        GlossaryCore.CO2TaxesValue: GlossaryCore.CO2Taxes,
        # Employment rate param
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3, 'unit': '-'},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3, 'unit': '-'},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3, 'unit': '-'},
        'usable_capital_ref': {'type': 'float', 'unit': 'T$', 'default': 0.3, 'user_level': 3,
                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref'},
        'energy_capital': {'type': 'dataframe', 'unit': 'T$', 'visibility': 'Shared', 'namespace': 'ns_witness',
                           'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                    'energy_capital': ('float', None, False), }
                           },
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
    }

    DESC_OUT = {
        GlossaryCore.EconomicsDetailDfValue: GlossaryCore.EconomicsDetailDf,
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        GlossaryCore.EnergyInvestmentsValue: GlossaryCore.EnergyInvestments,
        GlossaryCore.EnergyInvestmentsWoRenewable['var_name']: GlossaryCore.EnergyInvestmentsWoRenewable, # todo : can be deleted
        GlossaryCore.WorkforceDfValue: {'type': GlossaryCore.WorkforceDf['type'], 'unit': GlossaryCore.WorkforceDf['unit']},
        GlossaryCore.CapitalDfValue: {'type': 'dataframe', 'unit': '-',
                                      'dataframe_descriptor':GlossaryCore.CapitalDf['dataframe_descriptor']},
        GlossaryCore.DetailedCapitalDfValue: {'type': 'dataframe', 'unit': '-',
                                              'dataframe_descriptor':GlossaryCore.DetailedCapitalDf['dataframe_descriptor']},
        GlossaryCore.ConstraintLowerBoundUsableCapital: {'type': 'array', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_functions'},
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

            if GlossaryCore.SectorListValue in self.get_data_in():
                sectorlist = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

        if sectorlist is not None:
            sector_gdg_desc = copy.deepcopy(GlossaryCore.SectorGdpDf)  # deepcopy not to modify dataframe_descriptor in Glossary
            for sector in sectorlist:
                sector_gdg_desc['dataframe_descriptor'].update({sector: ('float', [1.e-8, 1e30], True)})
            # make sure the namespaces references are good in case shared namespaces were reassociated
            sector_gdg_desc[SoSWrapp.NS_REFERENCE] = self.get_shared_ns_dict()[sector_gdg_desc[SoSWrapp.NAMESPACE]]
            dynamic_outputs.update({GlossaryCore.SectorGdpDfValue: sector_gdg_desc})

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
                                                       GlossaryCore.CO2TaxEfficiencyValue: CO2_tax_efficiency})

            share_non_energy_investment = pd.DataFrame(
                {GlossaryCore.Years: years,
                 GlossaryCore.ShareNonEnergyInvestmentsValue: [27.0 - 2.6] * len(years)})

            self.set_dynamic_default_values(
                {GlossaryCore.CO2TaxEfficiencyValue: co2_tax_efficiency_default,
                 GlossaryCore.ShareNonEnergyInvestmentsValue: share_non_energy_investment ,})

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.macro_model = MacroEconomics(param)

    def run(self):
        param = self.get_sosdisc_inputs()
        damage_df = param.pop(GlossaryCore.DamageDfValue)
        energy_production = param.pop(GlossaryCore.EnergyProductionValue)
        co2_emissions_Gt = param.pop(GlossaryCore.CO2EmissionsGtValue)
        co2_taxes = param.pop(GlossaryCore.CO2TaxesValue)
        co2_tax_efficiency = param.pop(GlossaryCore.CO2TaxEfficiencyValue)
        co2_invest_limit = param.pop('co2_invest_limit')
        population_df = param.pop(GlossaryCore.PopulationDfValue)
        working_age_population_df = param.pop(GlossaryCore.WorkingAgePopulationDfValue)
        energy_capital_df = param['energy_capital']
        compute_gdp: bool = param['assumptions_dict']['compute_gdp']
        sector_list = param[GlossaryCore.SectorListValue]
        macro_inputs = {GlossaryCore.DamageDfValue: damage_df[[GlossaryCore.Years, GlossaryCore.DamageFractionOutput]],
                        GlossaryCore.EnergyProductionValue: energy_production,
                        GlossaryCore.EnergyInvestmentsWoTaxValue: param[GlossaryCore.EnergyInvestmentsWoTaxValue],
                        GlossaryCore.ShareNonEnergyInvestmentsValue: param[GlossaryCore.ShareNonEnergyInvestmentsValue],
                        GlossaryCore.CO2EmissionsGtValue: co2_emissions_Gt,
                        GlossaryCore.CO2TaxesValue: co2_taxes,
                        GlossaryCore.CO2TaxEfficiencyValue: co2_tax_efficiency,
                        'co2_invest_limit': co2_invest_limit,
                        GlossaryCore.PopulationDfValue: population_df[[GlossaryCore.Years, GlossaryCore.PopulationValue]],
                        GlossaryCore.WorkingAgePopulationDfValue: working_age_population_df[[GlossaryCore.Years, GlossaryCore.Population1570]],
                        'energy_capital_df': energy_capital_df,
                        'compute_gdp': compute_gdp,
                        GlossaryCore.SectorListValue: sector_list
                        }

        if not compute_gdp:
            macro_inputs.update({'gross_output_in': param['gross_output_in']})

        # Model execution
        economics_detail_df, economics_df, energy_investment, energy_investment_wo_renewable, \
            workforce_df, capital_df, sector_gdp_df = \
            self.macro_model.compute(macro_inputs)

        # Store output data
        dict_values = {GlossaryCore.EconomicsDetailDfValue: economics_detail_df,
                       GlossaryCore.EconomicsDfValue: economics_df,
                       GlossaryCore.EnergyInvestmentsValue: energy_investment,
                       GlossaryCore.EnergyInvestmentsWoRenewableValue: energy_investment_wo_renewable,
                       GlossaryCore.SectorGdpDfValue: sector_gdp_df,
                       GlossaryCore.WorkforceDfValue: workforce_df,
                       GlossaryCore.DetailedCapitalDfValue: capital_df,
                       GlossaryCore.CapitalDfValue: capital_df[GlossaryCore.CapitalDf['dataframe_descriptor'].keys()],
                       GlossaryCore.ConstraintLowerBoundUsableCapital: self.macro_model.delta_capital_cons,
                       }

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable

        """

        inputs_dict = deepcopy(self.get_sosdisc_inputs())

        year_start = inputs_dict[GlossaryCore.YearStart]
        year_end = inputs_dict[GlossaryCore.YearEnd]
        time_step = inputs_dict[GlossaryCore.TimeStep]
        nb_years = len(np.arange(year_start, year_end + 1, time_step))
        npzeros = np.zeros((self.macro_model.nb_years, self.macro_model.nb_years))

        # Compute gradient for coupling variable co2_emissions_Gt
        d_energy_invest_d_co2_emissions, d_investment_d_co2_emissions = self.macro_model.d_investment_d_co2emissions()
        d_consumption_d_co2_emissions = self.macro_model.d_consumption_d_user_input(
            npzeros, d_investment_d_co2_emissions)
        d_consumption_pc_d_co2_emissions = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_co2_emissions)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            d_energy_invest_d_co2_emissions * 10.)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions), d_consumption_pc_d_co2_emissions)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            npzeros)

        # Compute gradient for coupling variable Total production
        d_gross_output_d_energy, d_usable_capital_d_energy, d_lower_bound_constraint_dE = self.macro_model.d_Y_Ku_Constraint_d_energy()

        d_net_output_d_energy = self.macro_model.d_net_output_d_user_input(d_gross_output_d_energy)
        d_energy_investment_d_energy, d_investment_d_energy, d_non_energy_investment_d_energy = self.macro_model.d_investment_d_user_input(d_net_output_d_energy)
        d_consumption_d_energy = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_energy, d_investment_d_energy)
        d_consumption_pc_d_energy = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_energy)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CapitalDfValue, GlossaryCore.UsableCapital),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_usable_capital_d_energy
        )
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_gross_output_d_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_net_output_d_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_consumption_pc_d_energy)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_energy_investment_d_energy)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_lower_bound_constraint_dE)

        # Compute gradient for coupling variable damage (column damage frac output)
        d_gross_output_d_damage_frac_output, d_Ku_d_dfo, d_Kne_d_dfo, d_lower_bound_constraint_d_dfo = \
            self.macro_model.d_gross_output_d_damage_frac_output()
        d_net_output_d_damage_frac_output = self.macro_model.d_net_output_d_damage_frac_output(d_gross_output_d_damage_frac_output)
        (denergy_investment_d_damage_frac_output,
         d_investment_d_damage_frac_output,
         d_non_energy_investment_d_damage_frac_output) = self.macro_model.d_investment_d_user_input(
            d_net_output_d_damage_frac_output)

        d_consumption_d_damage_frac_output = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_damage_frac_output, d_investment_d_damage_frac_output)
        d_consumption_pc_d_damage_frac_output = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_damage_frac_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            d_gross_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            d_net_output_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            d_consumption_pc_d_damage_frac_output)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            denergy_investment_d_damage_frac_output / 1e3)  # Invest from T$ to G$
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CapitalDfValue, GlossaryCore.UsableCapital),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            d_Ku_d_dfo)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.DamageDfValue, GlossaryCore.DamageFractionOutput),
            d_lower_bound_constraint_d_dfo)

        # Compute gradients wrt population_df
        d_consumption_pc_d_population = self.macro_model.d_consumption_pc_d_population()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            d_consumption_pc_d_population)

        # Compute gradients with respect to working age population
        d_workforce_d_working_age_population = self.macro_model.d_workforce_d_workagepop()
        d_Ku_d_wap, d_gross_output_d_working_age_population, d_lower_bound_constraint_d_wap = self.macro_model.d_gross_output_d_working_pop()
        d_net_output_d_work_age_population = self.macro_model.d_net_output_d_user_input(d_gross_output_d_working_age_population)
        d_energy_investment_d_working_age_population, \
        d_investment_d_working_age_population, \
        d_non_energy_investment_d_working_age_population = self.macro_model.d_investment_d_user_input(
            d_net_output_d_work_age_population)
        d_consumption_d_working_age_population = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_work_age_population, d_investment_d_working_age_population)
        d_consumption_pc_d_working_age_population = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkforceDfValue, GlossaryCore.Workforce),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_workforce_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CapitalDfValue, GlossaryCore.UsableCapital),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_Ku_d_wap)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_gross_output_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_net_output_d_work_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_consumption_pc_d_working_age_population)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_workforce_d_working_age_population * d_energy_investment_d_working_age_population / 1e3)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            d_lower_bound_constraint_d_wap)


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
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.EnergyInvestmentsWoTaxValue, GlossaryCore.EnergyInvestmentsWoTaxValue), npzeros)

        # Compute gradient CO2 Taxes
        d_energy_investment_d_co2_tax = self.macro_model.d_energy_investment_d_co2_tax()
        d_investment_d_co2_tax = d_energy_investment_d_co2_tax
        d_net_output_d_co2_tax = np.zeros((nb_years, nb_years))
        d_consumption_d_co2_tax = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_co2_tax, d_investment_d_co2_tax)
        d_consumption_pc_d_co2_tax = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_co2_tax)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyInvestmentsValue, GlossaryCore.EnergyInvestmentsValue),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax),
            d_energy_investment_d_co2_tax * 10.)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), d_consumption_pc_d_co2_tax)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.CO2TaxesValue, GlossaryCore.CO2Tax), npzeros)

        # Compute gradient WRT share investment non energy
        d_investment_d_share_investment_non_energy, d_non_energy_invest_d_share_investment_non_energy =\
            self.macro_model.d_investment_d_share_investment_non_energy()
        d_net_output_d_share_investment_non_energy = np.zeros((nb_years, nb_years))
        d_consumption_d_share_investment_non_energy = self.macro_model.d_consumption_d_user_input(
            d_net_output_d_share_investment_non_energy, d_investment_d_share_investment_non_energy)
        dconsumption_pc_d_share_investment_non_energy = self.macro_model.d_consumption_per_capita_d_user_input(
            d_consumption_d_share_investment_non_energy)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EconomicsDfValue, GlossaryCore.PerCapitaConsumption),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            dconsumption_pc_d_share_investment_non_energy)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ConstraintLowerBoundUsableCapital,),
            (GlossaryCore.ShareNonEnergyInvestmentsValue, GlossaryCore.ShareNonEnergyInvestmentsValue),
            npzeros)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.GrossOutput,
                      GlossaryCore.EnergyInvestmentsValue,
                      GlossaryCore.InvestmentsValue,
                      GlossaryCore.EnergyInvestmentsWoTaxValue,
                      GlossaryCore.OutputGrowth,
                      GlossaryCore.UsableCapital,
                      GlossaryCore.EnergyUsage,
                      GlossaryCore.Capital,
                      GlossaryCore.EmploymentRate,
                      GlossaryCore.Workforce,
                      GlossaryCore.Productivity,
                      GlossaryCore.EnergyEfficiency,
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
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue))
        co2_invest_limit, capital_utilisation_ratio = deepcopy(
            self.get_sosdisc_inputs(['co2_invest_limit', 'capital_utilisation_ratio']))
        workforce_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.WorkforceDfValue))
        sector_gdp_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.SectorGdpDfValue))
        economics_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.EconomicsDfValue))
        sectors_list = deepcopy(
            self.get_sosdisc_inputs(GlossaryCore.SectorListValue))

        if GlossaryCore.GrossOutput in chart_list:

            to_plot = [GlossaryCore.InvestmentsValue, GlossaryCore.Consumption, GlossaryCore.Damages]

            legend = {GlossaryCore.InvestmentsValue: 'Investments',
                      GlossaryCore.Consumption: 'Consumption',
                      GlossaryCore.Damages: 'Damages'}

            years = list(economics_detail_df.index)
            chart_name = 'Breakdown of gross output'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '[trillion $2020]',
                                                 chart_name=chart_name, stacked_bar=True)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'bar', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        
        if 'output of damage' in chart_list:

            to_plot = [GlossaryCore.GrossOutput, GlossaryCore.OutputNetOfDamage]

            legend = {GlossaryCore.GrossOutput: 'world gross output',
                      GlossaryCore.OutputNetOfDamage: 'world output net of damage'}

            years = list(economics_detail_df.index)
            chart_name = 'Economics output (Power Purchase Parity)'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'world output [trillion $2020]',
                                                 chart_name=chart_name)

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

        if GlossaryCore.UsableCapital in chart_list:
            capital_df = self.get_sosdisc_outputs(GlossaryCore.DetailedCapitalDfValue)
            first_serie = capital_df[GlossaryCore.NonEnergyCapital]
            second_serie = capital_df[GlossaryCore.UsableCapital]
            years = list(capital_df.index)

            chart_name = 'Productive capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Trillion $2020 PPP',
                                                 chart_name=chart_name)
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
                years, list(capital_df[GlossaryCore.UsableCapitalUnbounded]), 'Unbounded Usable capital', 'lines', visible_line)

            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, percentage_productive_capital_stock,
                f'{capital_utilisation_ratio * 100}% of Productive Capital Stock', 'lines', visible_line)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyUsage in chart_list:
            economics_df = self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue)

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'TWh',
                                                 chart_name=GlossaryCore.EnergyUsage,
                                                 stacked_bar=True)

            to_plot = [GlossaryCore.UsedEnergy, GlossaryCore.UnusedEnergy]
            for p in to_plot:
                new_series = InstanciatedSeries(
                    list(economics_df[GlossaryCore.Years]),
                    list(economics_df[p]),
                    p, 'bar', True)
                new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                list(economics_df[GlossaryCore.Years]),
                list(economics_df[GlossaryCore.OptimalEnergyProduction]),
                GlossaryCore.OptimalEnergyProduction, 'lines', True)
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.Capital in chart_list:
            energy_capital_df = self.get_sosdisc_inputs('energy_capital')
            capital_df = self.get_sosdisc_outputs(GlossaryCore.DetailedCapitalDfValue)
            first_serie = capital_df[GlossaryCore.NonEnergyCapital]
            second_serie = energy_capital_df['energy_capital']
            third_serie = capital_df[GlossaryCore.Capital]
            years = list(capital_df.index)

            chart_name = 'Capital stock per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Trillion $2020 PPP',
                                                 chart_name=chart_name, stacked_bar=True)
            visible_line = True
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.series.append(new_series)

            ordonate_data = list(first_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Non energy capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.series.append(new_series)

            ordonate_data_ter = list(third_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_ter, 'Total capital stock', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.EmploymentRate in chart_list:
            years = list(workforce_df.index)

            chart_name = 'Employment rate'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'employment rate',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(workforce_df[GlossaryCore.EmploymentRate])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.EmploymentRate, 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Workforce in chart_list:
            working_age_pop_df = self.get_sosdisc_inputs(
                GlossaryCore.WorkingAgePopulationDfValue)
            years = list(workforce_df.index)

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(workforce_df[GlossaryCore.Workforce])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)
            ordonate_data_bis = list(working_age_pop_df[GlossaryCore.Population1570])
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Working-age population', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if GlossaryCore.Productivity in chart_list:

            to_plot = [GlossaryCore.Productivity]

            years = list(economics_detail_df.index)

            chart_name = 'Total Factor Productivity'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Total Factor Productivity [no unit]',
                                                 chart_name=chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(economics_detail_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyEfficiency in chart_list:

            to_plot = [GlossaryCore.EnergyEfficiency]

            years = list(capital_df.index)

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'no unit',
                                                 chart_name=chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(capital_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)


        if GlossaryCore.OutputGrowth in chart_list:
            to_plot = [GlossaryCore.OutputGrowth]
            legend = {GlossaryCore.OutputGrowth: 'output growth rate from WITNESS'}
            years = list(economics_detail_df.index)
            chart_name = 'Output growth rate over the years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Output  growth rate',
                                                 chart_name=chart_name)

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
