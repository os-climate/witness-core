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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from copy import deepcopy


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
        'damage_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'year_start': {'type': 'int', 'default': 2020,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'productivity_start': {'type': 'float', 'default':0.27357, 'user_level': 2},
        'init_gross_output': {'type': 'float', 'unit': 'trillions $', 'visibility': 'Shared', 'default': 130.187,
                              'namespace': 'ns_witness', 'user_level': 2},
        'capital_start': {'type': 'float', 'unit': 'trillions $', 'default': 376.6387346, 'user_level': 2},
        'population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'working_age_population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'productivity_gr_start': {'type': 'float', 'default': 0.004781, 'user_level': 2},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02387787, 'user_level': 3},
        #Usable capital
        'capital_utilisation_ratio':  {'type': 'float', 'default': 0.8, 'user_level': 3},
        'energy_eff_k':  {'type': 'float', 'default': 0.05085, 'user_level': 3},
        'energy_eff_cst': {'type': 'float', 'default': 0.9835, 'user_level': 3},
        'energy_eff_xzero' : {'type': 'float', 'default': 2012.8327, 'user_level': 3},
        'energy_eff_max' : {'type': 'float', 'default': 3.5165, 'user_level': 3},
        #Production function param
        'output_alpha': {'type': 'float', 'default': 0.86537, 'user_level': 2},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2},
        'depreciation_capital': {'type': 'float', 'default': 0.07, 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        #Lower and upper bounds 
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
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'user_level': 2},
        'co2_emissions_Gt': {'type': 'dataframe', 'visibility': 'Shared',
                             'namespace': 'ns_energy_mix', 'unit': 'Gt'},
        'CO2_tax_efficiency': {'type': 'dataframe', 'unit': '%'},
        'co2_invest_limit': {'type': 'float', 'default': 2.0},
        'CO2_taxes': {'type': 'dataframe', 'unit': '$/tCO2', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        #Employment rate param 
        'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level' : 3},
        'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3},
        'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3} 
    }


    DESC_OUT = {
        'economics_detail_df': {'type': 'dataframe'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_investment': {'type': 'dataframe', 'visibility': 'Shared', 'unit': 'G$', 'namespace': 'ns_witness'},
        'energy_investment_wo_renewable': {'type': 'dataframe', 'unit': 'G$'},
        'global_investment_constraint': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'pc_consumption_constraint': {'type': 'array', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'workforce_df':  {'type': 'dataframe'}, 
        'usable_capital_df':  {'type': 'dataframe'}
    }

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

        macro_inputs = {'damage_frac_output': damage_df[['years', 'damage_frac_output']],
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
                        'working_age_population_df': working_age_population_df[['years', 'population_1570']]
                        }
        # Check inputs
        count = len(
            [i for i in list(share_energy_investment['share_investment']) if np.real(i) > 100.0])
        if count > 0:
            print(
                'For at least one year, the share of energy investment is above 100% of total investment')
        # Model execution
        economics_df, energy_investment, global_investment_constraint, energy_investment_wo_renewable, pc_consumption_constraint, workforce_df, usable_capital_df = \
            self.macro_model.compute(macro_inputs)

        # Store output data
        dict_values = {'economics_detail_df': economics_df,
                       'economics_df': economics_df[['years', 'gross_output', 'pc_consumption', 'net_output']],
                       'energy_investment': energy_investment,
                       'global_investment_constraint': global_investment_constraint,
                       'energy_investment_wo_renewable': energy_investment_wo_renewable,
                       'pc_consumption_constraint': pc_consumption_constraint, 
                       'workforce_df': workforce_df, 
                       'usable_capital_df' : usable_capital_df}
                       
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable 

        """
        scaling_factor_energy_production = self.get_sosdisc_inputs(
            'scaling_factor_energy_production')
        scaling_factor_energy_investment = self.get_sosdisc_inputs(
            'scaling_factor_energy_investment')
        ref_pc_consumption_constraint = self.get_sosdisc_inputs(
            'ref_pc_consumption_constraint')
        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        time_step = self.get_sosdisc_inputs('time_step')
        nb_years = len(np.arange(year_start, year_end + 1, time_step))
 
#     Compute gradient for coupling variable co2_emissions_Gt
        denergy_invest, dinvestment = self.macro_model.compute_dinvest_dco2emissions()
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), 
            ('co2_emissions_Gt', 'Total CO2 emissions'), denergy_invest / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$
        dconsumption = self.macro_model.compute_dconsumption(np.zeros((self.macro_model.nb_years, self.macro_model.nb_years)), dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
             ('output', 'pc_consumption'), ('co2_emissions_Gt', 'Total CO2 emissions'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
             ('pc_consumption_constraint',), ('co2_emissions_Gt', 'Total CO2 emissions'), - dconsumption_pc / ref_pc_consumption_constraint)

        #Compute gradient for coupling variable Total production
        dcapitalu_denergy = self.macro_model.dusablecapital_denergy()
        dgross_output = self.macro_model.dgrossoutput_denergy(dcapitalu_denergy)
        self.set_partial_derivative_for_other_types(
              ('economics_df', 'gross_output'), ('energy_production', 'Total production'), scaling_factor_energy_production * dgross_output)
        dnet_output = self.macro_model.dnet_output(dgross_output)
        self.set_partial_derivative_for_other_types(
              ('economics_df', 'net_output'), ('energy_production', 'Total production'), scaling_factor_energy_production * dnet_output)
        denergy_investment, dinvestment = self.macro_model.dinvestment(dnet_output)
        dconsumption = self.macro_model.compute_dconsumption(dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'pc_consumption'), ('energy_production', 'Total production'), scaling_factor_energy_production * dconsumption_pc)
        self.set_partial_derivative_for_other_types(
             ('pc_consumption_constraint',), ('energy_production', 'Total production'), - scaling_factor_energy_production \
                 * dconsumption_pc / ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
             ('energy_investment', 'energy_investment'), ('energy_production', 'Total production'), scaling_factor_energy_production * denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$

#        Compute gradient for coupling variable damage
        dproductivity = self.macro_model.compute_dproductivity()
        dgross_output = self.macro_model.dgross_output_ddamage(dproductivity)
        dnet_output = self.macro_model.dnet_output_ddamage(dgross_output)
        denergy_investment, dinvestment = self.macro_model.dinvestment(dnet_output)
        dconsumption = self.macro_model.compute_dconsumption(dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'gross_output'), ('damage_df', 'damage_frac_output'), dgross_output)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'net_output'), ('damage_df', 'damage_frac_output'), dnet_output)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'pc_consumption'), ('damage_df', 'damage_frac_output'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(
             ('pc_consumption_constraint',), ('damage_df', 'damage_frac_output'), - dconsumption_pc/ ref_pc_consumption_constraint)
        self.set_partial_derivative_for_other_types(
             ('energy_investment', 'energy_investment'), ('damage_df', 'damage_frac_output'), denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$

        #compute gradient for coupling variable population
        dconsumption_pc = self.macro_model.compute_dconsumption_pc_dpopulation()
        self.set_partial_derivative_for_other_types(('economics_df', 'pc_consumption'), ('population_df', 'population'), dconsumption_pc)
        self.set_partial_derivative_for_other_types(('pc_consumption_constraint',), ('population_df', 'population'), 
                                                    - dconsumption_pc / ref_pc_consumption_constraint)
  
        #compute gradient for coupling variable working age population
        dworkforce_dworkingagepop = self.macro_model.compute_dworkforce_dworkagepop()
        self.set_partial_derivative_for_other_types(
            ('workforce_df', 'workforce'), ('working_age_population_df', 'population_1570'),dworkforce_dworkingagepop)        
        dgross_output = self.macro_model.dgrossoutput_dworkingpop()
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'gross_output'), ('working_age_population_df', 'population_1570'),dworkforce_dworkingagepop * dgross_output)
        dnet_output = self.macro_model.dnet_output(dgross_output)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'net_output'), ('working_age_population_df', 'population_1570'),dworkforce_dworkingagepop * dnet_output)
        denergy_investment, dinvestment = self.macro_model.dinvestment(dnet_output)
        self.set_partial_derivative_for_other_types(
             ('energy_investment', 'energy_investment'), ('working_age_population_df', 'population_1570'), dworkforce_dworkingagepop * denergy_investment / scaling_factor_energy_investment * 1e3)
        dconsumption = self.macro_model.compute_dconsumption(dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
             ('economics_df', 'pc_consumption'), ('working_age_population_df', 'population_1570'),dworkforce_dworkingagepop * dconsumption_pc)
        self.set_partial_derivative_for_other_types(
             ('pc_consumption_constraint',), ('working_age_population_df', 'population_1570'), - dconsumption_pc / ref_pc_consumption_constraint * dworkforce_dworkingagepop)

        # compute gradients for share_energy_investment
        denergy_investment, denergy_investment_wo_renewable = self.macro_model.compute_denergy_investment_dshare_energy_investement()
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('share_energy_investment', 'share_investment'), denergy_investment * 1e3 / scaling_factor_energy_investment)
        self.set_partial_derivative_for_other_types(
            ('energy_investment_wo_renewable', 'energy_investment_wo_renewable'), ('share_energy_investment', 'share_investment'),
            denergy_investment_wo_renewable)
        dinvestment = self.macro_model.compute_dinvestment_dshare_energy_investement(denergy_investment)
        dnet_output = np.zeros((nb_years, nb_years))
        dconsumption = self.macro_model.compute_dconsumption(dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('share_energy_investment', 'share_investment'),
            dconsumption_pc)#OK
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('share_energy_investment', 'share_investment'),
            - dconsumption_pc / ref_pc_consumption_constraint )

        #compute gradient CO2 Taxes
        denergy_investment = self.macro_model.compute_denergy_investment_dco2_tax()
        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('CO2_taxes', 'CO2_tax'), denergy_investment * 1e3 / scaling_factor_energy_investment)
        dinvestment = denergy_investment
        dnet_output = np.zeros((nb_years, nb_years))
        dconsumption = self.macro_model.compute_dconsumption(dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(dconsumption)
        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('CO2_taxes', 'CO2_tax'),dconsumption_pc)
        self.set_partial_derivative_for_other_types(
            ('pc_consumption_constraint',), ('CO2_taxes', 'CO2_tax'),- dconsumption_pc / ref_pc_consumption_constraint)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['output of damage', 'gross output and gross output bis',
                      'investment', 'energy_investment', 'consumption', 
                      'Output growth rate', 'energy supply',
                      'usable capital', 'employment_rate', 'workforce', 'productivity']
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
        co2_invest_limit = deepcopy(
            self.get_sosdisc_inputs('co2_invest_limit'))
        workforce_df = deepcopy(
            self.get_sosdisc_outputs('workforce_df'))

        if 'output of damage' in chart_list:

            to_plot = ['gross_output', 'net_output']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'gross_output': 'world gross output',
                      'net_output': 'world output net of damage'}

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

            new_chart = TwoAxesInstanciatedChart('years', 'world output (trill $)',
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

            new_chart = TwoAxesInstanciatedChart('years', 'investment (trill $)',
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
                      'energy_investment_wo_tax': 'base invest from macroeconomic',
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

            new_chart = TwoAxesInstanciatedChart('years', 'investment (trill $)',
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

            usable_capital_df = self.get_sosdisc_outputs('usable_capital_df')
            first_serie = economics_df['capital']            
            second_serie = usable_capital_df['usable_capital']
            years = list(usable_capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]
            
            max_values = {}
            min_values = {}
            min_values['usable_capital'], max_values['usable_capital'] = self.get_greataxisrange(first_serie)
            min_values['capital'], max_values['capital'] =  self.get_greataxisrange(second_serie)

            min_value = min(min_values.values())
            max_value = max(max_values.values()) 

            chart_name = 'Capital stock and usable capital for production'

            new_chart = TwoAxesInstanciatedChart('years', 'Trillion dollars',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)


            visible_line = True
            ordonate_data = list(first_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Capital Stock', 'lines', visible_line)
            new_chart.series.append(new_series)
            ordonate_data_bis = list(second_serie)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, '', 'lines', visible_line) 
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

            new_chart = TwoAxesInstanciatedChart('years', ' global consumption (trill $)',
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

            min_value, max_value = 0,1 

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

            working_age_pop_df = self.get_sosdisc_inputs('working_age_population_df')
            years = list(workforce_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value =  self.get_greataxisrange(
                working_age_pop_df['population_1570'])

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart('years', 'Number of people in million',
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

            new_chart = TwoAxesInstanciatedChart('years', 'Total Factor Productivity',
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

            new_chart = TwoAxesInstanciatedChart('years', 'world output (trill $)',
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
