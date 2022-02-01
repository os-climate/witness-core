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
from climateeconomics.core.core_witness.macroeconomics_model import MacroEconomics
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from copy import deepcopy


class MacroeconomicsDiscipline(ClimateEcoDiscipline):
    "Macroeconomics discipline for WITNESS"
    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        'damage_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'year_start': {'type': 'int', 'default': 2020,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100,  'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'productivity_start': {'type': 'float', 'default': 0.974422, 'user_level': 2},
        'init_gross_output': {'type': 'float', 'unit': 'trillions $', 'visibility': 'Shared', 'default': 130.187,
                              'namespace': 'ns_witness', 'user_level': 2},
        'capital_start': {'type': 'float', 'unit': 'trillions $', 'default': 355.9210491, 'user_level': 2},
        'population_df': {'type': 'dataframe', 'unit': 'billions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'productivity_gr_start': {'type': 'float', 'default': 0.042925, 'user_level': 2},
        'decline_rate_tfp': {'type': 'float', 'default': 0.02351234, 'user_level': 3},
        'depreciation_capital': {'type': 'float', 'default': 0.08, 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'lo_capital': {'type': 'float', 'unit': 'trillions $', 'default': 1.0, 'user_level': 3},
        'lo_conso': {'type': 'float', 'unit': 'trillions $', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'trillions $', 'default': 0.01, 'user_level': 3},
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
        'output_k_exponent': {'type': 'float', 'default': 0.1924566, 'user_level': 3},
        'output_pop_exponent': {'type': 'float', 'default': 0.10810935, 'user_level': 3},
        'output_energy_exponent': {'type': 'float', 'default': 0.24858645, 'user_level': 3},
        'output_energy_share': {'type': 'float', 'default': 0.01, 'user_level': 3},
        'output_exponent': {'type': 'float', 'default':  2.75040992, 'user_level': 3},
        'output_pop_share': {'type': 'float', 'default': 0.29098974, 'user_level': 3},
        'pop_factor': {'type': 'float', 'default': 1e-3, 'user_level': 3},
        'energy_factor': {'type': 'float', 'default': 1e-4, 'user_level': 3},
        'decline_rate_energy_productivity': {'type': 'float', 'default':  0.01345699, 'user_level': 3},
        'init_energy_productivity': {'type': 'float', 'default': 3.045177, 'user_level': 2},
        'init_energy_productivity_gr': {'type': 'float', 'default':   0.0065567, 'user_level': 2},
        'co2_emissions_Gt': {'type': 'dataframe', 'visibility': 'Shared',
                             'namespace': 'ns_energy_mix', 'unit': 'Gt'},
        'CO2_tax_efficiency': {'type': 'dataframe', 'unit': '%'},
        'co2_invest_limit': {'type': 'float', 'default': 2.0},
        'CO2_taxes': {'type': 'dataframe', 'unit': '$/tCO2', 'visibility': 'Shared', 'namespace': 'ns_witness'},
    }

    DESC_OUT = {
        'economics_detail_df': {'type': 'dataframe'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_investment': {'type': 'dataframe', 'visibility': 'Shared', 'unit': 'G$', 'namespace': 'ns_witness'},
        'energy_investment_wo_renewable': {'type': 'dataframe', 'unit': 'G$'},
        'global_investment_constraint': {'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
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
        # rescaling from billion to million
        cols = [col for col in population_df.columns if col != 'years']
        population_df[cols] = population_df[cols] * 1e3

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
                        'population_df': population_df[['years', 'population']]
                        }
        # Check inputs
        count = len(
            [i for i in list(share_energy_investment['share_investment']) if np.real(i) > 100.0])
        if count > 0:
            print(
                'For at least one year, the share of energy investment is above 100% of total investment')
        # Model execution
        economics_df, energy_investment, global_investment_constraint, energy_investment_wo_renewable = self.macro_model.compute(
            macro_inputs)

        # Store output data
        dict_values = {'economics_detail_df': economics_df,
                       'economics_df': economics_df[['years', 'gross_output', 'pc_consumption', 'output_net_of_d']],
                       'energy_investment': energy_investment,
                       'global_investment_constraint': global_investment_constraint,
                       'energy_investment_wo_renewable': energy_investment_wo_renewable}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        economics_df
          - 'gross_output',
              - damage_df, damage_frac_output
              - energy_production, Total production 
          - 'output_growth'
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'output_net_of_d',
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'net_output',
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'consumption'
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'pc_consumption'
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'interest_rate'
              - damage_df, damage_frac_output
              - energy_production, Total production
          - 'productivity',
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
            ('economics_df', 'gross_output'), ('damage_df', 'damage_frac_output'), dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('damage_df', 'damage_frac_output'), dconsumption_pc)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('damage_df', 'damage_frac_output'), doutput_net_of_d)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('damage_df', 'damage_frac_output'), denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$

        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_denergy_supply()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('energy_production', 'Total production'), scaling_factor_energy_production * dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('energy_production', 'Total production'), scaling_factor_energy_production * dconsumption_pc)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('energy_production', 'Total production'), scaling_factor_energy_production * doutput_net_of_d)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('energy_production', 'Total production'), scaling_factor_energy_production * denergy_investment / scaling_factor_energy_investment * 1e3)  # Invest from T$ to G$
        # compute gradient for design variable share_energy_investment
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dshare_energy_investment()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('share_energy_investment', 'share_investment'), denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('share_energy_investment', 'share_investment'), dgross_output / 100)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('share_energy_investment', 'share_investment'), dconsumption_pc / 100)
        # compute gradient for design variable total_investment_share_of_gdp
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dtotal_investment_share_of_gdp()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('total_investment_share_of_gdp', 'share_investment'), dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('total_investment_share_of_gdp', 'share_investment'), dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('total_investment_share_of_gdp', 'share_investment'), doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('total_investment_share_of_gdp', 'share_investment'), denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for coupling variable co2_emissions_Gt
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dCO2_emission_gt()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('co2_emissions_Gt', 'Total CO2 emissions'), dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('co2_emissions_Gt', 'Total CO2 emissions'), dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('co2_emissions_Gt', 'Total CO2 emissions'), doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('co2_emissions_Gt', 'Total CO2 emissions'), denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for design variable CO2_taxes
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dCO2_taxes()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('CO2_taxes', 'CO2_tax'), dgross_output / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('CO2_taxes', 'CO2_tax'), dconsumption_pc / 100.0)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('CO2_taxes', 'CO2_tax'), doutput_net_of_d / 100.0)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('CO2_taxes', 'CO2_tax'), denergy_investment / scaling_factor_energy_investment * 1e3 / 100.0)  # Invest from T$ to G$

        # compute gradient for coupling variable population
        dgross_output, dinvestment, denergy_investment, dnet_output = self.macro_model.compute_dgross_output_dpopulation()
        dconsumption = self.macro_model.compute_dconsumption(
            dnet_output, dinvestment)
        dconsumption_pc = self.macro_model.compute_dconsumption_pc_dpopulation(
            dconsumption)
        doutput_net_of_d = self.macro_model.compute_doutput_net_of_d(
            dgross_output)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'gross_output'), ('population_df', 'population'), dgross_output * 1e3)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'pc_consumption'), ('population_df', 'population'), dconsumption_pc * 1e3)

        self.set_partial_derivative_for_other_types(
            ('economics_df', 'output_net_of_d'), ('population_df', 'population'), doutput_net_of_d * 1e3)

        self.set_partial_derivative_for_other_types(
            ('energy_investment', 'energy_investment'), ('population_df', 'population'), denergy_investment / scaling_factor_energy_investment * 1e6)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['output of damage', 'gross output and gross output bis',
                      'investment', 'energy_investment', 'population', 'productivity', 'consumption', 'Output growth rate', 'energy supply', 'energy productivity']
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

        if 'population' in chart_list:

            population_df = self.get_sosdisc_inputs('population_df')

            years = list(population_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                population_df['population'])

            chart_name = 'Population evolution over the years'

            new_chart = TwoAxesInstanciatedChart('years', ' population (billion)',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(population_df['population'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'population', 'lines', visible_line)

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

        if 'energy productivity' in chart_list:

            to_plot = ['energy_productivity']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(economics_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                economics_df[to_plot])

            chart_name = 'Energy Productivity'

            new_chart = TwoAxesInstanciatedChart('years', 'global productivity',
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
