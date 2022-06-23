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
from climateeconomics.core.core_witness.consumption_model import ConsumptionModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from copy import deepcopy
import pandas as pd
import numpy as np


class ConsumptionDiscipline(ClimateEcoDiscipline):
    "ConsumptionModel discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Consumption WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-child fa-fw',
        'version': '',
    }
    _maturity = 'Research'
    years = np.arange(2020, 2101)
    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'alpha': {'type': 'float', 'range': [0., 1.], 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'gamma': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 1},
        'welfare_obj_option': {'type': 'string', 'default': 'welfare', 'possible_values': ['last_utility', 'welfare'], 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'conso_elasticity': {'type': 'float', 'default': 1.45, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_rate_time_pref': {'type': 'float', 'default': 0.015, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'economics_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-'},
        'population_df': {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'energy_mean_price': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_energy_mix', 'unit': '$/MWh'},
        'initial_raw_energy_price': {'type': 'float', 'unit': '$/MWh', 'default': 110, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'init_discounted_utility': {'type': 'float', 'unit': '-', 'default': 3400, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
        'init_period_utility_pc': {'type': 'float', 'unit': '-', 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        'discounted_utility_ref': {'type': 'float', 'unit': '-', 'default': 1700, 'visibility': 'Shared', 'namespace': 'ns_ref', 'user_level': 2},
        'lo_conso': {'type': 'float', 'unit': 'T$', 'default': 2.0, 'user_level': 3},
        'lo_per_capita_conso': {'type': 'float', 'unit': 'k$', 'default': 0.01, 'user_level': 3},
        'total_investment_share_of_gdp': {'type': 'dataframe', 'unit': '%', 'dataframe_descriptor': {'years': ('float', None, False),
                                                                                                     'share_investment': ('float', None, True)}, 'dataframe_edition_locked': False, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'residential_energy_conso_ref' : {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_ref', 'unit': 'MWh', 'default': 21},
        'residential_energy' : {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_energy_mix', 'unit': 'MWh'},
    }
    DESC_OUT = {
        'utility_detail_df': {'type': 'dataframe', 'unit': '-'},
        'utility_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'welfare_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'negative_welfare_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'min_utility_objective': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness'}
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

            total_investment_share_of_gdp = pd.DataFrame(
                {'years': years, 'share_investment': np.ones(len(years)) * 27.0}, index=years)

            self.set_dynamic_default_values(
                {'total_investment_share_of_gdp': total_investment_share_of_gdp})

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.conso_m = ConsumptionModel(inp_dict)

    def run(self):
        # get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # compute utility
        economics_df = inp_dict.pop('economics_df')
        energy_mean_price = inp_dict['energy_mean_price']
        population_df = inp_dict.pop('population_df')
        total_investment_share_of_gdp = inp_dict.pop(
            'total_investment_share_of_gdp')
        residential_energy = inp_dict.pop(
            'residential_energy')

        utility_inputs = {'economics_df': economics_df[['years', 'output_net_of_d']],
                          'population_df': population_df[['years', 'population']],
                          'energy_mean_price': energy_mean_price,
                          'total_investment_share_of_gdp': total_investment_share_of_gdp,
                          'residential_energy': residential_energy
                        }
        utility_df = self.conso_m.compute(utility_inputs)

        # Compute objective function
        obj_option = inp_dict['welfare_obj_option']
        if obj_option in ['last_utility', 'welfare']:
            welfare_objective = self.conso_m.compute_welfare_objective()
        else:
            raise ValueError('obj_option = ' + str(obj_option) + ' not in ' +
                             str(self.DESC_IN['welfare_obj_option']['possible_values']))
        min_utility_objective = self.conso_m.compute_min_utility_objective()
        negative_welfare_objective = self.conso_m.compute_negative_welfare_objective()
        # store output data
        dict_values = {'utility_detail_df': utility_df,
                       'utility_df': utility_df[['years', 'u_discount_rate', 'period_utility_pc', 'discounted_utility', 'welfare', 'pc_consumption']],
                       'welfare_objective': welfare_objective,
                       'min_utility_objective': min_utility_objective,
                       'negative_welfare_objective' : negative_welfare_objective
                       }
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        utility_df
          - 'consumption':
                - economics_df : 'output_net_of_d'
                - energy_mean_price : 'energy_price'
                - residential_energy : 'residential_energy'
                - total_investment_share_of_gdp : 'share_investment'
          - 'pc_consumption':
                - economics_df : 'output_net_of_d'
                - energy_mean_price : 'energy_price'
                - residential_energy : 'residential_energy'
                - total_investment_share_of_gdp : 'share_investment'          
          - 'period_utility_pc':
                - economics_df : 'output_net_of_d'
                - energy_mean_price : 'energy_price'
                - residential_energy : 'residential_energy'
                - total_investment_share_of_gdp : 'share_investment'
          - 'discounted_utility',
                - economics_df : 'output_net_of_d'
                - energy_mean_price : 'energy_price'
                - residential_energy : 'residential_energy'
                - total_investment_share_of_gdp : 'share_investment'
          - 'welfare'
                - economics_df : 'output_net_of_d'
                - energy_mean_price : 'energy_price'
                - residential_energy : 'residential_energy'
                - total_investment_share_of_gdp : 'share_investment'
        """
        inputs_dict = self.get_sosdisc_inputs()
        obj_option = inputs_dict['welfare_obj_option']
        d_pc_consumption_d_output_net_of_d, d_pc_consumption_d_share_investment, d_pc_consumption_d_population, \
        d_period_utility_pc_d_output_net_of_d, d_period_utility_pc_d_share_investment, d_period_utility_d_population,\
        d_discounted_utility_d_output_net_of_d, d_discounted_utility_d_share_investment, d_discounted_utility_d_population, \
         d_welfare_d_output_net_of_d, d_welfare_d_share_investment, d_welfare_d_population = self.conso_m.compute_gradient()
        
        # d_pc_consumption_d_output_net_of_d, d_pc_consumption_d_share_investment, \
        # d_period_utility_d_pc_consumption, d_discounted_utility_d_pc_consumption, d_discounted_utility_d_population,\
        #     d_welfare_d_pc_consumption, d_welfare_d_population = self.conso_m.compute_gradient()
            
        d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, \
            d_welfare_d_energy_price = self.conso_m.compute_gradient_energy_mean_price()

        d_period_utility_d_residential_energy, d_discounted_utility_d_residential_energy, \
            d_welfare_d_residential_energy = self.conso_m.compute_gradient_residential_energy()
        d_obj_d_welfare, d_obj_d_period_utility_pc = self.conso_m.compute_gradient_objective()

        # fill jacobians
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'pc_consumption'), ('economics_df', 'output_net_of_d'),  d_pc_consumption_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'pc_consumption'), ('total_investment_share_of_gdp', 'share_investment'),  d_pc_consumption_d_share_investment)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'pc_consumption'), ('population_df', 'population'),  d_pc_consumption_d_population)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('economics_df', 'output_net_of_d'),  d_period_utility_pc_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('total_investment_share_of_gdp', 'share_investment'),  d_period_utility_pc_d_share_investment)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('energy_mean_price', 'energy_price'),  d_period_utility_d_energy_price)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('residential_energy', 'residential_energy'),  d_period_utility_d_residential_energy)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'period_utility_pc'), ('population_df', 'population'),  d_period_utility_d_population)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('economics_df', 'output_net_of_d'),  d_discounted_utility_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('total_investment_share_of_gdp', 'share_investment'),  d_discounted_utility_d_share_investment)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('energy_mean_price', 'energy_price'),  d_discounted_utility_d_energy_price)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('residential_energy', 'residential_energy'),  d_discounted_utility_d_residential_energy)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'discounted_utility'), ('population_df', 'population'),  d_discounted_utility_d_population)

        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('economics_df', 'output_net_of_d'),  d_welfare_d_output_net_of_d)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('total_investment_share_of_gdp', 'share_investment'),  d_welfare_d_share_investment)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('population_df', 'population'),  d_welfare_d_population)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('energy_mean_price', 'energy_price'),  d_welfare_d_energy_price)
        self.set_partial_derivative_for_other_types(
            ('utility_df', 'welfare'), ('residential_energy', 'residential_energy'),  d_welfare_d_residential_energy)

        if obj_option == 'last_utility':
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('economics_df', 'output_net_of_d'), d_obj_d_period_utility_pc.dot(d_period_utility_pc_d_output_net_of_d))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('total_investment_share_of_gdp', 'share_investment'), d_obj_d_period_utility_pc.dot(d_period_utility_pc_d_share_investment))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('energy_mean_price', 'energy_price'), d_obj_d_period_utility_pc.dot(d_period_utility_d_energy_price))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('residential_energy', 'residential_energy'), d_obj_d_period_utility_pc.dot(d_period_utility_d_residential_energy))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('population_df', 'population'),  d_obj_d_period_utility_pc.dot(d_period_utility_d_population))

        elif obj_option == 'welfare':
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('economics_df', 'output_net_of_d'), np.dot(d_obj_d_welfare, d_welfare_d_output_net_of_d))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('total_investment_share_of_gdp', 'share_investment'), np.dot(d_obj_d_welfare, d_welfare_d_share_investment))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('energy_mean_price', 'energy_price'), np.dot(d_obj_d_welfare, d_welfare_d_energy_price))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('residential_energy', 'residential_energy'), np.dot(d_obj_d_welfare, d_welfare_d_residential_energy))
            self.set_partial_derivative_for_other_types(
                ('welfare_objective',), ('population_df', 'population'),  np.dot(d_obj_d_welfare, d_welfare_d_population))

        else:
            pass

        d_neg_obj_d_welfare, x = self.conso_m.compute_gradient_negative_objective()

        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), ('economics_df', 'output_net_of_d'),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_output_net_of_d))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), ('total_investment_share_of_gdp', 'share_investment'),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_share_investment))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), ('energy_mean_price', 'energy_price'),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_energy_price))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), ('residential_energy', 'residential_energy'),
            np.dot(d_neg_obj_d_welfare, d_welfare_d_residential_energy))
        self.set_partial_derivative_for_other_types(
            ('negative_welfare_objective',), ('population_df', 'population'), np.dot(d_neg_obj_d_welfare, d_welfare_d_population))


        d_obj_d_discounted_utility, d_obj_d_period_utility_pc = self.conso_m.compute_gradient_min_utility_objective()

        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('economics_df', 'output_net_of_d'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_output_net_of_d))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('total_investment_share_of_gdp', 'share_investment'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_share_investment))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('energy_mean_price', 'energy_price'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_energy_price))
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('residential_energy', 'residential_energy'), np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_residential_energy))        
        self.set_partial_derivative_for_other_types(
            ('min_utility_objective',), ('population_df', 'population'),  np.dot(d_obj_d_discounted_utility, d_discounted_utility_d_population))
    
    
    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Consumption', 'Consumption PC', 'Utility', 'Utility of pc consumption',
                      'Energy effects on utility', 'Energy ratios']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Utility' in chart_list:

            to_plot = ['discounted_utility']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_df'))

            discounted_utility = utility_df['discounted_utility']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(discounted_utility)

            chart_name = 'Utility'

            new_chart = TwoAxesInstanciatedChart('years', 'Discounted Utility (trill $)',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Utility of pc consumption' in chart_list:

            to_plot = ['period_utility_pc']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_df'))

            utility = utility_df['period_utility_pc']

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(utility)

            chart_name = 'Utility of per capita consumption'

            new_chart = TwoAxesInstanciatedChart('years', 'Utility of pc consumption',
                                                 [year_start - 5, year_end + 5], [
                                                     min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                c_emission = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, c_emission, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Energy ratios' in chart_list:

            energy_mean_price = self.get_sosdisc_inputs('energy_mean_price')[
                'energy_price'].values

            energy_price_ref = self.get_sosdisc_inputs(
                'initial_raw_energy_price')
            
            
            residential_energy = self.get_sosdisc_inputs('residential_energy')[
                'residential_energy'].values

            residential_energy_conso_ref = self.get_sosdisc_inputs(
                'residential_energy_conso_ref')

            residential_energy_ratio = residential_energy / residential_energy_conso_ref

            energy_price_ratio = energy_price_ref / energy_mean_price

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy ratios'

            new_chart = TwoAxesInstanciatedChart('years', '',
                                                 chart_name=chart_name)

            visible_line = True

            new_series = InstanciatedSeries(
                years, residential_energy_ratio.tolist(), 'Residential energy availability ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, energy_price_ratio.tolist(), 'Energy price ratio', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Energy effects on utility' in chart_list:

            utility_df = deepcopy(self.get_sosdisc_outputs('utility_df'))

            discounted_utility_final = utility_df['discounted_utility'].values

            energy_mean_price = self.get_sosdisc_inputs('energy_mean_price')[
                'energy_price'].values

            energy_price_ref = self.get_sosdisc_inputs(
                'initial_raw_energy_price')
            
            
            residential_energy = self.get_sosdisc_inputs('residential_energy')[
                'residential_energy'].values

            residential_energy_conso_ref = self.get_sosdisc_inputs(
                'residential_energy_conso_ref')

            residential_energy_ratio = residential_energy / residential_energy_conso_ref

            energy_price_ratio = energy_price_ref / energy_mean_price

            discounted_utility_before = discounted_utility_final / energy_price_ratio / residential_energy_ratio

            discounted_utility_price_ratio = discounted_utility_final / residential_energy_ratio

            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            chart_name = 'Energy price ratio effect on discounted utility'

            new_chart = TwoAxesInstanciatedChart('years', 'Discounted Utility (trill $)',
                                                 chart_name=chart_name)

            visible_line = True

            new_series = InstanciatedSeries(
                years, discounted_utility_before.tolist(), 'Discounted Utility without residential energy and price ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, discounted_utility_price_ratio.tolist(), 'Discounted Utility without residential energy ratio', 'lines', visible_line)

            new_chart.series.append(new_series)

            new_series = InstanciatedSeries(
                years, discounted_utility_final.tolist(), 'Discounted Utility', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Consumption' in chart_list:

            to_plot = ['consumption']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_detail_df'))
            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                utility_df[to_plot])

            chart_name = 'Global consumption over the years'

            new_chart = TwoAxesInstanciatedChart('years', ' global consumption [trillion $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)
        if 'Consumption PC' in chart_list:

            to_plot = ['pc_consumption']
            utility_df = deepcopy(self.get_sosdisc_outputs('utility_detail_df'))
            years = list(utility_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                utility_df[to_plot])

            chart_name = 'Per capita consumption over the years'

            new_chart = TwoAxesInstanciatedChart('years', ' Per capita consumption [thousand $]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(utility_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, key, 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
