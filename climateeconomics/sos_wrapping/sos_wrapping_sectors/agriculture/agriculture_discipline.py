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
from climateeconomics.core.core_sectorization.sector_model import SectorModel
from sostrades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd
import numpy as np
from copy import deepcopy
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min
from sostrades_core.tools.cst_manager.constraint_manager import compute_delta_constraint, compute_ddelta_constraint
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline


class AgricultureDiscipline(ClimateEcoDiscipline):
    "Agriculture sector discpline"
    
    # ontology information
    _ontology_data = {
        'label': 'Agriculture sector WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-building-wheat',
        'version': '',
    }
    _maturity = 'Research'
    
    sector_name = 'Agriculture'
    prod_cap_unit = 'T$'
      
    DESC_IN = {
        'damage_df': {'type': 'dataframe', 'unit': 'G$',
                      'dataframe_descriptor':
                          {'years': ('float', None, False),'damages': ('float', None, True),
                              'damage_frac_output': ('float', None, True),'base_carbon_price': ('float', None, True),}
},
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'productivity_start': {'type': 'float', 'default': 1.31162, 'user_level': 2, 'unit': '-'},
        'capital_start': {'type': 'float', 'unit': 'T$', 'default': 6.92448579, 'user_level': 2},
        'workforce_df': {'type': 'dataframe', 'dataframe_descriptor': {},'dynamic_dataframe_columns': True,
                        'unit': 'millions of people', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                        'namespace': 'ns_witness'},
        'productivity_gr_start': {'type': 'float', 'default': 0.0027844, 'user_level': 2, 'unit': '-'},
        'decline_rate_tfp': {'type': 'float', 'default': 0.098585, 'user_level': 3, 'unit': '-'},
        # Usable capital
        'capital_utilisation_ratio':  {'type': 'float', 'default': 0.8, 'user_level': 3, 'unit': '-'},
        'max_capital_utilisation_ratio': {'type': 'float', 'default': 0.95, 'user_level': 3, 'unit': '-'},
        'energy_eff_k':  {'type': 'float', 'default': 0.1, 'user_level': 3, 'unit': '-'},
        'energy_eff_cst': {'type': 'float', 'default': 0.490463, 'user_level': 3, 'unit': '-'},
        'energy_eff_xzero': {'type': 'float', 'default': 1993, 'user_level': 3, 'unit': '-'},
        'energy_eff_max': {'type': 'float', 'default': 2.35832, 'user_level': 3, 'unit': '-'},
        # Production function param
        'output_alpha': {'type': 'float', 'default': 0.99, 'user_level': 2, 'unit': '-'},
        'output_gamma': {'type': 'float', 'default': 0.5, 'user_level': 2, 'unit': '-'},
        'depreciation_capital': {'type': 'float', 'default': 0.058, 'user_level': 2, 'unit': '-'},
        'damage_to_productivity': {'type': 'bool', 'default': True, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'unit': '-','namespace': 'ns_witness'},
        'frac_damage_prod': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': '-', 'default': 0.3, 'user_level': 2, 'unit': '-'},
        'sectors_investment_df': {'type': 'dataframe', 'unit': 'T$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                            'namespace': 'ns_witness', 'dataframe_descriptor': {}, 'dynamic_dataframe_columns': True },

        # energy_production stored in PetaWh for coupling variables scaling
        'energy_production': {'type': 'dataframe','unit': 'PWh',
                              'dataframe_descriptor':{'years': ('float', None, False),
                                      'Total production': ('float', None, True),}},
        'scaling_factor_energy_production': {'type': 'float', 'default': 1e3, 'unit': '-', 'user_level': 2, 'visibility': 'Shared', 'namespace': 'ns_witness'},
        'alpha': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': 'ns_witness',
                  'user_level': 1, 'unit': '-'},
        'init_output_growth': {'type': 'float', 'default': -0.046154, 'user_level': 2, 'unit': '-'},
        'ref_emax_enet_constraint': {'type': 'float', 'default': 60e3, 'user_level': 3,
                                     'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ref', 'unit': '-'},
         'prod_function_fitting': {'type': 'bool', 'default': False, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                    'unit': '-','namespace': 'ns_macro', 'structuring': True},
    }

    DESC_OUT = {
        'productivity_df': {'type': 'dataframe'},
        'production_df': {'type': 'dataframe', 'unit': 'T$',
                          'dataframe_descriptor': {'years': ('float', None, False),
                                                   'gross_output': ('float', None, False),}
               },
        'capital_df':  {'type': 'dataframe', 'unit': 'T$'},
        'detailed_capital_df': {'type': 'dataframe', 'unit': 'T$'},
        'growth_rate_df': {'type': 'dataframe', 'unit': '-'},
        'emax_enet_constraint':  {'type': 'array'},
    }

    def setup_sos_disciplines(self):

        self.update_default_with_years()
        dynamic_outputs = {}
        dynamic_inputs = {}
        if 'prod_function_fitting' in self.get_data_in():
            prod_function_fitting = self.get_sosdisc_inputs('prod_function_fitting')
            if prod_function_fitting == True:
                dynamic_inputs['energy_eff_max_range_ref'] = {'type': 'float', 'unit': '-', 'default': 20}
                dynamic_outputs['longterm_energy_efficiency'] =  {'type': 'dataframe', 'unit': '-','dataframe_descriptor': {},'dynamic_dataframe_columns': True}
                dynamic_outputs['range_energy_eff_constraint'] = {'type': 'array', 'unit': '-'}
                dynamic_inputs['hist_sector_investment'] = {'type': 'dataframe', 'unit': '-', 'dataframe_descriptor': {},'dynamic_dataframe_columns': True}

                self.add_outputs(dynamic_outputs)
                self.add_inputs(dynamic_inputs)

    def update_default_with_years(self):
        '''
        Update all default dataframes with years 
        '''
        if 'year_start' in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                ['year_start', 'year_end'])
            years = np.arange(year_start, year_end + 1)

    def init_execution(self):
        param = self.get_sosdisc_inputs(in_dict=True)
        self.agriculture_model = SectorModel()
        self.agriculture_model.configure_parameters(param, self.sector_name)

    def run(self):
        # Get inputs
        param = self.get_sosdisc_inputs(in_dict=True)
        #configure param
        self.agriculture_model.configure_parameters(param, self.sector_name)
        #coupling df 
        damage_df = param['damage_df']
        energy_production = param['energy_production']
        sectors_investment_df = param['sectors_investment_df']
        workforce_df = param['workforce_df']
        prod_function_fitting = param['prod_function_fitting']

        agriculture_inputs = {'damage_df': damage_df[['years', 'damage_frac_output']],
                              'energy_production': energy_production,
                              'sectors_investment_df': sectors_investment_df,
                              'workforce_df': workforce_df}
        # Model execution
        production_df, capital_df, productivity_df, growth_rate_df, emax_enet_constraint, lt_energy_eff, range_energy_eff_cstrt = self.agriculture_model.compute(
            agriculture_inputs)

        # Store output data
        dict_values = {'productivity_df': productivity_df,
                       'production_df': production_df[['years', 'output', 'output_net_of_damage']],
                       'capital_df': capital_df[['years', 'capital', 'usable_capital']],
                       'detailed_capital_df': capital_df,
                       'growth_rate_df': growth_rate_df,
                       'emax_enet_constraint': emax_enet_constraint
                       }
        
        if  prod_function_fitting == True: 
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
        scaling_factor_energy_production, ref_emax_enet_constraint = self.get_sosdisc_inputs(
            ['scaling_factor_energy_production', 'ref_emax_enet_constraint'])
        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        time_step = self.get_sosdisc_inputs('time_step')
        nb_years = len(np.arange(year_start, year_end + 1, time_step))

        # Gradients wrt energy
        dcapitalu_denergy = self.agriculture_model.dusablecapital_denergy()
        doutput_denergy = self.agriculture_model.doutput_denergy(dcapitalu_denergy)
        dnetoutput_denergy = self.agriculture_model.dnetoutput(doutput_denergy)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output'), ('energy_production', 'Total production'),scaling_factor_energy_production * doutput_denergy)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output_net_of_damage'), ('energy_production', 'Total production'), scaling_factor_energy_production * dnetoutput_denergy)
        self.set_partial_derivative_for_other_types(
            ('capital_df', 'usable_capital'), ('energy_production', 'Total production'), scaling_factor_energy_production * dcapitalu_denergy)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), ('energy_production', 'Total production'), - scaling_factor_energy_production * (np.identity(nb_years) / ref_emax_enet_constraint))

        # gradients wrt workforce
        doutput_dworkforce = self.agriculture_model.compute_doutput_dworkforce()
        dnetoutput_dworkforce = self.agriculture_model.dnetoutput(
            doutput_dworkforce)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output'), ('workforce_df', self.sector_name), doutput_dworkforce)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output_net_of_damage'), ('workforce_df', self.sector_name), dnetoutput_dworkforce)
        
        # gradients wrt damage:
        dproductivity_ddamage = self.agriculture_model.dproductivity_ddamage()
        doutput_ddamage = self.agriculture_model.doutput_ddamage(
            dproductivity_ddamage)
        dnetoutput_ddamage = self.agriculture_model.dnetoutput_ddamage(
            doutput_ddamage)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output'), ('damage_df', 'damage_frac_output'), doutput_ddamage)
        self.set_partial_derivative_for_other_types(
            ('production_df', 'output_net_of_damage'), ('damage_df', 'damage_frac_output'), dnetoutput_ddamage)

        # gradients wrt invest
        #If production fitting = true we use the investment from another input
        prod_function_fitting = self.get_sosdisc_inputs('prod_function_fitting')
        if prod_function_fitting == True:
            invest_df = 'hist_sector_investment'
        else:
            invest_df = 'sectors_investment_df'
        dcapital_dinvest = self.agriculture_model.dcapital_dinvest()
        demax_cstrt_dinvest = self.agriculture_model.demaxconstraint(
            dcapital_dinvest)
        self.set_partial_derivative_for_other_types(
            ('capital_df', 'capital'), (invest_df, self.sector_name), dcapital_dinvest)
        self.set_partial_derivative_for_other_types(
            ('emax_enet_constraint',), (invest_df, self.sector_name), demax_cstrt_dinvest)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['sector output', 'investment', 'output growth',
                      'usable capital', 'capital', 'employment_rate', 'workforce', 'productivity', 'energy efficiency', 'e_max']
        prod_func_fit = self.get_sosdisc_inputs('prod_function_fitting')
        if prod_func_fit == True: 
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

        production_df = self.get_sosdisc_outputs('production_df')
        capital_df = self.get_sosdisc_outputs('capital_df')
        capital_detail_df = self.get_sosdisc_outputs('detailed_capital_df')
        productivity_df = self.get_sosdisc_outputs('productivity_df')
        workforce_df = self.get_sosdisc_inputs('workforce_df')
        capital_utilisation_ratio = self.get_sosdisc_inputs('capital_utilisation_ratio')
        growth_rate_df = self.get_sosdisc_outputs('growth_rate_df')
        prod_func_fit = self.get_sosdisc_inputs('prod_function_fitting')
        if prod_func_fit == True:
            lt_energy_eff = self.get_sosdisc_outputs('longterm_energy_efficiency')


        if 'sector output' in chart_list:

            to_plot = ['output', 'output_net_of_damage']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            legend = {'output': 'sector gross output',
                      'output_net_of_damage': 'world output net of damage'}

            years = list(production_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                   production_df[to_plot])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Agriculture sector economics output'

            new_chart = TwoAxesInstanciatedChart('years', 'world output [trillion dollars]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(production_df[key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, legend[key], 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'usable capital' in chart_list:

            first_serie = capital_df['capital']
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

            new_chart = TwoAxesInstanciatedChart('years', 'Capital stock [Trillion dollars]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)

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
            serie = capital_df['capital']
            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(serie)

            chart_name = 'Agriculture capital stock per year'

            new_chart = TwoAxesInstanciatedChart('years', 'Capital stock [Trillion dollars]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name, stacked_bar=True)
            visible_line = True
            ordonate_data = list(serie)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Service capital stock', InstanciatedSeries.BAR_DISPLAY)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'workforce' in chart_list:

            years = list(workforce_df['years'])
            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                workforce_df[self.sector_name])

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart('years', 'Number of people [million]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(workforce_df[self.sector_name])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'productivity' in chart_list:

            to_plot = ['productivity']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(productivity_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                productivity_df[to_plot])

            chart_name = 'Total Factor Productivity'

            new_chart = TwoAxesInstanciatedChart('years', 'Total Factor Productivity [-]',
                                                [year_start, year_end], [
                                                     min_value, max_value], chart_name)

            ordonate_data = list(productivity_df['productivity'])
            visible_line = True

            new_series = InstanciatedSeries(
                years, ordonate_data, 'Total Factor productivity', 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'energy efficiency' in chart_list:

            to_plot = capital_detail_df['energy_efficiency']
            #economics_df = discipline.get_sosdisc_outputs('economics_df')

            years = list(capital_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(to_plot)

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart('years', 'Capital energy efficiency [-]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)

        
            visible_line = True

            ordonate_data = list(to_plot)

            new_series = InstanciatedSeries(
                    years, ordonate_data, 'Energy efficiency', 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'long term energy efficiency' in chart_list: 

            to_plot = ['energy_efficiency']

            years = list(lt_energy_eff['years'])

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(lt_energy_eff[to_plot])

            chart_name = 'Capital energy efficiency over the years'

            new_chart = TwoAxesInstanciatedChart('years', 'Capital energy efficiency [-]',
                                                 [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)

            for key in to_plot:
                visible_line = True

                ordonate_data = list(lt_energy_eff[key])

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

            years = list(capital_detail_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            max_values = {}
            min_values = {}
            min_values['e_max'], max_values['e_max'] = self.get_greataxisrange(
                capital_detail_df[to_plot])
            min_values['energy'], max_values['energy'] = self.get_greataxisrange(
                total_production)

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'E_max value and Net Energy'

            new_chart = TwoAxesInstanciatedChart('years', '[Twh]',
                                                [year_start, year_end],
                                                 [min_value, max_value], chart_name)
            visible_line = True

            ordonate_data = list(capital_detail_df[to_plot])
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

        if 'output growth' in chart_list:

            to_plot = ['net_output_growth_rate']
            years = list(growth_rate_df.index)
            year_start = years[0]
            year_end = years[len(years) - 1]
            min_value, max_value = self.get_greataxisrange(growth_rate_df[to_plot])
            chart_name = 'Net output growth rate over years'
            new_chart = TwoAxesInstanciatedChart('years', ' growth rate [-]',
                                                [year_start, year_end],
                                                 [min_value, max_value],
                                                 chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(growth_rate_df[key])
                new_series = InstanciatedSeries(years, ordonate_data, key, 'lines', visible_line)
                new_chart.series.append(new_series)
                
            instanciated_charts.append(new_chart)

        return instanciated_charts
