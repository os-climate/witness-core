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
from numpy import arange, asarray
from pandas import DataFrame
import numpy as np
import pandas as pd
from pathlib import Path

from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from os.path import join, dirname

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM


class DataStudy():
    def __init__(self, year_start=2020, year_end=2100, time_step=1):
        self.study_name = 'default_name'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.study_name_wo_extra_name = self.study_name

    def setup_usecase(self):
        setup_data_list = []
        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)

        # private values economics operator model
        witness_input = {}
        witness_input[self.study_name + '.year_start'] = self.year_start
        witness_input[self.study_name + '.year_end'] = self.year_end
        witness_input[self.study_name + '.time_step'] = self.time_step

        witness_input[self.study_name +
                      '.Damage.tipping_point'] = True
        witness_input[self.study_name +
                      '.Macroeconomics.damage_to_productivity'] = True
        witness_input[self.study_name +
                      '.frac_damage_prod'] = 0.30
        witness_input[self.study_name +
                      '.init_rate_time_pref'] = .015
        witness_input[self.study_name +
                      '.conso_elasticity'] = 1.45
        witness_input[self.study_name +
                      '.init_gross_output'] = 130.187
        # Relax constraint for 15 first years
        witness_input[self.study_name + '.Damage.damage_constraint_factor'] = np.concatenate(
            (np.linspace(1.0, 1.0, 20), np.asarray([1] * (len(years) - 20))))
#         witness_input[self.study_name +
#                       '.Damage.damage_constraint_factor'] = np.asarray([1] * len(years))

        # get population from csv file
        # get file from the data folder 3 folder up.
        global_data_dir = join(Path(__file__).parents[3], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        witness_input[self.study_name + '.population_df'] = population_df

        self.share_energy_investment_array = asarray([1.65] * nb_per)

        total_invest = asarray([25.0] * nb_per)
        total_invest = DataFrame(
            {'years': years, 'share_investment': total_invest})
        witness_input[self.study_name +
                      '.total_investment_share_of_gdp'] = total_invest
        share_energy_investment = DataFrame(
            {'years': years, 'share_investment': self.share_energy_investment_array}, index=years)
        witness_input[self.study_name +
                      '.share_energy_investment'] = share_energy_investment
        data = arange(1.0, nb_per + 1.0, 1)

        df_eco = DataFrame({'years': years,
                            'gross_output': data,
                            'pc_consumption': data,
                            'output_net_of_d': data},
                           index=arange(self.year_start, self.year_end + 1, self.time_step))

        witness_input[self.study_name + '.economics_df'] = df_eco

        nrj_invest = arange(1000, nb_per + 1000, 1)

        df_energy_investment = DataFrame({'years': years,
                                          'energy_investment': nrj_invest},
                                         index=arange(self.year_start, self.year_end + 1, self.time_step))
        df_energy_investment_before_year_start = DataFrame({'past_years': [2017, 2018, 2019],
                                                            'energy_investment_before_year_start': [1924, 1927, 1935]},
                                                           index=[2017, 2018, 2019])

        CO2_emitted_forest = pd.DataFrame()
        #GtCO2
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        CO2_emitted_forest['years'] = years
        CO2_emitted_forest['emitted_CO2_evol'] = emission_forest
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission
        witness_input[self.study_name +
                      '.CO2_emitted_forest_df'] = CO2_emitted_forest

        self.CO2_tax = np.asarray([50.] * len(years))

        witness_input[self.study_name +
                      '.energy_investment'] = df_energy_investment
        witness_input[self.study_name +
                      '.energy_investment_before_year_start'] = df_energy_investment_before_year_start

        intermediate_point = 30
        # CO2 taxes related inputs
        CO2_tax_efficiency = np.concatenate(
            (np.linspace(30, intermediate_point, 15), np.asarray([intermediate_point] * (len(years) - 15))))
        # CO2_tax_efficiency = 30.0
        default_co2_efficiency = pd.DataFrame(
            {'years': years, 'CO2_tax_efficiency': CO2_tax_efficiency})

        witness_input[f'{self.study_name}.Macroeconomics.CO2_tax_efficiency'] = default_co2_efficiency

        witness_input[f'{self.study_name}.delta_co2_price'] = 5.
        witness_input[f'{self.study_name}.beta'] = 1.0
        witness_input[f'{self.study_name}.gamma'] = 1.0
        witness_input[f'{self.study_name}.init_discounted_utility'] = 4000.0

        witness_input[f'{self.study_name}.init_rate_time_pref'] = 0.0
        witness_input[f'{self.study_name}.total_emissions_ref'] = 7.2
        witness_input[f'{self.study_name}.total_emissions_damage_ref'] = 18.0
        witness_input[f'{self.study_name}.temperature_change_ref'] = 1.0
        witness_input[f'{self.study_name_wo_extra_name}.NormalizationReferences.total_emissions_ref'] = 12.0
        #witness_input[f'{self.name}.CO2_emissions_Gt'] = co2_emissions_gt
#         self.exec_eng.dm.export_couplings(
#             in_csv=True, f_name='couplings.csv')

#         self.exec_eng.root_process.coupling_structure.graph.export_initial_graph(
#             "initial.pdf")
# self.exec_eng.root_process.coupling_structure.graph.export_reduced_graph(
# "reduced.pdf")
        setup_data_list.append(witness_input)

        return setup_data_list

    def setup_objectives(self):
        func_df = DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['welfare_objective', 'min_utility_objective', 'temperature_objective', 'CO2_objective', 'ppm_objective'])
        list_parent.extend(['utility_objective', 'utility_objective',
                            'CO2_obj', 'CO2_obj', 'CO2_obj'])
        list_ns.extend(['ns_functions', 'ns_functions',
                        'ns_functions', 'ns_functions', 'ns_functions'])
        list_ftype.extend(
            [OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE])
        list_weight.extend([1.0, 0.0, 0.0, 1.0, 0.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM, AGGR_TYPE_SMAX, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM])

        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        return func_df

    def setup_constraints(self):
        func_df = pd.DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['rockstrom_limit_constraint'])
        list_parent.extend([''])
        list_ns.extend(['ns_functions'])
        list_ftype.extend([INEQ_CONSTRAINT])
        list_weight.extend([0.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SMAX])

        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns
        return func_df
