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

from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc import CropDiscipline
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from os.path import join, dirname
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import \
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT
from climateeconomics.sos_processes.iam.witness.land_use_v2_process.usecase import Study as datacase_landuse
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import Study as datacase_agriculture_mix
from climateeconomics.sos_processes.iam.witness.resources_process.usecase import Study as datacase_resource

from climateeconomics.sos_processes.iam.witness.agriculture_process.usecase import update_dspace_dict_with
from climateeconomics.glossarycore import GlossaryCore

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_DELTA = FunctionManager.AGGR_TYPE_DELTA
AGGR_TYPE_LIN_TO_QUAD = FunctionManager.AGGR_TYPE_LIN_TO_QUAD


class DataStudy():
    def __init__(self, year_start=2020, year_end=2100, time_step=1,
                 agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT):
        self.study_name = 'default_name'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.techno_dict = agri_techno_list
        self.study_name_wo_extra_name = self.study_name
        self.dspace = {}
        self.dspace['dspace_size'] = 0

    def setup_usecase(self):
        setup_data_list = []
        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)

        forest_invest = np.linspace(5.0, 8.0, len(years))
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})

        # private values economics operator pyworld3
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
        witness_input[f'{self.study_name}.InvestmentDistribution.forest_investment'] = self.forest_invest_df
        # get population from csv file
        # get file from the data folder 3 folder up.
        global_data_dir = join(Path(__file__).parents[3], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        witness_input[self.study_name + '.population_df'] = population_df
        working_age_population_df = pd.DataFrame(
            {'years': years, 'population_1570': 6300}, index=years)
        witness_input[self.study_name +
                      '.working_age_population_df'] = working_age_population_df

        share_energy_investment = DataFrame(
            {'years': years,
             'energy': asarray([1.65] * nb_per)},
            index=years)

        share_non_energy_investment = DataFrame(
            {'years': years,
             'non_energy': asarray([27. - 1.65] * nb_per)},
            index=years)

        witness_input[f'{self.study_name}.share_energy_investment'] = share_energy_investment
        witness_input[f'{self.study_name}.share_non_energy_investment'] = share_non_energy_investment

        # deactive mortality due to undernutrition/overnutrition:
        diet_mortality = pd.read_csv(join(global_data_dir, 'diet_mortality_param.csv'))
        diet_mortality['undernutrition'] = 0.
        diet_mortality['overnutrition'] = 0.
        witness_input[f'{self.study_name}.Population.{GlossaryCore.DietMortalityParamDf["var_name"]}'] = diet_mortality

        witness_input[f'{self.study_name}.AgricultureMix.Crop.red_meat_calories_per_day'] = DataFrame(
            {'years': years,
             'red_meat_calories_per_day': [CropDiscipline.red_meat_average_ca_daily_intake] * len(years)}
        )

        witness_input[f'{self.study_name}.AgricultureMix.Crop.white_meat_calories_per_day'] = DataFrame(
            {'years': years,
             'white_meat_calories_per_day': [CropDiscipline.white_meat_average_ca_daily_intake] * len(years)}
        )

        witness_input[f'{self.study_name}.AgricultureMix.Crop.vegetables_and_carbs_calories_per_day'] = DataFrame(
            {'years': years,
             'vegetables_and_carbs_calories_per_day': [CropDiscipline.vegetables_and_carbs_average_ca_daily_intake] * len(years)}
        )

        witness_input[f'{self.study_name}.AgricultureMix.Crop.milk_and_eggs_calories_per_day'] = DataFrame(
            {'years': years,
             'milk_and_eggs_calories_per_day': [CropDiscipline.milk_eggs_average_ca_daily_intake] * len(years)}
        )

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

        witness_input[self.study_name +
                      '.agri_capital_techno_list'] = []

        CO2_emitted_land = pd.DataFrame()
        # GtCO2
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest)
        CO2_emitted_land['Crop'] = np.zeros(len(years))
        CO2_emitted_land['Forest'] = cum_emission

        witness_input[self.study_name +
                      '.CO2_land_emissions'] = CO2_emitted_land

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

        forest_invest = np.linspace(5.0, 8.0, len(years))
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})

        # -- load data from resource
        dc_resource = datacase_resource(
            self.year_start, self.year_end)
        dc_resource.study_name = self.study_name

        # -- load data from land use
        dc_landuse = datacase_landuse(
            self.year_start, self.year_end, self.time_step, name='.Land_Use_V2', extra_name='.EnergyMix')
        dc_landuse.study_name = self.study_name

        # -- load data from agriculture
        dc_agriculture_mix = datacase_agriculture_mix(
            self.year_start, self.year_end, self.time_step, agri_techno_list=self.techno_dict)
        dc_agriculture_mix.additional_ns = '.InvestmentDistribution'
        dc_agriculture_mix.study_name = self.study_name

        resource_input_list = dc_resource.setup_usecase()
        setup_data_list = setup_data_list + resource_input_list

        land_use_list = dc_landuse.setup_usecase()
        setup_data_list = setup_data_list + land_use_list

        agriculture_list = dc_agriculture_mix.setup_usecase()
        setup_data_list = setup_data_list + agriculture_list
        self.dspace_size = dc_agriculture_mix.dspace.pop('dspace_size')
        self.dspace.update(dc_agriculture_mix.dspace)
        nb_poles = 8
        update_dspace_dict_with(self.dspace, 'share_energy_investment_ctrl',
                                asarray([1.65] * nb_poles),
                                asarray([0.5] * nb_poles),
                                asarray([5.0] * nb_poles), enable_variable=False)
        # WITNESS
        # setup objectives
        share_energy_investment = DataFrame(
            {'years': years,
             'energy': asarray([1.65] * nb_per)},
            index=years)

        share_non_energy_investment = DataFrame(
            {'years': years,
             'non_energy': asarray([27. - 1.65] * nb_per)},
            index=years)

        witness_input[f'{self.study_name}.share_energy_investment'] = share_energy_investment
        witness_input[f'{self.study_name}.share_non_energy_investment'] = share_non_energy_investment
        witness_input[f'{self.study_name}.Macroeconomics.CO2_tax_efficiency'] = default_co2_efficiency

        witness_input[f'{self.study_name}.beta'] = 1.0
        witness_input[f'{self.study_name}.gamma'] = 0.5
        witness_input[f'{self.study_name}.init_discounted_utility'] = 4000.0

        witness_input[f'{self.study_name}.init_rate_time_pref'] = 0.0
        witness_input[f'{self.study_name}.total_emissions_ref'] = 7.2
        witness_input[f'{self.study_name}.total_emissions_damage_ref'] = 18.0
        witness_input[f'{self.study_name}.temperature_change_ref'] = 1.0
        witness_input[f'{self.study_name_wo_extra_name}.NormalizationReferences.total_emissions_ref'] = 12.0
        # 

        GHG_total_energy_emissions = pd.DataFrame({'years': years,
                                                   'Total CO2 emissions': np.linspace(37., 10., len(years)),
                                                   'Total N2O emissions': np.linspace(1.7e-3, 5.e-4, len(years)),
                                                   'Total CH4 emissions': np.linspace(0.17, 0.01, len(years))})
        witness_input[f'{self.study_name}.GHG_total_energy_emissions'] = GHG_total_energy_emissions
        # witness_input[f'{self.name}.CO2_emissions_Gt'] = co2_emissions_gt
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
            ['welfare_objective', 'gwp20_objective', 'gwp100_objective', 'non_use_capital_objective',
             'delta_capital_objective',
             'delta_capital_objective_weighted'])
        list_parent.extend(['utility_objective',
                            'GWP_short_term_obj',
                            'GWP_long_term_obj', 'non_use_capital_objective', 'delta_capital_objective',
                            'delta_capital_objective_weighted'])
        list_ns.extend(['ns_functions',
                        'ns_functions',
                        'ns_functions', 'ns_witness', 'ns_functions', 'ns_functions'])
        list_ftype.extend(
            [OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE])
        list_weight.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM])

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
        # -------------------------------------------------
        # CO2 ppm constraints
        list_var.extend(
            ['rockstrom_limit_constraint', 'minimum_ppm_constraint'])
        list_parent.extend(['CO2 ppm', 'CO2 ppm'])
        list_ns.extend(['ns_functions', 'ns_functions'])
        list_ftype.extend([INEQ_CONSTRAINT, INEQ_CONSTRAINT])
        list_weight.extend([0.0, -1.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SMAX, AGGR_TYPE_SMAX])

        # -------------------------------------------------
        # e_max_constraint
        list_var.append('emax_enet_constraint')
        list_parent.append('macroeconomics_constraints')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(-1.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        # -------------------------------------------------
        # calories_per_day_constraint
        list_var.append('calories_per_day_constraint')
        list_parent.append('agriculture_constraints')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(-1.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        # -------------------------------------------------
        # pc_consumption_constraint
        list_var.append('pc_consumption_constraint')
        list_parent.append('macroeconomics_constraints')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(0.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        list_var.extend(['delta_capital_constraint', 'delta_capital_constraint_dc', 'delta_capital_lintoquad'])
        list_parent.extend(['invests_constraints', 'invests_constraints', 'invests_constraints'])
        list_ns.extend(['ns_functions', 'ns_functions', 'ns_functions'])
        list_ftype.extend([INEQ_CONSTRAINT, INEQ_CONSTRAINT, EQ_CONSTRAINT])
        list_weight.extend([-1.0, 0.0, 0.0])
        list_aggr_type.extend([
            AGGR_TYPE_SMAX, AGGR_TYPE_SMAX, AGGR_TYPE_LIN_TO_QUAD])

        list_var.append('non_use_capital_cons')
        list_parent.append('invests_constraints')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(-1.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        list_var.append('forest_lost_capital_cons')
        list_parent.append('agriculture_constraint')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(-1.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        return func_df
