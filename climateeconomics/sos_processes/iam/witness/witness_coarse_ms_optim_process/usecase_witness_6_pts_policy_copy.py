'''
Copyright 2023 Capgemini

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
import numpy as np
import pandas as pd
from os.path import join, dirname

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.study_manager.study_manager import StudyManager
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import Study as witness_optim_usecase
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager


class Study(ClimateEconomicsStudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self):
        witness_ms_usecase = witness_optim_usecase(
            execution_engine=self.execution_engine)

        self.scatter_scenario = 'optimization scenarios'
        # Set public values at a specific namespace
        witness_ms_usecase.study_name = f'{self.study_name}.{self.scatter_scenario}'

        values_dict = {}
        scenario_list = []
        alpha_list = np.linspace(0, 125, 6, endpoint=True)
        for alpha_i in alpha_list:
            scenario_i = f'scenario_policy={alpha_i}%'
            scenario_i = scenario_i.replace('.', ',')
            scenario_list.append(scenario_i)
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.co2_damage_price_percentage'] = alpha_i
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.ccs_price_percentage'] = alpha_i

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = 6

        len_scenarios = len(scenario_list)
        scenario_df = pd.DataFrame({'selected_scenario': [True] * len_scenarios ,'scenario_name': scenario_list})

        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_df'] = scenario_df
        witness_uc_dict = {}
        for scenario in scenario_list:
            scenarioUseCase = witness_optim_usecase(
                bspline=self.bspline, execution_engine=self.execution_engine)
            scenarioUseCase.optim_name = f'{scenario}.{scenarioUseCase.optim_name}'
            scenarioUseCase.study_name = witness_ms_usecase.study_name
            scenarioData = scenarioUseCase.setup_usecase()
            witness_uc_dict[scenario] = scenarioUseCase
            default_func_df = scenarioUseCase.func_df
            # no CO2 obejctive in this formulation
            default_func_df.loc[default_func_df['variable'] == 'CO2_objective', 'weight'] = 0.0

            for dict_data in scenarioData:
                values_dict.update(dict_data)
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario}.{witness_ms_usecase.optim_name}' \
                        f'.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.FunctionsManager.function_df'] = default_func_df
            values_dict[
                    f'{self.study_name}.{self.scatter_scenario}.{scenario}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.alpha'] = 1.
        year_start = scenarioUseCase.year_start
        year_end = scenarioUseCase.year_end
        years = np.arange(year_start, year_end + 1)

        values_dict[f'{self.study_name}.{self.scatter_scenario}.NormalizationReferences.liquid_hydrogen_percentage'] = np.concatenate((np.ones(5)/1e-4,np.ones(len(years)-5)/4), axis=None)
        values_dict[f'{self.study_name}.{self.scatter_scenario}.builder_mode']= 'multi_instance'
        lifetime = 35
        construction_delay = 3
        dac_to_update = {'maturity': 0,
                                 'Opex_percentage': 0.25,
                                 'CO2_per_energy': 0.65,
                                 'CO2_per_energy_unit': 'kg/kg',
                                 'elec_demand': 2.,
                                 'elec_demand_unit': 'kWh/kgCO2',
                                 'heat_demand': 0.,
                                 'heat_demand_unit': 'kWh/kgCO2',
                                 'WACC': 0.1,
                                 'learning_rate': 0.1,
                                 'maximum_learning_capex_ratio': 0.33,
                                 'lifetime': lifetime,
                                 'lifetime_unit': GlossaryCore.Years,
                                 'Capex_init': 0.88,
                                 'Capex_init_unit': '$/kgCO2',
                                 'efficiency': 0.9,
                                 'CO2_capacity_peryear': 3.6E+8,
                                 'CO2_capacity_peryear_unit': 'kg CO2/year',
                                 'real_factor_CO2': 1.0,
                                 'transport_cost': 0.0,
                                 'transport_cost_unit': '$/kgCO2',
                                 'enthalpy': 1.124,
                                 'enthalpy_unit': 'kWh/kgC02',
                                 'energy_efficiency': 0.78,
                                 'construction_delay': construction_delay,
                                 'techno_evo_eff': 'no',
                                 'CO2_from_production': 0.0,
                                 'CO2_from_production_unit': 'kg/kg',
                                 }
        diet_mortality_df = pd.read_csv(join(dirname(__file__), 'data', 'diet_mortality.csv'))
        fossil_properties_df = pd.read_csv(join(dirname(__file__), 'data', 'fossil_properties.csv'))
        fossil_properties_dict = {}
        for index, row in fossil_properties_df.iterrows():
            try:
                fossil_properties_dict[row["variable"]] = eval(row["value"])
            except:
                fossil_properties_dict[row["variable"]] = row["value"]
        # overload values 
        values_dict_updt = {}
        for scenario in scenario_list:
            witness_uc = witness_uc_dict[scenario]
            dspace = witness_uc.witness_uc.dspace 
            list_design_var_to_clean = ['red_meat_calories_per_day_ctrl', 'white_meat_calories_per_day_ctrl', 'vegetables_and_carbs_calories_per_day_ctrl', 'milk_and_eggs_calories_per_day_ctrl', 'forest_investment_array_mix', 'deforestation_investment_ctrl']

            # clean dspace
            dspace.drop(dspace.loc[dspace['variable'].isin(list_design_var_to_clean)].index, inplace=True)

            # clean dspace descriptor 
            dvar_descriptor = witness_uc.witness_uc.design_var_descriptor
            
            updated_dvar_descriptor = {k:v for k,v in dvar_descriptor.items() if k not in list_design_var_to_clean}

            dspace_file_name = f'optimization scenarios.{scenario}.WITNESS_MDO.design_space_out.csv'
            dspace_out = pd.read_csv(join(dirname(__file__), 'data', 'data_dvar', dspace_file_name))
            
            for index, row in dspace.iterrows():
                variable = row["variable"]
                
                if variable in dspace_out["variable"].values:
                    valeur_str = dspace_out[dspace_out["variable"] == variable]["value"].iloc[0]
                    upper_bnd_str = dspace_out[dspace_out["variable"] == variable]["upper_bnd"].iloc[0]
                    lower_bnd_str = dspace_out[dspace_out["variable"] == variable]["lower_bnd"].iloc[0]
                    activated_elem_str = dspace_out[dspace_out["variable"] == variable]["activated_elem"].iloc[0]

                    valeur_array = np.array(eval(valeur_str))
                    upper_bnd_array = np.array(eval(upper_bnd_str))
                    lower_bnd_array = np.array(eval(lower_bnd_str))
                    activated_elem_array = eval(activated_elem_str)

                    dspace.at[index, "value"] = valeur_array
                    dspace.at[index, "upper_bnd"] = upper_bnd_array
                    dspace.at[index, "lower_bnd"] = lower_bnd_array
                    dspace.at[index, "activated_elem"] = activated_elem_array
                    values_dict_updt.update({f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.EnergyMix.{variable}':valeur_array,
                                             f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.CCUS.{variable}':valeur_array})
            dspace['enable_variable'] = False

            invest_mix_file = f'optimization scenarios.{scenario}.invest_mix.csv'
            invest_mix = pd.read_csv(join(dirname(__file__), 'data', 'invest_mix', invest_mix_file))
            forest_invest_file = f'optimization scenarios.{scenario}.forest_investment.csv'
            forest_invest = pd.read_csv(join(dirname(__file__), 'data', 'invest_mix', forest_invest_file))

            DAC_name = f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno'
            fossil_energy_name = f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.EnergyMix.fossil'
            values_dict_updt.update({f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.design_space' : dspace,
            f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.witness_uc.designvariable_name}.design_var_descriptor': updated_dvar_descriptor, 
            f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.InvestmentDistribution.invest_mix': invest_mix, 
            f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.InvestmentDistribution.forest_investment': forest_invest,
            f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.AgricultureMix.Forest.reforestation_cost_per_ha': 3800.,
            f'{self.study_name}.{self.scatter_scenario}.{witness_uc.optim_name}.{witness_uc.coupling_name}.WITNESS.Population.diet_mortality_param_df': diet_mortality_df,



            f'{DAC_name}.techno_infos_dict': dac_to_update, 
            f'{fossil_energy_name}.data_fuel_dict': fossil_properties_dict
            })


        values_dict.update(values_dict_updt)
        # overload design var with optimal points
            # delete diet design variables 
            
            # set investment values 

        # convert optimal diet to calories per day 


        # overload carbon capture properties 


        # overload kwh/kg CO2 of fossil (because of other uses)


        # change reforestation cost


        # deactivate diet effect on population
        

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()


#    for graph in graph_list:
#        graph.to_plotly().show()
