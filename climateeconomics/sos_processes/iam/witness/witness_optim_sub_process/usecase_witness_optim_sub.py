'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/06 Copyright 2023 Capgemini

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
from collections import defaultdict

import numpy as np
import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)

# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import (
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
)
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import (
    Study as witness_usecase,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
OPTIM_NAME = "WITNESS_MDO"
COUPLING_NAME = "WITNESS_Eval"
EXTRA_NAME = "WITNESS"


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault,
                 bspline=False, run_usecase=False,

                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[
                     2], techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT, agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 process_level='dev'):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        
        self.coupling_name = COUPLING_NAME
        self.designvariable_name = "DesignVariables"
        self.func_manager_name = "FunctionsManager"
        self.extra_name = EXTRA_NAME
        self.energy_mix_name = 'EnergyMix'
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.agri_techno_list = agri_techno_list
        self.process_level = process_level
        self.witness_uc = witness_usecase(
            year_start=self.year_start, year_end=self.year_end, bspline=self.bspline,
            execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict, process_level=process_level,
            agri_techno_list=agri_techno_list)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        """ Overloaded method to initialize witness multiscenario optimization process

        @return list of dictionary: [{str: *}]
        """
        setup_data_list = []

        # -- retrieve energy input data

        self.witness_uc.study_name = f'{self.study_name}.{self.coupling_name}.{self.extra_name}'
        self.witness_uc.study_name_wo_extra_name = f'{self.study_name}.{self.coupling_name}'
        witness_data_list = self.witness_uc.setup_usecase()
        setup_data_list = setup_data_list + witness_data_list

        dspace_df = self.witness_uc.dspace
        values_dict = {}

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        dv_arrays_dict = {}

        design_var_descriptor = {}
        years = np.arange(self.year_start, self.year_end + 1)

        for energy in self.witness_uc.energy_list:
            energy_wo_dot = energy.replace('.', '_')
            for technology in self.witness_uc.dict_technos[energy]:
                technology_wo_dot = technology.replace('.', '_')

                dvar_value = dspace_df[
                    f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix']['value']
                activated_dvar = dspace_df[
                    f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix']['activated_elem']
                activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'] = activated_value

                design_var_descriptor[f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'] = {
                    'out_name': GlossaryCore.invest_mix,
                    'out_type': 'dataframe',
                    'key': f'{energy}.{technology}',
                    'index': years,
                    'index_name': GlossaryCore.Years,
                    'namespace_in': GlossaryCore.NS_ENERGY_MIX,
                    'namespace_out': 'ns_invest'
                }

                design_var_utilization_ratio_value = dspace_df[f'{energy}_{technology}_utilization_ratio_array'][
                    'value']
                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}_{technology}_utilization_ratio_array'] = design_var_utilization_ratio_value
                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{technology}.{GlossaryCore.UtilisationRatioValue}'] = pd.DataFrame(
                    data={GlossaryCore.Years: years,
                          GlossaryCore.UtilisationRatioValue: 100.})
                # add design variable for utilization ratio per technology
                design_var_descriptor[f'{energy}_{technology}_utilization_ratio_array'] = {
                    'out_name': f'{energy}.{technology}.{GlossaryCore.UtilisationRatioValue}',
                    'out_type': 'dataframe',
                    'key': GlossaryCore.UtilisationRatioValue,
                    'index': years,
                    'index_name': GlossaryCore.Years,
                    'namespace_in': GlossaryCore.NS_ENERGY_MIX,
                    'namespace_out': GlossaryCore.NS_ENERGY_MIX
                }

        for ccs in self.witness_uc.ccs_list:
            ccs_wo_dot = ccs.replace('.', '_')
            for technology in self.witness_uc.dict_technos[ccs]:
                technology_wo_dot = technology.replace('.', '_')
                dvar_value = dspace_df[
                    f'{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix']['value']
                activated_dvar = dspace_df[
                    f'{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix']['activated_elem']
                activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix'] = \
                    activated_value
                design_var_descriptor[f'{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix'] = {
                    'out_name': GlossaryCore.invest_mix,
                    'out_type': 'dataframe',
                    'key': f'{ccs}.{technology}',
                    'index': years,
                    'index_name': GlossaryCore.Years,
                    'namespace_in': GlossaryCore.NS_CCS,
                    'namespace_out': 'ns_invest'
                }

                design_var_utilization_ratio_value = dspace_df[f'{ccs}.{technology}_utilization_ratio_array']['value']
                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}_utilization_ratio_array'] = design_var_utilization_ratio_value
                dv_arrays_dict[
                    f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}.{GlossaryCore.UtilisationRatioValue}'] = pd.DataFrame(
                    data={GlossaryCore.Years: years,
                          GlossaryCore.UtilisationRatioValue: 100.})
                # add design variable for utilization ratio per technology
                design_var_descriptor[f'{ccs}.{technology}_utilization_ratio_array'] = {
                    'out_name': f'{ccs}.{technology}.{GlossaryCore.UtilisationRatioValue}',
                    'out_type': 'dataframe',
                    'key': GlossaryCore.UtilisationRatioValue,
                    'index': years,
                    'index_name': GlossaryCore.Years,
                    'namespace_in': GlossaryCore.NS_CCS,
                    'namespace_out': GlossaryCore.NS_CCS
                }

        dv_arrays_dict[f'{self.witness_uc.study_name}.forest_investment_array_mix'] = \
            dspace_df['forest_investment_array_mix']['value']
        design_var_descriptor['forest_investment_array_mix'] = {'out_name': 'forest_investment',
                                                                'out_type': 'dataframe',
                                                                'key': 'forest_investment',
                                                                'index': years,
                                                                'index_name': GlossaryCore.Years,
                                                                'namespace_in': GlossaryCore.NS_WITNESS,
                                                                'namespace_out': 'ns_invest'
                                                                }
        if 'CropEnergy' in self.agri_techno_list:
            dv_arrays_dict[f'{self.witness_uc.study_name}.crop_investment_array_mix'] = \
                dspace_df['crop_investment_array_mix']['value']
            design_var_descriptor['crop_investment_array_mix'] = {'out_name': 'crop_investment',
                                                                  'out_type': 'dataframe',
                                                                  'key': GlossaryCore.InvestmentsValue,
                                                                  'index': years,
                                                                  'index_name': GlossaryCore.Years,
                                                                  'namespace_in': GlossaryCore.NS_WITNESS,
                                                                  'namespace_out': 'ns_crop'
                                                                  }
        if 'ManagedWood' in self.agri_techno_list:
            dv_arrays_dict[f'{self.witness_uc.study_name}.managed_wood_investment_array_mix'] = \
                dspace_df['managed_wood_investment_array_mix']['value']
            design_var_descriptor['managed_wood_investment_array_mix'] = {'out_name': 'managed_wood_investment',
                                                                          'out_type': 'dataframe',
                                                                          'key': GlossaryCore.InvestmentsValue,
                                                                          'index': years,
                                                                          'index_name': GlossaryCore.Years,
                                                                          'namespace_in': GlossaryCore.NS_WITNESS,
                                                                          'namespace_out': 'ns_forest'
                                                                          }
        dv_arrays_dict[f'{self.witness_uc.study_name}.deforestation_investment_ctrl'] = \
            dspace_df['deforestation_investment_ctrl']['value']
        design_var_descriptor['deforestation_investment_ctrl'] = {'out_name': 'deforestation_investment',
                                                                  'out_type': 'dataframe',
                                                                  'key': GlossaryCore.InvestmentsValue,
                                                                  'index': years,
                                                                  'index_name': GlossaryCore.Years,
                                                                  'namespace_in': GlossaryCore.NS_WITNESS,
                                                                  'namespace_out': 'ns_forest'
                                                                  }
        dv_arrays_dict[f'{self.witness_uc.study_name}.red_meat_calories_per_day_ctrl'] = \
            np.array(dspace_df['red_meat_calories_per_day_ctrl']['value'])
        design_var_descriptor['red_meat_calories_per_day_ctrl'] = {'out_name': 'red_meat_calories_per_day',
                                                                   'out_type': 'dataframe',
                                                                   'key': 'red_meat_calories_per_day',
                                                                   'index': years,
                                                                   'index_name': GlossaryCore.Years,
                                                                   'namespace_in': GlossaryCore.NS_WITNESS,
                                                                   'namespace_out': 'ns_crop'
                                                                   }
        dv_arrays_dict[f'{self.witness_uc.study_name}.white_meat_calories_per_day_ctrl'] = \
            np.array(dspace_df['white_meat_calories_per_day_ctrl']['value'])
        design_var_descriptor['white_meat_calories_per_day_ctrl'] = {'out_name': 'white_meat_calories_per_day',
                                                                     'out_type': 'dataframe',
                                                                     'key': 'white_meat_calories_per_day',
                                                                     'index': years,
                                                                     'index_name': GlossaryCore.Years,
                                                                     'namespace_in': GlossaryCore.NS_WITNESS,
                                                                     'namespace_out': 'ns_crop'
                                                                     }
        dv_arrays_dict[f'{self.witness_uc.study_name}.vegetables_and_carbs_calories_per_day_ctrl'] = \
            np.array(dspace_df['vegetables_and_carbs_calories_per_day_ctrl']['value'])
        design_var_descriptor['vegetables_and_carbs_calories_per_day_ctrl'] = {
            'out_name': 'vegetables_and_carbs_calories_per_day',
            'out_type': 'dataframe',
            'key': 'vegetables_and_carbs_calories_per_day',
            'index': years,
            'index_name': GlossaryCore.Years,
            'namespace_in': GlossaryCore.NS_WITNESS,
            'namespace_out': 'ns_crop'
        }
        dv_arrays_dict[f'{self.witness_uc.study_name}.milk_and_eggs_calories_per_day_ctrl'] = \
            np.array(dspace_df['milk_and_eggs_calories_per_day_ctrl']['value'])
        design_var_descriptor['milk_and_eggs_calories_per_day_ctrl'] = {
            'out_name': 'milk_and_eggs_calories_per_day',
            'out_type': 'dataframe',
            'key': 'milk_and_eggs_calories_per_day',
            'index': years,
            'index_name': GlossaryCore.Years,
            'namespace_in': GlossaryCore.NS_WITNESS,
            'namespace_out': 'ns_crop'
        }

        func_df = self.witness_uc.func_df
        func_df = func_df[~func_df['variable'].isin(['non_use_capital_cons', 'forest_lost_capital_cons'])]
        func_df.loc[func_df['variable'] == 'land_demand_constraint', 'weight'] = 0.
        func_df.loc[func_df['variable'] == 'calories_per_day_constraint', 'weight'] = 0.
        func_df.loc[func_df['variable'] == 'total_prod_minus_min_prod_constraint_df', 'weight'] = 0.

        # Display func_df after dropping rows

        self.func_df = func_df
        self.design_var_descriptor = design_var_descriptor
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.{FUNC_DF}'] = func_df

        values_dict[
            f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor

        values_dict[f'{self.study_name}.{self.coupling_name}.sub_mda_class'] = 'GSPureNewtonMDA'
        # values_dict[f'{self.study_name}.{self.coupling_name}.warm_start'] = True
        values_dict[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 50
        values_dict[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0
        # design space

        dspace = self.witness_uc.dspace
        self.dspace_size = dspace.pop('dspace_size')

        dspace_dict = defaultdict(list)
        for key, elem in dspace.items():
            dspace_dict['variable'].append(key)
            for column, value in elem.items():
                dspace_dict[column].append(value)

        self.dspace = pd.DataFrame(dspace_dict)
        """

        keys_to_update = ['carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
                          'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                          'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
        for key in keys_to_update:
            self.dspace.loc[self.dspace['variable'] == key, 'value'] = \
                np.array(self.dspace.loc[self.dspace['variable'] == key, 'lower_bnd'])
        """
        values_dict[
            f'{self.witness_uc.study_name}.{self.coupling_name}.{GlossaryCore.energy_list}'] = self.witness_uc.energy_list
        values_dict[f'{self.study_name}.design_space'] = self.dspace
        setup_data_list.append(values_dict)
        setup_data_list.append(dv_arrays_dict)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(process_level='dev')
    uc_cls.test()
