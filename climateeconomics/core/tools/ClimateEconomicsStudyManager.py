'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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
from typing import Any

import numpy as np
import pandas as pd
from energy_models.database_witness_energy import DatabaseWitnessEnergy
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class ClimateEconomicsStudyManager(StudyManager):
    '''
    Class that overloads study manager to define a specific check for climate economics usecases
    '''
    def should_be_lower(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value > ref_value:
            msg = f"\n{varname:140} should be lower than {ref_value} but is not. Value = {actual_value}"
        return msg

    def should_be_greater(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value < ref_value:
            msg = f"\n{varname:>140} should be greater than {ref_value} but is not. Value = {actual_value}"
        return msg

    @staticmethod
    def update_dataframes_with_year_star(values_dict: dict,  year_start: int) -> dict:
        """truncate all dataframe rows that are before year start"""
        year_start_varnames = list(filter(lambda x: f".{GlossaryCore.YearStart}" in x, values_dict.keys()))
        values_dict.update({varname: year_start for varname in year_start_varnames})

        values_dict_2023 = {}
        for key, value in values_dict.items():
            if isinstance(value, pd.DataFrame) and GlossaryCore.Years in value.columns:
                new_value = value.loc[value[GlossaryCore.Years] >= year_start]
                values_dict_2023[key] = new_value
        values_dict.update(values_dict_2023)
        return values_dict

    @staticmethod
    def get_share_invest_by_techno_of_total_energy_invest_for_coarse(selected_year: int):
        """
        returns the share invested in each witnes coarse techno relative to total invests in energy sector at the year selected
        """

        technos_coarse = []

        for stream, technos in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items():
            technos_coarse.extend(technos['value'])
        total_energy_invest_selected_year = 0
        invests = []
        for techno in technos_coarse:
            a, b = DatabaseWitnessEnergy.get_techno_invest_before_year_start(techno_name=techno,
                                                                             year_start=selected_year + 1,
                                                                             construction_delay=1)
            invests.append(a['invest'].values[0])
            total_energy_invest_selected_year += a['invest'].values[0]


        share_invest_by_techno = np.round(np.array(invests) / total_energy_invest_selected_year * 100, 2)

        share_invest_by_techno = dict(zip(technos_coarse, share_invest_by_techno))
        return share_invest_by_techno

    @staticmethod
    def get_share_invest_in_eneryg_relative_to_gdp(selected_year: int):
        """returns the share invested in energy relative to total gdp"""

        technos_coarse = []

        for stream, technos in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items():
            technos_coarse.extend(technos['value'])
        total_energy_invest_selected_year = 0
        invests = []
        for techno in technos_coarse:
            a, b = DatabaseWitnessEnergy.get_techno_invest_before_year_start(techno_name=techno,
                                                                             year_start=selected_year + 1,
                                                                             construction_delay=1)
            invests.append(a['invest'].values[0])
            total_energy_invest_selected_year += a['invest'].values[0]

        world_gdp = DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(selected_year)

        percent_of_gdp = total_energy_invest_selected_year / 1000 / world_gdp * 100

        return np.round(percent_of_gdp, 2)

    def get_dvar_descriptor_energy_mix(self, dspace: dict) -> tuple[dict, dict]:
        """returns design var array dict and design var descriptor"""
        years = np.arange(self.year_start, self.year_end + 1)
        dv_arrays_dict = {}
        design_var_descriptor = {}

        for energy in self.witness_uc.energy_list:
            energy_wo_dot = energy.replace('.', '_')
            for technology in self.witness_uc.dict_technos[energy]:
                technology_wo_dot = technology.replace('.', '_')

                dvar_value = dspace[f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix']['value']
                activated_dvar = dspace[f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix']['activated_elem']
                activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'] = activated_value

                design_var_descriptor[f'{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'] = {
                    'out_name': GlossaryCore.invest_mix,
                    'out_type': 'dataframe',
                    'key': f'{energy}.{technology}',
                    'index': years,
                    'index_name': GlossaryCore.Years,
                    'namespace_in': GlossaryCore.NS_ENERGY_MIX,
                    'namespace_out': 'ns_invest'
                }

                design_var_utilization_ratio_value = dspace[f'{energy}_{technology}_utilization_ratio_array'][
                    'value']
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}_{technology}_utilization_ratio_array'] = design_var_utilization_ratio_value
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{technology}.{GlossaryCore.UtilisationRatioValue}'] = pd.DataFrame(
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
                dvar_value = dspace[f'{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix']['value']
                activated_dvar = dspace[f'{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix']['activated_elem']
                activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

                dv_arrays_dict[f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix'] = \
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

                design_var_utilization_ratio_value = dspace[f'{ccs}.{technology}_utilization_ratio_array']['value']
                dv_arrays_dict[f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}_utilization_ratio_array'] = design_var_utilization_ratio_value
                dv_arrays_dict[f'{self.witness_uc.study_name}.{GlossaryCore.ccus_type}.{ccs}.{technology}.{GlossaryCore.UtilisationRatioValue}'] = pd.DataFrame(
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

        return dv_arrays_dict, design_var_descriptor

    def get_var_in_values_dict(self, values_dict: dict, varname: str) -> list:
        return [values_dict[x] for x in list(filter(lambda key: varname in key, values_dict.keys()))]

    def get_dict_values(self, values_dict: dict, varname: str) -> dict:
        return {x : values_dict[x] for x in list(filter(lambda key: varname in key, values_dict.keys()))}

    def get_fullname_in_values_dict(self, values_dict: dict, varname: str) -> list[str]:
        return list(filter(lambda key: varname in key, values_dict.keys()))

    def remove_all_variables_in_values_dict(self, values_dict: dict, shortvarname: str) -> dict:
        variables_to_remove = self.get_fullname_in_values_dict(values_dict=values_dict, varname=shortvarname)
        for var in variables_to_remove:
            del values_dict[var]

    def dspace_dict_to_dataframe(self, dspace: dict) -> tuple[int, pd.DataFrame]:
        dspace_size = dspace.pop("dspace_size")
        dspace_dict = defaultdict(list)
        for key, elem in dspace.items():
            dspace_dict['variable'].append(key)
            for column, value in elem.items():
                dspace_dict[column].append(value)

        dspace_df = pd.DataFrame(dspace_dict)
        return dspace_size, dspace_df

    def merge_design_spaces_dict(self, dspace_list: list[dict[str, Any]]) -> dict:
        """Update the design space from a list of other design spaces.

        It is necessary to use a set difference here, instead of dictionary update,
        to correctly update the design space size.

        Args:
            dspace_list: The list of design spaces to add.

        Raises:
            ValueError: If some variables are duplicated in several design spaces.
        """
        dspace_out = {}
        dspace_size_out = 0
        for dspace in dspace_list:
            dspace_size = dspace.pop("dspace_size")
            duplicated_variables = set(dspace.keys()).intersection(self.dspace.keys())
            if duplicated_variables:
                msg = (
                    "Failed to merge the design spaces; "
                    f"the following variables are present multiple times: {' '.join(duplicated_variables)}"
                )
                raise ValueError(msg)
            dspace_size_out += dspace_size
            dspace_out.update(dspace)
        dspace_out["dspace_size"] = dspace_size_out
        return dspace_out

    def load_data(self, from_path=None, from_input_dict=None, display_treeview=True, from_datasets_mapping=None):
        parameter_changes = super().load_data(from_path=from_path, from_input_dict=from_input_dict,
                                              display_treeview=display_treeview,
                                              from_datasets_mapping=from_datasets_mapping)
        lin_mode_dict = {}
        linearization_mode_dict = self.add_auto_linearization_mode_rec(self.execution_engine.root_process,
                                                                       lin_mode_dict)
        lin_parameter_changes = self.execution_engine.load_study_from_input_dict(linearization_mode_dict)
        parameter_changes.extend(lin_parameter_changes)
        return parameter_changes

    def add_auto_linearization_mode_rec(self, disc, lin_mode_dict):
        lin_mode_dict[f"{disc.get_disc_full_name()}.linearization_mode"] = 'auto'
        for sub_disc in disc.proxy_disciplines:
            lin_mode_dict = self.add_auto_linearization_mode_rec(sub_disc, lin_mode_dict)
        return lin_mode_dict
