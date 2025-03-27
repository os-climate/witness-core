'''
Copyright 2024 Capgemini

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
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization.usecase_witness_coarse_sectorization import (
    Study as witness_usecase_secto,
)

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
OPTIM_NAME = "MDO"
COUPLING_NAME = "WITNESS_Eval"
EXTRA_NAME = "WITNESS"


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault,
                 bspline=True, run_usecase=False,
                 execution_engine=None, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.dspace_size: int = 0
        self.year_start = year_start
        self.year_end = year_end
        
        self.coupling_name = COUPLING_NAME
        self.designvariable_name = "DesignVariables"
        self.func_manager_name = "FunctionsManager"
        self.extra_name = EXTRA_NAME
        self.energy_mix_name = 'EnergyMix'
        self.bspline = bspline
        self.techno_dict = techno_dict
        self.witness_uc = witness_usecase_secto(
            year_start=self.year_start, year_end=self.year_end, bspline=self.bspline,
            execution_engine=execution_engine, techno_dict=techno_dict)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict
        self.test_post_procs = True

    def dspace_sectorization(self) -> dict:
        dspace_dict = {}
        dspace_size = 0

        invest_val_year_start = {
            GlossaryCore.SectorServices: DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(year=self.year_start) * DatabaseWitnessCore.InvestServicespercofgdpYearStart.value / 100.,
            GlossaryCore.SectorAgriculture: DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(year=self.year_start) * DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value / 100.,
            GlossaryCore.SectorIndustry: DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(year=self.year_start) * DatabaseWitnessCore.InvestInduspercofgdp2020.value / 100.
        }

        for sector, val_year_start in invest_val_year_start.items():
            design_var_name = f"{sector}_invest_array"
            dspace_size += GlossaryCore.NB_POLES_SECTORS_DVAR
            dspace_dict[design_var_name] = {
                'value': [np.round(val_year_start, 2)] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'activated_elem': [False] + [True] * (GlossaryEnergy.NB_POLES_SECTORS_DVAR - 1),
                'lower_bnd': [np.round(val_year_start / 10., 2) ] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'upper_bnd': [np.round(val_year_start * 5, 2) ] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'enable_variable': True
            }
        share_energy_val_year_start = {
            GlossaryCore.SectorServices: DatabaseWitnessCore.EnergyshareServicesYearStart.value,
            GlossaryCore.SectorAgriculture: DatabaseWitnessCore.EnergyshareAgricultureYearStart.value,
        }

        for sector, val_year_start in share_energy_val_year_start.items():
            design_var_name = f"{sector}_share_energy_array"
            dspace_size += GlossaryCore.NB_POLES_SECTORS_DVAR
            dspace_dict[design_var_name] = {
                'value': [np.round(val_year_start, 2)] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'activated_elem': [False] + [True] * (GlossaryEnergy.NB_POLES_SECTORS_DVAR - 1),
                'lower_bnd': [np.round(val_year_start / 10., 2)] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'upper_bnd': [np.round(val_year_start * 1.6, 2)] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'enable_variable': True
            }

        dspace_dict["dspace_size"] = dspace_size
        return dspace_dict

    def get_dvar_descriptor_sectorization(self, dspace: dict) -> tuple[dict, dict]:
        """returns design var array dict and design var descriptor"""
        years = np.arange(self.year_start, self.year_end + 1)
        dv_arrays_dict = {}
        design_var_descriptor = {}

        # share invest dvars
        for sector in GlossaryCore.SectorsPossibleValues:
            dvar_value = dspace[f'{sector}_invest_array']['value']
            activated_dvar = dspace[f'{sector}_invest_array']['activated_elem']
            activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

            dv_arrays_dict[f'{self.study_name}.{self.coupling_name}.{self.extra_name}.Macroeconomics.{sector}_invest_array'] = activated_value

            design_var_descriptor[f'{sector}_invest_array'] = {
                'out_name': f"{sector}.invest_mdo_df",
                'out_type': 'dataframe',
                'key': GlossaryCore.InvestmentsValue,
                'index': years,
                'index_name': GlossaryCore.Years,
                'namespace_in': GlossaryCore.NS_SECTORS,
                'namespace_out': GlossaryCore.NS_SECTORS
            }
            # share invest dvars

        for sector in GlossaryCore.SectorsValueOptim:
            dvar_value = dspace[f'{sector}_share_energy_array']['value']
            activated_dvar = dspace[f'{sector}_share_energy_array']['activated_elem']
            activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

            dv_arrays_dict[
                f'{self.study_name}.{self.coupling_name}.{self.extra_name}.Macroeconomics.{sector}_share_energy_array'] = activated_value

            design_var_descriptor[f'{sector}_share_energy_array'] = {
                'out_name': f"{sector}.{GlossaryCore.ShareSectorEnergyDfValue}",
                'out_type': 'dataframe',
                'key': GlossaryCore.ShareSectorEnergy,
                'index': years,
                'index_name': GlossaryCore.Years,
                'namespace_in': GlossaryCore.NS_SECTORS,
                'namespace_out': GlossaryCore.NS_SECTORS
            }

        return dv_arrays_dict, design_var_descriptor

    def setup_func_df(self):
        constraints_energy_mix = {}

        anti_decreasing_net_gdp_obj = {
            'variable': [GlossaryCore.DecreasingGdpIncrementsObjectiveValue],
            'parent': [GlossaryCore.DecreasingGdpIncrementsObjectiveValue],
            'ftype': [FunctionManagerDisc.OBJECTIVE],
            'weight': [3],
            FunctionManagerDisc.AGGR_TYPE: [FunctionManager.AGGR_TYPE_SUM],
            'namespace': [GlossaryCore.NS_FUNCTIONS]
        }

        welfare_secto = {
            'variable': [f"{sector}.{GlossaryCore.UtilityObjectiveName}" for sector in GlossaryCore.SectorsPossibleValues],
            'parent': ["sectorized welfare"] * 3,
            'ftype': [FunctionManagerDisc.OBJECTIVE] * 3,
            'weight': [1/3] * 3,
            FunctionManagerDisc.AGGR_TYPE: [FunctionManager.AGGR_TYPE_SUM] * 3,
            'namespace': [GlossaryCore.NS_FUNCTIONS] * 3
        }


        func_df = pd.concat([pd.DataFrame(var) for var in [
            welfare_secto, constraints_energy_mix, anti_decreasing_net_gdp_obj
        ]])

        return func_df

    def setup_usecase(self, study_folder_path=None):
        """ Overloaded method to initialize witness multiscenario optimization process

        @return list of dictionary: [{str: *}]
        """
        setup_data_list = {}

        # -- retrieve energy input data

        self.witness_uc.study_name = f'{self.study_name}.{self.coupling_name}.{self.extra_name}'
        self.witness_uc.study_name_wo_extra_name = f'{self.study_name}.{self.coupling_name}'
        witness_data_list = self.witness_uc.setup_usecase()
        setup_data_list.update(witness_data_list)

        setup_data_list[f'{self.study_name}.epsilon0'] = 1.0
        setup_data_list[f'{self.study_name}.{self.coupling_name}.inner_mda_name'] = 'MDAGaussSeidel'
        setup_data_list[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 50
        setup_data_list[f'{self.study_name}.{self.coupling_name}.tolerance'] = 1e-10
        setup_data_list[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        setup_data_list[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0
        setup_data_list[f'{self.witness_uc.study_name}.{self.coupling_name}.{GlossaryCore.energy_list}'] = self.witness_uc.energy_list

        setup_data_list[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.function_df'] = self.setup_func_df()

        dspace_energy_mix = self.witness_uc.dspace
        dspace_sectorization = self.dspace_sectorization()

        dv_arrays_dict_energy_mix, design_var_descriptor_energy_mix = self.get_dvar_descriptor_energy_mix(dspace=dspace_energy_mix)
        dv_arrays_dict_sectorization, design_var_descriptor_sectorization = self.get_dvar_descriptor_sectorization(dspace=dspace_sectorization)

        design_var_descriptor = {}
        design_var_descriptor.update(design_var_descriptor_energy_mix)
        design_var_descriptor.update(design_var_descriptor_sectorization)

        setup_data_list.update(dv_arrays_dict_energy_mix)
        setup_data_list.update(dv_arrays_dict_sectorization)

        dspace = self.merge_design_spaces_dict(dspace_list=[dspace_energy_mix, dspace_sectorization])
        self.dspace_size, self.dspace = self.dspace_dict_to_dataframe(dspace)

        setup_data_list[f'{self.study_name}.design_space'] = self.dspace
        setup_data_list[f'{self.study_name}.{self.coupling_name}.{self.extra_name}.mdo_mode'] = True
        setup_data_list[f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor


        self.remove_all_variables_in_values_dict(values_dict=setup_data_list, shortvarname=GlossaryCore.ShareSectorInvestmentDfValue)
        self.remove_all_variables_in_values_dict(values_dict=setup_data_list, shortvarname=GlossaryCore.ShareSectorEnergyDfValue)
        self.remove_all_variables_in_values_dict(values_dict=setup_data_list, shortvarname=GlossaryCore.EnergyInvestmentsWoTaxValue)


        agri_subsector_invests = pd.DataFrame({
            GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
            GlossaryCore.Crop: 90.,
            GlossaryCore.Forestry: 10.,
        })
        setup_data_list[f'{self.study_name}.{self.coupling_name}.WITNESS.Macroeconomics.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}'] = agri_subsector_invests
        setup_data_list[f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
