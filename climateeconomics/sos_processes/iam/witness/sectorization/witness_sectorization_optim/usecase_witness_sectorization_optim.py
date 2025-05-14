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
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim_sub_process.datacase import (
    COUPLING_NAME,
    EXTRA_NAME,
)
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim_sub_process.datacase import (
    Study as witness_sectorization_optim_sub_usecase,
)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, bspline=True, run_usecase=False,
                 execution_engine=None, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT, filename=__file__):
        super().__init__(filename, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.optim_name = "MDO"
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.designvariable_name = "DesignVariables"
        self.func_manager_name = "FunctionsManager"
        self.bspline = bspline
        self.techno_dict = techno_dict
        self.sub_uc = witness_sectorization_optim_sub_usecase(
            self.year_start, self.year_end, bspline=self.bspline, execution_engine=execution_engine,
            techno_dict=techno_dict,)
        self.witness_uc = self.sub_uc.witness_uc
        self.sub_study_path_dict = self.sub_uc.sub_study_path_dict
        self.test_post_procs = False

    def setup_process(self):
        witness_sectorization_optim_sub_usecase.setup_process(self)

    def dspace_sectorization(self) -> dict:
        dspace_dict = {}
        dspace_size = 0

        invest_val_year_start = {
            GlossaryCore.SectorServices: DatabaseWitnessCore.InvestServicespercofgdpYearStart.value,
            GlossaryCore.SectorAgriculture: DatabaseWitnessCore.InvestAgriculturepercofgdpYearStart.value,
            GlossaryCore.SectorIndustry: DatabaseWitnessCore.InvestInduspercofgdp2020.value
        }

        for sector, val_year_start in invest_val_year_start.items():
            design_var_name = f"{sector}_invest_array"
            dspace_size += GlossaryCore.NB_POLES_SECTORS_DVAR
            dspace_dict[design_var_name] = {
                'value': [np.round(val_year_start, 2)] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'activated_elem': [False] + [True] * (GlossaryEnergy.NB_POLES_SECTORS_DVAR - 1),
                'lower_bnd': [0.01] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
                'upper_bnd': [25.] * GlossaryEnergy.NB_POLES_SECTORS_DVAR,
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
        for sector in [GlossaryCore.SectorIndustry, GlossaryCore.SectorServices]:
            dvar_value = dspace[f'{sector}_invest_array']['value']
            activated_dvar = dspace[f'{sector}_invest_array']['activated_elem']
            activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

            dv_arrays_dict[f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.extra_name}.Macroeconomics.{sector}_invest_array'] = activated_value

            design_var_descriptor[f'{sector}_invest_array'] = {
                'out_name': f"{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}",
                'out_type': 'dataframe',
                'key': GlossaryCore.ShareInvestment,
                'index': years,
                'index_name': GlossaryCore.Years,
                'namespace_in': GlossaryCore.NS_SECTORS,
                'namespace_out': GlossaryCore.NS_SECTORS
            }
            # share invest dvars

        # Agriculture
        sector = GlossaryCore.SectorAgriculture
        dvar_value = dspace[f'{sector}_invest_array']['value']
        activated_dvar = dspace[f'{sector}_invest_array']['activated_elem']
        activated_value = np.array([elem for i, elem in enumerate(dvar_value) if activated_dvar[i]])

        dv_arrays_dict[
            f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.extra_name}.Macroeconomics.{sector}_invest_array'] = activated_value

        design_var_descriptor[f'{sector}_invest_array'] = {
            'out_name': f"{GlossaryCore.ShareSectorInvestmentDfValue}",
            'out_type': 'dataframe',
            'key': sector,
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
        ns = self.study_name

        self.sub_uc.study_name = f'{ns}.{self.optim_name}'
        self.coupling_name = self.sub_uc.coupling_name
        values_dict = self.sub_uc.setup_usecase()

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        values_dict[f'{self.study_name}.{self.coupling_name}.inner_mda_name'] = 'MDAGaussSeidel'
        values_dict[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 2
        values_dict[f'{self.study_name}.{self.coupling_name}.tolerance'] = 1e-10
        values_dict[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0

        values_dict[
            f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.func_manager_name}.function_df'] = self.setup_func_df()

        dspace_energy_mix = self.witness_uc.dspace
        dspace_sectorization = self.dspace_sectorization()

        dv_arrays_dict_energy_mix, design_var_descriptor_energy_mix = self.get_dvar_descriptor_energy_mix(
            dspace=dspace_energy_mix)
        dv_arrays_dict_sectorization, design_var_descriptor_sectorization = self.get_dvar_descriptor_sectorization(
            dspace=dspace_sectorization)

        design_var_descriptor = {}
        design_var_descriptor.update(design_var_descriptor_energy_mix)
        design_var_descriptor.update(design_var_descriptor_sectorization)

        values_dict.update(dv_arrays_dict_energy_mix)
        values_dict.update(dv_arrays_dict_sectorization)

        dspace = self.merge_design_spaces_dict(dspace_list=[dspace_energy_mix, dspace_sectorization])
        self.dspace_size, self.dspace = self.dspace_dict_to_dataframe(dspace)

        self.remove_all_variables_in_values_dict(values_dict=values_dict,shortvarname="design_space")
        self.remove_all_variables_in_values_dict(values_dict=values_dict,shortvarname="design_var_descriptor")

        values_dict[f'{self.study_name}.{self.optim_name}.design_space'] = self.dspace
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor


        agri_subsector_invests = pd.DataFrame({
            GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
            GlossaryCore.Crop: 98.,
            GlossaryCore.Forestry: 2.,
        })
        values_dict[f'{self.study_name}.{self.coupling_name}.WITNESS.Macroeconomics.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}'] = agri_subsector_invests
        values_dict[f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor
        values_dict[f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.extra_name}.mdo_mode_energy'] = True


        values_dict.update(
            {f'{ns}.epsilon0': 1,
             f'{ns}.cache_type': 'SimpleCache',
             f'{ns}.{self.optim_name}.objective_name': FunctionManagerDisc.OBJECTIVE_LAGR,
             f'{ns}.{self.optim_name}.eq_constraints': [],
             f'{ns}.{self.optim_name}.ineq_constraints': [],

             # optimization parameters:
             f'{ns}.{self.optim_name}.max_iter': 1,
             f'{ns}.{self.optim_name}.eval_mode': True,
             f'{ns}.warm_start': True,
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.warm_start': True,
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.cache_type': 'SimpleCache',
             # SLSQP, NLOPT_SLSQP
             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
             f'{ns}.{self.optim_name}.differentiation_method': 'user',
             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                      "ftol_abs": 3e-16,
                                                      "normalize_design_space": True,
                                                      "maxls": 3 * self.sub_uc.dspace_size,
                                                      "maxcor": self.sub_uc.dspace_size,
                                                      "pg_tol": 1e-16,
                                                      "xtol_rel": 1e-16,
                                                      "xtol_abs": 1e-16,
                                                      "max_iter": 700,
                                                      "disp": 30},
             # f'{ns}.{self.optim_name}.{witness_uc.coupling_name}.linear_solver_MDO':
             # 'GMRES',
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.linear_solver_MDO_options': {
                 'tol': 1.0e-10,
                 'max_iter': 10000},
             # f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA':
             # 'GMRES',
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.linear_solver_MDA_options': {
                 'tol': 1.0e-10,
                 'max_iter': 50000},
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.epsilon0': 1.0,
             f'{ns}.{self.optim_name}.{self.sub_uc.coupling_name}.tolerance': 1.0e-10,
             f'{ns}.{self.optim_name}.parallel_options': {"parallel": False,  # True
                                                          "n_processes": 32,
                                                          "use_threading": False,
                                                          "wait_time_between_fork": 0},
             f'{self.sub_uc.witness_uc.study_name}.DesignVariables.is_val_level': False})

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()