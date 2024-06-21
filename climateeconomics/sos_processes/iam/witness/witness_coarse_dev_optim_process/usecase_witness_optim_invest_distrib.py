'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import (
    COARSE_AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    COUPLING_NAME,
    EXTRA_NAME,
    OPTIM_NAME,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_optim_sub_usecase,
)
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugin.models.design_var.design_var_disc import (
    DesignVarDiscipline,
)
from sostrades_optimization_plugin.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
EXPORT_CSV = FunctionManagerDisc.EXPORT_CSV
WRITE_XVECT = DesignVarDiscipline.WRITE_XVECT


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1, bspline=False, run_usecase=False,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
                 agri_techno_list=COARSE_AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 process_level='dev',
                 file_path=__file__):
        super().__init__(file_path=file_path, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.optim_name = OPTIM_NAME
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.process_level = process_level
        self.witness_uc = witness_optim_sub_usecase(
            year_start=self.year_start, year_end=self.year_end, time_step=self.time_step, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict, process_level=process_level,
            agri_techno_list=agri_techno_list)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict

    def setup_process(self):
        witness_optim_sub_usecase.setup_process(self)

    def make_dspace_invests(self, dspace_dict: dict[str: list], overwrite_invest_index: list[int] = []) -> pd.DataFrame:
        """
        :param dspace_dict: {variable_name: [value, lower_bnd, upper_bnd, enable_variable]}
        """
        out = {
            "variable": [],
            "value": [],
            "lower_bnd": [],
            "upper_bnd": [],
            "enable_variable": [],
            "activated_elem": [],
        }
        initial_values_first_pole = {
            'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': DatabaseWitnessCore.InvestFossil2020.value,
            'renewable.RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix': DatabaseWitnessCore.InvestCleanEnergy2020.value,
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': DatabaseWitnessCore.InvestCCUS2020.value / 3,
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': DatabaseWitnessCore.InvestCCUS2020.value / 3,
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': DatabaseWitnessCore.InvestCCUS2020.value / 3,
        }

        for var, infos in dspace_dict.items():
            out['variable'].append(var)
            out['value'].append([initial_values_first_pole[var]] + [infos[0]] * (GlossaryCore.NB_POLES_COARSE - 1))
            out['lower_bnd'].append([infos[1]] * GlossaryCore.NB_POLES_COARSE)
            out['upper_bnd'].append([infos[2]] * GlossaryCore.NB_POLES_COARSE)
            out['enable_variable'].append(infos[3])
            out['activated_elem'].append([False] + [True] * (GlossaryCore.NB_POLES_COARSE - 1))

        for index in overwrite_invest_index:
            out['activated_elem'][index] = [False] * GlossaryCore.NB_POLES_COARSE
        out = pd.DataFrame(out)
        return out

    def make_dspace_utilization_ratio(self, dspace_dict: dict[str: list]) -> pd.DataFrame:
        """
        :param dspace_dict: {variable_name: [value, lower_bnd, upper_bnd, enable_variable]}
        """
        out = {
            "variable": [],
            "value": [],
            "lower_bnd": [],
            "upper_bnd": [],
            "enable_variable": [],
            "activated_elem": [],
        }

        for var, infos in dspace_dict.items():
            out['variable'].append(var)
            out['value'].append([100.] + [infos[0]] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO - 1))
            out['lower_bnd'].append([infos[1]] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO))
            out['upper_bnd'].append([infos[2]] * GlossaryCore.NB_POLES_UTILIZATION_RATIO)
            out['enable_variable'].append(infos[3])
            out['activated_elem'].append([False] + [True] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO - 1))

        out = pd.DataFrame(out)
        return out

    def make_dspace_Ine(self):
        return pd.DataFrame({
            "variable": ["share_non_energy_invest_ctrl"],
            "value": [[25.5] * GlossaryCore.NB_POLES_COARSE],
            "lower_bnd": [[5.0] * GlossaryCore.NB_POLES_COARSE],
            "upper_bnd": [[30.0] * GlossaryCore.NB_POLES_COARSE],
            "enable_variable": [True],
            "activated_elem": [[False] + [True] * (GlossaryCore.NB_POLES_COARSE - 1)]
        })

    def get_ine_dvar_descr(self):
        return {
            'out_name': GlossaryCore.ShareNonEnergyInvestmentsValue,
            'out_type': "dataframe",
            'key': GlossaryCore.ShareNonEnergyInvestmentsValue,
            'index': np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1),
            'index_name': GlossaryCore.Years,
            'namespace_in': GlossaryCore.NS_WITNESS,
            'namespace_out': GlossaryCore.NS_WITNESS,
        }

    def setup_usecase(self, study_folder_path=None):
        ns = self.study_name

        values_dict = {}

        self.witness_uc.study_name = f'{ns}.{self.optim_name}'
        self.coupling_name = self.witness_uc.coupling_name
        witness_uc_data = self.witness_uc.setup_usecase()
        for dict_data in witness_uc_data:
            values_dict.update(dict_data)

        # design space WITNESS
        dspace_df = self.witness_uc.dspace
        self.func_df = self.witness_uc.func_df
        # df_xvect = pd.read_pickle('df_xvect.pkl')
        # df_xvect.columns = [df_xvect.columns[0]]+[col.split('.')[-1] for col in df_xvect.columns[1:]]
        # dspace_df_xvect=pd.DataFrame({'variable':df_xvect.columns, 'value':df_xvect.drop(0).values[0]})
        # dspace_df.update(dspace_df_xvect)

        dspace_size = self.witness_uc.dspace_size
        # optimization functions:
        optim_values_dict = {f'{ns}.epsilon0': 1,
                             f'{ns}.cache_type': 'SimpleCache',
                             f'{ns}.{self.optim_name}.design_space': dspace_df,
                             f'{ns}.{self.optim_name}.objective_name': FunctionManagerDisc.OBJECTIVE_LAGR,
                             f'{ns}.{self.optim_name}.eq_constraints': [],
                             f'{ns}.{self.optim_name}.ineq_constraints': [],

                             # optimization parameters:
                             f'{ns}.{self.optim_name}.max_iter': 400,
                             f'{ns}.warm_start': True,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.warm_start': True,
                             # SLSQP, NLOPT_SLSQP
                             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
                             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
                             f'{ns}.{self.optim_name}.differentiation_method': 'user',
                             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                                      "ftol_abs": 3e-16,
                                                                      "normalize_design_space": True,
                                                                      "maxls": 3 * dspace_size,
                                                                      "maxcor": dspace_size,
                                                                      "factr": 1,
                                                                      "pg_tol": 1e-16,
                                                                      "xtol_rel": 1e-16,
                                                                      "xtol_abs": 1e-16,
                                                                      "max_iter": 1,
                                                                      "disp": 30},
                             # f'{ns}.{self.optim_name}.{witness_uc.coupling_name}.linear_solver_MDO':
                             # 'GMRES',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDO_options': {
                                 'tol': 1.0e-10,
                                 'max_iter': 10000},
                             # f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA':
                             # 'GMRES',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA_options': {
                                 'tol': 1.0e-10,
                                 'max_iter': 50000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.epsilon0': 1.0,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.tolerance': 1.0e-10,
                             f'{ns}.{self.optim_name}.parallel_options': {"parallel": False,  # True
                                                                          "n_processes": 32,
                                                                          "use_threading": False,
                                                                          "wait_time_between_fork": 0},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.sub_mda_class': 'MDAGaussSeidel',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.max_mda_iter': 50,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.cache_type': 'SimpleCache',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.propagate_cache_to_children': True,
                             f'{self.witness_uc.witness_uc.study_name}.DesignVariables.is_val_level': False}

        # ---- NORMALIZATION REFERENCES -> Specific to each optim usecase
        ref_value_dict = {
            f'{self.witness_uc.witness_uc.study_name}.NormalizationReferences.land_use_constraint_ref': 0.1}

        # f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.{WRITE_XVECT}':
        # True}

        # print("Design space dimension is ", dspace_size)

        out = {}
        out.update(values_dict)
        out.update(optim_values_dict)
        out.update(ref_value_dict)

        dspace = out[f'{self.study_name}.{self.optim_name}.design_space']
        list_design_var_to_clean = ['red_meat_calories_per_day_ctrl',
                                    'white_meat_calories_per_day_ctrl', 'vegetables_and_carbs_calories_per_day_ctrl',
                                    'milk_and_eggs_calories_per_day_ctrl', 'forest_investment_array_mix',
                                    'deforestation_investment_ctrl']

        # clean dspace
        dspace.drop(dspace.loc[dspace['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        # clean dspace descriptor
        dvar_descriptor = out[
            f'{self.study_name}.{self.optim_name}.{self.coupling_name}.DesignVariables.design_var_descriptor']

        updated_dvar_descriptor = {k: v for k, v in dvar_descriptor.items() if k not in list_design_var_to_clean}


        # Ajout design var Share Non Energy invest


        out.update({
            f'{self.study_name}.{self.optim_name}.design_space': dspace,
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.design_var_descriptor': updated_dvar_descriptor
        })


        import numpy as np
        a = {
          'out_name': GlossaryCore.ShareNonEnergyInvestmentsValue,
            'out_type': "dataframe",
            'key': GlossaryCore.ShareNonEnergyInvestmentsValue,
            'index': np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1),
            'index_name': GlossaryCore.Years,
            'namespace_in': "",
            'namespace_out': GlossaryCore.NS_WITNESS,
        }
        return out


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()

    # df_xvect = pd.read_pickle('df_xvect.pkl')
    # df_xvect.columns = [
    # f'{uc_cls.study_name}.{uc_cls.optim_name}.{uc_cls.coupling_name}.DesignVariables' + col for col in df_xvect.columns]
    # dict_xvect = df_xvect.iloc[-1].to_dict()
    # dict_xvect[f'{uc_cls.study_name}.{uc_cls.optim_name}.eval_mode'] = True
    # uc_cls.load_data(from_input_dict=dict_xvect)
    # f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables'
    # uc_cls.execution_engine.root_process.proxy_disciplines[0].set_opt_scenario()
    # uc_cls.execution_engine.set_debug_mode()

#     uc_cls.execution_engine.root_process.proxy_disciplines[0].coupling_structure.graph.export_reduced_graph(
#         "reduced.pdf")
#     uc_cls.execution_engine.root_process.proxy_disciplines[0].coupling_structure.graph.export_initial_graph(
#         "initial.pdf")
