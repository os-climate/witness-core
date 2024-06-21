'''
Copyright 2022 Airbus SAS
Modifications on 06/07/2024 Copyright 2024 Capgemini

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
from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.diet.diet_optim_sub_process._usecase_diet_optim_sub import (
    Study as diet_optim_sub_usecase,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    COUPLING_NAME,
    EXTRA_NAME,
    OPTIM_NAME,
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
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.optim_name = OPTIM_NAME
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict

        self.witness_uc = diet_optim_sub_usecase(
            self.year_start, self.year_end, self.time_step, bspline=self.bspline, execution_engine=execution_engine)

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
                             f'{ns}.{self.optim_name}.max_iter': 800,
                             f'{ns}.warm_start': True,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.warm_start': True,
                             # SLSQP, NLOPT_SLSQP
                             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
                             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
                             f'{ns}.{self.optim_name}.differentiation_method': 'user',
                             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                                      "ftol_abs": 3e-16,
                                                                      "normalize_design_space": True,
                                                                      "max_ls_step_nb": 3 * dspace_size,
                                                                      "maxcor": dspace_size,
                                                                      "pg_tol": 1e-16,
                                                                      "xtol_rel": 1e-16,
                                                                      "xtol_abs": 1e-16,
                                                                      "max_iter": 700,
                                                                      "disp": 30},

                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDO_options': {'tol': 1.0e-10,
                                                                                                                   'max_iter': 10000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA_options': {'tol': 1.0e-10,
                                                                                                                   'max_iter': 50000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.epsilon0': 1.0,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.tolerance': 1.0e-10,

                             f'{ns}.{self.optim_name}.parallel_options': {"parallel": False,  # True
                                                                          "n_processes": 32,
                                                                          "use_threading": False,
                                                                          "wait_time_between_fork": 0},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.sub_mda_class': 'GSPureNewtonMDA',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.max_mda_iter': 50,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.{WRITE_XVECT}': False}

        # print("Design space dimension is ", dspace_size)

        return [values_dict] + [optim_values_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()

    uc_cls.run()

