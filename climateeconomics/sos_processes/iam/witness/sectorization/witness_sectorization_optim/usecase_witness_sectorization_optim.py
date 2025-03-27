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
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim_sub_process.usecase import (
    COUPLING_NAME,
    EXTRA_NAME,
)
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim_sub_process.usecase import (
    Study as witness_sectorization_optim_sub_usecase,
)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, bspline=True, run_usecase=False,
                 execution_engine=None, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.optim_name = "MDO"
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.techno_dict = techno_dict
        self.witness_uc = witness_sectorization_optim_sub_usecase(
            self.year_start, self.year_end, bspline=self.bspline, execution_engine=execution_engine,
            techno_dict=techno_dict,)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict
        self.test_post_procs = False

    def setup_process(self):
        witness_sectorization_optim_sub_usecase.setup_process(self)

    def setup_usecase(self, study_folder_path=None):
        ns = self.study_name

        self.witness_uc.study_name = f'{ns}.{self.optim_name}'
        self.coupling_name = self.witness_uc.coupling_name
        values_dict = self.witness_uc.setup_usecase()

        values_dict.update(
            {f'{ns}.epsilon0': 1,
             f'{ns}.cache_type': 'SimpleCache',
             f'{ns}.{self.optim_name}.design_space': self.witness_uc.dspace,
             f'{ns}.{self.optim_name}.objective_name': FunctionManagerDisc.OBJECTIVE_LAGR,
             f'{ns}.{self.optim_name}.eq_constraints': [],
             f'{ns}.{self.optim_name}.ineq_constraints': [],

             # optimization parameters:
             f'{ns}.{self.optim_name}.max_iter': 1500,
             f'{ns}.warm_start': True,
             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.warm_start': True,
             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.cache_type': 'SimpleCache',
             # SLSQP, NLOPT_SLSQP
             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
             f'{ns}.{self.optim_name}.differentiation_method': 'user',
             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                      "ftol_abs": 3e-16,
                                                      "normalize_design_space": True,
                                                      "maxls": 3 * self.witness_uc.dspace_size,
                                                      "maxcor": self.witness_uc.dspace_size,
                                                      "pg_tol": 1e-16,
                                                      "xtol_rel": 1e-16,
                                                      "xtol_abs": 1e-16,
                                                      "max_iter": 700,
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
             f'{self.witness_uc.witness_uc.study_name}.DesignVariables.is_val_level': False})

        self.get_fullname_in_values_dict(values_dict, 'invest_level')
        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
