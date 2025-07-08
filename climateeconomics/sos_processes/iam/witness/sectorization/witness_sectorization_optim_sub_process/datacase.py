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
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
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
        GlossaryCore.EnergyMix = 'EnergyMix'
        self.bspline = bspline
        self.techno_dict = techno_dict
        self.witness_uc = witness_usecase_secto(
            year_start=self.year_start, year_end=self.year_end, bspline=self.bspline,
            execution_engine=execution_engine, techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2])
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict
        self.test_post_procs = True


    def setup_func_df(self):

        return pd.DataFrame({
            'variable': [],
            'parent': [],
            'ftype': [],
            'weight': [],
            FunctionManagerDisc.AGGR_TYPE: [],
            'namespace': []
        })

    def setup_usecase(self, study_folder_path=None):
        """ Overloaded method to initialize witness multiscenario optimization process

        @return list of dictionary: [{str: *}]
        """
        values_dict = {}

        # -- retrieve energy input data

        self.witness_uc.study_name = f'{self.study_name}.{self.coupling_name}.{self.extra_name}'
        self.witness_uc.study_name_wo_extra_name = f'{self.study_name}.{self.coupling_name}'
        witness_data_list = self.witness_uc.setup_usecase()
        values_dict.update(witness_data_list)

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        values_dict[f'{self.study_name}.{self.coupling_name}.inner_mda_name'] = 'MDAGaussSeidel'
        values_dict[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 100
        values_dict[f'{self.study_name}.{self.coupling_name}.tolerance'] = 1e-12
        values_dict[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0
        values_dict[f'{self.witness_uc.study_name}.{self.coupling_name}.{GlossaryCore.energy_list}'] = self.witness_uc.energy_list

        values_dict[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.function_df'] = self.setup_func_df()


        dspace = pd.DataFrame({
            'variable': [],
            'value': [],
            'lower_bnd': [],
            'upper_bnd': [],
            'enable_variable': [],
            'activated_elem': [],
        })
        values_dict[f'{self.study_name}.design_space'] = dspace

        agri_subsector_invests = pd.DataFrame({
            GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
            GlossaryCore.Crop: 98.,
            GlossaryCore.Forestry: 2.,
        })
        values_dict[f'{self.study_name}.{self.coupling_name}.WITNESS.Macroeconomics.Agriculture.{GlossaryCore.ShareSectorInvestmentDfValue}'] = agri_subsector_invests

        values_dict[f'{self.study_name}.{self.coupling_name}.{self.extra_name}.mdo_mode_energy'] = False
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.extra_name}.tp_a3'] = 3.5
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.extra_name}.assumptions_dict'] = ClimateEcoDiscipline.assumptions_dict_default

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
