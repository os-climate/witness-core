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
import pandas as pd
import numpy as np
from climateeconomics.database import DatabaseWitnessCore
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager import FunctionManager
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
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim.usecase_witness_sectorization_optim import (
    Study as StudyOptim,
)


class Study(StudyOptim):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(filename=__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        values_dict = super().setup_usecase(study_folder_path=study_folder_path)

        # For energy prod vs demand calibration, remove all invests in individual energy technos,
        # allow utilisation ratio for technos
        # control only global invest in energy sector
        design_var_descriptor = self.get_var_in_values_dict(values_dict=values_dict, varname="design_var_descriptor")
        design_space = self.get_var_in_values_dict(values_dict=values_dict, varname="design_space")
        a = 1
        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
