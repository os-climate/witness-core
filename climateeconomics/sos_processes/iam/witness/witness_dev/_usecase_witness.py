"""
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

"""

import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import (
    Study as datacase_energy,
)
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness_wo_energy_dev.datacase_witness_wo_energy import (
    DataStudy as datacase_witness_dev,
)

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(ClimateEconomicsStudyManager):

    def __init__(
        self,
        year_start=GlossaryCore.YearStartDefault,
        year_end=GlossaryCore.YearEndDefault,
        time_step=1,
        bspline=True,
        run_usecase=False,
        execution_engine=None,
        invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
        techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT_DEV,
        process_level="dev",
    ):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.process_level = process_level
        self.dc_energy = datacase_energy(
            self.year_start,
            self.year_end,
            self.time_step,
            bspline=self.bspline,
            execution_engine=execution_engine,
            invest_discipline=self.invest_discipline,
            techno_dict=techno_dict,
        )
        self.sub_study_path_dict = self.dc_energy.sub_study_path_dict

    def setup_constraint_land_use(self):
        func_df = pd.DataFrame(columns=["variable", "parent", "ftype", "weight", AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(["land_demand_constraint_df"])
        list_parent.extend([None])
        list_ftype.extend([INEQ_CONSTRAINT])
        list_weight.extend([-1.0])
        list_aggr_type.extend([AGGR_TYPE_SUM])
        list_ns.extend([GlossaryCore.NS_FUNCTIONS])
        func_df["variable"] = list_var
        func_df["parent"] = list_parent
        func_df["ftype"] = list_ftype
        func_df["weight"] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df["namespace"] = list_ns

        return func_df

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []

        # -- load data from energy pyworld3
        # -- Start with energy to have it at first position in the list...

        self.dc_energy.study_name = self.study_name
        self.energy_mda_usecase = self.dc_energy
        # -- load data from witness
        dc_witness = datacase_witness_dev(self.year_start, self.year_end, self.time_step)
        dc_witness.study_name = self.study_name

        witness_input_list = dc_witness.setup_usecase()
        setup_data_list = setup_data_list + witness_input_list

        energy_input_list = self.dc_energy.setup_usecase()
        setup_data_list = setup_data_list + energy_input_list

        dspace_energy = self.dc_energy.dspace

        self.merge_design_spaces([dspace_energy, dc_witness.dspace])

        # constraint land use
        land_use_df_constraint = self.setup_constraint_land_use()

        # WITNESS
        # setup objectives
        self.func_df = pd.concat(
            [
                dc_witness.setup_objectives(),
                dc_witness.setup_constraints(),
                self.dc_energy.setup_constraints(),
                self.dc_energy.setup_objectives(),
                land_use_df_constraint,
            ]
        )

        self.energy_list = self.dc_energy.energy_list
        self.ccs_list = self.dc_energy.ccs_list
        self.dict_technos = self.dc_energy.dict_technos

        numerical_values_dict = {
            f"{self.study_name}.epsilon0": 1.0,
            f"{self.study_name}.max_mda_iter": 50,
            f"{self.study_name}.tolerance": 1.0e-10,
            f"{self.study_name}.n_processes": 1,
            f"{self.study_name}.linearization_mode": "adjoint",
            f"{self.study_name}.sub_mda_class": "GSPureNewtonMDA",
            f"{self.study_name}.cache_type": "SimpleCache",
        }

        setup_data_list.append(numerical_values_dict)

        return setup_data_list


if "__main__" == __name__:
    uc_cls = Study()
    uc_cls.test()
