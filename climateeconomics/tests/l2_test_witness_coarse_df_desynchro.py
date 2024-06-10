"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

import unittest

from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_optim_process.usecase_witness_optim_invest_distrib import (
    Study as witness_proc_usecase,
)


class WitnessCoarseDesynchro(unittest.TestCase):

    def test_01_desynchro_on_witness_coarse_optim_outputs(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness",
            "witness_coarse_optim_process",
            techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1],
        )
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_proc_usecase(
            bspline=True,
            execution_engine=self.ee,
            techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1],
        )
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        algo_options = {
            "ftol_rel": 3e-16,
            "normalize_design_space": False,
            "maxls": 2 * 55,
            "maxcor": 55,
            "pg_tol": 1.0e-8,
            "max_iter": 2,
            "disp": 110,
        }

        full_values_dict["Test.WITNESS_MDO.algo_options"] = algo_options
        full_values_dict["Test.WITNESS_MDO.max_iter"] = 2
        self.ee.load_study_from_input_dict(full_values_dict)

        sub_mda_class_list = [
            "MDAJacobi",
            "MDAGaussSeidel",
            "MDANewtonRaphson",
            "GSNewtonMDA",
            "GSPureNewtonMDA",
            "GSorNewtonMDA",
        ]

        for sub_mda_class in sub_mda_class_list:

            dict_values = {}
            dict_values["Test.WITNESS_MDO.WITNESS_Eval.sub_mda_class"] = sub_mda_class

            if sub_mda_class == "MDAJacobi":
                dict_values["Test.WITNESS_MDO.WITNESS_Eval.max_mda_iter"] = 20
            else:
                dict_values["Test.WITNESS_MDO.WITNESS_Eval.max_mda_iter"] = 5

            self.ee.load_study_from_input_dict(dict_values)

            # execute process with sub_mda_class
            self.ee.execute()

            df_coupled = self.ee.dm.get_value(
                f"Test.WITNESS_MDO.WITNESS_Eval.WITNESS.{GlossaryCore.TemperatureDfValue}"
            )
            df_ncoupled = self.ee.dm.get_value(
                "Test.WITNESS_MDO.WITNESS_Eval.WITNESS.Temperature_change.temperature_detail_df"
            )

            # test synchronisation of coupled variable temperature_df and
            # non-coupled output temperature_detail_df
            self.assertListEqual(
                list(df_ncoupled[GlossaryCore.TempAtmo].values),
                list(df_coupled[GlossaryCore.TempAtmo].values),
                msg="desynchro of dataframes detected",
            )


if "__main__" == __name__:
    cls = WitnessCoarseDesynchro()
    cls.test_01_desynchro_on_witness_coarse_optim_outputs()
