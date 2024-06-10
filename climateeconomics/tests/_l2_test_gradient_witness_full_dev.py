"""
Copyright 2022 Airbus SAS
Modifications on 27/11/2023 Copyright 2023 Capgemini

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

from os.path import join

from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT_DEV
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)
from climateeconomics.tests.witness_jacobian_disc_test import WitnessJacobianDiscTest


class WitnessFullJacobianDiscTest(WitnessJacobianDiscTest):

    def setUp(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):

        return [
            self.test_01_gradient_all_disciplines_witness_full_dev(),
        ]

    def test_01_gradient_all_disciplines_witness_full_dev(self):
        """ """
        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness",
            "witness_optim_sub_process",
            process_level="dev",
            techno_dict=DEFAULT_TECHNO_DICT_DEV,
        )
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True,
            execution_engine=self.ee,
            process_level="dev",
            techno_dict=DEFAULT_TECHNO_DICT_DEV,
        )
        usecase.study_name = self.name

        directory = join(AbstractJacobianUnittest.PICKLE_DIRECTORY, "witness_full_dev")

        excluded_disc = []

        excluded_outputs = [
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.energy_detailed_techno_prices",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.liquid_fuel.{GlossaryCore.EnergyProductionDetailedValue}",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.energy_detailed_techno_prices",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.hydrotreated_oil_fuel.{GlossaryCore.EnergyProductionDetailedValue}",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.energy_detailed_techno_prices",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.biodiesel.{GlossaryCore.EnergyProductionDetailedValue}",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.ethanol.energy_detailed_techno_prices",
            f"Test.WITNESS_Eval.WITNESS.EnergyMix.fuel.ethanol.{GlossaryCore.EnergyProductionDetailedValue}",
        ]
        disc_to_test_list = ["WITNESS.Resources"]  #'WITNESS.AgricultureMix.Forest'

        self.all_usecase_disciplines_jacobian_test(
            usecase,
            directory=directory,
            excluded_disc=excluded_disc,
            max_mda_iter=3,
            excluded_outputs=excluded_outputs,
            #    optional_disciplines_list=disc_to_test_list
        )
