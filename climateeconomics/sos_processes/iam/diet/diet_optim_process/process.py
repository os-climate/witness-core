"""
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2024/06/24 Copyright 2023 Capgemini

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

from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.witness_sub_process_builder import (
    WITNESSSubProcessBuilder,
)

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    OPTIM_NAME,
)


class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        "label": "Diet Optimization Process",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):

        optim_name = OPTIM_NAME

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_TECHNO_DICT

        coupling_builder = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.diet", "diet_optim_sub_process"
        )

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(optim_name, after_name=self.ee.study_name)  # optim_name

        # -- set optim builder
        opt_builder = self.ee.factory.create_optim_builder(optim_name, [coupling_builder])

        return opt_builder
