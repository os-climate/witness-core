"""
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/21 Copyright 2023 Capgemini
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

from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.sos_processes.witness_sub_process_builder import (
    WITNESSSubProcessBuilder,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import (
    DEFAULT_COARSE_TECHNO_DICT,
)


class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        "label": "WITNESS Coarse Dev Process",
        "description": "",
        "category": "",
        "version": "",
    }

    def __init__(self, ee, process_level="dev"):
        WITNESSSubProcessBuilder.__init__(self, ee)
        self.invest_discipline = INVEST_DISCIPLINE_OPTIONS[2]
        self.process_level = process_level
        # Running an mda only, not a mdo

    def get_builders(self):

        chain_builders = []
        # retrieve energy process
        chain_builders_witness = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam", "witness_wo_energy_dev"
        )
        chain_builders.extend(chain_builders_witness)

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = DEFAULT_COARSE_TECHNO_DICT

        chain_builders_energy = self.ee.factory.get_builder_from_process(
            "energy_models.sos_processes.energy.MDA",
            "energy_process_v0_mda",
            techno_dict=techno_dict,
            invest_discipline=self.invest_discipline,
            process_level=self.process_level,
        )

        chain_builders.extend(chain_builders_energy)

        # Update namespace regarding land use and energy mix coupling
        ns_dict = {
            "ns_land_use": f"{self.ee.study_name}.EnergyMix",
            GlossaryCore.NS_FUNCTIONS: f"{self.ee.study_name}.EnergyMix",
            GlossaryCore.NS_REFERENCE: f"{self.ee.study_name}.NormalizationReferences",
            "ns_dashboard": f"{self.ee.study_name}",
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            "ns_dashboard", "climateeconomics.sos_wrapping.post_procs.dashboard"
        )

        return chain_builders
