"""
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
"""

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        "label": "WITNESS Coarse Dev Story Telling mda subprocess for sensitivity",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):
        """
        # Builders for Witness Coarse Story Telling MDA usecase 7 (NZE inspired) for Sensitivity Analysis demo.
        """
        builder_cdf_list = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "witness_coarse_dev_story_telling"
        )

        # Add input/output analysis disciplines
        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_witness_coarse_for_sensitivity.fossil_techno_infos.FossilTechnoInfos"
        builder_cdf_list.append(self.ee.factory.get_builder_from_module("FossilTechnoInfo", mod_path))
        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_witness_coarse_for_sensitivity.renewable_techno_infos.RenewableTechnoInfos"
        builder_cdf_list.append(self.ee.factory.get_builder_from_module("RenewableTechnoInfo", mod_path))
        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_witness_coarse_for_sensitivity.witness_indicators.WitnessIndicators"
        builder_cdf_list.append(self.ee.factory.get_builder_from_module("Indicators", mod_path))

        # Add new namespaces needed for these disciplines
        ns_dict = {
            "ns_fossil_techno": f"{self.ee.study_name}.EnergyMix.fossil.FossilSimpleTechno",
            "ns_renewable_techno": f"{self.ee.study_name}.EnergyMix.renewable.RenewableSimpleTechno",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)
        return builder_cdf_list
