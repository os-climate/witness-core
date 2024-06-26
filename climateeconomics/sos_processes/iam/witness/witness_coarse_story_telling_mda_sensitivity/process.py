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
        "label": "WITNESS Coarse Dev Story Telling Sensitivity Analysis Process",
        "description": "",
        "category": "",
        "version": "",
    }

    COUPLING_NAME = "NZE"
    DRIVER_NAME = "AnalysisWITNESS"

    def get_builders(self):
        """ """
        builder_cdf_list = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "witness_coarse_story_telling_mda_sensitivity_subprocess"
        )
        coupling_builder = self.ee.factory.create_builder_coupling(self.COUPLING_NAME)
        coupling_builder.set_builder_info("cls_builder", builder_cdf_list)
        # # modify namespaces defined in the shared_dict
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            f"{self.COUPLING_NAME}", after_name=self.ee.study_name, clean_existing=False
        )
        sensitivity_analysis = self.ee.factory.create_mono_instance_driver(self.DRIVER_NAME, coupling_builder)

        return sensitivity_analysis
