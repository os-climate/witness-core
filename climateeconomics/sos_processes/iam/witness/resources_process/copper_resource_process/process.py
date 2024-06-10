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

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

from climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc import (
    CopperResourceDiscipline,
)


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        "label": "WITNESS Copper Resource Process",
        "description": "",
        "category": "",
        "version": "",
    }

    COPPER_NAME = CopperResourceDiscipline.resource_name

    def get_builders(self):

        ns_scatter = self.ee.study_name

        ns_dict = {
            "ns_copper_resource": ns_scatter,
            "ns_public": ns_scatter,
            "ns_resource": ns_scatter,
        }

        mods_dict = {
            "CopperResourceModel": "climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc.CopperResourceDiscipline"
        }
        # chain_builders_resource = self.ee.factory.get_builder_from_module()
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)
        # builder_list.append(chain_builders_resource)
        return builder_list
