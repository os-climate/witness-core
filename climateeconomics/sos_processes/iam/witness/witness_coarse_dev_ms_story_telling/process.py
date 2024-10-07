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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Coarse Dev Multi-Scenario Story Telling mda Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        """
        # scenario build map
        scenario_map = {'input_name': 'scenario_list',

                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': [GlossaryCore.NS_WITNESS,
                                         GlossaryCore.NS_FUNCTIONS,
                                         GlossaryCore.NS_ENERGY_MIX,
                                         'ns_public',
                                         'ns_optim',
                                         'ns_flue_gas',
                                         'ns_energy_study',
                                         'ns_energy',
                                         'ns_carbon_capture',
                                         'ns_carbon_storage',
                                         'ns_land_use',
                                         'ns_clean_energy',
                                         'ns_fossil',
                                         GlossaryCore.NS_CCS,
                                         'ns_resource',
                                         'ns_invest',
                                         'ns_agriculture',
                                         'ns_crop',
                                         'ns_forest',
                                         'ns_coal_resource',
                                         'ns_natural_gas_resource',
                                         'ns_oil_resource',
                                         'ns_uranium_resource'
                                         ]}

        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        """
        builder_cdf_list = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_story_telling')

        scatter_scenario_name = 'mda_scenarios'

        # Add new namespaces needed for the scatter multiscenario
        ns_dict = {'ns_scatter_scenario': f'{self.ee.study_name}.{scatter_scenario_name}',
                   'ns_post_processing': f'{self.ee.study_name}.{scatter_scenario_name}',
                   }


        self.ee.ns_manager.add_ns_def(ns_dict)
        self.ee.scattermap_manager.add_build_map('new_map', {'ns_not_to_update': ['ns_post_processing',
                                                                                  'ns_scatter_scenario']})

        multi_scenario = self.ee.factory.create_multi_instance_driver(
            scatter_scenario_name, builder_cdf_list, map_name='new_map'
        )
        self.ee.post_processing_manager.add_post_processing_module_to_namespace('ns_post_processing',
                                                                                'climateeconomics.sos_wrapping.post_procs.witness_ms.post_processing_witness_coarse_mda')

        return multi_scenario
