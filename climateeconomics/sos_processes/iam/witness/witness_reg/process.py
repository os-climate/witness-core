'''
Copyright 2022 Airbus SAS

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

from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from energy_models.sos_processes.witness_sub_process_builder import WITNESSSubProcessBuilder
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from copy import copy

class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def __init__(self, ee, process_level='val'):
        WITNESSSubProcessBuilder.__init__(
            self, ee)
        self.invest_discipline = INVEST_DISCIPLINE_OPTIONS[2]
        self.process_level = process_level

    def get_builders(self):

        chain_builders = []
        # retrieve energy process


        chain_builders_witness = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness')
        for build in chain_builders_witness:
            build.set_disc_name(f'US.{build.sos_name}')
        chain_builders.extend(chain_builders_witness)
            
        all_dict_US = copy(self.ee.ns_manager.all_ns_dict)

        list_ns_to_associate = self.ee.ns_manager.update_namespace_list_with_extra_ns('US', self.ee.study_name, namespace_list = list(self.ee.ns_manager.shared_ns_dict.values()))
        for build in chain_builders_witness:
            build.associate_namespaces(list_ns_to_associate)
            build.set_builder_info('local_namespace_database', True)

        for ns in list_ns_to_associate:
            ns_obj = self.ee.ns_manager.all_ns_dict[ns]
            ns_obj.get_from_database = True

        chain_builders_witness_UE = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness')
        for build in chain_builders_witness_UE:
            build.set_disc_name(f'UE.{build.sos_name}')
        chain_builders.extend(chain_builders_witness_UE)

        all_dict_updt = copy(self.ee.ns_manager.all_ns_dict)
        difference_dict = {key: all_dict_updt[key] for key in set(all_dict_updt.keys()) & set(all_dict_US.keys())}

        list_ns_to_associate = self.ee.ns_manager.update_namespace_list_with_extra_ns('UE', self.ee.study_name, namespace_list = list(difference_dict.values()))
        for build in chain_builders_witness_UE:
            build.associate_namespaces(list_ns_to_associate)
            build.set_builder_info('local_namespace_database', True)

        for ns in list_ns_to_associate:
            ns_obj = self.ee.ns_manager.all_ns_dict[ns]
            ns_obj.get_from_database = True

        self.ee.ns_manager.database_activated = True 

        return chain_builders
