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

from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Resources Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        ns_scatter = self.ee.study_name + '.Resources'

        ns_dict = {'coal_resource': f'{ns_scatter}.coal_resource',
                   'oil_resource': f'{ns_scatter}.oil_resource',
                   'natural_gas_resource': f'{ns_scatter}.natural_gas_resource',
                   'uranium_resource': f'{ns_scatter}.uranium_resource',
                   'ns_public': self.ee.study_name,
                   'ns_resource': ns_scatter,
                   'ns_demand': ns_scatter
                   }
        #'oil_availability_and_price': f'{ns_scatter}.oil.oil_availability_and_price',
        mods_dict = {'Resources.coal_resource': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_coal_resource.coal_resource_model.coal_resource_disc.CoalDiscipline',
                     'Resources.oil_resource': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_oil_resource.oil_resource_model.oil_resource_disc.OilDiscipline',
                     'Resources.natural_gas_resource': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_gas_resource.gas_resource_model.gas_resource_disc.GasDiscipline',
                     'Resources.uranium_resource': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_uranium_resource.uranium_resource_model.uranium_resource_disc.UraniumDiscipline',
                     'Resources': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_all_resource.all_resource_model.all_resource_disc.AllResourceDiscipline'
                     }
        #chain_builders_resource = self.ee.factory.get_builder_from_module()
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)
        # builder_list.append(chain_builders_resource)
        return builder_list
