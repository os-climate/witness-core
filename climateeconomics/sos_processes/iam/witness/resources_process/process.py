'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023-2024/06/24 Copyright 2023 Capgemini

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

from climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc import (
    CoalResourceDiscipline,
)
from climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc import (
    CopperResourceDiscipline,
)
from climateeconomics.core.core_resources.models.natural_gas_resource.natural_gas_resource_disc import (
    NaturalGasResourceDiscipline,
)
from climateeconomics.core.core_resources.models.oil_resource.oil_resource_disc import (
    OilResourceDiscipline,
)
from climateeconomics.core.core_resources.models.platinum_resource.platinum_resource_disc import (
    PlatinumResourceDiscipline,
)
from climateeconomics.core.core_resources.models.uranium_resource.uranium_resource_disc import (
    UraniumResourceDiscipline,
)


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Resources Process',
        'description': '',
        'category': '',
        'version': '',
    }

    COAL_NAME = CoalResourceDiscipline.resource_name
    OIL_NAME = OilResourceDiscipline.resource_name
    NATURAL_GAS_NAME = NaturalGasResourceDiscipline.resource_name
    URANIUM_NAME = UraniumResourceDiscipline.resource_name
    COPPER_NAME = CopperResourceDiscipline.resource_name
    PLATINUM_NAME = PlatinumResourceDiscipline.resource_name

    def __init__(self, ee):
        self.associate_namespace = False
        BaseProcessBuilder.__init__(self, ee)

    def setup_process(self, associate_namespace=False):
        self.associate_namespace = associate_namespace

    def get_builders(self):

        ns_scatter = self.ee.study_name + '.Resources'

        ns_dict = {'ns_coal_resource': f'{ns_scatter}.{self.COAL_NAME}',
                   'ns_oil_resource': f'{ns_scatter}.{self.OIL_NAME}',
                   'ns_natural_gas_resource': f'{ns_scatter}.{self.NATURAL_GAS_NAME}',
                   'ns_uranium_resource': f'{ns_scatter}.{self.URANIUM_NAME}',
                   'ns_copper_resource': f'{ns_scatter}.{self.COPPER_NAME}',
                   #'ns_platinum_resource': f'{ns_scatter}.{self.PLATINUM_NAME}',
                   'ns_public': self.ee.study_name,
                   'ns_witness': self.ee.study_name,
                   'ns_resource': ns_scatter,
                   }
        #'oil_availability_and_price': f'{ns_scatter}.oil.oil_availability_and_price',
        mods_dict = {f'Resources.{self.COAL_NAME}': 'climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc.CoalResourceDiscipline',
                     f'Resources.{self.OIL_NAME}': 'climateeconomics.core.core_resources.models.oil_resource.oil_resource_disc.OilResourceDiscipline',
                     f'Resources.{self.NATURAL_GAS_NAME}': 'climateeconomics.core.core_resources.models.natural_gas_resource.natural_gas_resource_disc.NaturalGasResourceDiscipline',
                     f'Resources.{self.URANIUM_NAME}': 'climateeconomics.core.core_resources.models.uranium_resource.uranium_resource_disc.UraniumResourceDiscipline',
                     f'Resources.{self.COPPER_NAME}': 'climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc.CopperResourceDiscipline',
                     #f'Resources.{self.PLATINUM_NAME}': 'climateeconomics.core.core_resources.models.platinum_resource.platinum_resource_disc.PlatinumResourceDiscipline',
                     'Resources': 'climateeconomics.core.core_resources.resource_mix.resource_mix_disc.ResourceMixDiscipline'
                     }
        #chain_builders_resource = self.ee.factory.get_builder_from_module()
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict, associate_namespace=self.associate_namespace)
        # builder_list.append(chain_builders_resource)
        return builder_list
