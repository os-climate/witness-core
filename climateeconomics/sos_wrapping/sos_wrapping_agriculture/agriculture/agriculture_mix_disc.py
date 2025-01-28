'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/27-2023/11/03 Copyright 2023 Capgemini

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
from energy_models.core.stream_type.energy_disc import EnergyDiscipline

from climateeconomics.glossarycore import GlossaryCore
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry


class AgricultureMixDiscipline(EnergyDiscipline):
    # ontology information
    _ontology_data = {
        'label': 'Agriculture Mix Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-tractor fa-fw',
        'version': '',
    }


    DESC_IN = {GlossaryCore.techno_list: {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                                     'possible_values': ['Crop', 'Forest'],
                                     'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_agriculture',
                                     'structuring': True,
                                     'unit': '-'},
               'data_fuel_dict': {'type': 'dict', 'unit': 'defined in dict',
                                  'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                  'namespace': 'ns_agriculture', 'default': BiomassDry.data_energy_dict},
               }
    DESC_IN.update(EnergyDiscipline.DESC_IN)
    name = 'AgricultureMix'
    stream_name = BiomassDry.name

    # -- add specific techno outputs to this
    DESC_OUT = {}
    DESC_OUT.update(EnergyDiscipline.DESC_OUT)

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.energy_model = BiomassDry(self.energy_name)
        self.energy_model.configure_parameters(inputs_dict)
