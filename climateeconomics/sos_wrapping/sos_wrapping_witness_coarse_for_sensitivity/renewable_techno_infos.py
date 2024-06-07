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
from copy import deepcopy

from energy_models.models.renewable.renewable_simple_techno.renewable_simple_techno_disc import (
    RenewableSimpleTechnoDiscipline,
)
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

RENEWABLE_DEFAULT_TECHNO_DICT = RenewableSimpleTechnoDiscipline.techno_infos_dict_default

class RenewableTechnoInfos(SoSWrapp):
    """
    Utility discipline to analyze Sensitivity Analysis demonstrator outputs in Witness Coarse Storytelling MDA.
    """
    # ontology information
    _ontology_data = {
        'label': 'Renewable Techno Infos',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    DESC_IN = {'Opex_percentage': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 0.12},
               'Initial_capex': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 230.0},
               'Energy_costs': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 70.0},
               }

    DESC_OUT = {'techno_infos_dict': {SoSWrapp.TYPE: 'dict',
                                      SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                                      SoSWrapp.NAMESPACE: 'ns_renewable_techno',
                                      SoSWrapp.DEFAULT: deepcopy(RENEWABLE_DEFAULT_TECHNO_DICT),
                                      SoSWrapp.UNIT: 'defined in dict'}
                }

    def run(self):
        techno_infos_dict = deepcopy(RENEWABLE_DEFAULT_TECHNO_DICT)
        techno_infos_dict['Opex_percentage'] = self.get_sosdisc_inputs('Opex_percentage')
        techno_infos_dict['Capex_init'] = self.get_sosdisc_inputs('Initial_capex')
        techno_infos_dict['resource_price'] = self.get_sosdisc_inputs('Energy_costs')
        self.store_sos_outputs_values({'techno_infos_dict': techno_infos_dict})
