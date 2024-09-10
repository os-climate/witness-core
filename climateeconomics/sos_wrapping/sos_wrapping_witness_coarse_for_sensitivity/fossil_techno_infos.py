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

from energy_models.models.fossil.fossil_simple_techno.fossil_simple_techno_disc import (
    FossilSimpleTechnoDiscipline,
)
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

FOSSIL_DEFAULT_TECHNO_DICT = FossilSimpleTechnoDiscipline.techno_infos_dict_default


class FossilTechnoInfos(SoSWrapp):
    """
    Utility discipline to analyze Sensitivity Analysis demonstrator outputs in Witness Coarse Storytelling MDA.
    """
    # ontology information
    _ontology_data = {
        'label': 'Fossil Techno Infos',
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

    DESC_IN = {'Opex_percentage': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 0.024},
               'Initial_capex': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 100.0},
               'Energy_costs': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 75.0},
               'CO2_from_production': {SoSWrapp.TYPE: 'float', SoSWrapp.DEFAULT: 0.37077040550222284},
               }

    DESC_OUT = {'techno_infos_dict': {SoSWrapp.TYPE: 'dict',
                                      SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                                      SoSWrapp.NAMESPACE: 'ns_fossil_techno',
                                      SoSWrapp.DEFAULT: deepcopy(FOSSIL_DEFAULT_TECHNO_DICT),
                                      SoSWrapp.UNIT: 'defined in dict'}
                }

    def run(self):
        techno_infos_dict = deepcopy(FOSSIL_DEFAULT_TECHNO_DICT)
        techno_infos_dict['Opex_percentage'] = self.get_sosdisc_inputs('Opex_percentage')
        techno_infos_dict['Capex_init'] = self.get_sosdisc_inputs('Initial_capex')
        techno_infos_dict['resource_price'] = self.get_sosdisc_inputs('Energy_costs')
        techno_infos_dict['CO2_from_production'] = self.get_sosdisc_inputs('CO2_from_production')
        self.store_sos_outputs_values({'techno_infos_dict': techno_infos_dict})
