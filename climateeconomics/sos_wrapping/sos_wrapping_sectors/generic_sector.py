'''
Copyright 2024 Capgemini
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.glossarycore import GlossaryCore


class GenericSector(AutodifferentiedDisc):
    """Generic sector class for witness sectorized version"""
    _ontology_data = {
        'label': 'Agriculture sector model for WITNESS Sectorized version',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-seedling fa-fw',
        'version': '',
    }

    sub_sectors = []
    sub_sector_commun_variables = []

    name = ''

    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'default': GlossaryCore.YearStartDefault, 'structuring': True,
                                 'unit': GlossaryCore.Years, 'visibility': 'Shared', 'namespace': 'ns_public',
                                 'range': [1950, 2080]},
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
    }

    for sub_sector in sub_sectors:
        for commun_variable_name, commun_variable_descr in sub_sector_commun_variables:
            DESC_IN.update({
                f"{sub_sector}.{commun_variable_name}": GlossaryCore.get_subsector_variable(
                    subsector_name=sub_sector, sector_namespace=GlossaryCore.NS_AGRI, var_descr=commun_variable_descr),
            })

    DESC_OUT = {}

    for commun_variable_name in sub_sector_commun_variables:
        DESC_OUT.update()


