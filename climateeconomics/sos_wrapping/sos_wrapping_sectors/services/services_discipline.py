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
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline import SectorDiscipline


class ServicesDiscipline(SectorDiscipline):
    "Services sector discpline"

    # ontology information
    _ontology_data = {
        'label': 'Services sector WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-building-user',
        'version': '',
    }
    _maturity = 'Research'
    
    sector_name = 'Services'
    prod_cap_unit = 'T$'
