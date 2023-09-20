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


class IndustrialDiscipline(SectorDiscipline):
    "Industrial sector discpline"

    # ontology information
    _ontology_data = {
        'label': 'Industrial sector WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-industry fa-fw',
        'version': '',
    }
    _maturity = 'Research'
    
    sector_name = 'Industry'
    prod_cap_unit = 'T$'

    # update default values:
    DESC_IN = SectorDiscipline.DESC_IN
    DESC_IN['productivity_start']['default'] = 0.4903228
    DESC_IN['capital_start']['default'] = 88.5051
    DESC_IN['productivity_gr_start']['default'] = 0.00019
    DESC_IN['decline_rate_tfp']['default'] = 1e-05
    DESC_IN['energy_eff_k']['default'] = 0.026986
    DESC_IN['energy_eff_cst']['default'] = 0.171694
    DESC_IN['energy_eff_xzero']['default'] = 2015
    DESC_IN['energy_eff_max']['default'] = 3.1562276
    DESC_IN['output_alpha']['default'] = 0.909985
    DESC_IN['depreciation_capital']['default'] = 0.075
