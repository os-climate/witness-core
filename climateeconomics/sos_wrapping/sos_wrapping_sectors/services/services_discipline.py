'''
Copyright 2022 Airbus SAS
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
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline import (
    SectorDiscipline,
)


class ServicesDiscipline(SectorDiscipline):
    "Services sector discpline"
    sector_name = GlossaryCore.SectorServices

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

    # Update default values
    def setup_sos_disciplines(self):
        SectorDiscipline.setup_sos_disciplines(self)
        self.update_default_value('capital_start', 'in', DatabaseWitnessCore.SectorServiceCapitalStart.value)
        self.update_default_value('productivity_start', 'in', DatabaseWitnessCore.SectorServiceProductivityStart.value)
        self.update_default_value('productivity_gr_start', 'in', DatabaseWitnessCore.SectorServiceProductivityGrowthStart.value)
        self.update_default_value('decline_rate_tfp', 'in', 0.088925)
        self.update_default_value('energy_eff_k', 'in', 0.04383)
        self.update_default_value('energy_eff_cst', 'in', 3.12565)
        self.update_default_value('energy_eff_xzero', 'in', 2044.09)
        self.update_default_value('energy_eff_max', 'in', 12.5229)
        self.update_default_value('output_alpha', 'in', 0.594575)
        self.update_default_value('depreciation_capital', 'in', 0.058)
