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

from climateeconomics.glossarycore import GlossaryCore


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS economics sectorization process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        ns_macro = f"{self.ee.study_name}.{'Macroeconomics'}"
        ns_scatter = self.ee.study_name

        ns_dict = {GlossaryCore.NS_WITNESS: ns_scatter,
                   GlossaryCore.NS_MACRO: ns_macro,
                   'ns_public': ns_scatter,
                   GlossaryCore.NS_FUNCTIONS: ns_scatter,
                   GlossaryCore.NS_REFERENCE: ns_scatter,
                   GlossaryCore.NS_SECTORS: ns_macro,
                   GlossaryCore.NS_ENERGY_MIX: ns_scatter,
                   GlossaryCore.NS_GHGEMISSIONS: ns_scatter,
                   GlossaryCore.NS_HOUSEHOLDS_EMISSIONS: self.ee.study_name}

        mods_dict = {'Macroeconomics': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline',
                     f'Macroeconomics.{GlossaryCore.SectorServices}': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline' ,
                     f'Macroeconomics.{GlossaryCore.SectorAgriculture}':'climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline.AgricultureDiscipline',
                     f'Macroeconomics.{GlossaryCore.SectorIndustry}':'climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline.IndustrialDiscipline'
                     }
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        return builder_list
