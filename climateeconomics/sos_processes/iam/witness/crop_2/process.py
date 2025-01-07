'''
Copyright 2024 Capgemini
Modifications on 2023/04/19-2024/06/24 Copyright 2024 Capgemini

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


from energy_models.core.energy_process_builder import EnergyProcessBuilder

from climateeconomics.glossarycore import GlossaryCore


class ProcessBuilder(EnergyProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Crop 2 process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        ns_study = self.ee.study_name
        model_name = 'AgricultureMix.Crop'
        ns_dict = {
            'ns_public': ns_study,
            GlossaryCore.NS_WITNESS: ns_study,
            'ns_crop': f'{ns_study}.{model_name}',
            'ns_sectors': f'{ns_study}.Macroeconomics',
            GlossaryCore.NS_ENERGY_MIX: f'{ns_study}',
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mods_dict = {
            model_name: 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop_2.crop_disc_2.CropDiscipline',
            f'Macroeconomics.{GlossaryCore.SectorAgriculture}': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_economy_discipline.AgricultureEconomyDiscipline',
                     }
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        return builder_list
