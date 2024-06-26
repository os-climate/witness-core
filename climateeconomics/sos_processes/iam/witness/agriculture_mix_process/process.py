'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2024/06/24 Copyright 2023 Capgemini

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

import re

from energy_models.core.energy_process_builder import EnergyProcessBuilder

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import (
    AGRI_MIX_MODEL_LIST,
)


class ProcessBuilder(EnergyProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Agriculture Mix',
        'description': '',
        'category': '',
        'version': '',
    }

    def __init__(self, ee):
        EnergyProcessBuilder.__init__(self, ee)
        self.model_list = AGRI_MIX_MODEL_LIST

    def get_builders(self):

        ns_study = self.ee.study_name

        ns_agriculture_mix = 'AgricultureMix'
        ns_crop = 'Crop'
        ns_forest = 'Forest'
        ns_dict = {'ns_agriculture': f'{ns_study}.{ns_agriculture_mix}',
                   'ns_energy': f'{ns_study}.{ns_agriculture_mix}',
                   'ns_energy_study': f'{ns_study}',
                   'ns_public': f'{ns_study}',
                   GlossaryCore.NS_WITNESS: f'{ns_study}',
                    GlossaryCore.NS_REFERENCE: f'{ns_study}',
                   GlossaryCore.NS_FUNCTIONS: f'{ns_study}',
                   'ns_biomass_dry': f'{ns_study}',
                   'ns_land_use': f'{ns_study}',
                   'ns_forest': f'{ns_study}.{ns_agriculture_mix}.{ns_forest}',
                   'ns_invest': f'{ns_study}'}
        self.ee.ns_manager.add_ns_def(ns_dict)
        self.ee.ns_manager.add_ns('ns_crop', f'{ns_study}.{ns_agriculture_mix}.{ns_crop}')
        builder_list = []

        agricultureDiscPath = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_mix_disc.AgricultureMixDiscipline'

        agrimix_builder = self.ee.factory.get_builder_from_module(
            ns_agriculture_mix, agricultureDiscPath)
        builder_list.append(agrimix_builder)

        for model_name in self.model_list:
            # technoDiscPath = self.get_techno_disc_path(techno_name,agricultureDiscPath)
            # fix just while crop and forest are not in the right folder
            if model_name == "Crop":
                technoDiscPath = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
                builder = self.ee.factory.get_builder_from_module(
                    f'{ns_agriculture_mix}.{model_name}', technoDiscPath)
            elif model_name == "Forest":
                technoDiscPath = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'

            builder = self.ee.factory.get_builder_from_module(
                f'{ns_agriculture_mix}.{model_name}', technoDiscPath)
            # Replace the display name of Crop by Food
            # Crop is necessary to deal with energy model genericity but food is more appropriate to understand what is behind the model
            # Diet + Crop energy with residues from diet
            # if model_name == "Crop":
            #     self.ee.ns_manager.add_display_ns_to_builder(
            #         builder, f'{ns_study}.{ns_agriculture_mix}.Food')
            builder_list.append(builder)

        return builder_list

    def get_techno_disc_path(self, techno_name, techno_dir_path, sub_dir=None):
        list_name = re.findall('[A-Z][^A-Z]*', techno_name)
        test = [len(element) for element in list_name]
        # -- in case only one letter is capital, support all are capital and don't add _
        if 1 in test:
            mod_name = "".join(element.lower() for element in list_name)
        else:
            mod_name = "_".join(element.lower() for element in list_name)
        # --case of CO2... to be generalized
        if '2' in mod_name:
            mod_name = "2_".join(mod_name.split('2'))
        # -- try to find rule for electrolysis case
        # -- get correct disc name in case of dot in name
        dot_plit = mod_name.split('.')
        dot_name = "_".join(dot_plit)
        disc_name = f'{dot_name}_disc'
        # -- fix techno name in case of dot in name
        dot_tech_split = techno_name.split('.')
        mod_techno_name = "".join(dot_tech_split)

        if sub_dir is not None:
            techno_path = f'{techno_dir_path}.{sub_dir}.{mod_name}.{disc_name}.{mod_techno_name}Discipline'
        else:
            techno_path = f'{techno_dir_path}.{mod_name}.{disc_name}.{mod_techno_name}Discipline'
        return techno_path
