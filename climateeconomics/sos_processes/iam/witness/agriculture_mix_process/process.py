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

from energy_models.core.energy_process_builder import EnergyProcessBuilder,\
    INVEST_DISCIPLINE_OPTIONS
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process._usecase import TECHNOLOGIES_LIST_FOR_OPT
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
import re

class ProcessBuilder(EnergyProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Agriculture Mix - Biomass Dry Mix',
        'description': '',
        'category': '',
        'version': '',
    }
    def __init__(self, ee):
        EnergyProcessBuilder.__init__(self, ee)
        self.techno_list = TECHNOLOGIES_LIST_FOR_OPT

    def get_builders(self):

        ns_study = self.ee.study_name

        biomass_dry_name = BiomassDry.name
        agriculture_mix = 'AgricultureMix'
        ns_dict = {'ns_agriculture': f'{ns_study}.{agriculture_mix}',
                   'ns_energy': f'{ns_study}.{agriculture_mix}',
                   'ns_energy_study': f'{ns_study}',
                   'ns_public': f'{ns_study}',
                   'ns_witness': f'{ns_study}',
                   'ns_resource': f'{ns_study}.{agriculture_mix}'}
        mods_dict = {}
        agricultureDiscPath = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture'
        mods_dict[f'{agriculture_mix}'] = agricultureDiscPath + '.agriculture_mix_disc.AgricultureMixDiscipline'
        for techno_name in self.techno_list:
            #technoDiscPath = self.get_techno_disc_path(techno_name,agricultureDiscPath)
            # fix just while crop and forest are not in the right folder
            if techno_name == "Crop":
                technoDiscPath ='climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
            elif techno_name == "Forest":
                technoDiscPath ='climateeconomics.sos_wrapping.sos_wrapping_forest.forest_v2.forest_disc.ForestDiscipline'
            mods_dict[f'{agriculture_mix}.{techno_name}'] = technoDiscPath
            print(technoDiscPath)

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        return builder_list


    def get_techno_disc_path(self, techno_name, techno_dir_path, sub_dir=None):
        list_name = re.findall('[A-Z][^A-Z]*', techno_name)
        test = [len(l) for l in list_name]
        #-- in case only one letter is capital, support all are capital and don't add _
        if 1 in test:
            mod_name = "".join(l.lower() for l in list_name)
        else:
            mod_name = "_".join(l.lower() for l in list_name)
        #--case of CO2... to be generalized
        if '2' in mod_name:
            mod_name = "2_".join(mod_name.split('2'))
        #-- try to find rule for electrolysis case
        #-- get correct disc name in case of dot in name
        dot_plit = mod_name.split('.')
        dot_name = "_".join(dot_plit)
        disc_name = f'{dot_name}_disc'
        #-- fix techno name in case of dot in name
        dot_tech_split = techno_name.split('.')
        mod_techno_name = "".join(dot_tech_split)

        if sub_dir is not None:
            techno_path = f'{techno_dir_path}.{sub_dir}.{mod_name}.{disc_name}.{mod_techno_name}Discipline'
        else:
            techno_path = f'{techno_dir_path}.{mod_name}.{disc_name}.{mod_techno_name}Discipline'
        return techno_path