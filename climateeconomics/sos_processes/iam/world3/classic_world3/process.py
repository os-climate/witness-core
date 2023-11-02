'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/20-2023/11/02 Copyright 2023 Capgemini

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
# -*- coding: utf-8 -*-
from climateeconomics.glossarycore import GlossaryCore
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'World 3',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        study_name = self.ee.study_name

        ns_dict = {'ns_data': f'{study_name}',
                   'ns_coupling': f'{study_name}',
                   'ns_obj': f'{study_name}'}

        mods_dict = {GlossaryCore.SectorAgriculture: 'climateeconomics.sos_wrapping.sos_wrapping_world3.agriculture_discipline.AgricultureDiscipline',
                     'Capital': 'climateeconomics.sos_wrapping.sos_wrapping_world3.capital_discipline.CapitalDiscipline',
                     'Population': 'climateeconomics.sos_wrapping.sos_wrapping_world3.population_discipline.PopulationDiscipline',
                     'Resource': 'climateeconomics.sos_wrapping.sos_wrapping_world3.resource_discipline.ResourceDiscipline',
                     'Pollution': 'climateeconomics.sos_wrapping.sos_wrapping_world3.pollution_discipline.PollutionDiscipline'}

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)


        return builder_list