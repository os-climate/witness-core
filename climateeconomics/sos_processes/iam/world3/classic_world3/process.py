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