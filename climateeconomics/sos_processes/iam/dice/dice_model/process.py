'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/03 Copyright 2023 Capgemini

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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'DICE Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        ns_scatter = self.ee.study_name

        ns_dict = {'ns_dice': ns_scatter, GlossaryCore.NS_WITNESS: ns_scatter, 'ns_scenario': ns_scatter}

        mods_dict = {
            'Carboncycle': 'climateeconomics.sos_wrapping.sos_wrapping_dice.carboncycle.carboncycle_discipline.CarbonCycleDiscipline',
            'Macroeconomics': 'climateeconomics.sos_wrapping.sos_wrapping_dice.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline',

            'Temperature_change': 'climateeconomics.sos_wrapping.sos_wrapping_dice.tempchange.tempchange_discipline.TempChangeDiscipline',
            'Damage': 'climateeconomics.sos_wrapping.sos_wrapping_dice.damagemodel.damagemodel_discipline.DamageDiscipline',
            'Carbon_emissions': 'climateeconomics.sos_wrapping.sos_wrapping_dice.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline',
            'Utility': 'climateeconomics.sos_wrapping.sos_wrapping_dice.utilitymodel.utilitymodel_discipline.UtilityModelDiscipline'}

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)
        return builder_list
