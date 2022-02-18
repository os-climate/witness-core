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
# -*- coding: utf-8 -*-
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# Copyright (C) 2020 Airbus SAS.
# All rights reserved.
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Process without Energy',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        ns_scatter = self.ee.study_name

        ns_dict = {'ns_witness': ns_scatter,
                   'ns_energy_mix': ns_scatter,
                   'ns_ref': f'{ns_scatter}.NormalizationReferences'}

        mods_dict = {'Macroeconomics': 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline',
                     'Carboncycle': 'climateeconomics.sos_wrapping.sos_wrapping_witness.carboncycle.carboncycle_discipline.CarbonCycleDiscipline',
                     'Carbon_emissions': 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline',
                     'Damage': 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline',
                     'Temperature_change': 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline',
                     'Utility': 'climateeconomics.sos_wrapping.sos_wrapping_witness.utilitymodel.utilitymodel_discipline.UtilityModelDiscipline',
                     'Policy': 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'}

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        return builder_list
