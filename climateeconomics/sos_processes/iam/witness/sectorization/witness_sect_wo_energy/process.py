'''
Copyright 2024 Capgemini

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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline import (
    GHGemissionsDiscipline,
)


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

        ns_dict = {GlossaryCore.NS_WITNESS: ns_scatter,
                   GlossaryCore.NS_ENERGY_MIX: ns_scatter,
                   GlossaryCore.NS_REGIONALIZED_POST_PROC: ns_scatter,
                   GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS: ns_scatter,
                   GlossaryCore.NS_SECTORS_POST_PROC_GDP: ns_scatter,
                   'ns_agriculture': ns_scatter,
                   'ns_forestry': ns_scatter}

        mods_dict = {
            'GHGCycle': 'climateeconomics.sos_wrapping.sos_wrapping_witness.ghgcycle.ghgcycle_discipline.GHGCycleDiscipline',
            'Damage': 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline',
            'Temperature change': 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline',
            'Policy': 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'}

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        chain_builders_landuse = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'land_use_v2_process')
        builder_list.extend(chain_builders_landuse)

        chain_builders_sect = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness.sectorization', 'sectorization_process')
        builder_list.extend(chain_builders_sect)

        ns_dict = {'ns_land_use': f'{self.ee.study_name}.EnergyMix',
                   GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}.EnergyMix',
                   'ns_resource': f'{self.ee.study_name}.EnergyMix',
                   GlossaryCore.NS_GHGEMISSIONS: f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}",
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)


        # emissions post proc modules :
        self.ee.ns_manager.add_ns(GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS,
                                  f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}")
        sectors_post_proc_module = 'climateeconomics.sos_wrapping.post_procs.sectors.emissions.economics_emissions'
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS, sectors_post_proc_module
        )
        for sector in GlossaryCore.DefaultSectorListGHGEmissions:
            ns = f'ns_{sector.lower()}_emissions'
            self.ee.ns_manager.add_ns(f'ns_{sector.lower()}_gdp', f"{self.ee.study_name}.Macroeconomics.{sector}")
            self.ee.ns_manager.add_ns(ns,
                                      f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}.{sector}")
            post_proc_module = f'climateeconomics.sos_wrapping.post_procs.sectors.emissions.{sector.lower()}'
            self.ee.post_processing_manager.add_post_processing_module_to_namespace(
                ns, post_proc_module
            )

        # gdp for sectors post proc modules :
        for sector in GlossaryCore.SectorsPossibleValues:
            ns = f'ns_{sector.lower()}_gdp'
            self.ee.ns_manager.add_ns(ns, f"{self.ee.study_name}.Macroeconomics.{sector}")

        return builder_list
