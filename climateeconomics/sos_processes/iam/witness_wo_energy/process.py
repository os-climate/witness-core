'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/06 Copyright 2023 Capgemini

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
from climateeconomics.sos_wrapping.sos_wrapping_emissions.agriculture_emissions.agriculture_emissions_discipline import (
    AgricultureEmissionsDiscipline,
)
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
                   'ns_agriculture': ns_scatter,
                   GlossaryCore.NS_MACRO: ns_scatter,
                   'ns_forest': ns_scatter}

        mods_dict = {
            'Macroeconomics': 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline',
            'GHGCycle': 'climateeconomics.sos_wrapping.sos_wrapping_witness.ghgcycle.ghgcycle_discipline.GHGCycleDiscipline',
            'Damage': 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline',
            'Temperature change': 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline',
            'Utility': 'climateeconomics.sos_wrapping.sos_wrapping_witness.utilitymodel.utilitymodel_discipline.UtilityModelDiscipline',
            'Policy': 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'}

        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        chain_builders_landuse = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'land_use_v2_process')
        builder_list.extend(chain_builders_landuse)

        chain_builders_agriculture = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'agriculture_mix_process')
        builder_list.extend(chain_builders_agriculture)

        chain_builders_population = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'population_process')
        builder_list.extend(chain_builders_population)

        ns_dict = {'ns_land_use': f'{self.ee.study_name}.EnergyMix',
                   GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}.EnergyMix',
                   'ns_resource': f'{self.ee.study_name}.EnergyMix',
                   GlossaryCore.NS_GHGEMISSIONS: f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}",
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)


        mods_dict = {
            GHGemissionsDiscipline.name: 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline',
            AgricultureEmissionsDiscipline.name: 'climateeconomics.sos_wrapping.sos_wrapping_emissions.agriculture_emissions.agriculture_emissions_discipline.AgricultureEmissionsDiscipline',
        }
        non_use_capital_list = self.create_builder_list(
            mods_dict, ns_dict=ns_dict)
        builder_list.extend(non_use_capital_list)

        self.ee.ns_manager.add_ns(GlossaryCore.NS_REGIONALIZED_POST_PROC,
                                  f"{self.ee.study_name}.Macroeconomics.Regions")
        region_post_proc_module = 'climateeconomics.sos_wrapping.post_procs.regions'
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            GlossaryCore.NS_REGIONALIZED_POST_PROC, region_post_proc_module
        )

        # sector breakdown GDP post proc module :
        self.ee.ns_manager.add_ns(GlossaryCore.NS_SECTORS_POST_PROC_GDP,
                                  f"{self.ee.study_name}.Macroeconomics.{GlossaryCore.EconomicSectors}")
        sectors_post_proc_module = 'climateeconomics.sos_wrapping.post_procs.sectors.gdp_non_sectorized.economics_gdp'
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            GlossaryCore.NS_SECTORS_POST_PROC_GDP, sectors_post_proc_module
        )

        # emissions post proc modules :
        self.ee.ns_manager.add_ns(GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS,
                                  f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}")
        sectors_post_proc_module = 'climateeconomics.sos_wrapping.post_procs.sectors.emissions.economics_emissions'
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS, sectors_post_proc_module
        )

        ns = f'ns_{GlossaryCore.Households.lower()}_emissions'
        self.ee.ns_manager.add_ns(ns,
                                  f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}.{GlossaryCore.Households}")
        post_proc_module = f'climateeconomics.sos_wrapping.post_procs.sectors.emissions.{GlossaryCore.Households.lower()}'
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            ns, post_proc_module
        )

        for sector in GlossaryCore.DefaultSectorListGHGEmissions:
            ns = f'ns_{sector.lower()}_emissions'
            self.ee.ns_manager.add_ns(ns,
                                      f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}.{sector}")
            post_proc_module = f'climateeconomics.sos_wrapping.post_procs.sectors.emissions.{sector.lower()}'
            self.ee.post_processing_manager.add_post_processing_module_to_namespace(
                ns, post_proc_module
            )

        # gdp for sectors post proc modules :
        for sector in GlossaryCore.SectorsPossibleValues:
            ns = f'ns_{sector.lower()}_gdp'
            self.ee.ns_manager.add_ns(ns,
                                      f"{self.ee.study_name}.Macroeconomics.{GlossaryCore.EconomicSectors}.{sector}")
            post_proc_module = f'climateeconomics.sos_wrapping.post_procs.sectors.gdp_non_sectorized.{sector.lower()}'
            self.ee.post_processing_manager.add_post_processing_module_to_namespace(
                ns, post_proc_module
            )
        return builder_list
