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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline import (
    GHGemissionsDiscipline,
)


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Sectorization process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        ns_macro = self.ee.study_name + '.Macroeconomics'
        ns_scatter = self.ee.study_name 

        ns_dict = {GlossaryCore.NS_WITNESS: ns_scatter,
                   GlossaryCore.NS_MACRO: ns_macro,
                   GlossaryCore.NS_ENERGY_MIX: ns_scatter,
                   "ns_energy_market": ns_scatter,
                   'ns_public': ns_scatter,
                   GlossaryCore.NS_FUNCTIONS: ns_scatter,
                   GlossaryCore.NS_SECTORS: ns_macro,
                   GlossaryCore.NS_GHGEMISSIONS: f"{self.ee.study_name}.{GHGemissionsDiscipline.name}.{GlossaryCore.EconomicSectors}",

                   }

        # economics sector process called here before create_builder_list to avoid process overwriting namespace
        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness.sectorization', 'economics_sector_process')

        mods_dict = {'Population':'climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline.PopulationDiscipline',
                     'LaborMarket': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.labor_market.labor_market_discipline.LaborMarketDiscipline',
                     'Utility':'climateeconomics.sos_wrapping.sos_wrapping_sectors.utility.sectorized_utility_discipline.SectorizedUtilityDiscipline',
                     'Consumption':'climateeconomics.sos_wrapping.sos_wrapping_sectors.consumption.consumption_discipline.ConsumptionDiscipline',
                     'SectorsInvestDistribution': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_invests.sectors_redistribution_invest_discipline.SectorsRedistributionInvestsDiscipline',
                     GHGemissionsDiscipline.name: 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline'}
                           
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        chain_builders.extend(builder_list)

        return chain_builders
