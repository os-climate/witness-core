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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS sectorization process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):

        ns_macro = self.ee.study_name + '.Macroeconomics'
        ns_scatter = self.ee.study_name 

        ns_dict = {'ns_witness': ns_scatter,
                   'ns_macro': ns_macro,
                   'ns_energy_mix': ns_scatter,
                   'ns_public': ns_scatter,
                   'ns_functions': ns_scatter,
                   'ns_ref': ns_scatter,
                   'ns_sectors': ns_macro
                   }
        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'economics_sector_process')
        mods_dict = {'Population':'climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline.PopulationDiscipline',
                     'LaborMarket': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.labor_market.labor_market_discipline.LaborMarketDiscipline',
                     'Consumption':'climateeconomics.sos_wrapping.sos_wrapping_witness.consumption.consumption_discipline.ConsumptionDiscipline',
                     'SectorsInvestDistribution': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_invests.sectors_redistribution_invest_discipline.SectorsRedistributionInvestsDiscipline',
                     'SectorsEnergyDistribution': 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_energy.sectors_redistribution_energy_discipline.SectorsRedistributionEnergyDiscipline'}
                           
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)



        chain_builders.append(builder_list)

        return chain_builders
