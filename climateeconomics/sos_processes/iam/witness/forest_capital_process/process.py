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

from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from energy_models.sos_processes.witness_sub_process_builder import WITNESSSubProcessBuilder
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS


class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Forest and capital Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        chain_builders = []
        # retrieve energy process

        chain_builders_forest = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'forest_v2_process')
        chain_builders.extend(chain_builders_forest)

        chain_builders_capital = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'non_use_capital_process')
        chain_builders.extend(chain_builders_capital)

        return chain_builders
