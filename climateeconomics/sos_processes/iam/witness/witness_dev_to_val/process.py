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

from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT, DEFAULT_TECHNO_DICT_DEV
from energy_models.sos_processes.witness_sub_process_builder import WITNESSSubProcessBuilder
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from copy import deepcopy


DEFAULT_TECHNO_DICT = deepcopy(DEFAULT_TECHNO_DICT)
streams_to_add=['fuel.ethanol']
technos_to_add = ['Methanation', 'BiomassFermentation']
for key in DEFAULT_TECHNO_DICT_DEV.keys():
    if key not in DEFAULT_TECHNO_DICT.keys() and key in streams_to_add:
        DEFAULT_TECHNO_DICT[key]=dict({'type': DEFAULT_TECHNO_DICT_DEV[key]['type'], 'value':[]})
    for value in DEFAULT_TECHNO_DICT_DEV[key]['value']:
        try:
            if value not in DEFAULT_TECHNO_DICT[key]['value'] and value in technos_to_add:
                DEFAULT_TECHNO_DICT[key]['value']+=[value,]
        except:
            pass

class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Dev to Val Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def __init__(self, ee, process_level='val'):
        WITNESSSubProcessBuilder.__init__(
            self, ee)
        self.invest_discipline = INVEST_DISCIPLINE_OPTIONS[2]
        self.process_level = process_level

    def get_builders(self):

        chain_builders = []
        # retrieve witness dev processes
        chain_builders_witness_dev = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam', 'witness_wo_energy_dev')
        # retrieve witness val processes
        chain_builders_witness_val = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam', 'witness_wo_energy')
        i_disc_to_pop, i_disc_to_add = [], []
        for i, disc in enumerate(chain_builders_witness_val):
            if 'name of unwanted disc' in disc.sos_name:
                i_disc_to_pop += [i,]
        for i, disc in enumerate(chain_builders_witness_dev):
            if 'name of wanted disc' in disc.sos_name:
                i_disc_to_add += [i,]
        i_disc_to_pop.sort(reverse=True)
        for i in i_disc_to_pop:
            chain_builders_witness_val.pop(i)
        for i in i_disc_to_add:
            chain_builders_witness_val.append(chain_builders_witness_dev[i])
        chain_builders.extend(chain_builders_witness_val)

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = DEFAULT_TECHNO_DICT

        chain_builders_energy = self.ee.factory.get_builder_from_process(
            'energy_models.sos_processes.energy.MDA', 'energy_process_v0_mda',
            techno_dict=techno_dict, invest_discipline=self.invest_discipline)

        chain_builders.extend(chain_builders_energy)

        # Update namespace regarding land use and energy mix coupling
        ns_dict = {'ns_land_use': f'{self.ee.study_name}.EnergyMix',
                   'ns_functions': f'{self.ee.study_name}.EnergyMix',
                   'ns_ref': f'{self.ee.study_name}.NormalizationReferences',
                   'ns_invest': f'{self.ee.study_name}.InvestmentDistribution',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_witness',
            'climateeconomics.sos_wrapping.sos_wrapping_witness.post_proc_witness_optim.post_processing_witness_full')

        for resource_namespace in ['ns_coal_resource', 'ns_oil_resource', 'ns_natural_gas_resource', 'ns_uranium_resource']:
            self.ee.post_processing_manager.add_post_processing_module_to_namespace(
                resource_namespace, 'climateeconomics.sos_wrapping.sos_wrapping_resources.post_proc_resource.post_processing_resource')

        return chain_builders
