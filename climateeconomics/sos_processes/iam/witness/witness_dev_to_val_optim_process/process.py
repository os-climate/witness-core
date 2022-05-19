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

from climateeconomics.sos_processes.iam.witness.witness_dev_to_val_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT, DEFAULT_TECHNO_DICT_DEV
from energy_models.sos_processes.witness_sub_process_builder import WITNESSSubProcessBuilder
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from copy import deepcopy

DEFAULT_TECHNO_DICT = deepcopy(DEFAULT_TECHNO_DICT)
streams_to_add=[]
technos_to_add = ['Methanation', 'PlasmaCracking', 'Pyrolysis', 'AutothermalReforming', 'CoElectrolysis', 'BiogasFired',
                  'OilGen', 'BiomassFired', 'flue_gas_capture.ChilledAmmoniaProcess', 'flue_gas_capture.CO2Membranes',
                  'flue_gas_capture.PiperazineProcess', 'flue_gas_capture.PressureSwingAdsorption', 'BiomassBuryingFossilization',
                  'DeepOceanInjection', 'EnhancedOilRecovery', 'PureCarbonSolidStorage']
for key in DEFAULT_TECHNO_DICT_DEV.keys():
    if key not in DEFAULT_TECHNO_DICT.keys() and key in streams_to_add:
        DEFAULT_TECHNO_DICT[key]=dict({'type': DEFAULT_TECHNO_DICT_DEV[key]['type'], 'value':[]})
    for value in DEFAULT_TECHNO_DICT_DEV[key]['value']:
        if value not in DEFAULT_TECHNO_DICT[key]['value'] and value in technos_to_add:
            DEFAULT_TECHNO_DICT[key]['value']+=[value,]

class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Dev to Val Optimization Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        optim_name = OPTIM_NAME

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = DEFAULT_TECHNO_DICT

        coupling_builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_dev_to_val_optim_sub_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='val')

        # modify namespaces defined in the child process
        for ns in self.ee.ns_manager.ns_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, optim_name, after_name=self.ee.study_name)  # optim_name

        #-- set optim builder
        opt_builder = self.ee.factory.create_optim_builder(
            optim_name, [coupling_builder])

        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_optim',
            'climateeconomics.sos_wrapping.sos_wrapping_witness.post_proc_witness_optim.post_processing_witness_full')

        return opt_builder
