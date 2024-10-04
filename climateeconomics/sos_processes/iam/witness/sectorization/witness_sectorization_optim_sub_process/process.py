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
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import (
    INVEST_DISC_NAME,
)
from energy_models.sos_processes.witness_sub_process_builder import (
    WITNESSSubProcessBuilder,
)

from climateeconomics.glossarycore import GlossaryCore


class ProcessBuilder(WITNESSSubProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Optimization Sub-Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        coupling_name = "WITNESS_Eval"
        designvariable_name = "DesignVariables"
        func_manager_name = "FunctionsManager"
        extra_name = 'WITNESS'
        self.invest_discipline = INVEST_DISCIPLINE_OPTIONS[2]

        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness.sectorization', 'witness_sectorization',
            techno_dict=self.techno_dict)

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=self.ee.study_name, clean_existing=True)
        self.ee.factory.update_builder_list_with_extra_name(
            extra_name, builder_list=chain_builders)

        # design variables builder
        design_var_path = 'sostrades_optimization_plugins.models.design_var.design_var_disc.DesignVarDiscipline'
        design_var_builder = self.ee.factory.get_builder_from_module(
            f'{designvariable_name}', design_var_path)
        chain_builders.append(design_var_builder)

        # function manager builder
        fmanager_path = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fmanager_builder = self.ee.factory.get_builder_from_module(
            f'{func_manager_name}', fmanager_path)
        chain_builders.append(fmanager_builder)

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            coupling_name, after_name=self.ee.study_name, clean_existing = True)

        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}.{coupling_name}.{extra_name}',
                   'ns_optim': f'{self.ee.study_name}',
                   GlossaryCore.NS_REFERENCE: f'{self.ee.study_name}.NormalizationReferences',
                   'ns_invest': f'{self.ee.study_name}.{coupling_name}.{extra_name}.{INVEST_DISC_NAME}', }
        self.ee.ns_manager.add_ns_def(ns_dict)

        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(
            coupling_name)
        coupling_builder.set_builder_info('cls_builder', chain_builders)
        #coupling_builder.set_builder_info('with_data_io', True)

        return coupling_builder
