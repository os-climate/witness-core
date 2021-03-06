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

from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Sectorization Opt process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        
        coupling_name = "Sectorization_Eval"
        designvariable_name = "DesignVariables"
        func_manager_name = "FunctionsManager"
        optim_name = "SectorsOpt"
        objectives_name = "Objectives"
        macro_name = "Macroeconomics"


        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'sectorization_process')
        
        # design variables builder
        design_var_path = 'sos_trades_core.execution_engine.design_var.design_var_disc.DesignVarDiscipline'
        design_var_builder = self.ee.factory.get_builder_from_module(f'{designvariable_name}', design_var_path)
        chain_builders.append(design_var_builder)

        # function manager builder
        fmanager_path = 'sos_trades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
        fmanager_builder = self.ee.factory.get_builder_from_module(f'{func_manager_name}', fmanager_path)
        chain_builders.append(fmanager_builder)
        
        #Add objective discipline 
        obj_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.objectives.objectives_discipline.ObjectivesDiscipline'
        obj_builder = self.ee.factory.get_builder_from_module(f'{objectives_name}', obj_path)
        chain_builders.append(obj_builder)
        
        #ns_macro = self.ee.study_name + coupling_name + '.Macroeconomics'
        ns_scatter = self.ee.study_name 
        
                # modify namespaces defined in the child process
        for ns in self.ee.ns_manager.ns_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, optim_name + '.' +coupling_name, after_name=self.ee.study_name)
        
        ns_dict = {'ns_optim': ns_scatter + '.' + optim_name,
                   'ns_services':  ns_scatter + '.' + optim_name + '.' + coupling_name + '.' + macro_name + '.Services',
                   'ns_indus':  ns_scatter + '.' + optim_name + '.' + coupling_name + '.' + macro_name + '.Industry',
                   'ns_agri':  ns_scatter + '.' + optim_name + '.' + coupling_name + '.' + macro_name + '.Agriculture',
                   'ns_obj': ns_scatter + '.' + optim_name + '.' + coupling_name + '.Objectives',}
        self.ee.ns_manager.add_ns_def(ns_dict)
    
        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(coupling_name)
        # coupling
        coupling_builder.set_builder_info('cls_builder', chain_builders)
        coupling_builder.set_builder_info('with_data_io', True)

        opt_builder = self.ee.factory.create_optim_builder('SectorsOpt', [coupling_builder])
    

        return opt_builder