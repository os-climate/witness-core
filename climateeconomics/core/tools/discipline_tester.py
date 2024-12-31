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

from typing import List

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)


def discipline_test_function(module_path: str, name: str, model_name: str,
                             inputs_dict: dict, namespaces_dict: dict,
                             jacobian_test: bool, coupling_inputs: List[str] = [], coupling_outputs: List[str] = [],
                             pickle_name: str = None, pickle_directory: str = None,
                             override_dump_jacobian: bool = False,
                             show_graphs: bool = True):
    """
    Function to perform a discipline test, mimicking the behavior of the DisciplineTestTemplate class.

    :param module_path: Path to the module containing the discipline
    :param model_name: Name of the model to be tested
    :param coupling_inputs: List of coupling input variable names
    :param coupling_outputs: List of coupling output variable names
    :param jacobian_test: Flag indicating if Jacobian testing is required
    :param name: Name of the execution engine
    :param show_graphs: Flag to display graphs generated during post-processing
    :param inputs_dict: Dictionary of input values for the study
    :param namespaces_dict: Dictionary of namespaces definitions
    """

    # Initialize the execution engine
    if jacobian_test and (pickle_name is None or pickle_directory is None):
        raise ValueError("pickle_name and pickle_directory must be filled.")
    ee = ExecutionEngine(name)
    ee.ns_manager.add_ns_def(namespaces_dict)

    # Build and configure the discipline
    builder = ee.factory.get_builder_from_module(model_name, module_path)
    ee.factory.set_builders_to_coupling_builder(builder)

    ee.configure()
    ee.display_treeview_nodes()

    # Load inputs and execute the study
    ee.load_study_from_input_dict(inputs_dict)
    ee.execute()

    # Retrieve discipline for post-processing
    disc = ee.dm.get_disciplines_with_name(f'{name}.{model_name}')[0]
    filter = disc.get_chart_filter_list()
    graph_list = disc.get_post_processing_list(filter)

    # Show generated graphs
    if show_graphs:
        for graph in graph_list:
            graph.to_plotly().show()

    # Perform Jacobian test if required
    if jacobian_test:
        def get_full_varnames(variables: list[str]):
            return [ee.dm.get_all_namespaces_from_var_name(varname)[0] for varname in variables]

        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        jacobian_test_instnace = AbstractJacobianUnittest()
        jacobian_test_instnace.override_dump_jacobian = override_dump_jacobian
        jacobian_test_instnace.check_jacobian(
            location=pickle_directory,
            filename=pickle_name,
            local_data=disc_techno.local_data,
            discipline=disc_techno,
            step=1e-15,
            derr_approx='complex_step',
            inputs=get_full_varnames(coupling_inputs),
            outputs=get_full_varnames(coupling_outputs)
        )
