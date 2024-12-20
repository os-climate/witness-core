'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/08-2023/11/03 Copyright 2023 Capgemini

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
import pickle

from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import (
    Study as MDA_Coarse,
)


def launch_data_pickle_generation(directory=''):
    # Run MDA WITNESS Coarse
    name = 'Data_Generator'
    ee = ExecutionEngine(name)
    model_name = 'EnergyMix'

    repo = 'climateeconomics.sos_processes.iam.witness'
    builder = ee.factory.get_builder_from_process(
        repo, 'witness_coarse')

    ee.factory.set_builders_to_coupling_builder(builder)
    ee.configure()
    usecase = MDA_Coarse(execution_engine=ee)
    usecase.study_name = name
    values_dict = usecase.setup_usecase()

    ee.display_treeview_nodes()
    full_values_dict = {}
    for dict_v in values_dict:
        full_values_dict.update(dict_v)

    full_values_dict[f'{name}.epsilon0'] = 1.0
    full_values_dict[f'{name}.tolerance'] = 1.0e-8
    full_values_dict[f'{name}.inner_mda_name'] = 'MDAGaussSeidel'
    full_values_dict[f'{name}.max_mda_iter'] = 200

    ee.load_study_from_input_dict(full_values_dict)

    ee.execute()

    Energy_Mix_disc = ee.dm.get_disciplines_with_name(
        f'{name}.{model_name}')[0]
    energy_list = Energy_Mix_disc.get_sosdisc_inputs(
        GlossaryCore.energy_list)

# Inputs
    mda_coarse_data_energymix_input_dict = {}
    # Check if the input is a coupling
    full_inputs = Energy_Mix_disc.get_input_data_names()
    # For the coupled inputs and outputs, test inputs/outputs on all
    # namespaces
    coupled_inputs = []
    namespaces = [f'{name}.', f'{name}.{model_name}.']
    for namespace in namespaces:
        coupled_inputs += [input[len(namespace):] for input in full_inputs if ee.dm.get_data(
            input, 'coupling')]
    # Loop on inputs and fill the dict with value and boolean is_coupling
    for key in Energy_Mix_disc.get_sosdisc_inputs().keys():
        is_coupling = False
        if key in coupled_inputs:
            is_coupling = True
        mda_coarse_data_energymix_input_dict[key] = {
            'value': Energy_Mix_disc.get_sosdisc_inputs(key), 'is_coupling': is_coupling}
    # Output
    mda_coarse_data_energymix_output_dict = {}
    full_outputs = Energy_Mix_disc.get_output_data_names()
    # For the coupled inputs and outputs, test inputs/outputs on all
    # namespaces
    coupled_outputs = []
    namespaces = [f'{name}.', f'{name}.{model_name}.']
    for namespace in namespaces:
        coupled_outputs += [output[len(namespace):] for output in full_outputs if ee.dm.get_data(
            output, 'coupling')]
    for key in Energy_Mix_disc.get_sosdisc_outputs().keys():
        is_coupling = False
        if key in coupled_outputs:
            is_coupling = True
        mda_coarse_data_energymix_output_dict[key] = {
            'value': Energy_Mix_disc.get_sosdisc_outputs(key), 'is_coupling': is_coupling}

    # Collect input and output data from each energy and each techno
    mda_coarse_data_streams_input_dict, mda_coarse_data_streams_output_dict = {}, {}
    mda_coarse_data_technologies_input_dict, mda_coarse_data_technologies_output_dict = {}, {}

    ############
    # Energies #
    ############
    for energy in energy_list:
        # Loop on energies
        energy_disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}.{energy}')[0]
        # Inputs
        mda_coarse_data_streams_input_dict[energy] = {}
        # Check if the input is a coupling
        full_inputs = energy_disc.get_input_data_names()
        # For the coupled inputs and outputs, test inputs/outputs on all
        # namespaces
        coupled_inputs = []
        namespaces = [f'{name}.', f'{name}.{model_name}.',
                      f'{name}.{model_name}.{energy}.']
        for namespace in namespaces:
            coupled_inputs += [input[len(namespace):] for input in full_inputs if ee.dm.get_data(
                input, 'coupling')]
        # Loop on inputs and fill the dict with value and boolean is_coupling
        for key in energy_disc.get_sosdisc_inputs().keys():
            is_coupling = False
            if key in coupled_inputs:
                is_coupling = True
            mda_coarse_data_streams_input_dict[energy][key] = {
                'value': energy_disc.get_sosdisc_inputs(key), 'is_coupling': is_coupling}
        # Output
        mda_coarse_data_streams_output_dict[energy] = {}
        full_outputs = energy_disc.get_output_data_names()
        # For the coupled inputs and outputs, test inputs/outputs on all
        # namespaces
        coupled_outputs = []
        namespaces = [f'{name}.', f'{name}.{model_name}.',
                      f'{name}.{model_name}.{energy}.']
        for namespace in namespaces:
            coupled_outputs += [output[len(namespace):] for output in full_outputs if ee.dm.get_data(
                output, 'coupling')]
        for key in energy_disc.get_sosdisc_outputs().keys():
            is_coupling = False
            if key in coupled_outputs:
                is_coupling = True
            mda_coarse_data_streams_output_dict[energy][key] = {
                'value': energy_disc.get_sosdisc_outputs(key), 'is_coupling': is_coupling}
        ################
        # Technologies #
        ################
        technologies_list = energy_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
        for techno in technologies_list:
            # Loop on technologies
            techno_disc = ee.dm.get_disciplines_with_name(
                f'{name}.{model_name}.{energy}.{techno}')[0]
            # Inputs
            mda_coarse_data_technologies_input_dict[techno] = {}
            full_inputs = techno_disc.get_input_data_names()
            # For the coupled inputs and outputs, test inputs/outputs on all
            # namespaces
            coupled_inputs = []
            namespaces = [f'{name}.', f'{name}.{model_name}.', f'{name}.{model_name}.{energy}.',
                          f'{name}.{model_name}.{energy}.{techno}.']
            for namespace in namespaces:
                coupled_inputs += [input[len(namespace):] for input in full_inputs if ee.dm.get_data(
                    input, 'coupling')]
            for key in techno_disc.get_sosdisc_inputs().keys():
                is_coupling = False
                if key in coupled_inputs:
                    is_coupling = True
                mda_coarse_data_technologies_input_dict[techno][key] = {
                    'value': techno_disc.get_sosdisc_inputs(key), 'is_coupling': is_coupling}
            # Output
            mda_coarse_data_technologies_output_dict[techno] = {}
            full_outputs = techno_disc.get_output_data_names()
            # For the coupled inputs and outputs, test inputs/outputs on all
            # namespaces
            coupled_outputs = []
            namespaces = [f'{name}.', f'{name}.{model_name}.', f'{name}.{model_name}.{energy}.',
                          f'{name}.{model_name}.{energy}.{techno}.']
            for namespace in namespaces:
                coupled_outputs += [output[len(namespace):] for output in full_outputs if ee.dm.get_data(
                    output, 'coupling')]
            for key in techno_disc.get_sosdisc_outputs().keys():
                is_coupling = False
                if key in coupled_outputs:
                    is_coupling = True
                mda_coarse_data_technologies_output_dict[techno][key] = {
                    'value': techno_disc.get_sosdisc_outputs(key), 'is_coupling': is_coupling}

    ccs_list = Energy_Mix_disc.get_sosdisc_inputs(
        GlossaryCore.ccs_list)
    ###############
    # CCS Streams #
    ###############
    for stream in ccs_list:
        stream_disc = ee.dm.get_disciplines_with_name(
            f'{name}.CCUS.{stream}')[0]
        # Inputs
        mda_coarse_data_streams_input_dict[stream] = {}
        full_inputs = stream_disc.get_input_data_names()
        # For the coupled inputs and outputs, test inputs/outputs on all
        # namespaces
        coupled_inputs = []
        namespaces = [f'{name}.', f'{name}.CCUS.', f'{name}.CCUS.{stream}.', ]
        for namespace in namespaces:
            coupled_inputs += [input[len(namespace):] for input in full_inputs if ee.dm.get_data(
                input, 'coupling')]
        for key in stream_disc.get_sosdisc_inputs().keys():
            is_coupling = False
            if key in coupled_inputs:
                is_coupling = True
            mda_coarse_data_streams_input_dict[stream][key] = {
                'value': stream_disc.get_sosdisc_inputs(key), 'is_coupling': is_coupling}
        # Output
        mda_coarse_data_streams_output_dict[stream] = {}
        full_outputs = stream_disc.get_output_data_names()
        # For the coupled inputs and outputs, test inputs/outputs on all
        # namespaces
        coupled_outputs = []
        namespaces = [f'{name}.', f'{name}.CCUS.', f'{name}.CCUS.{stream}.']
        for namespace in namespaces:
            coupled_outputs += [output[len(namespace):] for output in full_outputs if ee.dm.get_data(
                output, 'coupling')]
        for key in stream_disc.get_sosdisc_outputs().keys():
            is_coupling = False
            if key in coupled_outputs:
                is_coupling = True
            mda_coarse_data_streams_output_dict[stream][key] = {
                'value': stream_disc.get_sosdisc_outputs(key), 'is_coupling': is_coupling}
        ################
        # Technologies #
        ################
        technologies_list = stream_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
        for techno in technologies_list:
            # Loop on technologies
            techno_disc = ee.dm.get_disciplines_with_name(
                f'{name}.CCUS.{stream}.{techno}')[0]
            # Inputs
            mda_coarse_data_technologies_input_dict[techno] = {}
            full_inputs = techno_disc.get_input_data_names()
            # For the coupled inputs and outputs, test inputs/outputs on all
            # namespaces
            coupled_inputs = []
            namespaces = [f'{name}.', f'{name}.CCUS.', f'{name}.CCUS.{stream}.',
                          f'{name}.CCUS.{stream}.{techno}.']
            for namespace in namespaces:
                coupled_inputs += [input[len(namespace):] for input in full_inputs if ee.dm.get_data(
                    input, 'coupling')]
            for key in techno_disc.get_sosdisc_inputs().keys():
                is_coupling = False
                if key in coupled_inputs:
                    is_coupling = True
                mda_coarse_data_technologies_input_dict[techno][key] = {
                    'value': techno_disc.get_sosdisc_inputs(key), 'is_coupling': is_coupling}
            # Output
            mda_coarse_data_technologies_output_dict[techno] = {}
            full_outputs = techno_disc.get_output_data_names()
            # For the coupled inputs and outputs, test inputs/outputs on all
            # namespaces
            coupled_outputs = []
            namespaces = [f'{name}.', f'{name}.CCUS.', f'{name}.CCUS.{stream}.',
                          f'{name}.CCUS.{stream}.{techno}.']
            for namespace in namespaces:
                coupled_outputs += [output[len(namespace):] for output in full_outputs if ee.dm.get_data(
                    output, 'coupling')]
            for key in techno_disc.get_sosdisc_outputs().keys():
                is_coupling = False
                if key in coupled_outputs:
                    is_coupling = True
                mda_coarse_data_technologies_output_dict[techno][key] = {
                    'value': techno_disc.get_sosdisc_outputs(key), 'is_coupling': is_coupling}

    if directory == '':
        prefix = '.'
    else:
        prefix = f'./{directory}'

    output = open(f'{prefix}/mda_coarse_data_streams_input_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_streams_input_dict, output)
    output.close()

    output = open(f'{prefix}/mda_coarse_data_streams_output_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_streams_output_dict, output)
    output.close()

    output = open(
        f'{prefix}/mda_coarse_data_technologies_input_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_technologies_input_dict, output)
    output.close()

    output = open(
        f'{prefix}/mda_coarse_data_technologies_output_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_technologies_output_dict, output)
    output.close()

    output = open(
        f'{prefix}/mda_coarse_data_energymix_input_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_energymix_input_dict, output)
    output.close()

    output = open(
        f'{prefix}/mda_coarse_data_energymix_output_dict.pkl', 'wb')
    pickle.dump(mda_coarse_data_energymix_output_dict, output)
    output.close()


if '__main__' == __name__:
    launch_data_pickle_generation()
