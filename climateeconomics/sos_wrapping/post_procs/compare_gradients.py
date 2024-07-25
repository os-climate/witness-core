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
import logging
import os
import pickle
from copy import deepcopy
from math import floor
from os.path import dirname, join

import numpy as np
from gemseo.utils.derivatives_approx import DisciplineJacApprox
from gemseo.utils.pkl_tools import dump_compressed_pickle, load_compressed_pickle
from sostrades_core.execution_engine.sos_mda_chain import (
    SoSMDAChain,
)
from sostrades_core.sos_processes.script_test_all_usecases import test_compare_dm
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

'''
Post-processing that compares the analytical vs the approximated gradient of the objective lagrangian function wrt the design 
variables defined in the design_space. Therefore, this post-processing only works on processes that inherit from 
an optim process, namely optimization and multi-scenario optimization processes
The gradient is computed at the last iteration of the optimization problem
Does not work for optim sub-processes, since mdo_disc._differentiated_inputs=[] 
'''

TEMP_PKL_PATH = 'temp_pkl'


def find_mdo_disc(execution_engine, scenario_name, class_to_check):
    '''
    recover for the scenario name the mdo_discipline at the lowest level that respects the class_to_check
    ex:
    for an optim process:
        mdo_disc = execution_engine.root_process.proxy_disciplines[0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
    for a ms_optim_process:
            mdo_disc = execution_engine.root_process.proxy_disciplines[1].proxy_disciplines[0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

    this post-processing is assumed to be linked to the namespace ns_witness which value ends by .WITNESS. Therefore,
    the scenario name value = namespace value is something.WITNESS
    The mdo_discipline should be in WITNESS_EVAL which is one step above witness
    '''
    scenario_name_trimmed = scenario_name[:scenario_name.rfind('.')] # remove the .WITNESS of the namespace value
    levels = [execution_engine.root_process]
    while levels:
        current_level = levels.pop(0)
        # Check if current level has the required attribute
        if hasattr(current_level, 'mdo_discipline_wrapp'):
            if hasattr(current_level.mdo_discipline_wrapp, 'mdo_discipline'):
                mdo_disc = current_level.mdo_discipline_wrapp.mdo_discipline
                if isinstance(mdo_disc, class_to_check) and scenario_name_trimmed == mdo_disc.name:
                    logging.debug(f"The object at {current_level} is an instance of {class_to_check.__name__}")
                    return mdo_disc

        # Check if current level has proxy_disciplines and add them to the list
        if hasattr(current_level, 'proxy_disciplines'):
            levels.extend(current_level.proxy_disciplines)

    logging.debug(f"No instance of {class_to_check.__name__} found for mdo_discipline")
    return None


def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ['No_grad_check', 'Check_grad_Obj_Lagr']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts_grad', chart_list, 'No_grad_check', 'No_grad_check', multiple_selection=False)) # name 'Charts' is already used by ssp_comparison post-proc

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None): #scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
    finite_differences_step = 1e-7

    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'No_grad_check':
                chart_list = chart_filter.selected_values

    if 'Check_grad_Obj_Lagr' in chart_list:
        '''
        The l1s_test compare the variables before and after the post-processing. 
        In the post-processing below, the approx gradient computes the mda in X+h as opposed to X for the initial mda
        Therefore, data in the datamanager change and the l1s_test would fail. To prevent that, the data manager is recovered 
        at the end of post-processing  
        '''
        # The hierarchical level of the mdo discipline depends on the process used (sub-optim, optim, ms_optim, etc.)
        # the mdo discipline used to compute the gradients is a SoSMDAChain instance, not a SoSMDOScenario isntance
        # in case of multi-scenario, the proxy_disciplines contains as many elements as there are scenarios
        mdo_disc = find_mdo_disc(execution_engine, scenario_name, SoSMDAChain)
        inputs = mdo_disc._differentiated_inputs
        outputs = mdo_disc._differentiated_outputs
        dm_data_dict_before = deepcopy(execution_engine.dm.get_data_dict_values())
        logging.info('Post-processing: Computing analytical gradient')
        grad_analytical = mdo_disc.linearize(force_no_exec=True) #can add data of the converged point (see sos_mdo_discipline.py)

        # computed approximated gradient in 2nd step so that the analytical gradient is not computed in X0 + h
        # warm start must have been removed so that mda and convergence criteria are the same for analytical and approximated gradients
        # to avoid unnecessary computation of the approx gradient in case the of updated post-processing, the inputs are
        # compared to values in pkl file. If they are different, approx jac is recomputed, otherwise the pkl values are considered
        # Checking inputs
        pkl_path = join(dirname(__file__), TEMP_PKL_PATH)
        if not os.path.isdir(pkl_path):
            os.mkdir(pkl_path)
        recompute_gradients = True # initialize
        input_pkl = join(pkl_path, scenario_name + '_inputs_dm_data_dict.pkl')
        output_pkl = join(pkl_path, scenario_name + '_approx_jacobian.pkl')
        try:
            with open(input_pkl, 'rb') as f:
                dm_data_dict_pkl = pickle.load(f)
            f.close()
        except:
            logging.info(f'Cannot open file {input_pkl}. Must recompute approximated gradients')
            dm_data_dict_pkl = None
        if dm_data_dict_pkl is not None:
            compare_test_passed, error_msg_compare = test_compare_dm(dm_data_dict_before, dm_data_dict_pkl, scenario_name, 'pkl vs case dm' )
            logging.info(error_msg_compare)
            recompute_gradients = not compare_test_passed
        if recompute_gradients:
            logging.info('Input data have changed. Must recompute approx gradients')
            with open(input_pkl, 'wb') as f:
                pickle.dump(dm_data_dict_before, f)
            f.close()
        grad_approx = None
        try:
            grad_approx = load_compressed_pickle(output_pkl)
        except:
            logging.info('Missing grad approx pkl file. Must recompute approx gradients')
            recompute_gradients = True
        if recompute_gradients:
            logging.info('Post-processing: computing approximated gradient')
            approx = DisciplineJacApprox(mdo_disc, approx_method=DisciplineJacApprox.FINITE_DIFFERENCES, step=finite_differences_step)
            grad_approx = approx.compute_approx_jac(outputs, inputs)
            dump_compressed_pickle(output_pkl, grad_approx)
        else:
            logging.info(f'Post-processing: recovering approximated gradient from pkl file {output_pkl}')


        logging.info('Post-processing: generating gradient charts')
        # extract the gradient values for the objective lagrangian only
        output = None
        for key in outputs:
            if key.split('.')[-1] == OBJECTIVE_LAGR:
                output = key
        grad_analytical_dict = grad_analytical[output]
        grad_approx_dict = grad_approx[output]

        # Plot values from adjoint and finite differences gradient vs poles
        # keep same color for adjoint and finite difference for a given variable
        color_dict = {
            "Blue": dict(color='blue'),
            "Green": dict(color='green'),
            "Orange": dict(color='orange'),
            "Red": dict(color='red'),
            "Black": dict(color='black'),
            "Purple": dict(color='purple'),
            "Magenta": dict(color='magenta'),
            "Yellow": dict(color='yellow'),
            "Cyan": dict(color='cyan'),
        }
        # put in a box the symbols used for adjoint and finite difference gradients
        note = {'______': 'Adjoint',
                '- - - - - - ': 'Finite differences',
                }
        chart_name = f'Gradient of {OBJECTIVE_LAGR} vs years. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variables pole number (corresponding to years) [-]', 'Gradient [with unit]',
                                         chart_name=chart_name, y_min_zero=False, show_legend=False)

        visible_line = True
        for index, k in enumerate(inputs):
            color_dict_index = index - floor(index / len(color_dict.keys())) * len(color_dict.keys()) # when reach last color, start again with first color
            color_name = list(color_dict.keys())[color_dict_index]
            color_code = color_dict[color_name]
            v_analytical = grad_analytical_dict[k]
            v_approx = grad_approx_dict[k]
            x_analytical = list(range(len(v_analytical[0])))
            y_analytical = list(v_analytical[0])
            x_approx = list(range(len(v_approx[0])))
            y_approx = list(v_approx[0])
            # add the variable name when scrolling on the data
            var_name = k.split('.')[-1]
            new_series = InstanciatedSeries(
                x_analytical, y_analytical, var_name, 'lines', visible_line, line=color_code, text=[var_name] * len(y_approx))
            new_chart.add_series(new_series)
            new_series = InstanciatedSeries(
                x_approx, y_approx, var_name, 'dash_lines', visible_line, line=color_code, text=[var_name] * len(y_approx))
            new_chart.add_series(new_series)
        new_chart.annotation_upper_left = note
        instanciated_charts.append(new_chart)

        # Plot adjoint absolute grad error vs poles
        chart_name = f'Adjoint error (of {OBJECTIVE_LAGR}) vs years. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variables pole number (corresponding to years) [-]', 'Gradient error [with unit] ',
                                         chart_name=chart_name, y_min_zero=False)
        for k, v_approx in grad_approx_dict.items():
            v_analytical = grad_analytical_dict[k]
            x = list(range(len(v_analytical[0])))
            y = list(v_analytical[0] - v_approx[0])
            new_series = InstanciatedSeries(x, y, f"{k.split('.')[-1]}", 'lines', visible_line)
            new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # Plot adjoint relative grad error vs poles
        chart_name = f'Adjoint relative error (of {OBJECTIVE_LAGR}) vs years. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variables pole number (corresponding to years) [-]', 'Gradient relative error [%]',
                                         chart_name=chart_name, y_min_zero=False)
        for k, v_approx in grad_approx_dict.items():
            v_analytical = grad_analytical_dict[k]
            x = list(range(len(v_analytical[0])))
            y = list((v_analytical[0] - v_approx[0]) / v_approx[0] * 100.)
            new_series = InstanciatedSeries(x, y, f"{k.split('.')[-1]}", 'lines', visible_line)
            new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        var_list_with_ns_val = list(grad_approx_dict.keys())
        var_list = [var.split('.')[-1] for var in var_list_with_ns_val] #only keep var name without full namespace value in front
        x = list(range(len(var_list)))

        # Plot gradient norm vs each var
        chart_name = f'Gradient of {OBJECTIVE_LAGR} vs design variables. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variables index [-]', 'Gradient (L2 norm on years) [with unit]',
                                         chart_name=chart_name, y_min_zero=False)
        val_list_analytical = []
        val_list_approx = []
        for k in inputs:
            v_analytical = grad_analytical_dict[k]
            v_approx = grad_approx_dict[k]
            val_list_analytical += [np.linalg.norm(v_analytical[0])]
            val_list_approx += [np.linalg.norm(v_approx[0])]
        new_series = InstanciatedSeries(x, val_list_analytical, 'Adjoint', 'lines', visible_line, text=var_list)
        new_chart.add_series(new_series)
        new_series = InstanciatedSeries(x, val_list_approx, 'Finite Differences', 'dash_lines', visible_line, text=var_list)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # Plot gradient error  norm vs each var
        chart_name = f'Adjoint error (of {OBJECTIVE_LAGR}) vs design variables. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variables index [-]', 'Error of the gradient (L2 norm on years) [with unit]',
                                         chart_name=chart_name, y_min_zero=False)
        val_list = []
        for k, v_approx in grad_approx_dict.items():
            v_analytical = grad_analytical_dict[k]
            val_list += [np.linalg.norm(v_analytical[0] - v_approx[0])]
        new_series = InstanciatedSeries(x, val_list, '', 'lines+markers', visible_line, text=var_list)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # Plot gradient relative error of the norm vs each var
        chart_name = f'Relative error of the adjoint (of {OBJECTIVE_LAGR}) vs design variables. Finite diff step={finite_differences_step}'

        new_chart = TwoAxesInstanciatedChart('Design variable index [-]', 'Relative error of the Gradient (L2 norm on years) [%]',
                                         chart_name=chart_name, y_min_zero=False)
        val_list = []
        for k, v_approx in grad_approx_dict.items():
            v_analytical = grad_analytical_dict[k]
            val_list += [np.linalg.norm((v_analytical[0] - v_approx[0]) / v_approx[0] * 100.)]
        new_series = InstanciatedSeries(x, val_list, '', 'lines+markers', visible_line, text=var_list)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # reset the data manager to initial value to pass the l1s tests
        execution_engine.dm.set_values_from_dict(dm_data_dict_before)
        execution_engine.dm.create_reduced_dm()

        '''
        NB: on the GUI, to visualize the gradients graph once the computation has run, it is necessary first to
        close the study and reopen it
        '''

    return instanciated_charts
