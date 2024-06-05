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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline as MacroEconomics
import climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline as Population
from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from climateeconomics.core.tools.post_proc import get_scenario_value
from climateeconomics.glossarycore import GlossaryCore
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart
from gemseo.utils.derivatives_approx import DisciplineJacApprox
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart

'''
Post-processing designe to compare the analytical vs the approximated gradient of the objective function wrt the design 
variables
'''
def post_processing_filters(execution_engine, namespace):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    '''
    chart_filters = []

    chart_list = ['Objective Lagrangian']
    # First filter to deal with the view : program or actor
    chart_filters.append(ChartFilter(
        'Charts_grad', chart_list, chart_list, 'Charts_grad')) # name 'Charts' is already used by ssp_comparison post-proc

    return chart_filters


def post_processings(execution_engine, scenario_name, chart_filters=None): #scenario_name, chart_filters=None):
    '''
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    '''
    OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR

    instanciated_charts = []
    chart_list = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'Charts_grad':
                chart_list = chart_filter.selected_values

    if 'Objective Lagrangian' in chart_list:
        n_profiles = get_scenario_value(execution_engine, 'n_profiles', scenario_name)
        mdo_disc = execution_engine.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        inputs = [f'{scenario_name}.InvestmentsProfileBuilderDisc.coeff_{i}' for i in range(n_profiles)]
        outputs = [f"{scenario_name.split('.')[0]}.{OBJECTIVE_LAGR}"]
        # compute analytical gradient first at the converged point of the MDA
        mdo_disc.add_differentiated_inputs(inputs)
        mdo_disc.add_differentiated_outputs(outputs)
        grad_analytical = mdo_disc.linearize(force_no_exec=True) #can add data of the converged point (see sos_mdo_discipline.py)

        # computed approximated gradient in 2nd step so that the analytical gradient is not computed in X0 + h
        # warm start must have been removed so that mda and convergence criteria are the same for analytical and approximated gradients
        approx = DisciplineJacApprox(mdo_disc, approx_method=DisciplineJacApprox.FINITE_DIFFERENCES)
        grad_approx = approx.compute_approx_jac(outputs, inputs)

        coeff_i = [i for i in range(n_profiles)]
        output = outputs[0]
        # data type: grad_approx = {output:{coeff_i:array([[X]])}}
        grad_analytical_list = [grad_analytical[output][inputs[i]][0][0] for i in range(n_profiles)]
        grad_approx_list = [grad_approx[output][inputs[i]][0][0] for i in range(n_profiles)]

        # plot the gradients in abs value
        chart_name = f'df/dx({output})'

        new_chart = TwoAxesInstanciatedChart('Design variables (coeff_i)', 'Gradient',
                                         chart_name=chart_name, y_min_zero=False)

        visible_line = True
        new_series = InstanciatedSeries(
            coeff_i, grad_analytical_list, 'Adjoint Gradient', 'lines', visible_line)
        new_chart.add_series(new_series)
        new_series = InstanciatedSeries(
            coeff_i, grad_approx_list, 'Finite Differences', 'lines', visible_line)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

        # plot the relative error
        chart_name = f'Relative error of df/dx({output})'

        new_chart = TwoAxesInstanciatedChart('Design variables (coeff_i)', 'Gradient relative error [%]',
                                         chart_name=chart_name, y_min_zero=False)

        visible_line = True
        # unless the optimum is reached, a non-zero gradient is expected
        error = (np.array(grad_analytical_list) - np.array(grad_approx_list))/np.array(grad_approx_list) * 100.
        new_series = InstanciatedSeries(
            coeff_i, list(error), '', 'lines', visible_line)
        new_chart.add_series(new_series)
        instanciated_charts.append(new_chart)

    return instanciated_charts


