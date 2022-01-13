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
from climateeconomics.core.design_variables_translation.witness_bspline_invest_distrib.design_var import Design_var
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import plotly.colors as plt_color

from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from energy_models.core.stream_type.carbon_models.carbon_capture import CarbonCapture
from energy_models.core.stream_type.carbon_models.carbon_storage import CarbonStorage

color_list = plt_color.qualitative.Plotly
color_list.extend(plt_color.qualitative.Alphabet)


class Design_Var_Discipline(SoSDiscipline):
    WRITE_XVECT = 'write_xvect'
    LOG_DVAR = 'log_designvar'

    DESC_IN = {
        'year_start': {'type': 'int', 'default': 2020, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'year_end': {'type': 'int', 'default': 2100, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'time_step': {'type': 'int', 'default': 1, 'visibility': 'Shared', 'unit': 'year', 'namespace': 'ns_witness'},
        'energy_list': {'type': 'string_list', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_energy_study', 'editable': False, 'structuring': True},
        'ccs_list': {'type': 'string_list', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_energy_study',
                     'possible_values': [CarbonCapture.name, CarbonStorage.name],
                     'default': [CarbonCapture.name, CarbonStorage.name], 'editable': False, 'structuring': True},
        'livestock_usage_factor_array': {'type': 'array', 'unit': '%', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'design_space': {'type': 'dataframe', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_optim'},
        WRITE_XVECT: {'type': 'bool', 'default': False, 'user_level': 3},
        LOG_DVAR: {'type': 'bool', 'default': True, 'user_level': 3}
    }

    DESC_OUT = {
        'invest_mix': {'type': 'dataframe', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_invest'},
        'livestock_usage_factor_df': {'type': 'dataframe', 'unit': '%', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'design_space_last_ite': {'type': 'dataframe', 'user_level': 3}
    }

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        dynamic_outputs = {}

        if 'energy_list' in self._data_in:
            energy_list = self.get_sosdisc_inputs('energy_list')
            if energy_list is not None:
                for energy in energy_list:
                    energy_wo_dot = energy.replace('.', '_')

                    dynamic_inputs[f'{energy}.technologies_list'] = {'type': 'string_list',
                                                                     'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_energy_mix', 'structuring': True}

                    if f'{energy}.technologies_list' in self._data_in:
                        technology_list = self.get_sosdisc_inputs(
                            f'{energy}.technologies_list')

                        if technology_list is not None:
                            for techno in technology_list:
                                techno_wo_dot = techno.replace('.', '_')
                                dynamic_inputs[f'{energy}.{techno}.{energy_wo_dot}_{techno_wo_dot}_array_mix'] = {
                                    'type': 'array', 'unit': '%', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_energy_mix'}
        if 'ccs_list' in self._data_in:
            ccs_list = self.get_sosdisc_inputs('ccs_list')
            if ccs_list is not None:
                for ccs_name in ccs_list:
                    ccs_name_wo_dot = ccs_name.replace('.', '_')
                    dynamic_inputs[f'{ccs_name}.technologies_list'] = {'type': 'string_list',
                                                                       'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ccs', 'structuring': True}

                    if f'{ccs_name}.technologies_list' in self._data_in:
                        technology_list = self.get_sosdisc_inputs(
                            f'{ccs_name}.technologies_list')

                        if technology_list is not None:
                            for techno in technology_list:
                                techno_wo_dot = techno.replace('.', '_')
                                dynamic_inputs[f'{ccs_name}.{techno}.{ccs_name_wo_dot}_{techno_wo_dot}_array_mix'] = {
                                    'type': 'array', 'unit': '%', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ccs'}

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        self.iter = 0

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()

        self.design = Design_var(inputs_dict)
        self.dict_last_ite = None

    def run(self):
        inputs_dict = self.get_sosdisc_inputs()

        self.design.configure(inputs_dict)
        outputs_dict = self.design.output_dict

        # retrieve the design space and update values with current iteration
        # values
        dspace_in = self.get_sosdisc_inputs('design_space')
        dspace_out = dspace_in.copy(deep=True)
        dict_diff = {}
        dict_current = {}
        for dvar in dspace_in.variable:
            for disc_var in self._data_in:
                if dvar in disc_var:
                    val = self.get_sosdisc_inputs(disc_var)
                    if isinstance(val, np.ndarray):
                        dict_current[dvar] = val

                        val = str(val.tolist())
                        # dict_diff[dvar]
                        # dict_diff[dvar]
                    if isinstance(val, list):
                        dict_current[dvar] = np.array(val)
                    dspace_out.loc[dspace_out.variable ==
                                   dvar, "value"] = str(val)

        # option to log difference between two iterations to track optimization
        if inputs_dict[self.LOG_DVAR]:
            if self.dict_last_ite is None:
                self.logger.info('x0' + str(dict_current))

                self.dict_last_ite = dict_current

            else:
                dict_diff = {
                    key: dict_current[key] - self.dict_last_ite[key] for key in dict_current}
                self.logger.info(
                    'difference between two iterations' + str(dict_diff))
        # update output dictionary with dspace
        outputs_dict.update({'design_space_last_ite': dspace_out})

        # dump design space into a csv
        if self.get_sosdisc_inputs(self.WRITE_XVECT):
            dspace_out.to_csv(f"dspace_ite_{self.iter}.csv", index=False)

        self.store_sos_outputs_values(self.design.output_dict)
        self.iter += 1

    def compute_sos_jacobian(self):

        inputs_dict = self.get_sosdisc_inputs()

        for energy in inputs_dict['energy_list']:
            energy_wo_dot = energy.replace('.', '_')
            for techno in inputs_dict[f'{energy}.technologies_list']:
                techno_wo_dot = techno.replace('.', '_')
                self.set_partial_derivative_for_other_types(
                    (f'invest_mix', f'{energy}.{techno}'), (
                        f'{energy}.{techno}.{energy_wo_dot}_{techno_wo_dot}_array_mix',),
                    self.design.bspline_dict[f'{energy}.{techno}.{energy_wo_dot}_{techno_wo_dot}_array_mix']['b_array'])

        for ccs in inputs_dict['ccs_list']:
            ccs_wo_dot = ccs.replace('.', '_')
            for techno in inputs_dict[f'{ccs}.technologies_list']:
                techno_wo_dot = techno.replace('.', '_')
                self.set_partial_derivative_for_other_types(
                    (f'invest_mix',  f'{ccs}.{techno}'), (
                        f'{ccs}.{techno}.{ccs_wo_dot}_{techno_wo_dot}_array_mix',),
                    self.design.bspline_dict[f'{ccs}.{techno}.{ccs_wo_dot}_{techno_wo_dot}_array_mix']['b_array'])

        self.set_partial_derivative_for_other_types(
            (f'livestock_usage_factor_df', 'percentage'), (f'livestock_usage_factor_array',),  self.design.bspline_dict['livestock_usage_factor_array']['b_array'])

    def get_chart_filter_list(self):

        chart_filters = []
        chart_list = ['Others', 'Array mixes']
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        initial_xvect_list = ['Standard', 'With initial xvect']
        chart_filters.append(ChartFilter(
            'Initial xvect', initial_xvect_list, ['Standard', ], 'initial_xvect'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        # For the outputs, making a graph for block fuel vs range and blocktime vs
        # range

        instanciated_charts = []
        charts = []
        initial_xvect_list = ['Standard', ]
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values
                if chart_filter.filter_key == 'initial_xvect':
                    initial_xvect_list = chart_filter.selected_values
        if 'With initial xvect' in initial_xvect_list:
            init_xvect = True
        else:
            init_xvect = False
        if 'Others' in charts:
            list_dv = ['livestock_usage_factor_array']
            for parameter in list_dv:
                new_chart = self.get_chart_BSpline(
                    parameter, init_xvect)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)

        if 'Array mixes' in charts:

            invest_mix = self.get_sosdisc_outputs('invest_mix')
            for column in invest_mix:
                if column != 'years':

                    new_chart = self.get_chart_BSpline_array_mix(
                        column, init_xvect)
                    if new_chart is not None:
                        instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_chart_BSpline(self, parameter, init_xvect=False):
        """
        Function to create post-proc for the design variables with display of the control points used to 
        calculate the B-Splines.
        The activation/deactivation of control points is accounted for by inserting the values from the design space
        dataframe into the ctrl_pts if need be (activated_elem==False) and at the appropriate index.
        Input: parameter (name), parameter values, design_space
        Output: InstantiatedPlotlyNativeChart
        """
        design_space = self.get_sosdisc_inputs('design_space')
        if parameter not in design_space['variable'].to_list():
            return None
        ctrl_pts = list(self.get_sosdisc_inputs(parameter))
        starting_pts = list(
            design_space[design_space['variable'] == parameter]['value'].values[0])
        for i, activation in enumerate(design_space.loc[design_space['variable']
                                                        == parameter, 'activated_elem'].to_list()[0]):
            if not activation and len(design_space.loc[design_space['variable'] == parameter, 'value'].to_list()[0]) > i:
                ctrl_pts.insert(i, design_space.loc[design_space['variable']
                                                    == parameter, 'value'].to_list()[0][i])
        eval_pts = None
        for key in self.get_sosdisc_outputs().keys():
            if key in parameter or parameter[:-6] in key:
                for column in self.get_sosdisc_outputs(key).columns:
                    if column not in ['years', ]:
                        eval_pts = self.get_sosdisc_outputs(key)[column].values
                        years = self.get_sosdisc_outputs(key)['years'].values
        if eval_pts is None:
            print('eval pts not found in sos_disc_outputs')
            return None
        else:
            chart_name = f'B-Spline for {parameter}'
            fig = go.Figure()
            if 'complex' in str(type(ctrl_pts[0])):
                ctrl_pts = [np.real(value) for value in ctrl_pts]
            if 'complex' in str(type(eval_pts[0])):
                eval_pts = [np.real(value) for value in eval_pts]
            if 'complex' in str(type(starting_pts[0])):
                starting_pts = [np.real(value) for value in starting_pts]
            x_ctrl_pts = np.linspace(
                years[0], years[-1], len(ctrl_pts))
            marker_dict = dict(size=150 / len(ctrl_pts), line=dict(
                width=150 / (3 * len(ctrl_pts)), color='DarkSlateGrey'))
            fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                     y=list(ctrl_pts), name='Poles',
                                     line=dict(color=color_list[0]),
                                     mode='markers',
                                     marker=marker_dict))
            fig.add_trace(go.Scatter(x=list(years), y=list(eval_pts), name='B-Spline',
                                     line=dict(color=color_list[0]),))
            if init_xvect:
                marker_dict['opacity'] = 0.5
                fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                         y=list(starting_pts), name='Initial Poles',
                                         mode='markers',
                                         line=dict(color=color_list[0]),
                                         marker=marker_dict))
            fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                              xaxis_title='years', yaxis_title=f'value of {parameter}')
            new_chart = InstantiatedPlotlyNativeChart(
                fig, chart_name=chart_name, default_title=True)
        return new_chart

    def get_chart_BSpline_array_mix(self, column_name, init_xvect=False):
        """
        Function to create post-proc for the design variables with display of the control points used to 
        calculate the B-Splines.
        The activation/deactivation of control points is accounted for by inserting the values from the design space
        dataframe into the ctrl_pts if need be (activated_elem==False) and at the appropriate index.
        Input: parameter (name), parameter values, design_space
        Output: InstantiatedPlotlyNativeChart
        """
        design_space = self.get_sosdisc_inputs('design_space')

        techno_array = column_name.replace('.', '_')
        design_space_name = f'{techno_array}_array_mix'
        if design_space_name not in design_space['variable'].to_list():
            return None
        inputs_dict = self.get_sosdisc_inputs()

        inputs_dict_keys = list(inputs_dict.keys())

        key_array_mix = [
            key for key in inputs_dict_keys if key.endswith(design_space_name)]
        if len(key_array_mix) == 1:
            key_array_mix = key_array_mix[0]
        else:
            return None
        ctrl_pts = inputs_dict[key_array_mix]
        starting_pts = list(
            design_space[design_space['variable'] == design_space_name]['value'].values[0])
        for i, activation in enumerate(design_space.loc[design_space['variable']
                                                        == design_space_name, 'activated_elem'].to_list()[0]):
            if not activation and len(design_space.loc[design_space['variable'] == design_space_name, 'value'].to_list()[0]) > i:
                ctrl_pts.insert(i, design_space.loc[design_space['variable']
                                                    == design_space_name, 'value'].to_list()[0][i])

        invest_mix = self.get_sosdisc_outputs('invest_mix')

        eval_pts = invest_mix[column_name].values
        years = invest_mix['years'].values

        chart_name = f'B-Spline for {design_space_name}'
        fig = go.Figure()
        if 'complex' in str(type(ctrl_pts[0])):
            ctrl_pts = [np.real(value) for value in ctrl_pts]
        if 'complex' in str(type(eval_pts[0])):
            eval_pts = [np.real(value) for value in eval_pts]
        if 'complex' in str(type(starting_pts[0])):
            starting_pts = [np.real(value) for value in starting_pts]
        x_ctrl_pts = np.linspace(
            years[0], years[-1], len(ctrl_pts))
        marker_dict = dict(size=150 / len(ctrl_pts), line=dict(
            width=150 / (3 * len(ctrl_pts)), color='DarkSlateGrey'))
        fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                 y=list(ctrl_pts), name='Poles',
                                 line=dict(color=color_list[0]),
                                 mode='markers',
                                 marker=marker_dict))
        fig.add_trace(go.Scatter(x=list(years), y=list(eval_pts), name='B-Spline',
                                 line=dict(color=color_list[0]),))
        if init_xvect:
            marker_dict['opacity'] = 0.5
            fig.add_trace(go.Scatter(x=list(x_ctrl_pts),
                                     y=list(starting_pts), name='Initial Poles',
                                     mode='markers',
                                     line=dict(color=color_list[0]),
                                     marker=marker_dict))
        fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                          xaxis_title='years', yaxis_title=f'value of {design_space_name}')
        new_chart = InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name, default_title=True)
        return new_chart
