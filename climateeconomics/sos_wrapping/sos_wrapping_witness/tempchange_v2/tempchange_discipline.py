'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2024/03/05 Copyright 2023 Capgemini

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

from copy import deepcopy

import numpy as np
import sostrades_core.tools.post_processing.post_processing_tools as ppt
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.tempchange_model_v2 import TempChange
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class TempChangeDiscipline(ClimateEcoDiscipline):
    "     Temperature evolution"

    # ontology information
    _ontology_data = {
        'label': 'Temperature Change WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-thermometer-three-quarters fa-fw',
        'version': '',
    }
    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'init_temp_ocean': {'type': 'float', 'default': 0.02794825, 'user_level': 2, 'unit': '°C'},
        'init_temp_atmo': {'type': 'float', 'default': DatabaseWitnessCore.TemperatureAnomalyPreIndustrialYearStart.value, 'user_level': 2, 'unit': '°C'},
        'eq_temp_impact': {'type': 'float', 'unit': '-', 'default': 3.1, 'user_level': 3},
        'temperature_model': {'type': 'string', 'default': 'FUND', 'possible_values': ['DICE', 'FUND', 'FAIR'],
                              'structuring': True},
        'climate_upper': {'type': 'float', 'default': 0.1005, 'user_level': 3, 'unit': '-'},
        'transfer_upper': {'type': 'float', 'default': 0.088, 'user_level': 3, 'unit': '-'},
        'transfer_lower': {'type': 'float', 'default': 0.025, 'user_level': 3, 'unit': '-'},
        'forcing_eq_co2': {'type': 'float', 'default': 3.74, 'user_level': 3, 'unit': '-'},
        'pre_indus_co2_concentration_ppm': {'type': 'float', 'default': DatabaseWitnessCore.CO2PreIndustrialConcentration.value, 'unit': 'ppm', 'user_level': 3},
        'lo_tocean': {'type': 'float', 'default': -1.0, 'user_level': 3, 'unit': '°C'},
        'up_tatmo': {'type': 'float', 'default': 12.0, 'user_level': 3, 'unit': '°C'},
        'up_tocean': {'type': 'float', 'default': 20.0, 'user_level': 3, 'unit': '°C'},
        GlossaryCore.GHGCycleDfValue: GlossaryCore.GHGCycleDf,
        'alpha': ClimateEcoDiscipline.ALPHA_DESC_IN,
        'beta': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-',
                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS},
        'temperature_obj_option': {'type': 'string',
                                   'possible_values': [TempChange.LAST_TEMPERATURE_OBJECTIVE,
                                                       TempChange.INTEGRAL_OBJECTIVE],
                                   'default': TempChange.INTEGRAL_OBJECTIVE,
                                   'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'temperature_change_ref': {'type': 'float', 'default': 0.2, 'unit': '°C',
                                   'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                   'namespace': GlossaryCore.NS_REFERENCE, 'user_level': 2},

        'scale_factor_atmo_conc': {'type': 'float', 'default': 1e-2, 'unit': '-', 'user_level': 2,
                                   'visibility': 'Shared',
                                   'namespace': GlossaryCore.NS_WITNESS},
        'temperature_end_constraint_limit': {'type': 'float', 'default': 1.5, 'unit': '°C', 'user_level': 2},
        'temperature_end_constraint_ref': {'type': 'float', 'default': 3., 'unit': '°C', 'user_level': 2},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
        GlossaryCore.TemperatureDetailedDfValue: GlossaryCore.TemperatureDetailedDf,
        'forcing_detail_df': {'type': 'dataframe', 'unit': 'W.m-2'},
        'temperature_constraint': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS}}

    _maturity = 'Research'

    def setup_sos_disciplines(self):
        dynamic_inputs = {}

        if 'temperature_model' in self.get_data_in():
            temperature_model = self.get_sosdisc_inputs('temperature_model')
            if temperature_model == 'DICE':

                dynamic_inputs['forcing_model'] = {'type': 'string',
                                                   'default': 'DICE',
                                                   'possible_values': ['DICE', 'Etminan', 'Meinshausen'],
                                                   }
                dynamic_inputs['init_forcing_nonco'] = {
                    'type': 'float', 'default': 0.83, 'unit': 'W.m-2', 'user_level': 2}
                dynamic_inputs['hundred_forcing_nonco'] = {
                    'type': 'float', 'default': 1.1422, 'unit': 'W.m-2', 'user_level': 2}

                # test for other forcing models
                dynamic_inputs['pre_indus_ch4_concentration_ppm'] = {
                    'type': 'float', 'default': 722., 'unit': 'ppm', 'user_level': 2}
                dynamic_inputs['pre_indus_n2o_concentration_ppm'] = {
                    'type': 'float', 'default': 273., 'unit': 'ppm', 'user_level': 2}

            elif temperature_model == 'FUND':

                dynamic_inputs['forcing_model'] = {'type': 'string',
                                                   'default': 'Meinshausen',
                                                   'possible_values': ['Myhre', 'Etminan', 'Meinshausen'],
                                                   }
                dynamic_inputs['pre_indus_ch4_concentration_ppm'] = {
                    'type': 'float', 'default': 790., 'unit': 'ppm', 'user_level': 2}
                dynamic_inputs['pre_indus_n2o_concentration_ppm'] = {
                    'type': 'float', 'default': 285., 'unit': 'ppm', 'user_level': 2}

            elif temperature_model == 'FAIR':

                dynamic_inputs['forcing_model'] = {'type': 'string',
                                                   'default': 'Meinshausen',
                                                   'possible_values': ['Myhre', 'Etminan', 'Meinshausen'],
                                                   }
                dynamic_inputs['pre_indus_ch4_concentration_ppm'] = {
                    'type': 'float', 'default': 722., 'unit': 'ppm', 'user_level': 2}
                dynamic_inputs['pre_indus_n2o_concentration_ppm'] = {
                    'type': 'float', 'default': 273., 'unit': 'ppm', 'user_level': 2}
        # var_names = ['forcing_model','init_forcing_nonco','hundred_forcing_nonco','pre_indus_ch4_concentration_ppm','pre_indus_n2o_concentration_ppm']
        # for var_name in var_names:
        #     if var_name in self.get_data_in():
        #         self.clean_variables([var_name], self.IO_TYPE_IN)
        self.add_inputs(dynamic_inputs)

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.model = TempChange(in_dict)

    def run(self):
        ''' pyworld3 execution '''
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        # todo: for sensitivity, generalise ?
        self.model.init_temp_atmo = in_dict['init_temp_atmo']
        

        # pyworld3 execution
        temperature_df = self.model.compute(in_dict)

        # store output data
        out_dict = {GlossaryCore.TemperatureDetailedDfValue: temperature_df,
                    GlossaryCore.TemperatureDfValue: temperature_df[GlossaryCore.TemperatureDf['dataframe_descriptor'].keys()],
                    'forcing_detail_df': self.model.forcing_df,
                    'temperature_constraint': self.model.temperature_end_constraint}
        
        self.store_sos_outputs_values(out_dict)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable
        """
        temperature_model = self.get_sosdisc_inputs('temperature_model')
        forcing_model = self.get_sosdisc_inputs('forcing_model')
        year_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
        year_end = self.get_sosdisc_inputs(GlossaryCore.YearEnd)
        temperature_constraint_ref = self.get_sosdisc_inputs('temperature_end_constraint_ref')
        identity = np.identity(year_end - year_start + 1)

        # forcing_detail
        self.model.compute_d_forcing()
        d_forcing_datmo_conc = self.model.d_forcing_datmo_conc_dict

        if forcing_model == 'DICE':
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CO2 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                identity * d_forcing_datmo_conc['CO2 forcing'], )

        elif forcing_model == 'Myhre':
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CO2 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                identity * d_forcing_datmo_conc['CO2 forcing'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CH4 and N2O forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration),
                identity * d_forcing_datmo_conc['CH4 forcing'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CH4 and N2O forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
                identity * d_forcing_datmo_conc['N2O forcing'], )

        elif forcing_model == 'Etminan' or forcing_model == 'Meinshausen':
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CO2 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                identity * d_forcing_datmo_conc['CO2 forcing CO2 ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CO2 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
                identity * d_forcing_datmo_conc['CO2 forcing N2O ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CH4 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration),
                identity * d_forcing_datmo_conc['CH4 forcing CH4 ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'CH4 forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
                identity * d_forcing_datmo_conc['CH4 forcing N2O ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'N2O forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                identity * d_forcing_datmo_conc['N2O forcing CO2 ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'N2O forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration),
                identity * d_forcing_datmo_conc['N2O forcing CH4 ppm'], )
            self.set_partial_derivative_for_other_types(
                ('forcing_detail_df', 'N2O forcing'), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
                identity * d_forcing_datmo_conc['N2O forcing N2O ppm'], )

        if temperature_model == 'DICE':
            d_tempatmo_d_atmoconc, _ = self.model.compute_d_temp_atmo()

            # temperature_df
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration), d_tempatmo_d_atmoconc, )

            # temperature_constraint
            d_tempatmo_d_atmoconc, _ = self.model.compute_d_temp_atmo()
            self.set_partial_derivative_for_other_types(
                ('temperature_constraint',), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                -d_tempatmo_d_atmoconc[-1] / temperature_constraint_ref, )

        elif temperature_model == 'FUND':

            # temperature_df
            d_temp_d_forcing_fund = self.model.compute_d_temp_d_forcing_fund()

            if forcing_model == 'Myhre':
                d_temp_d_co2_ppm = np.matmul(d_temp_d_forcing_fund, identity * d_forcing_datmo_conc['CO2 forcing'])
                d_temp_d_ch4_ppm = np.matmul(d_temp_d_forcing_fund, identity * d_forcing_datmo_conc['CH4 forcing'])
                d_temp_d_n2o_ppm = np.matmul(d_temp_d_forcing_fund, identity * d_forcing_datmo_conc['N2O forcing'])

            elif forcing_model == 'Etminan' or forcing_model == 'Meinshausen':

                d_temp_d_co2_ppm = np.matmul(d_temp_d_forcing_fund, identity * (
                            d_forcing_datmo_conc['CO2 forcing CO2 ppm'] + d_forcing_datmo_conc['N2O forcing CO2 ppm']))
                d_temp_d_ch4_ppm = np.matmul(d_temp_d_forcing_fund, identity * (
                            d_forcing_datmo_conc['CH4 forcing CH4 ppm'] + d_forcing_datmo_conc['N2O forcing CH4 ppm']))
                d_temp_d_n2o_ppm = np.matmul(d_temp_d_forcing_fund, identity * (
                            d_forcing_datmo_conc['CO2 forcing N2O ppm'] + d_forcing_datmo_conc['CH4 forcing N2O ppm'] +
                            d_forcing_datmo_conc['N2O forcing N2O ppm']))

            else:

                raise Exception("forcing model not in available models")

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration), d_temp_d_co2_ppm, )
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration), d_temp_d_ch4_ppm, )
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration), d_temp_d_n2o_ppm, )

            # temperature_constraint
            self.set_partial_derivative_for_other_types(
                ('temperature_constraint',), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                -d_temp_d_co2_ppm[-1] / temperature_constraint_ref, )
            self.set_partial_derivative_for_other_types(
                ('temperature_constraint',), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration),
                -d_temp_d_ch4_ppm[-1] / temperature_constraint_ref, )
            self.set_partial_derivative_for_other_types(
                ('temperature_constraint',), (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
                -d_temp_d_n2o_ppm[-1] / temperature_constraint_ref, )

        elif temperature_model == 'FAIR':
            # temperature df

            # temperature_constraint
            d_tempatmo_d_atmoconc, _ = self.model.compute_d_temp_atmo()
            d_tempatmoobj_d_temp_atmo = self.model.compute_d_temp_atmo_objective()
            self.set_partial_derivative_for_other_types(
                ('temperature_constraint',), (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
                -d_tempatmo_d_atmoconc[-1] / temperature_constraint_ref, )

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Temperature evolution', 'Radiative forcing']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'Temperature evolution' in chart_list:

            model = self.get_sosdisc_inputs('temperature_model')
            temperature_df = deepcopy(
                self.get_sosdisc_outputs(GlossaryCore.TemperatureDetailedDfValue))

            instanciated_charts = temperature_evolution(model, temperature_df, instanciated_charts)

        if 'Radiative forcing' in chart_list:

            forcing_df = self.get_sosdisc_outputs('forcing_detail_df')

            instanciated_charts = radiative_forcing(forcing_df, instanciated_charts)

        return instanciated_charts

def temperature_evolution(model, temperature_df, instanciated_charts):
    if model == 'DICE':
        to_plot = [GlossaryCore.TempAtmo, GlossaryCore.TempOcean]
        legend = {GlossaryCore.TempAtmo: 'Atmosphere',
                      GlossaryCore.TempOcean: 'Ocean'}

    elif model == 'FUND':
        to_plot = [GlossaryCore.TempAtmo]
        legend = {GlossaryCore.TempAtmo: 'Atmosphere'}

    elif model == 'FAIR':
        raise NotImplementedError('Model not implemented yet')

    else:
        raise Exception("forcing model not in available models")

    years = list(temperature_df.index)

    year_start = years[0]
    year_end = years[len(years) - 1]

    max_values = {}
    min_values = {}
    for key in to_plot:
        min_values[key], max_values[key] = ppt.get_greataxisrange(
            temperature_df[to_plot])

    min_value = min(min_values.values())
    max_value = max(max_values.values()) + DatabaseWitnessCore.ENSOTemperatureAnomaly.value

    chart_name = 'Temperature anomaly since pre-industrial era (1850-1900)'

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                             '°C',
                                             [year_start - 5, year_end + 5], [
                                                 min_value, max_value],
                                             chart_name)

    for key in to_plot:
        visible_line = True

        ordonate_data = list(temperature_df[key])

        new_series = InstanciatedSeries(
            years, ordonate_data, legend[key], 'lines', visible_line)

        new_chart.series.append(new_series)

    new_chart = new_chart.to_plotly()

    el_nino_max_temp = temperature_df[GlossaryCore.TempAtmo].values + DatabaseWitnessCore.ENSOTemperatureAnomaly.value

    import plotly.graph_objects as go
    new_chart.add_trace(go.Scatter(
        x=years,
        y=list(el_nino_max_temp),
        fill='tonexty',  # fill area between trace0 and trace1
        mode='lines',
        fillcolor='rgba(200,200,200,0.25)',
        line={'dash': 'dash', 'color': "rgba(200,200,200,0.3)"},
        opacity=0.2,
        name='Natural variations (El Niño/La Niña)', ))

    la_nina_min_temp = temperature_df[GlossaryCore.TempAtmo].values - DatabaseWitnessCore.ENSOTemperatureAnomaly.value
    new_chart.add_trace(go.Scatter(
        x=years,
        y=list(la_nina_min_temp),
        fill='tonexty',  # fill area between trace0 and trace1
        mode='lines',
        fillcolor='rgba(200,200,200,0.25)',
        line={'dash': 'dash', 'color': "rgba(200,200,200,0.3)"},
        opacity=0.2,
        showlegend=False))

    new_chart.update_layout(legend_traceorder="normal")

    new_chart = InstantiatedPlotlyNativeChart(fig=new_chart, chart_name=chart_name)
    instanciated_charts.append(new_chart)

    # Seal level chart for FUND pyworld3
    if model == 'FUND':
        chart_name = 'Sea level'
        min_value, max_value = ppt.get_greataxisrange(temperature_df['sea_level'])
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 'Seal level evolution',
                                                 [year_start - 5, year_end + 5], [min_value, max_value],
                                                 chart_name)
        visible_line = True
        ordonate_data = list(temperature_df['sea_level'])
        new_series = InstanciatedSeries(
                years, ordonate_data, 'Seal level evolution [m]', 'lines', visible_line)
        new_chart.series.append(new_series)

        instanciated_charts.append(new_chart)

    return instanciated_charts

def radiative_forcing(forcing_df, instanciated_charts):
    years = forcing_df[GlossaryCore.Years].values.tolist()

    chart_name = 'Gas Radiative forcing'

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Radiative forcing (W.m-2)',
                                             chart_name=chart_name)

    for forcing in forcing_df.columns:
        if forcing != GlossaryCore.Years:
            new_series = InstanciatedSeries(
                years, forcing_df[forcing].values.tolist(), forcing, 'lines')

            new_chart.series.append(new_series)

    instanciated_charts.append(new_chart)

    return instanciated_charts

