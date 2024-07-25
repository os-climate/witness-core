'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.carbon_emissions_model import CarbonEmissions
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class CarbonemissionsDiscipline(ClimateEcoDiscipline):
    "carbonemissions discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'Carbon Emission WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-smog fa-fw',
        'version': '',
    }

    _maturity = 'Research'
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'init_gr_sigma': {'type': 'float', 'default': -0.0152, 'user_level': 2, 'unit': '-'},
        'decline_rate_decarbo': {'type': 'float', 'default': -0.001, 'user_level': 2, 'unit': '-'},
        'init_indus_emissions': {'type': 'float', 'default': 34, 'unit': 'GtCO2 per year', 'user_level': 2},
        GlossaryCore.InitialGrossOutput['var_name']: GlossaryCore.InitialGrossOutput,
        'init_cum_indus_emissions': {'type': 'float', 'default': 577.31, 'unit': 'GtCO2', 'user_level': 2},
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        'energy_emis_share': {'type': 'float', 'default': 0.9, 'user_level': 2, 'unit': '-'},
        'land_emis_share': {'type': 'float', 'default': 0.0636, 'user_level': 2, 'unit': '-'},
        'alpha': ClimateEcoDiscipline.ALPHA_DESC_IN,
        'beta': {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'unit': '-',
                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS},
        'min_co2_objective': {'type': 'float', 'default': -1000., 'unit': 'GtCO2', 'user_level': 2},
        'total_emissions_ref': {'type': 'float', 'default': 39.6, 'unit': 'GtCO2', 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'co2_emissions_ccus_Gt': {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_CCS,
                                  'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                           'carbon_storage Limited by capture (Gt)': ('float', None, False),}},
        'CO2_emissions_by_use_sources': {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_CCS,
                                         'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                  'carbon_capture from energy mix (Gt)': ('float', None, False),
                                                                  'Total CO2 by use (Gt)': ('float', None, False),
                                                                  'Total CO2 from Flue Gas (Gt)': ('float', None, False),
                                                                  'CO2 from energy mix (Gt)': ('float', None, False)}},
        'CO2_emissions_by_use_sinks':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_CCS,
                                        'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                 'CO2 removed by energy mix (Gt)': ('float', None, False),}},
        'co2_emissions_needed_by_energy_mix': {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_energy',
                                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                        'carbon_capture needed by energy mix (Gt)': ('float', None, False),}},

        # Ref in 2020 is around 34 Gt, the objective is normalized with this
        # reference
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2): {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                               'dataframe_descriptor': {
                                   GlossaryCore.Years: ('float', [2000, 2200], False),
                                   'emitted_CO2_evol': ('float', [-1e5, 1e7], False),
                                   'emitted_CO2_evol_cumulative': ('float', None, False),}},
        GlossaryCore.CheckRangeBeforeRunBoolName : GlossaryCore.CheckRangeBeforeRunBool
    }
    DESC_OUT = {
        GlossaryCore.CO2EmissionsDetailDfValue: GlossaryCore.CO2EmissionsDetailDf,
        GlossaryCore.CO2EmissionsDfValue: GlossaryCore.CO2EmissionsDf,
        'CO2_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': ''},
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
    }

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.emissions_model = CarbonEmissions(in_dict)

    def run(self):
        # Get inputs
        in_dict = self.get_sosdisc_inputs()
        

        # Compute de emissions_model
        CO2_emissions_df, CO2_objective = self.emissions_model.compute(in_dict)
        self.emissions_model.compute_total_CO2_emissions()
        # Store output data
        dict_values = {GlossaryCore.CO2EmissionsDetailDfValue: CO2_emissions_df,
                       GlossaryCore.CO2EmissionsDfValue: CO2_emissions_df[GlossaryCore.CO2EmissionsDf['dataframe_descriptor'].keys()],
                       'CO2_objective': CO2_objective,
                       GlossaryCore.CO2EmissionsGtValue: self.emissions_model.co2_emissions[[GlossaryCore.Years, GlossaryCore.TotalCO2Emissions]]}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        CO2_emissions_df
          - 'indus_emissions':
                - economics_df, GlossaryCore.GrossOutput
                - co2_emissions_Gt, GlossaryCore.TotalCO2Emissions
          - 'cum_indus_emissions'
                - economics_df, GlossaryCore.GrossOutput
                - co2_emissions_Gt, GlossaryCore.TotalCO2Emissions
          - 'total_emissions',
                - CO2_emissions_df, land_emissions
                - economics_df, GlossaryCore.GrossOutput
                - co2_emissions_Gt, Total CO2 emissions
          - 'cum_total_emissions'
                - CO2_emissions_df, land_emissions
                - economics_df, GlossaryCore.GrossOutput
                - co2_emissions_Gt, Total CO2 emissions
          - 'CO2_objective'
                - total_emissions:
                    - CO2_emissions_df, land_emissions
                    - economics_df, GlossaryCore.GrossOutput
                    - co2_emissions_Gt, Total CO2 emissions
        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(
            inputs_dict[GlossaryCore.YearStart], inputs_dict[GlossaryCore.YearEnd] + 1, inputs_dict[GlossaryCore.TimeStep])

        d_indus_emissions_d_gross_output, d_cum_indus_emissions_d_gross_output, d_cum_indus_emissions_d_total_CO2_emitted = self.emissions_model.compute_d_indus_emissions()
        d_CO2_obj_d_total_emission = self.emissions_model.compute_d_CO2_objective()
        dobjective_exp_min = self.emissions_model.compute_dobjective_with_exp_min()
        d_total_emissions_C02_emitted_land = self.emissions_model.compute_d_land_emissions()
        columns_sources = self.get_sosdisc_inputs(
            'CO2_emissions_by_use_sources').columns
        # fill jacobians
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsDfValue, 'total_emissions'),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            d_indus_emissions_d_gross_output)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsDfValue, 'cum_total_emissions'),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            d_cum_indus_emissions_d_gross_output)

        for column_sources in columns_sources:
            if column_sources != GlossaryCore.Years:
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsDfValue, 'total_emissions'),
                    ('CO2_emissions_by_use_sources', column_sources),
                    np.identity(len(years)))
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
                    ('CO2_emissions_by_use_sources', column_sources),  np.identity(len(years)))

                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsDfValue, 'cum_total_emissions'),
                    ('CO2_emissions_by_use_sources', column_sources),
                    d_cum_indus_emissions_d_total_CO2_emitted)

                self.set_partial_derivative_for_other_types(
                    ('CO2_objective',),
                    ('CO2_emissions_by_use_sources', column_sources),
                    d_CO2_obj_d_total_emission * dobjective_exp_min)
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
                    ('CO2_emissions_by_use_sources', column_sources),
                    np.identity(len(years)))

        sinks_dict = {'CO2_emissions_by_use_sinks': "CO2 removed by energy mix (Gt)", 'co2_emissions_needed_by_energy_mix':
                      'carbon_capture needed by energy mix (Gt)', 'co2_emissions_ccus_Gt': 'carbon_storage Limited by capture (Gt)'}

        for df_name, col_name in sinks_dict.items():
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2EmissionsDfValue, 'total_emissions'),
                (df_name, col_name),
                - np.identity(len(years)))
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
                (df_name, col_name),
                - np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2EmissionsDfValue, 'cum_total_emissions'),
                (df_name, col_name),
                - d_cum_indus_emissions_d_total_CO2_emitted)

            self.set_partial_derivative_for_other_types(
                ('CO2_objective',), (df_name, col_name),
                - d_CO2_obj_d_total_emission * dobjective_exp_min)

        self.set_partial_derivative_for_other_types(
            ('CO2_objective',),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.GrossOutput),
            dobjective_exp_min * d_CO2_obj_d_total_emission.dot(d_indus_emissions_d_gross_output))


        #land emissions
        CO2_land_emissions = inputs_dict[GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)]
        for column in CO2_land_emissions.columns:
            if column != GlossaryCore.Years:
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsDfValue, 'total_emissions'),
                    (GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2), column),
                    np.identity(len(years)))

                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.CO2EmissionsDfValue, 'cum_total_emissions'),
                    (GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2), column),
                    d_total_emissions_C02_emitted_land)

                self.set_partial_derivative_for_other_types(
                    ('CO2_objective',),
                    (GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2), column),
                    dobjective_exp_min * d_CO2_obj_d_total_emission)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Carbon emissions',
                      'Sources and sinks', 'Cumulated CO2 emissions']
        #chart_list = ['sectoral energy carbon emissions cumulated']
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

        if 'Carbon emissions' in chart_list:
            new_chart = self.get_chart_co2_emissions()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Sources and sinks' in chart_list:
            new_chart = self.get_chart_sources_and_sinks()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

            new_chart = self.get_chart_sources_and_sinks(detailed=True)
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Cumulated CO2 emissions' in chart_list:
            new_chart = self.get_chart_cumulated_co2_emissions()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_chart_co2_emissions(self):

        to_plot = ['total_emissions', 'land_emissions', 'indus_emissions']

        CO2_emissions_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.CO2EmissionsDetailDfValue))

        total_emission = CO2_emissions_df['total_emissions']
        land_emissions = CO2_emissions_df['land_emissions']
        indus_emissions = CO2_emissions_df['indus_emissions']

        years = list(CO2_emissions_df.index)

        year_start = years[0]
        year_end = years[len(years) - 1]

        min_value_e, max_value_e = self.get_greataxisrange(total_emission)
        min_value_l, max_value_l = self.get_greataxisrange(land_emissions)
        min_value_i, max_value_i = self.get_greataxisrange(indus_emissions)
        min_value = min(min_value_e, min_value_l, min_value_i)
        max_value = max(max_value_e, max_value_l, max_value_i)

        chart_name = 'Total carbon emissions'

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 emissions [GtCO2]',
                                             [year_start - 5, year_end + 5],
                                             [min_value, max_value],
                                             chart_name)

        for key in to_plot:
            visible_line = True

            c_emission = list(CO2_emissions_df[key])

            new_series = InstanciatedSeries(
                years, c_emission, key, 'lines', visible_line)

            new_chart.series.append(new_series)

        return new_chart

    def get_chart_cumulated_co2_emissions(self):

        CO2_emissions_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.CO2EmissionsDetailDfValue))

        total_emission_cum = CO2_emissions_df['total_emissions'].cumsum()

        years = list(CO2_emissions_df.index)

        chart_name = f'Cumulated carbon emissions since {years[0]}'

        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 emissions (GtCO2)',
                                             chart_name=chart_name)

        new_series = InstanciatedSeries(
            years, total_emission_cum.values.tolist(), 'lines')

        new_chart.series.append(new_series)

        return new_chart

    def get_chart_sources_and_sinks(self, detailed=False):

        CO2_emissions_df = deepcopy(
            self.get_sosdisc_outputs(GlossaryCore.CO2EmissionsDetailDfValue))
        years = list(CO2_emissions_df.index)

        CO2_emissions_breakdown = pd.DataFrame({GlossaryCore.Years: years})
        # Energy emissions
        #----------------
        cols_to_sum = []
        # Get all the sources and put them as columns in df
        CO2_emissions_by_use_sources = self.get_sosdisc_inputs(
            'CO2_emissions_by_use_sources')
        for col in CO2_emissions_by_use_sources.columns:
            if col != GlossaryCore.Years:
                CO2_emissions_breakdown[col] = CO2_emissions_by_use_sources[col].values
                cols_to_sum += [col, ]
        # Sum all the sources columns
        CO2_emissions_breakdown['category energy sources (Gt)'] = CO2_emissions_breakdown[cols_to_sum].sum(
            axis=1)

        cols_to_sum = []
        # Get all the sinks and put them as columns in df
        CO2_emissions_by_use_sinks = self.get_sosdisc_inputs(
            'CO2_emissions_by_use_sinks')
        for col in CO2_emissions_by_use_sinks.columns:
            if col != GlossaryCore.Years:
                CO2_emissions_breakdown[col] = - \
                    CO2_emissions_by_use_sinks[col].values
                cols_to_sum += [col, ]
        co2_emissions_needed_by_energy_mix = self.get_sosdisc_inputs(
            'co2_emissions_needed_by_energy_mix')
        for col in co2_emissions_needed_by_energy_mix.columns:
            if col != GlossaryCore.Years:
                CO2_emissions_breakdown[col] = - \
                    co2_emissions_needed_by_energy_mix[col].values
                cols_to_sum += [col, ]
        co2_emissions_ccus_Gt = self.get_sosdisc_inputs(
            'co2_emissions_ccus_Gt')
        for col in co2_emissions_ccus_Gt.columns:
            if col != GlossaryCore.Years:
                CO2_emissions_breakdown[col] = - \
                    co2_emissions_ccus_Gt[col].values
                cols_to_sum += [col, ]
        # Sum all the sources columns
        CO2_emissions_breakdown['category energy sinks (Gt)'] = CO2_emissions_breakdown[cols_to_sum].sum(
            axis=1)

        # Industrial emissions
        #---------------------
        # To be replaced by sources and sinks from pyworld3
        cols_to_sum = []
        # Get all the sources and put them as columns in df
        sigma = CO2_emissions_df['sigma'].values
        gross_output_ter = self.get_sosdisc_inputs(
            GlossaryCore.EconomicsDfValue)[GlossaryCore.GrossOutput].values
        energy_emis_share = self.get_sosdisc_inputs('energy_emis_share')
        share_land_emis = self.get_sosdisc_inputs('land_emis_share')
        indus_emissions = sigma * gross_output_ter * \
            (1 - energy_emis_share - share_land_emis)
        CO2_emissions_breakdown['industrial emissions (Gt)'] = indus_emissions
        # Sum all the sources columns
        CO2_emissions_breakdown['category industrial sources (Gt)'] = indus_emissions

        # Land use emissions
        #-----------------
        # To be replaced by sources and sinks from models (Forest,
        # Agriculture,...)
        cols_to_sum = []
        # Get all the sources and put them as columns in df
        CO2_emissions_breakdown['land_emissions (Gt)'] = CO2_emissions_df['land_emissions'].values
        # Sum all the sources columns
        CO2_emissions_breakdown['category land_use sources (Gt)'] = CO2_emissions_df[
            'land_emissions'].values

        chart_name = 'CO2 emissions breakdown'
        if detailed:
            chart_name = 'CO2 emissions breakdown detailed'
        new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 emissions [Gt]',
                                             chart_name=chart_name, stacked_bar=True)

        if detailed:
            for col in CO2_emissions_breakdown.columns:
                if 'category' not in col and col != GlossaryCore.Years:
                    legend_title = f'{col}'.replace(
                        " (Gt)", "")
                    serie = InstanciatedSeries(
                        CO2_emissions_breakdown[GlossaryCore.Years].values.tolist(),
                        CO2_emissions_breakdown[col].values.tolist(), legend_title, 'bar')
                    new_chart.series.append(serie)
        else:
            for col in CO2_emissions_breakdown.columns:
                if 'category' in col and col != GlossaryCore.Years:
                    legend_title = f'{col}'.replace(
                        " (Gt)", "")
                    serie = InstanciatedSeries(
                        CO2_emissions_breakdown[GlossaryCore.Years].values.tolist(),
                        CO2_emissions_breakdown[col].values.tolist(), legend_title, 'bar')
                    new_chart.series.append(serie)

        return new_chart
