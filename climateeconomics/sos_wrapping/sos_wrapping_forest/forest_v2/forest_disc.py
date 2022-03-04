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
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
from climateeconomics.core.core_forest.forest_v2 import Forest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


class ForestDiscipline(ClimateEcoDiscipline):
    ''' Forest discipline
    '''

    # ontology information
    _ontology_data = {
        'label': 'Forest',
        'type': '',
        'source': '',
        'validated': '',
        'validated_by': '',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-tree fa-fw',
        'version': '',
    }
    default_year_start = 2020
    default_year_end = 2050

    deforestation_limit = 1000
    initial_emissions = 3.21

    construction_delay = 3  # years, time for wood to dry

    # available planted forests in 2020: 294 Mha (worldbioenergy.org)

    # reference:
    # https://qtimber.daf.qld.gov.au/guides/wood-density-and-hardness
    wood_density = 600.0  # kg/m3
    residues_density = 200.0  # kg/m3

    # reference :
    # https://www.eubia.org/cms/wiki-biomass/biomass-resources/challenges-related-to-biomass/recovery-of-forest-residues/
    # average of 155 and 310 divided by 5
    residue_density_m3_per_ha = 46.5
    # average of 360 and 600 divided by 5
    wood_density_m3_per_ha = 96

    # in litterature, average price of residue is 30-50euro/t
    # wood price is 100-200euro/t => 26% between
    wood_residue_price_percent_dif = 0.34

    # 1,62% of managed wood is used for energy purpose
    # (3% of global forest is used for energy purpose and
    # 54% of global forest are managed forests)
    wood_percentage_for_energy = 0.48
    residue_percentage_for_energy = 0.48

    density_per_ha = residue_density_m3_per_ha + \
        wood_density_m3_per_ha

    wood_percentage = wood_density_m3_per_ha / density_per_ha
    residue_percentage = residue_density_m3_per_ha / density_per_ha

    mean_density = wood_percentage * wood_density + \
        residue_percentage * residues_density

    # reference :
    # https://www.eubia.org/cms/wiki-biomass/biomass-resources/challenges-related-to-biomass/recovery-of-forest-residues/
    years_between_harvest = 20

    recycle_part = 0.52  # 52%
#     mean_calorific_value = BiomassDryTechnoDiscipline.data_energy_dict[
#         'calorific_value']

    managed_wood_techno_dict = {'maturity': 5,
                                'wood_residues_moisture': 0.35,  # 35% moisture content
                                'wood_residue_colorific_value': 4.356,
                                # teagasc : 235euro/ha/year for planting 5% and spot spraying and manual cleaning
                                # +  chipping + off_road transport 8 euro/Mwh (www.eubia.org )
                                # for wood + residues
                                'Opex_percentage': 0.045,
                                # Capex init: 12000 $/ha to buy the land (CCUS-report_V1.30)
                                # + 2564.128 euro/ha (ground preparation, planting) (www.teagasc.ie)
                                # 1USD = 0,87360 euro in 2019
                                'Managed_wood_price_per_ha': 13047,
                                'Unmanaged_wood_price_per_ha': 10483,
                                'Price_per_ha_unit': 'euro/ha',
                                'full_load_hours': 8760.0,
                                'euro_dollar': 1.1447,  # in 2019, date of the paper
                                'percentage_production': 0.52,

                                'residue_density_percentage': residue_percentage,
                                'non_residue_density_percentage': wood_percentage,
                                'density_per_ha': density_per_ha,
                                'wood_percentage_for_energy': wood_percentage_for_energy,
                                'residue_percentage_for_energy': residue_percentage_for_energy,
                                'density': mean_density,
                                'wood_density': wood_density,
                                'residues_density': residues_density,
                                'density_per_ha_unit': 'm^3/ha',
                                'techno_evo_eff': 'no',  # yes or no
                                'years_between_harvest': years_between_harvest,
                                'wood_residue_price_percent_dif': wood_residue_price_percent_dif,
                                'recycle_part': recycle_part,
                                'construction_delay': construction_delay}
    # invest: 0.19 Mha are planted each year at 13047.328euro/ha, and 28% is
    # the share of wood (not residue)
    invest_before_year_start = pd.DataFrame(
        {'past_years': np.arange(-construction_delay, 0), 'invest': [1.135081, 1.135081, 1.135081]})
    # www.fao.org : forest under long-term management plans = 2.05 Billion Ha
    # 31% of All forests is used for production : 0.31 * 4.06 = 1.25
    # 92% of the production come from managed wood. 8% from unmanaged wood
    # 3.36 : calorific value of wood kwh/kg
    # 4.356 : calorific value of residues
    # initial_production = 1.25 * 0.92 * density_per_ha * density * 3.36  # in
    # Twh
#     initial_production = 1.25 * 0.92 * \
#         (residue_density_m3_per_ha * residues_density * 4.356 + wood_density_m3_per_ha *
# wood_density * 3.36) / years_between_harvest / (1 - recycle_part)  # in
# Twh
    mw_initial_production = 1.25 * 0.92 * \
        density_per_ha * mean_density * 3.6 / \
        years_between_harvest / (1 - recycle_part)  # in Twh

    uw_initial_production = 1.25 * 0.08 * \
        density_per_ha * mean_density * 3.6 / \
        years_between_harvest / (1 - recycle_part)

    DESC_IN = {Forest.YEAR_START: {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.YEAR_END: {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               Forest.TIME_STEP: {'type': 'int', 'default': 1, 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
               Forest.DEFORESTATION_SURFACE: {'type': 'dataframe', 'unit': 'Mha',
                                              'dataframe_descriptor': {'years': ('float', None, False),
                                                                       'deforested_surface': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               Forest.LIMIT_DEFORESTATION_SURFACE: {'type': 'float', 'unit': 'Mha', 'default': deforestation_limit,
                                                    'namespace': 'ns_forest', },
               Forest.INITIAL_CO2_EMISSIONS: {'type': 'float', 'unit': 'GtCO2', 'default': initial_emissions,
                                              'namespace': 'ns_forest', },
               Forest.CO2_PER_HA: {'type': 'float', 'unit': 'kgCO2/ha/year', 'default': 4000, 'namespace': 'ns_forest'},
               Forest.REFORESTATION_COST_PER_HA: {'type': 'float', 'unit': '$/ha', 'default': 3800, 'namespace': 'ns_forest'},
               Forest.REFORESTATION_INVESTMENT: {'type': 'dataframe', 'unit': 'G$',
                                                 'dataframe_descriptor': {'years': ('float', None, False),
                                                                          'forest_investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                                 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               'wood_techno_dict': {'type': 'dict', 'unit': '-', 'default': managed_wood_techno_dict, 'namespace': 'ns_forest'},
               'managed_wood_initial_prod': {'type': 'float', 'unit': 'TWh', 'default': mw_initial_production, 'namespace': 'ns_forest'},
               'managed_wood_initial_surface': {'type': 'float', 'unit': 'Gha', 'namespace': 'ns_forest'},
               'managed_wood_invest_before_year_start': {'type': 'dataframe', 'unit': 'G$',
                                                         'dataframe_descriptor': {'past_years': ('float', None, False),
                                                                                  'investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                                         'namespace': 'ns_forest'},
               'managed_wood_investment': {'type': 'dataframe', 'unit': 'G$',
                                           'dataframe_descriptor': {'years': ('float', None, False),
                                                                    'investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                           'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               'unmanaged_wood_initial_prod': {'type': 'float', 'unit': 'TWh', 'default': uw_initial_production, 'namespace': 'ns_forest'},
               'unmanaged_wood_initial_surface': {'type': 'float', 'unit': 'Gha', 'namespace': 'ns_forest'},
               'unmanaged_wood_invest_before_year_start': {'type': 'dataframe', 'unit': 'G$',
                                                           'dataframe_descriptor': {'past_years': ('float', None, False),
                                                                                    'investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                                           'namespace': 'ns_forest'},
               'unmanaged_wood_investment': {'type': 'dataframe', 'unit': 'G$',
                                             'dataframe_descriptor': {'years': ('float', None, False),
                                                                      'investment': ('float', [0, 1e9], True)}, 'dataframe_edition_locked': False,
                                             'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
               }

    DESC_OUT = {
        'CO2_emissions_detail_df': {
            'type': 'dataframe', 'unit': 'GtCO2', 'namespace': 'ns_forest'},
        Forest.FOREST_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        Forest.FOREST_DETAIL_SURFACE_DF: {
            'type': 'dataframe', 'unit': 'Gha'},
        Forest.CO2_EMITTED_FOREST_DF: {
            'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'managed_wood_df': {
            'type': 'dataframe', 'unit': 'Gha', 'namespace': 'ns_forest'},
        'unmanaged_wood_df': {
            'type': 'dataframe', 'unit': 'Gha',  'namespace': 'ns_forest'},
    }

    FOREST_CHARTS = 'Forest chart'

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.forest_model = Forest(param)

    def run(self):

        #-- get inputs
        #         inputs = list(self.DESC_IN.keys())
        #         inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        outputs_dict = {
            Forest.CO2_EMITTED_DETAIL_DF: self.forest_model.CO2_emitted_df,
            Forest.FOREST_DETAIL_SURFACE_DF: self.forest_model.forest_surface_df,
            Forest.FOREST_SURFACE_DF: self.forest_model.forest_surface_df[['years', 'forest_surface_evol']],
            Forest.CO2_EMITTED_FOREST_DF: self.forest_model.CO2_emitted_df[['years', 'emitted_CO2_evol_cumulative']],
            'managed_wood_df': self.forest_model.managed_wood_df,
            'unmanaged_wood_df': self.forest_model.unmanaged_wood_df,

        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        # gradient for deforestation rate
        d_deforestation_surface_d_deforestation_surface = self.forest_model.d_deforestation_surface_d_deforestation_surface()
        d_cum_deforestation_d_deforestation_surface = self.forest_model.d_cum(
            d_deforestation_surface_d_deforestation_surface)
        d_forest_surface_d_invest = self.forest_model.d_forestation_surface_d_invest()
        d_cun_forest_surface_d_invest = self.forest_model.d_cum(
            d_forest_surface_d_invest)

        # forest surface vs deforestation grad
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, 'forest_surface_evol'), (
                Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_deforestation_surface_d_deforestation_surface)
#         self.set_partial_derivative_for_other_types(
#             (Forest.FOREST_SURFACE_DF,
#              'forest_surface_evol_cumulative'),
#             (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
#             d_cum_deforestation_d_deforestation_surface)

        # forest surface vs forest invest
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, 'forest_surface_evol'), (
                Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
            d_forest_surface_d_invest)
#         self.set_partial_derivative_for_other_types(
#             (Forest.FOREST_SURFACE_DF,
#              'forest_surface_evol_cumulative'),
#             (Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
#             d_cun_forest_surface_d_invest)

        # d_CO2 d deforestation
        d_CO2_emitted_d_deforestation_surface = self.forest_model.d_CO2_emitted(
            d_deforestation_surface_d_deforestation_surface)
        d_cum_CO2_emitted_d_deforestation_surface = self.forest_model.d_cum(
            d_CO2_emitted_d_deforestation_surface)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, 'emitted_CO2_evol_cumulative'),
            (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
            d_cum_CO2_emitted_d_deforestation_surface)

        # d_CO2 d invest
        d_CO2_emitted_d_invest = self.forest_model.d_CO2_emitted(
            d_forest_surface_d_invest)
        d_cum_CO2_emitted_d_invest = self.forest_model.d_cum(
            d_CO2_emitted_d_invest)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, 'emitted_CO2_evol_cumulative'),
            (Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
            d_cum_CO2_emitted_d_invest)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [ForestDiscipline.FOREST_CHARTS]

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        '''
        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then
        '''
        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        forest_surface_df = self.get_sosdisc_outputs(
            Forest.FOREST_DETAIL_SURFACE_DF)
        managed_wood_df = self.get_sosdisc_outputs(
            'managed_wood_df')
        unmanaged_wood_df = self.get_sosdisc_outputs(
            'unmanaged_wood_df')

        if ForestDiscipline.FOREST_CHARTS in chart_list:

            years = forest_surface_df['years'].values.tolist()
            # values are *1000 to convert from Gha to Mha
            delta_reforestation = forest_surface_df['delta_reforestation_surface'].values * 1000
            reforestation = forest_surface_df['reforestation_surface'].values * 1000

            delta_deforestation = forest_surface_df['delta_deforestation_surface'].values * 1000
            deforestation = forest_surface_df['deforested_surface'].values * 1000

            delta_managed_wood_surface = managed_wood_df['delta_surface'].values * 1000
            managed_wood_surface = managed_wood_df['cumulative_surface'].values * 1000

            delta_unmanaged_wood_surface = unmanaged_wood_df['delta_surface'].values * 1000
            unmanaged_wood_surface = unmanaged_wood_df['cumulative_surface'].values * 1000

            delta_global = forest_surface_df['delta_global_forest_surface'].values * 1000
            global_surface = forest_surface_df['global_forest_surface'].values * 1000

            # forest evolution year by year chart
            new_chart = TwoAxesInstanciatedChart('years', 'Yearly delta of forest surface evolution [Mha / year]',
                                                 chart_name='Yearly delta of forest surface evolution', stacked_bar=True)

            deforested_series = InstanciatedSeries(
                years, delta_deforestation.tolist(), 'Deforestation', 'bar')
            forested_series = InstanciatedSeries(
                years, delta_reforestation.tolist(), 'Reforestation', 'bar')
            total_series = InstanciatedSeries(
                years, delta_global.tolist(), 'Global forest surface', InstanciatedSeries.LINES_DISPLAY)
            managed_wood_series = InstanciatedSeries(
                years, delta_managed_wood_surface.tolist(), 'Managed wood', 'bar')
            unmanaged_wood_series = InstanciatedSeries(
                years, delta_unmanaged_wood_surface.tolist(), 'Unmanaged wood', 'bar')

            new_chart.add_series(deforested_series)
            new_chart.add_series(total_series)
            new_chart.add_series(forested_series)
            new_chart.add_series(managed_wood_series)
            new_chart.add_series(unmanaged_wood_series)

            instanciated_charts.append(new_chart)

            # forest cumulative evolution chart
            new_chart = TwoAxesInstanciatedChart('years', 'Forest surface evolution [Mha]',
                                                 chart_name='Global forest surface evolution', stacked_bar=True)

            deforested_series = InstanciatedSeries(
                years, deforestation.tolist(), 'Deforested surface', 'bar')
            forested_series = InstanciatedSeries(
                years, reforestation.tolist(), 'Reforested surface', 'bar')
            total_series = InstanciatedSeries(
                years, global_surface.tolist(), 'Forest surface evolution', InstanciatedSeries.LINES_DISPLAY)
            managed_wood_series = InstanciatedSeries(
                years, managed_wood_surface.tolist(), 'Managed wood', 'bar')
            unmanaged_wood_series = InstanciatedSeries(
                years, unmanaged_wood_surface.tolist(), 'Unmanaged wood', 'bar')

            new_chart.add_series(deforested_series)
            new_chart.add_series(total_series)
            new_chart.add_series(forested_series)
            new_chart.add_series(managed_wood_series)
            new_chart.add_series(unmanaged_wood_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            CO2_emissions_df = self.get_sosdisc_outputs(
                'CO2_emissions_detail_df')

            delta_reforestation = CO2_emissions_df['delta_CO2_reforestation'].values
            reforestation = CO2_emissions_df['CO2_reforestation'].values

            delta_deforestation = CO2_emissions_df['delta_CO2_deforestation'].values
            deforestation = CO2_emissions_df['CO2_deforestation'].values

            delta_managed_wood_surface = managed_wood_df['delta_CO2_emitted'].values
            managed_wood_surface = managed_wood_df['CO2_emitted'].values

            delta_unmanaged_wood_surface = unmanaged_wood_df['delta_CO2_emitted'].values
            unmanaged_wood_surface = unmanaged_wood_df['CO2_emitted'].values

            delta_global = forest_surface_df['delta_CO2_emitted'].values
            global_surface = forest_surface_df['global_CO2_emission_balance'].values

            new_chart = TwoAxesInstanciatedChart('years', 'CO2 emission & capture [GtCO2 / year]',
                                                 chart_name='Yearly forest delta CO2 emissions', stacked_bar=True)

            CO2_deforestation_series = InstanciatedSeries(
                years, delta_deforestation.tolist(), 'Deforestation emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_reforestation_series = InstanciatedSeries(
                years, delta_reforestation.tolist(), 'Reforestation emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_total_series = InstanciatedSeries(
                years, delta_global.tolist(), 'Global CO2 balance', InstanciatedSeries.LINES_DISPLAY)
            CO2_managed_wood_series = InstanciatedSeries(
                years, delta_managed_wood_surface.tolist(), 'Managed wood emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_unmanaged_wood_series = InstanciatedSeries(
                years, delta_unmanaged_wood_surface.tolist(), 'Unmanaged wood emissions', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(CO2_deforestation_series)
            new_chart.add_series(CO2_total_series)
            new_chart.add_series(CO2_reforestation_series)
            new_chart.add_series(CO2_managed_wood_series)
            new_chart.add_series(CO2_unmanaged_wood_series)

            instanciated_charts.append(new_chart)

            # in Gt
            new_chart = TwoAxesInstanciatedChart('years', 'CO2 emission & capture [GtCO2]',
                                                 chart_name='Forest CO2 emissions', stacked_bar=True)
            CO2_deforestation_series = InstanciatedSeries(
                years, deforestation.tolist(), 'Deforestation emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_reforestation_series = InstanciatedSeries(
                years, reforestation.tolist(), 'Reforestation emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_total_series = InstanciatedSeries(
                years, global_surface.tolist(), 'Global CO2 balance', InstanciatedSeries.LINES_DISPLAY)
            CO2_managed_wood_series = InstanciatedSeries(
                years, managed_wood_surface.tolist(), 'Managed wood emissions', InstanciatedSeries.BAR_DISPLAY)
            CO2_unmanaged_wood_series = InstanciatedSeries(
                years, unmanaged_wood_surface.tolist(), 'Unmanaged wood emissions', InstanciatedSeries.BAR_DISPLAY)

            new_chart.add_series(CO2_deforestation_series)
            new_chart.add_series(CO2_total_series)
            new_chart.add_series(CO2_reforestation_series)
            new_chart.add_series(CO2_managed_wood_series)
            new_chart.add_series(CO2_unmanaged_wood_series)
            instanciated_charts.append(new_chart)
        return instanciated_charts
