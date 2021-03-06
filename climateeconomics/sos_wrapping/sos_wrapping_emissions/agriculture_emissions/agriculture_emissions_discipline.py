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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from copy import deepcopy
import pandas as pd
import numpy as np
from climateeconomics.core.core_emissions.indus_emissions_model import IndusEmissions
from climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline import GHGemissionsDiscipline


class AgricultureEmissionsDiscipline(ClimateEcoDiscipline):
    """Agriculture Emissions discipline"""

    # ontology information
    _ontology_data = {
        'label': 'Agricultural Emission WITNESS Model',
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
    years = np.arange(2020, 2101)

    name = f'{GHGemissionsDiscipline.name}.Agriculture'
    _maturity = 'Research'
    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'technologies_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                              'possible_values': ['Crop', 'Forest'],
                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                              'namespace': 'ns_agriculture',
                              'structuring': True},
        'other_land_CO2_emissions': {'type': 'float', 'unit': 'GtCO2',  'default': 10.1, },
        # other land emissions = land use change emission - Forest initial
        # emission - computed crop emissions = 3.2(initial) + 7.6(frorest) -
        # 0.7(crop)
    }
    DESC_OUT = {
        'CO2_land_emissions': {'type': 'dataframe', 'unit': 'GtCO2',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'CH4_land_emissions': {'type': 'dataframe', 'unit': 'GtCH4',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'N2O_land_emissions': {'type': 'dataframe', 'unit': 'GtN2O',
                                       'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        if 'technologies_list' in self._data_in:
            techno_list = self.get_sosdisc_inputs('technologies_list')
            if techno_list is not None:
                for techno in techno_list:
                    dynamic_inputs[f'{techno}.CO2_land_emission_df'] = {
                        'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'}
                    if techno == 'Crop':
                        dynamic_inputs[f'{techno}.CH4_land_emission_df'] = {
                            'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'}
                        dynamic_inputs[f'{techno}.N2O_land_emission_df'] = {
                            'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_agriculture'}

        self.add_inputs(dynamic_inputs, clean_inputs=False)

    def run(self):
        # -- get CO2 emissions inputs
        CO2_emitted_crop_df = self.get_sosdisc_inputs(
            'Crop.CO2_land_emission_df')
        CH4_emitted_crop_df = self.get_sosdisc_inputs(
            'Crop.CH4_land_emission_df')
        N2O_emitted_crop_df = self.get_sosdisc_inputs(
            'Crop.N2O_land_emission_df')
        CO2_emitted_forest_df = self.get_sosdisc_inputs(
            'Forest.CO2_land_emission_df')
        other_land_CO2_emissions = self.get_sosdisc_inputs(
            'other_land_CO2_emissions')

        # init dataframes
        CO2_emissions_land_use_df = pd.DataFrame()
        CH4_emissions_land_use_df = pd.DataFrame()
        N2O_emissions_land_use_df = pd.DataFrame()

        # co2 aggregation
        CO2_emissions_land_use_df['years'] = CO2_emitted_crop_df['years']
        CO2_emissions_land_use_df['Crop'] = CO2_emitted_crop_df['emitted_CO2_evol_cumulative']
        CO2_emissions_land_use_df['Forest'] = CO2_emitted_forest_df['emitted_CO2_evol_cumulative']
        CO2_emissions_land_use_df['Other_emissions'] = other_land_CO2_emissions

        # ch4 aggregation
        CH4_emissions_land_use_df['years'] = CH4_emitted_crop_df['years']
        CH4_emissions_land_use_df['Crop'] = CH4_emitted_crop_df['emitted_CH4_evol_cumulative']

        # n2o aggregation
        N2O_emissions_land_use_df['years'] = N2O_emitted_crop_df['years']
        N2O_emissions_land_use_df['Crop'] = N2O_emitted_crop_df['emitted_N2O_evol_cumulative']

        # write outputs
        outputs_dict = {
            'CO2_land_emissions': CO2_emissions_land_use_df,
            'CH4_land_emissions': CH4_emissions_land_use_df,
            'N2O_land_emissions': N2O_emissions_land_use_df,
        }
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        inputs_dict = self.get_sosdisc_inputs()
        np_years = inputs_dict['year_end'] - inputs_dict['year_start'] + 1
        techno_list = self.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            self.set_partial_derivative_for_other_types(
                ('CO2_land_emissions', f'{techno}'), (
                    f'{techno}.CO2_land_emission_df', 'emitted_CO2_evol_cumulative'),
                np.identity(np_years))
            if techno == 'Crop':
                self.set_partial_derivative_for_other_types(
                    ('CH4_land_emissions', f'{techno}'),
                    (f'{techno}.CH4_land_emission_df', 'emitted_CH4_evol_cumulative'), np.identity(np_years))
                self.set_partial_derivative_for_other_types(
                    ('N2O_land_emissions', f'{techno}'),
                    (f'{techno}.N2O_land_emission_df', 'emitted_N2O_evol_cumulative'), np.identity(np_years))

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['GHG emissions']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        charts = []
        if chart_filters is not None:
            for chart_filter in chart_filters:
                charts = chart_filter.selected_values

        if 'GHG emissions' in charts:
            new_charts = self.get_chart_ghg_emissions()
            if new_charts is not None:
                instanciated_charts += new_charts

        return instanciated_charts

    def get_chart_ghg_emissions(self):
        new_charts = []

        # all ghg graph
        chart_name = f'Greenhouse Gas emissions of agriculture lands'
        new_chart = TwoAxesInstanciatedChart(
            'years', 'GHG emissions (Gt)', chart_name=chart_name, stacked_bar=True)

        technology_list = self.get_sosdisc_inputs('technologies_list')

        CO2_emissions_land_use_df = self.get_sosdisc_outputs(
            'CO2_land_emissions')
        CH4_emissions_land_use_df = self.get_sosdisc_outputs(
            'CH4_land_emissions')
        N2O_emissions_land_use_df = self.get_sosdisc_outputs(
            'N2O_land_emissions')
        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        year_list = np.arange(year_start, year_end + 1).tolist()

        CO2_emissions_land_use = CO2_emissions_land_use_df.drop(
            'years', axis=1).sum(axis=1).values.tolist()
        CH4_emissions_land_use = CH4_emissions_land_use_df.drop(
            'years', axis=1).sum(axis=1).values.tolist()
        N2O_emissions_land_use = N2O_emissions_land_use_df.drop(
            'years', axis=1).sum(axis=1).values.tolist()

        serie = InstanciatedSeries(
            year_list, CO2_emissions_land_use, 'CO2 Emissions [GtCO2]', 'bar')
        new_chart.series.append(serie)

        serie = InstanciatedSeries(
            year_list, CH4_emissions_land_use, 'CH4 Emissions [GtCH4]', 'bar')
        new_chart.series.append(serie)

        serie = InstanciatedSeries(
            year_list, N2O_emissions_land_use, 'N2O Emissions [GtN2O]', 'bar')
        new_chart.series.append(serie)

        new_charts.append(new_chart)

        # co2 per crop/forest
        chart_name = f'Comparison of CO2 emissions of agriculture lands'
        new_chart = TwoAxesInstanciatedChart(
            'years', 'CO2 emissions (GtCO2)', chart_name=chart_name, stacked_bar=True)

        technology_list = self.get_sosdisc_inputs('technologies_list')

        CO2_emissions_land_use_df = self.get_sosdisc_outputs(
            'CO2_land_emissions')
        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        year_list = np.arange(year_start, year_end + 1).tolist()
        for column in CO2_emissions_land_use_df.columns:
            if column != 'years':
                techno_emissions = CO2_emissions_land_use_df[column]
                serie = InstanciatedSeries(
                    year_list, techno_emissions.tolist(), column, 'bar')
                new_chart.series.append(serie)

        new_charts.append(new_chart)

        return new_charts
