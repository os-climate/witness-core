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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import numpy as np
from climateeconomics.core.core_emissions.ghg_emissions_model import GHGEmissions


class GHGemissionsDiscipline(ClimateEcoDiscipline):
    "GHGemissions discipline for DICE"

    # ontology information
    _ontology_data = {
        'label': 'GHG Emission WITNESS Model',
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
    name = 'GHGEmissions'
    _maturity = 'Research'

    # https://ghgprotocol.org/sites/default/files/ghgp/Global-Warming-Potential-Values%20%28Feb%2016%202016%29_1.pdf
    # From IPCC AR5
    GWP_100_default = {'CO2': 1.0,
                       'CH4': 28.,
                       'N2O': 265.}

    GWP_20_default = {'CO2': 1.0,
                      'CH4': 85.,
                      'N2O': 265.}
    DESC_IN = {
        'year_start': ClimateEcoDiscipline.YEAR_START_DESC_IN,
        'year_end': ClimateEcoDiscipline.YEAR_END_DESC_IN,
        'time_step': ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'GHG_global_warming_potential20':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_20_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness', 'user_level': 3},
        'GHG_global_warming_potential100':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_100_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness', 'user_level': 3},
        'CO2_land_emissions': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'CH4_land_emissions': {'type': 'dataframe', 'unit': 'GtCH4', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'N2O_land_emissions': {'type': 'dataframe', 'unit': 'GtN2O', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'CO2_indus_emissions_df':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
        'GHG_total_energy_emissions':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'},
    }
    DESC_OUT = {
        'co2_emissions_Gt': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_energy_mix', 'unit': 'Gt'},
        'GHG_emissions_df': {'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness', 'unit': 'Gt'},
        'GHG_emissions_detail_df': {'type': 'dataframe', 'unit': 'Gt'},
        'GWP_emissions': {'type': 'dataframe', 'unit': 'GtCO2eq'}
    }

    def init_execution(self, proxy):
        in_dict = proxy.get_sosdisc_inputs()
        self.emissions_model = GHGEmissions(in_dict)

    def run(self):
        # Get inputs
        inputs_dict = self.get_sosdisc_inputs()
        self.emissions_model.configure_parameters_update(inputs_dict)
        # Compute de emissions_model
        self.emissions_model.compute()
        # Store output data

        # co2_emissions_df = self.emissions_model.compute_co2_emissions_for_carbon_cycle()
        cols = ['years'] + [f'Total {ghg} emissions' for ghg in self.emissions_model.GHG_TYPE_LIST]
        emissions_df = self.emissions_model.ghg_emissions_df[cols]

        dict_values = {'GHG_emissions_detail_df': self.emissions_model.ghg_emissions_df,
                       'co2_emissions_Gt': self.emissions_model.GHG_total_energy_emissions[['years', 'Total CO2 emissions']],
                       'GHG_emissions_df': emissions_df,
                       'GWP_emissions': self.emissions_model.gwp_emissions}

        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        co2_emissions_Gt

        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(
            inputs_dict['year_start'], inputs_dict['year_end'] + 1, inputs_dict['time_step'])

        # land emissions
        for ghg in self.emissions_model.GHG_TYPE_LIST:
            ghg_land_emissions = inputs_dict[f'{ghg}_land_emissions']
            for column in ghg_land_emissions.columns:
                if column != "years":
                    self.set_partial_derivative_for_other_types(
                        ('GHG_emissions_df', f'Total {ghg} emissions'), (f'{ghg}_land_emissions', column),  np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                ('GHG_emissions_df', f'Total {ghg} emissions'), ('GHG_total_energy_emissions', f'Total {ghg} emissions'),
                np.identity(len(years)))

        self.set_partial_derivative_for_other_types(
            ('co2_emissions_Gt', 'Total CO2 emissions'), ('GHG_total_energy_emissions', 'Total CO2 emissions'),  np.identity(len(years)))
        self.set_partial_derivative_for_other_types(
            ('GHG_emissions_df', 'Total CO2 emissions'), ('CO2_indus_emissions_df', 'indus_emissions'),
            np.identity(len(years)))

    def get_chart_filter_list(self, proxy):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['GHG emissions per sector', 'Global Warming Potential']
        #chart_list = ['sectoral energy carbon emissions cumulated']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, proxy, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        charts = []
        if chart_filters is not None:
            for chart_filter in chart_filters:
                charts = chart_filter.selected_values

        if 'GHG emissions per sector' in charts:
            for ghg in GHGEmissions.GHG_TYPE_LIST:
                new_chart = self.get_chart_emission_per_sector(ghg)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)

        if 'Global Warming Potential' in charts:
            for gwp_year in [20, 100]:
                new_chart = self.get_chart_gwp(gwp_year)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)
        return instanciated_charts

    def get_chart_gwp(self, gwp_year):
        GWP_emissions = self.get_sosdisc_outputs(
            'GWP_emissions')

        chart_name = f'Global warming potential at {gwp_year} years'
        new_chart = TwoAxesInstanciatedChart(
            'years', 'GWP [GtCO2]', chart_name=chart_name, stacked_bar=True)

        for ghg in GHGEmissions.GHG_TYPE_LIST:
            new_serie = InstanciatedSeries(list(GWP_emissions['years'].values), list(GWP_emissions[f'{ghg}_{gwp_year}'].values),
                                           ghg, 'bar')

            new_chart.series.append(new_serie)

        return new_chart

    def get_chart_emission_per_sector(self, ghg):
        GHG_emissions_detail_df = self.get_sosdisc_outputs(
            'GHG_emissions_detail_df')

        chart_name = f'{ghg} emissions per sector'
        new_chart = TwoAxesInstanciatedChart(
            'years', f'{ghg} Emissions [Gt]', chart_name=chart_name, stacked_bar=True)

        sector_list = ['energy', 'land', 'industry']
        for sector in sector_list:
            if f'{ghg} {sector}_emissions' in GHG_emissions_detail_df:
                new_serie = InstanciatedSeries(list(GHG_emissions_detail_df['years'].values), list(GHG_emissions_detail_df[f'{ghg} {sector}_emissions'].values),
                                               sector, 'bar')

                new_chart.series.append(new_serie)

        return new_chart
