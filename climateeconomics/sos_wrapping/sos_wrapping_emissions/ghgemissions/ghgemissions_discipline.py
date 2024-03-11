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
import numpy as np

from climateeconomics.core.core_emissions.ghg_emissions_model import GHGEmissions
from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart


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
    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
    name = 'GHGEmissions'
    _maturity = 'Research'

    # https://ghgprotocol.org/sites/default/files/ghgp/Global-Warming-Potential-Values%20%28Feb%2016%202016%29_1.pdf
    # From IPCC AR5
    GWP_100_default = {GlossaryCore.CO2: 1.0,
                       GlossaryCore.CH4: 28.,
                       GlossaryCore.N2O: 265.}

    GWP_20_default = {GlossaryCore.CO2: 1.0,
                      GlossaryCore.CH4: 85.,
                      GlossaryCore.N2O: 265.}
    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'GHG_global_warming_potential20':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_20_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        'GHG_global_warming_potential100':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_100_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        'CO2_land_emissions': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                  'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Forest': ('float', None, False),
                                                          'Crop': ('float', None, False)}},
        'CH4_land_emissions': {'type': 'dataframe', 'unit': 'GtCH4', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Crop': ('float', None, False),
                                                          'Forest': ('float', None, False),}},
        'N2O_land_emissions': {'type': 'dataframe', 'unit': 'GtN2O', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Crop': ('float', None, False),
                                                          'Forest': ('float', None, False),}},
        'CO2_indus_emissions_df':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                             'indus_emissions': ('float', None, False),}
                                    },
        'GHG_total_energy_emissions':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                                        'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                 GlossaryCore.TotalCO2Emissions: ('float', None, False),
                                                                 GlossaryCore.TotalN2OEmissions: ('float', None, False),
                                                                 GlossaryCore.TotalCH4Emissions: ('float', None, False), }
                                        },
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.CO2EmissionsRef['var_name']: GlossaryCore.CO2EmissionsRef,
        'affine_co2_objective': {'type': 'bool','default': True, 'user_level': 2, 'namespace': GlossaryCore.NS_WITNESS},

    }
    DESC_OUT = {
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
        GlossaryCore.GHGEmissionsDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': 'Gt'},
        'GHG_emissions_detail_df': {'type': 'dataframe', 'unit': 'Gt'},
        'GWP_emissions': {'type': 'dataframe', 'unit': 'GtCO2eq'},
        GlossaryCore.CO2EmissionsObjectiveValue: GlossaryCore.CO2EmissionsObjective,
        GlossaryCore.TotalEnergyEmissions: GlossaryCore.TotalEnergyCO2eqEmissionsDf
    }

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.emissions_model = GHGEmissions(in_dict)

    def run(self):
        # Get inputs
        inputs_dict = self.get_sosdisc_inputs()
        if inputs_dict[GlossaryCore.CheckRangeBeforeRunBoolName]:
            dict_ranges = self.get_ranges_input_var()
            self.check_ranges(inputs_dict, dict_ranges)
        self.emissions_model.configure_parameters_update(inputs_dict)
        # Compute de emissions_model
        self.emissions_model.compute()
        # Store output data

        # co2_emissions_df = self.emissions_model.compute_co2_emissions_for_carbon_cycle()
        cols = [GlossaryCore.Years] + [f'Total {ghg} emissions' for ghg in self.emissions_model.GHG_TYPE_LIST]
        emissions_df = self.emissions_model.ghg_emissions_df[cols]

        dict_values = {'GHG_emissions_detail_df': self.emissions_model.ghg_emissions_df,
                       GlossaryCore.CO2EmissionsGtValue: self.emissions_model.GHG_total_energy_emissions[[GlossaryCore.Years, GlossaryCore.TotalCO2Emissions]],
                       GlossaryCore.GHGEmissionsDfValue: emissions_df,
                       'GWP_emissions': self.emissions_model.gwp_emissions,
                       GlossaryCore.CO2EmissionsObjectiveValue: self.emissions_model.co2_emissions_objective,
                       GlossaryCore.TotalEnergyEmissions: self.emissions_model.total_energy_co2eq_emissions
                       }
        if inputs_dict[GlossaryCore.CheckRangeBeforeRunBoolName]:
            dict_ranges = self.get_ranges_output_var()
            self.check_ranges(dict_values, dict_ranges)
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        co2_emissions_Gt

        """
        inputs_dict = self.get_sosdisc_inputs()
        years = np.arange(
            inputs_dict[GlossaryCore.YearStart], inputs_dict[GlossaryCore.YearEnd] + 1, inputs_dict[GlossaryCore.TimeStep])

        # land emissions
        for ghg in self.emissions_model.GHG_TYPE_LIST:
            ghg_land_emissions = inputs_dict[f'{ghg}_land_emissions']
            for column in ghg_land_emissions.columns:
                if column != GlossaryCore.Years:
                    self.set_partial_derivative_for_other_types(
                        (GlossaryCore.GHGEmissionsDfValue, f'Total {ghg} emissions'),
                        (f'{ghg}_land_emissions', column),
                        np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.GHGEmissionsDfValue, f'Total {ghg} emissions'),
                ('GHG_total_energy_emissions', f'Total {ghg} emissions'),
                np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TotalEnergyEmissions, GlossaryCore.TotalEnergyEmissions),
                ('GHG_total_energy_emissions', f'Total {ghg} emissions'),
                np.identity(len(years))* self.emissions_model.gwp_100[ghg])

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            ('GHG_total_energy_emissions', GlossaryCore.TotalCO2Emissions),  np.identity(len(years)))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            ('CO2_indus_emissions_df', 'indus_emissions'),
            np.identity(len(years)))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsObjectiveValue,), ('GHG_total_energy_emissions', GlossaryCore.TotalCO2Emissions),
            self.emissions_model.d_CO2_emissions_objective_d_total_co2_emissions())

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['GHG emissions per sector', 'Global Warming Potential', 'Total CO2 emissions']
        #chart_list = ['sectoral energy carbon emissions cumulated']
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

        if 'Total CO2 emissions' in charts:
            new_chart = self.get_chart_total_CO2()
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_chart_gwp(self, gwp_year):
        GWP_emissions = self.get_sosdisc_outputs(
            'GWP_emissions')

        chart_name = f'Global warming potential at {gwp_year} years'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, 'GWP [GtCO2]', chart_name=chart_name, stacked_bar=True)

        for ghg in GHGEmissions.GHG_TYPE_LIST:
            new_serie = InstanciatedSeries(list(GWP_emissions[GlossaryCore.Years].values), list(GWP_emissions[f'{ghg}_{gwp_year}'].values),
                                           ghg, 'bar')

            new_chart.series.append(new_serie)

        return new_chart

    def get_chart_emission_per_sector(self, ghg):
        GHG_emissions_detail_df = self.get_sosdisc_outputs(
            'GHG_emissions_detail_df')

        chart_name = f'{ghg} emissions per sector'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, f'{ghg} Emissions [Gt]', chart_name=chart_name, stacked_bar=True)

        sector_list = ['energy', 'land', 'industry']
        for sector in sector_list:
            if f'{ghg} {sector}_emissions' in GHG_emissions_detail_df:
                new_serie = InstanciatedSeries(list(GHG_emissions_detail_df[GlossaryCore.Years].values), list(GHG_emissions_detail_df[f'{ghg} {sector}_emissions'].values),
                                               sector, 'bar')

                new_chart.series.append(new_serie)

        return new_chart

    def get_chart_total_CO2(self):
        """
        Chart with total CO2 emissions per type
        """
        total_ghg_df = self.get_sosdisc_outputs(GlossaryCore.GHGEmissionsDfValue)
        chart_name = f'Total CO2 emissions'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, f'Total CO2 emissions [Gt]', chart_name=chart_name)
        new_serie = InstanciatedSeries(list(total_ghg_df[GlossaryCore.Years].values),
                                       total_ghg_df[f'Total CO2 emissions'].to_list(),
                                       f'CO2 emissions', 'lines')
        new_chart.series.append(new_serie)
        return new_chart



