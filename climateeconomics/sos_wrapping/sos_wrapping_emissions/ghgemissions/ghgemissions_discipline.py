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
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_emissions.ghg_emissions_model import GHGEmissions
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


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

    sector_list_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionList)
    sector_list_variable['default'] = GlossaryCore.DefaultSectorListGHGEmissions
    del sector_list_variable['visibility']
    del sector_list_variable['namespace']
    DESC_IN = {
        'cheat_var_to_update_ns_dashboard_in_ms_mdo': {'type': 'float', 'namespace': GlossaryCore.NS_GHGEMISSIONS,
                                                       'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                                                       'user_level': 3},
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'GHG_global_warming_potential20':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_20_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        'GHG_global_warming_potential100':  {'type': 'dict','subtype_descriptor': {'dict':'float'}, 'unit': 'kgCO2eq/kg', 'default': GWP_100_default, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2): {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                  'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Forest': ('float', None, False),
                                                          'Crop': ('float', None, False)}},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4): {'type': 'dataframe', 'unit': 'GtCH4', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Crop': ('float', None, False),
                                                          'Forest': ('float', None, False),}},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O): {'type': 'dataframe', 'unit': 'GtN2O', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                          'Crop': ('float', None, False),
                                                          'Forest': ('float', None, False),}},
        'GHG_total_energy_emissions':  {'type': 'dataframe', 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                                        'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                 GlossaryCore.TotalCO2Emissions: ('float', None, False),
                                                                 GlossaryCore.TotalN2OEmissions: ('float', None, False),
                                                                 GlossaryCore.TotalCH4Emissions: ('float', None, False), }
                                        },
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.CO2EmissionsRef['var_name']: GlossaryCore.CO2EmissionsRef,
        'affine_co2_objective': {'type': 'bool','default': True, 'user_level': 2, 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.EnergyProductionValue: GlossaryCore.EnergyProductionDf,
        GlossaryCore.SectorListValue: sector_list_variable,
        GlossaryCore.ResidentialEnergyConsumptionDfValue: GlossaryCore.ResidentialEnergyConsumptionDf
    }

    DESC_OUT = {
        GlossaryCore.CO2EmissionsGtValue: GlossaryCore.CO2EmissionsGt,
        GlossaryCore.GHGEmissionsDfValue: {'type': 'dataframe', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': 'Gt'},
        GlossaryCore.GHGEmissionsDetailedDfValue: {'type': 'dataframe', 'unit': 'Gt'},
        GlossaryCore.TotalGWPEmissionsDfValue: GlossaryCore.GWPEmissionsDf,
        GlossaryCore.CO2EmissionsObjectiveValue: GlossaryCore.CO2EmissionsObjective,
        GlossaryCore.TotalEnergyEmissions: GlossaryCore.TotalEnergyCO2eqEmissionsDf,
        GlossaryCore.EnergyCarbonIntensityDfValue: GlossaryCore.EnergyCarbonIntensityDf,
        GlossaryCore.EconomicsEmissionDfValue: GlossaryCore.EmissionDf,
        GlossaryCore.ResidentialEmissionsDfValue: GlossaryCore.ResidentialEmissionsDf
    }

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        dynamic_outputs = {}
        if GlossaryCore.SectorListValue in self.get_data_in():
            sectorlist = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            if sectorlist is not None:
                for sector in sectorlist:
                    # section energy consumption
                    section_energy_consumption_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionEnergyConsumptionDf)
                    section_energy_consumption_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    dynamic_inputs[f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}"] = section_energy_consumption_df_variable

                    section_non_energy_emissions_gdp_var = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionNonEnergyEmissionGdpDf)
                    section_non_energy_emissions_gdp_var.update({'namespace': GlossaryCore.NS_GHGEMISSIONS})
                    section_non_energy_emissions_gdp_var.update({'visibility': "Shared"})
                    dynamic_inputs[f"{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}"] = section_non_energy_emissions_gdp_var

                    section_gdp_var = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionGdpDf)
                    section_gdp_var["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    dynamic_inputs[f"{sector}.{GlossaryCore.SectionGdpDfValue}"] = section_gdp_var

                    section_energy_emissions_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionEnergyEmissionDf)
                    section_energy_emissions_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    section_energy_emissions_df_variable['namespace'] = GlossaryCore.NS_GHGEMISSIONS
                    section_energy_emissions_df_variable['visibility'] = "Shared"
                    dynamic_outputs[f"{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}"] = section_energy_emissions_df_variable

                    section_non_energy_emissions_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionNonEnergyEmissionDf)
                    section_non_energy_emissions_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    section_non_energy_emissions_df_variable['namespace'] = GlossaryCore.NS_GHGEMISSIONS
                    section_non_energy_emissions_df_variable['visibility'] = "Shared"
                    dynamic_outputs[f"{sector}.{GlossaryCore.SectionNonEnergyEmissionDfValue}"] = section_non_energy_emissions_df_variable

                    section_emissions_df_variable = GlossaryCore.get_dynamic_variable(GlossaryCore.SectionEmissionDf)
                    section_emissions_df_variable['namespace'] = GlossaryCore.NS_GHGEMISSIONS
                    section_emissions_df_variable['visibility'] = "Shared"
                    section_emissions_df_variable["dataframe_descriptor"].update({section: ('float', [0., 1e30], True) for section in GlossaryCore.SectionDictSectors[sector]})
                    dynamic_outputs[f"{sector}.{GlossaryCore.SectionEmissionDfValue}"] = section_emissions_df_variable

                    emission_df_disc = GlossaryCore.get_dynamic_variable(GlossaryCore.EmissionDf)
                    emission_df_disc['namespace'] = GlossaryCore.NS_GHGEMISSIONS
                    emission_df_disc['visibility'] = "Shared"
                    dynamic_outputs[f"{sector}.{GlossaryCore.EmissionsDfValue}"] = emission_df_disc

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.emissions_model = GHGEmissions(in_dict)

    def run(self):
        # Get inputs
        inputs_dict = self.get_sosdisc_inputs()
        
        self.emissions_model.configure_parameters_update(inputs_dict)
        # Compute de emissions_model
        self.emissions_model.compute(inputs_dict)
        # Store output data

        dict_values = {
            GlossaryCore.GHGEmissionsDetailedDfValue: self.emissions_model.ghg_emissions_df,
            GlossaryCore.CO2EmissionsGtValue: self.emissions_model.GHG_total_energy_emissions[[GlossaryCore.Years, GlossaryCore.TotalCO2Emissions]],
            GlossaryCore.GHGEmissionsDfValue: self.emissions_model.ghg_emissions_df[GlossaryCore.GHGEmissionsDf['dataframe_descriptor'].keys()],
            GlossaryCore.TotalGWPEmissionsDfValue: self.emissions_model.gwp_emissions,
            GlossaryCore.CO2EmissionsObjectiveValue: self.emissions_model.co2_emissions_objective,
            GlossaryCore.TotalEnergyEmissions: self.emissions_model.total_energy_co2eq_emissions,
            GlossaryCore.EnergyCarbonIntensityDfValue: self.emissions_model.carbon_intensity_of_energy_mix,
            GlossaryCore.EconomicsEmissionDfValue: self.emissions_model.total_economics_emisssions[GlossaryCore.EmissionDf['dataframe_descriptor'].keys()],
            GlossaryCore.ResidentialEmissionsDfValue: self.emissions_model.energy_emission_households_df
        }

        for sector in self.emissions_model.new_sector_list:
            dict_values.update({f"{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}": self.emissions_model.dict_sector_sections_energy_emissions[sector]})
            dict_values.update({f"{sector}.{GlossaryCore.SectionNonEnergyEmissionDfValue}": self.emissions_model.dict_sector_sections_non_energy_emissions[sector]})
            dict_values.update({f"{sector}.{GlossaryCore.SectionEmissionDfValue}": self.emissions_model.dict_sector_sections_emissions[sector]})
            dict_values.update({f"{sector}.{GlossaryCore.EmissionsDfValue}": self.emissions_model.dict_sector_emissions[sector][GlossaryCore.EmissionDf['dataframe_descriptor'].keys()]})

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
        d_energy_carbon_intensity_d_ghg_total_emissions = {}
        for ghg in self.emissions_model.GHG_TYPE_LIST:
            ghg_land_emissions = inputs_dict[GlossaryCore.insertGHGAgriLandEmissions.format(ghg)]
            for column in ghg_land_emissions.columns:
                if column != GlossaryCore.Years:
                    self.set_partial_derivative_for_other_types(
                        (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                        (GlossaryCore.insertGHGAgriLandEmissions.format(ghg), column),
                        np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                np.identity(len(years)))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.TotalEnergyEmissions, GlossaryCore.TotalEnergyEmissions),
                ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                np.identity(len(years))* self.emissions_model.gwp_100[ghg])

            d_energy_carbon_intensity_d_ghg_total_emissions[ghg] = self.emissions_model.d_carbon_intensity_of_energy_mix_d_ghg_energy_emissions(ghg=ghg)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EnergyCarbonIntensityDfValue, GlossaryCore.EnergyCarbonIntensityDfValue),
                ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                d_energy_carbon_intensity_d_ghg_total_emissions[ghg])

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsGtValue, GlossaryCore.TotalCO2Emissions),
            ('GHG_total_energy_emissions', GlossaryCore.TotalCO2Emissions),  np.identity(len(years)))
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.CO2EmissionsObjectiveValue,), ('GHG_total_energy_emissions', GlossaryCore.TotalCO2Emissions),
            self.emissions_model.d_CO2_emissions_objective_d_total_co2_emissions())

        d_carbon_intensity_d_energy_prod = self.emissions_model.d_carbon_intensity_of_energy_mix_d_energy_production()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.EnergyCarbonIntensityDfValue, GlossaryCore.EnergyCarbonIntensityDfValue),
            (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
            d_carbon_intensity_d_energy_prod)

        d_sector_energy_emissions_d_ghg_emissions = {}
        d_sector_energy_emissions_d_energy_prod = {}
        d_section_energy_emissions_d_section_energy_consumption = self.emissions_model.d_section_energy_emissions_d_section_energy_consumption()
        for sector in self.emissions_model.economic_sectors_except_agriculture:
            d_sector_energy_emissions_d_ghg_emissions[sector] = {}
            d_energy_emissions_sections_d_energy_prod_list = []

            d_sections_energy_emissions_d_ghg_emisssions = {ghg: [] for ghg in GlossaryCore.GreenHouseGases}
            #d_sections_non_energy_emissions_d_gdp_sections = {ghg: [] for ghg in GlossaryCore.GreenHouseGases}
            for section in GlossaryCore.SectionDictSectors[sector]:

                d_section_energy_emissions_d_energy_prod = self.emissions_model.d_section_energy_emissions_d_user_input(section_name=section, sector_name=sector, d_carbon_intensity_d_user_input=d_carbon_intensity_d_energy_prod)
                d_energy_emissions_sections_d_energy_prod_list.append(d_section_energy_emissions_d_energy_prod)

                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.EnergyEmissions),
                    (f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}", section),
                    d_section_energy_emissions_d_section_energy_consumption)

                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.TotalEmissions),
                    (f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}", section),
                    d_section_energy_emissions_d_section_energy_consumption)

                d_sector_section_non_energy_emissions_d_section_gdp = self.emissions_model.d_section_non_energy_emissions_d_gdp_section(sector=sector, section=section)

                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.NonEnergyEmissions),
                    (f"{sector}.{GlossaryCore.SectionGdpDfValue}", section),
                    d_sector_section_non_energy_emissions_d_section_gdp)
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.insertGHGTotalEmissions.format(GlossaryCore.CO2)),
                    (f"{sector}.{GlossaryCore.SectionGdpDfValue}", section),
                    d_sector_section_non_energy_emissions_d_section_gdp)
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.TotalEmissions),
                    (f"{sector}.{GlossaryCore.SectionGdpDfValue}", section),
                    d_sector_section_non_energy_emissions_d_section_gdp)

                for ghg in GlossaryCore.GreenHouseGases:
                    d_sections_energy_emissions_d_ghg_emisssions[ghg].append(self.emissions_model.d_section_energy_emissions_d_user_input(section_name=section, sector_name=sector, d_carbon_intensity_d_user_input=d_energy_carbon_intensity_d_ghg_total_emissions[ghg]))
                    self.set_partial_derivative_for_other_types(
                        (f"{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}", section),
                        ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                        d_sections_energy_emissions_d_ghg_emisssions[ghg][-1])

            d_sector_energy_emissions_d_energy_prod[sector] = np.sum(d_energy_emissions_sections_d_energy_prod_list, axis=0)

            for ghg in GlossaryCore.GreenHouseGases:
                d_sector_energy_emissions_d_ghg_emissions[sector][ghg] = np.sum(d_sections_energy_emissions_d_ghg_emisssions[ghg], axis=0)

        if self.emissions_model.economic_sectors_except_agriculture:
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.EnergyEmissions),
                (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
                np.sum(list(d_sector_energy_emissions_d_energy_prod.values()), axis=0))

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.TotalEmissions),
                (GlossaryCore.EnergyProductionValue, GlossaryCore.TotalProductionValue),
                np.sum(list(d_sector_energy_emissions_d_energy_prod.values()), axis=0))

            for ghg in GlossaryCore.GreenHouseGases:
                temp = [d_sector_energy_emissions_d_ghg_emissions[sector][ghg] for sector in self.emissions_model.economic_sectors_except_agriculture]
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.EnergyEmissions),
                    ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                    np.sum(temp, axis=0))
                self.set_partial_derivative_for_other_types(
                    (GlossaryCore.EconomicsEmissionDfValue, GlossaryCore.TotalEmissions),
                    ('GHG_total_energy_emissions', GlossaryCore.insertGHGTotalEmissions.format(ghg)),
                    np.sum(temp, axis=0))

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['Global Warming Potential (20-year basis) emissions per GHG',
                      'Global Warming Potential (100-year)',
                      'Global Warming Potential (20-year)',

                      'Global Warming Potential (100-year basis) emissions per source'
                      'Global Warming Potential (20-year basis) emissions per source'
                      'Total CO2 emissions',
                      GlossaryCore.EnergyCarbonIntensityDfValue]

        selected_values = ['Global Warming Potential (20-year basis) emissions per GHG',
                           'Global Warming Potential (100-year)',
                           'Global Warming Potential (100-year basis) emissions per source'
                           'Total CO2 emissions', ]

        chart_filters.append(ChartFilter(
            'Charts', chart_list, selected_values, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        charts = []
        if chart_filters is not None:
            for chart_filter in chart_filters:
                charts = chart_filter.selected_values

        if 'Global Warming Potential (100-year)' in charts:
            new_chart = self.get_chart_gwp(100)
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Global Warming Potential (20-year)' in charts:
            new_chart = self.get_chart_gwp(20)
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Global Warming Potential (20-year basis) emissions per GHG' in charts:
            for gwp in GHGEmissions.GHG_TYPE_LIST:
                new_chart = self.get_chart_emission_per_source(gwp)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)

        if 'Global Warming Potential (100-year basis) emissions per source' in charts:
            new_chart = self.get_chart_gwp_per_source(100)
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if 'Global Warming Potential (20-year basis) emissions per source' in charts:
            new_chart = self.get_chart_gwp_per_source(20)
            if new_chart is not None:
                instanciated_charts.append(new_chart)

        if GlossaryCore.EnergyCarbonIntensityDfValue in charts:
            carbon_intensity_df = self.get_sosdisc_outputs(GlossaryCore.EnergyCarbonIntensityDfValue)

            chart_name = "Carbon equivalent intensity of energy mix"
            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                GlossaryCore.EnergyCarbonIntensityDf['unit'], chart_name=chart_name)

            new_serie = InstanciatedSeries(list(carbon_intensity_df[GlossaryCore.Years].values),
                                           list(carbon_intensity_df[GlossaryCore.EnergyCarbonIntensityDfValue].values),
                                           GlossaryCore.EnergyCarbonIntensityDfValue,
                                           'lines')

            new_chart.series.append(new_serie)
            instanciated_charts.append(new_chart)

        return instanciated_charts

    def get_chart_gwp(self, gwp_year):
        GWP_emissions = self.get_sosdisc_outputs(
            GlossaryCore.TotalGWPEmissionsDfValue)

        chart_name = f'Global warming potential at {gwp_year} years - GHG breakdown'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, f"GWP {GlossaryCore.GWPEmissionsDf['unit']}", chart_name=chart_name, stacked_bar=True)

        for ghg in GHGEmissions.GHG_TYPE_LIST:
            new_serie = InstanciatedSeries(list(GWP_emissions[GlossaryCore.Years].values), list(GWP_emissions[f'{ghg}_{gwp_year}'].values),
                                           ghg, 'bar')

            new_chart.series.append(new_serie)

        new_serie = InstanciatedSeries(list(GWP_emissions[GlossaryCore.Years].values),
                                       list(GWP_emissions[f'Total GWP ({gwp_year}-year basis)'].values),
                                       "Total", 'lines')

        new_chart.series.append(new_serie)

        return new_chart

    def get_chart_emission_per_source(self, ghg):
        GHG_emissions_detail_df = self.get_sosdisc_outputs(
            GlossaryCore.GHGEmissionsDetailedDfValue)

        chart_name = f'{ghg} emissions per source'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, f"{ghg} Emissions [{GlossaryCore.GHGEmissionsDf['unit']}]", chart_name=chart_name, stacked_bar=True)

        emission_type = [GlossaryCore.insertGHGAgriLandEmissions,
                         GlossaryCore.insertGHGEnergyEmissions,
                         GlossaryCore.insertGHGNonEnergyEmissions]
        for e_t in emission_type:
            new_serie = InstanciatedSeries(
                list(GHG_emissions_detail_df[GlossaryCore.Years].values),
                list(GHG_emissions_detail_df[e_t.format(ghg)].values), e_t.format(ghg), 'bar')

            new_chart.series.append(new_serie)

        new_serie = InstanciatedSeries(
            list(GHG_emissions_detail_df[GlossaryCore.Years].values),
            list(GHG_emissions_detail_df[GlossaryCore.insertGHGTotalEmissions.format(ghg)].values),
            GlossaryCore.insertGHGTotalEmissions.format(ghg),
            'lines')

        new_chart.series.append(new_serie)

        return new_chart

    def get_chart_gwp_per_source(self, gwp_year):
        GWP_emissions = self.get_sosdisc_outputs(GlossaryCore.TotalGWPEmissionsDfValue)

        chart_name = f'Global warming potential ({gwp_year}-year basis) breakdown per source'
        new_chart = TwoAxesInstanciatedChart(
            GlossaryCore.Years, f"GWP {GlossaryCore.GWPEmissionsDf['unit']}", chart_name=chart_name, stacked_bar=True)

        for sector in [GlossaryCore.AgricultureAndLandUse, GlossaryCore.Energy, GlossaryCore.NonEnergy]:
            new_serie = InstanciatedSeries(list(GWP_emissions[GlossaryCore.Years].values),
                                           list(GWP_emissions[f'{sector}_{gwp_year}'].values),
                                           sector, 'bar')

            new_chart.series.append(new_serie)

        new_serie = InstanciatedSeries(list(GWP_emissions[GlossaryCore.Years].values),
                                       list(GWP_emissions[f'Total GWP ({gwp_year}-year basis)'].values),
                                       'Total', 'lines')
        new_chart.series.append(new_serie)
        return new_chart

