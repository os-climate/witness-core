'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2025/01/10 Copyright 2025 Capgemini

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

from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.core.core_forest.forest import ForestAutodiff
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class ForestDiscipline(AutodifferentiedDisc):
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
        'version': 'Version 0',
    }

    coupling_inputs = [
        'deforestation_investment',
        'reforestation_investment',
        GlossaryCore.CropProductivityReductionName,
        'managed_wood_investment',
    ]
    coupling_outputs = [
        'forest_surface_df',
        'land_use_required',
        'CO2_land_emission_df',
        'CO2_emissions',
        'techno_production',
        'techno_consumption',
        GlossaryCore.TechnoConsumptionWithoutRatioValue,
        'techno_prices',
        'forest_lost_capital',
        'forest_lost_capital',
    ]
    AGRI_CAPITAL_TECHNO_LIST = []
    biomass_cal_val = BiomassDry.data_energy_dict[
        'calorific_value']
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050

    deforestation_limit = 1000
    initial_emissions = 3.21
    construction_delay = 3

    # 31% of land area is covered by Forest = 4.06 GHa in 2020 (https://www.fao.org/state-of-forests/en/)
    # www.fao.org : forest under long-term management plans = 2.05 Billion Ha
    # 31% of All forests is used for production : 0.31 * 4.06 = 1.25
    # other source: 1150MHa of forest are production forest (https://research.wri.org/gfr/forest-designation-indicators/production-forests#how-much-production-forest-exists-globally)
    # 92% of the production come from managed wood. 8% from unmanaged wood
    # GHa
    wood_production_surface = 1.15  # GHa

    # Roundwood demand for energy FAO in the world see fao.org (2020) = 1697 (non coniferous) + 229 (coniferous) = 1926 Mm3
    energy_wood_production_2020 = 1697 + 229  # Mm3
    # Roundwood demand for industry FAO in the world see fao.org (2020) = 1984 Mm3 ; Total = 1926 + 1984 = 3910 Mm3
    industry_wood_production_2020 = 1984  # Mm3
    # then % of wood for energy = 1926/(1926+1984)= 49.2 %
    total_wood_production_2020 = energy_wood_production_2020 + industry_wood_production_2020  # 3910 Mm3
    wood_percentage_for_energy = energy_wood_production_2020 / total_wood_production_2020  # 49.2%
    # FAO 2020 : Chips = 262 Mm3, Residues = 233Mm3, Total Wood fuel : 1926 Mm3
    # % of residues  + chips = (233+262)/1926 = 25.7%
    residues_wood_production = 233 + 262  # Mm3
    residue_percentage = residues_wood_production / total_wood_production_2020  # 12.6%
    plantation_forest_surface_mha_2020 = 131.
    plantation_forest_supply_Mm3_2020 = 654.
    # Based on FAO, 2020 plantations forest are 131MHa and supply 654Mm3
    # then managed yield or plantation forest yield (which are forest where you invest in for managed wood) is : 654/131 m3/Ha
    # A fraction of the managed wood surface is harvested (and replanted) every year to keep a constant managed forest surface
    # (when managed forest invest = 0).
    managed_yield = plantation_forest_supply_Mm3_2020 / plantation_forest_surface_mha_2020  # 4.99 m3/ha
    # However actually roundwood production is not only plantation forests, then the yield is lower and can be computed with 2020 data (FAO)
    # 1.15GHa supply the total roundwood production which is 3910Mm3 in 2020 https://openknowledge.fao.org/server/api/core/bitstreams/5da0482f-d8b2-44e3-9cbb-8e9412b4ea86/content
    actual_yield = total_wood_production_2020 * 1e6 / (wood_production_surface * 1e9)  # 3.4 m3/ha
    # Deforested surfaces are either replaced by crop or pasture (all trees have been harvested and not replanted) or by tree plantations (for logging, soy, palm oil)
    unmanaged_yield = (total_wood_production_2020 - plantation_forest_supply_Mm3_2020) * 1e6 / (wood_production_surface * 1e9 - plantation_forest_surface_mha_2020 * 1e6)  # 3.19 m3/ha

    # reference:
    # https://qtimber.daf.qld.gov.au/guides/wood-density-and-hardness
    wood_density = 600   # kg/m3
    residues_density = 200  # kg/m3

    wood_techno_dict = {'maturity': 5,
                        'wood_residues_moisture': 0.35,  # 35% moisture content
                        # afforestation costs = monitoring and maintenance costs + establishment costs + harvesting costs = 500+2000+1000 $/Ha
                        'managed_wood_price_per_ha': 3500.,
                        'Price_per_ha_unit': '$/ha',  # in 2019, date of the paper
                        'residues_density_percentage': residue_percentage,
                        'wood_percentage_for_energy': wood_percentage_for_energy,
                        'actual_yield_year_start': actual_yield,
                        'managed_wood_yield_year_start': managed_yield,
                        'unmanaged_wood_yield_year_start': unmanaged_yield,
                        'yield_unit': 'm3/Ha',
                        'density_unit': 'm^3/ha',
                        'deforestation_cost_per_ha': 8000,
                        'reforestation_cost_per_ha': 13800,
                        'CO2_per_ha': 4000,
                        'wood_density': wood_density,
                        'biomass_dry_calorific_value': BiomassDry.data_energy_dict['calorific_value'],
                        'biomass_dry_high_calorific_value': BiomassDry.data_energy_dict['high_calorific_value'],
                        'residues_density': residues_density,
                        'residue_calorific_value': 5.61,  # 4.356,
                        'residue_calorific_value_unit': 'kWh/kg',
                        GlossaryCore.ConstructionDelay: construction_delay,
                        'WACC': 0.07,
                        # CO2 from production from tractor is taken
                        # into account into the energy net factor
                        # land CO2 absorption is computed in land_emission with
                        # the CO2_per_ha parameter
                        'CO2_from_production': - 0.425 * 44.01 / 12.0,
                        'CO2_from_production_unit': 'kg/kg'}

    # invest: 0.19 Mha are planted each year at 13047.328euro/ha, and 28% is
    # the share of wood (not residue)
    invest_before_year_start = DatabaseWitnessCore.get_reforestation_invest_before_year_start(year_start=GlossaryCore.YearStartDefault, construction_delay=construction_delay)[0]

    # protected forest are 21% of total forest
    # https://research.wri.org/gfr/forest-designation-indicators/protected-forests
    # FAO states 18% of total forest, namely 0.7GHa (https://www.fao.org/state-of-forests/en/)
    initial_protected_forest_surface = 4 * 0.21
    initial_unmanaged_forest_surface = 4 - 1.25 - initial_protected_forest_surface

    # reforestation costs: 10k$/ha of land and 3800$/ha to plant trees

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        'deforestation_investment': {'type': 'dataframe', 'unit': 'G$',
                                          'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                   GlossaryCore.InvestmentsValue: (
                                                                       'float', [0, 1e9], True)},
                                          'dataframe_edition_locked': False,
                                          'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                          'namespace': 'ns_forest'},
        'initial_co2_emissions': {'type': 'float', 'unit': 'GtCO2'},
        'co2_per_ha': {'type': 'float', 'unit': 'kgCO2/ha/year', 'default': 4000, 'namespace': 'ns_forest'},
        'reforestation_cost_per_ha': {'type': 'float', 'unit': '$/ha', 'default': 13800,
                                           'namespace': 'ns_forest'},
        'reforestation_investment': {'type': 'dataframe', 'unit': 'G$',
                                          'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                   'reforestation_investment': ('float', [0, 1e9], True)},
                                          'dataframe_edition_locked': False,
                                          'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                          'namespace': 'ns_invest'},
        'params': {'type': 'dict', 'unit': '-', 'default': wood_techno_dict,
                                  'namespace': 'ns_forest'},
        'managed_wood_initial_surface': {'type': 'float', 'unit': 'Gha', 'default': wood_production_surface,
                                    'namespace': 'ns_forest'},
        'managed_wood_invest_before_year_start': {'type': 'dataframe', 'unit': 'G$',
                                             'dataframe_descriptor': {GlossaryCore.InvestmentsValue: ('float', [0, 1e9], True)},
                                             'dataframe_edition_locked': False,
                                             'default': invest_before_year_start,
                                             'namespace': 'ns_forest'},
        'managed_wood_investment': {'type': 'dataframe', 'unit': 'G$',
                               'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                        GlossaryCore.InvestmentsValue: ('float', [0, 1e9], True)},
                               'dataframe_edition_locked': False,
                               'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_forest'},
        'transport_cost': {'type': 'dataframe', 'unit': '$/t', 'namespace': GlossaryCore.NS_WITNESS,
                                'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                         'transport': ('float', [0, 1e9], True)},
                                'dataframe_edition_locked': False},
        'margin': {'type': 'dataframe', 'unit': '%', 'namespace': GlossaryCore.NS_WITNESS,
                        'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                        'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                 'margin': ('float', [0, 1e9], True)},
                        'dataframe_edition_locked': False},
        'initial_unmanaged_forest_surface': {'type': 'float', 'unit': 'Gha', 'default': initial_unmanaged_forest_surface,
                                  'namespace': 'ns_forest'},
        'initial_protected_forest_surface': {'type': 'float', 'unit': 'Gha', 'default': initial_protected_forest_surface,
                                  'namespace': 'ns_forest'},
        'scaling_factor_techno_consumption': {'type': 'float', 'default': 1e3, 'unit': '-',
                                              'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                              'namespace': 'ns_public', 'user_level': 2},
        'scaling_factor_techno_production': {'type': 'float', 'default': 1e3, 'unit': '-',
                                             'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                             'namespace': 'ns_public', 'user_level': 2},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.CropProductivityReductionName: GlossaryCore.CropProductivityReductionDf
    }
    subsector_production_df = GlossaryCore.get_dynamic_variable(GlossaryCore.SubsectorProductionDf)
    subsector_production_df["namespace"] = GlossaryCore.NS_AGRI

    damages_df = GlossaryCore.get_dynamic_variable(GlossaryCore.SubsectorDamagesDf)
    damages_df["namespace"] = GlossaryCore.NS_AGRI
    DESC_OUT = {
        'CO2_emissions_detail_df': {
            'type': 'dataframe', 'unit': 'GtCO2', 'namespace': 'ns_forest'},
        'forest_surface_detail_df': {
            'type': 'dataframe', 'unit': 'Gha'},
        'forest_surface_df': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': GlossaryCore.NS_WITNESS},
        'CO2_land_emission_df': {
            'type': 'dataframe', 'unit': 'GtCO2', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'biomass_dry_df': {
            'type': 'dataframe', 'unit': '-', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': GlossaryCore.NS_WITNESS},
        'managed_wood_df': {
            'type': 'dataframe', 'unit': 'Gha', 'namespace': 'ns_forest'},
        'biomass_dry_detail_df': {
            'type': 'dataframe', 'unit': '-', 'namespace': 'ns_forest'},

        'techno_production': {
            'type': 'dataframe', 'unit': 'TWh or Mt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'techno_prices': {
            'type': 'dataframe', 'unit': '$/MWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'techno_consumption': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_forest',
            'unit': 'TWh or Mt'},
        'techno_consumption_woratio': {
            'type': 'dataframe', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_forest',
            'unit': 'TWh or Mt',
        },
        'land_use_required': {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'CO2_emissions': {
            'type': 'dataframe', 'unit': 'kg/kWh', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'forest_lost_capital': {
            'type': 'dataframe', 'unit': 'G$', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
            'namespace': 'ns_forest'},
        'yields': {'type': 'dataframe', 'unit': 'm^3/Ha', 'description': 'evolution of yields. Yields are affected by temperature change'},
        f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}": subsector_production_df,
        f"{GlossaryCore.Forest}.{GlossaryCore.DamageDfValue}": damages_df,
        GlossaryCore.DamageDetailedDfValue: {'type': 'dataframe', 'unit': 'G$', 'description': 'Economical damages details.'},
        GlossaryCore.EconomicsDetailDfValue: {'type': 'dataframe', 'unit': 'G$', 'description': 'Net economical output details.'},
    }

    FOREST_CHARTS = 'Forest chart'

    def setup_sos_disciplines(self):
        self.update_default_values()

    def update_default_values(self):
        disc_in = self.get_data_in()
        if disc_in is not None and GlossaryCore.YearStart in disc_in:
            year_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
            if year_start is not None:
                self.update_default_value('initial_co2_emissions', 'in', DatabaseWitnessCore.ForestEmissions.get_value_at_year(year_start))

    def init_execution(self):
        self.model = ForestAutodiff()

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [ForestDiscipline.FOREST_CHARTS,
                      "Economical output",
                      "Investments",
                      "Surface",
                      "Biomass price"
                      "Emissions",
                      "Capital",
                      "Damages"]

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
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        forest_surface_df = self.get_sosdisc_outputs('forest_surface_detail_df')
        managed_wood_df = self.get_sosdisc_outputs('managed_wood_df')
        years = managed_wood_df[GlossaryCore.Years]
        if "Economical output" in chart_list:
            economical_output_df = self.get_sosdisc_outputs(f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}")
            economical_damages_df = self.get_sosdisc_outputs(f"{GlossaryCore.Forest}.{GlossaryCore.DamageDfValue}")
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SubsectorProductionDf['unit'], chart_name='Economical output of forestry', stacked_bar=True)

            for col in economical_output_df.columns:
                if col != GlossaryCore.Years:
                    new_chart.add_series(InstanciatedSeries(years, economical_output_df[col], pimp_string(col), "lines"))

            new_chart.add_series(InstanciatedSeries(years, -economical_damages_df[GlossaryCore.Damages], "Damages", "bar"))
            new_chart.post_processing_section_name = "Economical output"
            instanciated_charts.append(new_chart)

        if "Economical output" in chart_list:
            economical_detail = self.get_sosdisc_outputs(GlossaryCore.EconomicsDetailDfValue)
            economical_output_df = self.get_sosdisc_outputs(f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}")
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SubsectorProductionDf['unit'], chart_name='Net economical output breakdown', stacked_bar=True)

            for col in economical_detail.columns:
                if col != GlossaryCore.Years:
                    new_chart.add_series(InstanciatedSeries(years, economical_detail[col], pimp_string(col), "bar"))

            new_chart.add_series(InstanciatedSeries(years, economical_output_df[GlossaryCore.OutputNetOfDamage], "Total", "lines"))
            new_chart.post_processing_section_name = "Economical output"
            instanciated_charts.append(new_chart)

        if "Damages" in chart_list:
            economical_damages_df = self.get_sosdisc_outputs(f"{GlossaryCore.Forest}.{GlossaryCore.DamageDfValue}")
            economical_damages_df_detailed = self.get_sosdisc_outputs(GlossaryCore.DamageDetailedDfValue)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.SubsectorDamagesDf['unit'], chart_name='Economical damages', stacked_bar=True)

            for col in economical_damages_df_detailed.columns:
                if col != GlossaryCore.Years:
                    new_chart.add_series(InstanciatedSeries(years, economical_damages_df_detailed[col], pimp_string(col), "bar"))

            new_chart.add_series(InstanciatedSeries(years, economical_damages_df[GlossaryCore.Damages], "Total", "lines"))
            new_chart.post_processing_section_name = "Damages"
            instanciated_charts.append(new_chart)

        if "Damages" in chart_list:
            yields_df = self.get_sosdisc_outputs('yields')
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'm^3/Ha', chart_name='Yields')

            for col in yields_df.columns:
                if col != GlossaryCore.Years:
                    new_chart.add_series(InstanciatedSeries(years, yields_df[col], pimp_string(col), "lines"))

            new_chart.post_processing_section_name = "Damages"
            instanciated_charts.append(new_chart)

        if "Damages" in chart_list:
            crop_productivity_reduction = self.get_sosdisc_inputs(GlossaryCore.CropProductivityReductionName)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.CropProductivityReductionDf['unit'], chart_name='Yields variation due to climate change')

            for col in crop_productivity_reduction.columns:
                if col != GlossaryCore.Years:
                    new_chart.add_series(InstanciatedSeries(years, - crop_productivity_reduction[col], pimp_string(col), "lines"))

            new_chart.post_processing_section_name = "Damages"
            instanciated_charts.append(new_chart)

        if "Investments" in chart_list:
            reforestation_investment_df = self.get_sosdisc_inputs('reforestation_investment')
            managed_wood_investment_df = self.get_sosdisc_inputs('managed_wood_investment')
            deforestation_investment_df = self.get_sosdisc_inputs('deforestation_investment')
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Investments [G$]',
                                                 chart_name='Investments in forests activities', stacked_bar=True)

            new_chart.add_series(InstanciatedSeries(years, reforestation_investment_df['reforestation_investment'], 'Reforestation invests', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, managed_wood_investment_df[GlossaryCore.InvestmentsValue], 'Managed wood invests', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, deforestation_investment_df[GlossaryCore.InvestmentsValue], 'Deforestation invests', 'bar'))
            new_chart.post_processing_section_name = "Investments & Capital"
            instanciated_charts.append(new_chart)

        if "Surface" in chart_list:
            # values are *1000 to convert from Gha to Mha
            delta_reforestation = forest_surface_df['delta_reforestation_surface'].values * 1e3
            reforestation = forest_surface_df['reforestation_surface'].values * 1e3

            delta_deforestation = forest_surface_df['delta_deforestation_surface'].values * 1e3
            deforestation = forest_surface_df['deforestation_surface'].values * 1e3

            delta_managed_wood_surface = managed_wood_df['delta_surface'].values * 1e3
            managed_wood_surface = managed_wood_df['cumulative_surface'].values * 1e3

            delta_global = forest_surface_df['delta_global_forest_surface'].values * 1e3
            global_surface = forest_surface_df['global_forest_surface'].values * 1e3

            unmanaged_forest = forest_surface_df['unmanaged_forest'].values * 1e3
            protected_forest = forest_surface_df['protected_forest_surface'].values * 1e3

            # forest evolution year by year chart
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                 'Yearly delta of forest surface evolution [Mha]',
                                                 chart_name='Yearly delta of forest surface evolution',
                                                 stacked_bar=True)

            new_chart.add_series(InstanciatedSeries(years, delta_deforestation.tolist(), 'Deforestation', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, delta_managed_wood_surface.tolist(), 'Managed wood', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, delta_global.tolist(), 'Global forest surface', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, delta_reforestation.tolist(), 'Reforestation', 'bar'))

            new_chart.post_processing_section_name = "Surface"
            instanciated_charts.append(new_chart)

            # forest cumulative evolution chart
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Forest surface evolution [Mha]',
                                                 chart_name='Global forest surface evolution', stacked_bar=True)

            new_chart.add_series(InstanciatedSeries(years, deforestation.tolist(), 'Deforested surface', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, reforestation.tolist(), 'Reforested surface', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, global_surface.tolist(), 'Forest surface evolution', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, managed_wood_surface.tolist(), 'Managed wood', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, unmanaged_forest.tolist(), 'Unmanaged forest', 'bar'))
            new_chart.add_series(InstanciatedSeries(years, protected_forest.tolist(), 'Protected forest', 'bar'))

            new_chart.post_processing_section_name = "Surface"
            instanciated_charts.append(new_chart)

        if "Emissions" in chart_list:

            CO2_emissions_df = self.get_sosdisc_outputs(GlossaryCore.CO2EmissionsDetailDfValue)
            delta_reforestation = CO2_emissions_df['delta_CO2_reforestation'].values
            reforestation = CO2_emissions_df['CO2_reforestation'].values

            delta_deforestation = CO2_emissions_df['delta_CO2_deforestation'].values
            deforestation = CO2_emissions_df['CO2_deforestation'].values

            init_balance = CO2_emissions_df['initial_CO2_land_use_change'].values

            delta_global = CO2_emissions_df['delta_CO2_emitted'].values
            global_surface = CO2_emissions_df['emitted_CO2_evol_cumulative'].values

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 emission & capture [GtCO2 / year]',
                                                 chart_name='Yearly forest delta CO2 emissions', stacked_bar=True)

            new_chart.add_series(InstanciatedSeries(years, delta_deforestation.tolist(), 'Deforestation emissions', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, delta_reforestation.tolist(), 'Reforestation emissions', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, delta_global.tolist(), 'Global CO2 balance', InstanciatedSeries.LINES_DISPLAY))

            new_chart.post_processing_section_name = "Emissions"
            instanciated_charts.append(new_chart)

            # in Gt
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'CO2 emission & capture [GtCO2]',
                                                 chart_name='Forest CO2 emissions', stacked_bar=True)
            new_chart.add_series(InstanciatedSeries(years, deforestation.tolist(), 'Deforestation emissions', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, reforestation.tolist(), 'Reforestation emissions', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, global_surface.tolist(), 'Global CO2 balance', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, init_balance.tolist(), 'initial forest emissions', InstanciatedSeries.BAR_DISPLAY))

            new_chart.post_processing_section_name = "Emissions"
            instanciated_charts.append(new_chart)

        if "Biomass and energy production" in chart_list:
            # biomass chart
            biomass_dry_df = self.get_sosdisc_outputs('biomass_dry_detail_df')

            # chart biomass dry for energy production
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Biomass dry [Mt]',
                                                 chart_name='Break down of biomass dry production for energy',
                                                 stacked_bar=True)
            mw_residues_energy = managed_wood_df['residues_production_for_energy (Mt)']
            mw_wood_energy = managed_wood_df['wood_production_for_energy (Mt)']
            biomass_dry_energy = biomass_dry_df['biomass_dry_for_energy (Mt)']
            deforestation_energy = biomass_dry_df['deforestation_for_energy']

            new_chart.add_series(InstanciatedSeries(years, mw_residues_energy.tolist(), 'Residues from managed wood', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, mw_wood_energy.tolist(), 'Wood from managed wood', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, deforestation_energy.tolist(), 'Biomass from deforestation', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, biomass_dry_energy.tolist(), 'Total biomass dry produced', InstanciatedSeries.LINES_DISPLAY))

            new_chart.post_processing_section_name = "Biomass and energy production"
            instanciated_charts.append(new_chart)

            # chart biomass dry for energy production
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Biomass dry [TWh]',
                                                 chart_name='Break down of biomass dry production for energy',
                                                 stacked_bar=True)
            mw_residues_energy_twh = managed_wood_df[
                                         'residues_production_for_energy (Mt)'] * ForestDiscipline.biomass_cal_val
            mw_wood_energy_twh = managed_wood_df['wood_production_for_energy (Mt)'] * ForestDiscipline.biomass_cal_val
            biomass_dry_energy_twh = biomass_dry_df['biomass_dry_for_energy (Mt)'] * ForestDiscipline.biomass_cal_val
            deforestation_energy_twh = biomass_dry_df['deforestation_for_energy'] * ForestDiscipline.biomass_cal_val

            new_chart.add_series(InstanciatedSeries(years, mw_residues_energy_twh.tolist(), 'Residues from managed wood', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, mw_wood_energy_twh.tolist(), 'Wood from managed wood', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, deforestation_energy_twh.tolist(), 'Biomass from deforestation', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, biomass_dry_energy_twh.tolist(), 'Total biomass dry produced', InstanciatedSeries.LINES_DISPLAY))

            new_chart.post_processing_section_name = "Biomass and energy production"
            instanciated_charts.append(new_chart)

            # chart total biomass dry production
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Biomass dry [Mt]',
                                                 chart_name='Break down of biomass dry production', stacked_bar=True)
            residues_industry = managed_wood_df[
                'residues_production_for_industry (Mt)'].values
            wood_industry = managed_wood_df['wood_production_for_industry (Mt)'].values
            deforestation_industry = biomass_dry_df['deforestation_for_industry']
            biomass_industry = residues_industry + wood_industry + deforestation_industry
            residues_energy = mw_residues_energy
            wood_energy = mw_wood_energy
            biomass_energy = residues_energy + wood_energy + deforestation_energy

            new_chart.add_series(InstanciatedSeries(years, biomass_industry.tolist(), 'Biomass dedicated to industry', InstanciatedSeries.BAR_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, biomass_energy.tolist(), 'Biomass dedicated to energy', InstanciatedSeries.BAR_DISPLAY))

            new_chart.post_processing_section_name = "Biomass and energy production"
            instanciated_charts.append(new_chart)

        if "Biomass price" in chart_list:
            # biomassdry price per kWh
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Price [$/MWh]',
                                                 chart_name='Biomass dry price evolution', stacked_bar=True)
            biomass_dry_df = self.get_sosdisc_outputs('biomass_dry_detail_df')
            mw_price = biomass_dry_df['managed_wood_price_per_MWh']
            deforestation_price = biomass_dry_df['deforestation_price_per_MWh']
            average_price = biomass_dry_df['price_per_MWh']

            new_chart.add_series(InstanciatedSeries(years, mw_price.tolist(), 'Managed wood', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, average_price.tolist(), 'Biomass dry', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, deforestation_price.tolist(), 'Deforestation', InstanciatedSeries.LINES_DISPLAY))

            new_chart.post_processing_section_name = "Biomass price"
            instanciated_charts.append(new_chart)

            # biomass dry price per ton
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Price [$/ton]',
                                                 chart_name='Biomass dry price evolution', stacked_bar=True)
            mw_price = biomass_dry_df['managed_wood_price_per_ton']
            deforestation_price = biomass_dry_df['deforestation_price_per_ton']
            average_price = biomass_dry_df['price_per_ton']

            new_chart.add_series(InstanciatedSeries(years, mw_price.tolist(), 'Managed wood', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, average_price.tolist(), 'Biomass dry', InstanciatedSeries.LINES_DISPLAY))
            new_chart.add_series(InstanciatedSeries(years, deforestation_price.tolist(), 'Deforestation', InstanciatedSeries.LINES_DISPLAY))

            new_chart.post_processing_section_name = "Biomass price"
            instanciated_charts.append(new_chart)

        if "Capital" in chart_list:
            # lost capital graph
            lost_capital_df = self.get_sosdisc_outputs('forest_lost_capital')
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Lost capital [G$]',
                                                 chart_name='Lost capital due to deforestation', stacked_bar=True)

            new_chart.add_series(InstanciatedSeries(years, lost_capital_df['deforestation'], 'Deforestation Lost Capital', 'bar'))
            new_chart.post_processing_section_name = "Investments & Capital"
            instanciated_charts.append(new_chart)

        return instanciated_charts


def pimp_string(val: str):
    val = val.replace("_", ' ')
    val = val.capitalize()
    return val
