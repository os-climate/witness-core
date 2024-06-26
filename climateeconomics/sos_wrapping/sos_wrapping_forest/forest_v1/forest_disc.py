"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_forest.forest_v1 import Forest
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class ForestDiscipline(ClimateEcoDiscipline):
    """Forest discipline"""

    # ontology information
    _ontology_data = {
        "label": "Forest",
        "type": "",
        "source": "",
        "validated": "",
        "validated_by": "",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fas fa-tree fa-fw",
        "version": "Version 1",
    }
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050

    deforestation_limit = 1000
    initial_emissions = 3.21

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        Forest.DEFORESTATION_SURFACE: {
            "type": "dataframe",
            "unit": "Mha",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                "deforested_surface": ("float", [0, 1e9], True),
            },
            "dataframe_edition_locked": False,
            "visibility": ClimateEcoDiscipline.SHARED_VISIBILITY,
            "namespace": GlossaryCore.NS_WITNESS,
        },
        Forest.LIMIT_DEFORESTATION_SURFACE: {
            "type": "float",
            "unit": "Mha",
            "default": deforestation_limit,
            "namespace": "ns_forest",
        },
        Forest.INITIAL_CO2_EMISSIONS: {
            "type": "float",
            "unit": "GtCO2",
            "default": initial_emissions,
            "namespace": "ns_forest",
        },
        Forest.CO2_PER_HA: {"type": "float", "unit": "kgCO2/ha/year", "default": 4000, "namespace": "ns_forest"},
        Forest.REFORESTATION_COST_PER_HA: {"type": "float", "unit": "$/ha", "default": 3800, "namespace": "ns_forest"},
        Forest.REFORESTATION_INVESTMENT: {
            "type": "dataframe",
            "unit": "G$",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                "forest_investment": ("float", [0, 1e9], True),
            },
            "dataframe_edition_locked": False,
            "visibility": ClimateEcoDiscipline.SHARED_VISIBILITY,
            "namespace": "ns_invest",
        },
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
    }

    DESC_OUT = {
        GlossaryCore.CO2EmissionsDetailDfValue: {"type": "dataframe", "unit": "GtCO2", "namespace": "ns_forest"},
        Forest.FOREST_SURFACE_DF: {
            "type": "dataframe",
            "unit": "Gha",
            "visibility": ClimateEcoDiscipline.SHARED_VISIBILITY,
            "namespace": GlossaryCore.NS_WITNESS,
        },
        Forest.FOREST_DETAIL_SURFACE_DF: {"type": "dataframe", "unit": "Gha"},
        Forest.CO2_EMITTED_FOREST_DF: {
            "type": "dataframe",
            "unit": "GtCO2",
            "visibility": ClimateEcoDiscipline.SHARED_VISIBILITY,
            "namespace": GlossaryCore.NS_WITNESS,
        },
    }

    FOREST_CHARTS = "Forest chart"

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.forest_model = Forest(param)

    def run(self):

        # -- get inputs
        #         inputs = list(self.DESC_IN.keys())
        #         inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # -- compute
        in_dict = self.get_sosdisc_inputs()

        self.forest_model.compute(in_dict)

        outputs_dict = {
            Forest.CO2_EMITTED_DETAIL_DF: self.forest_model.CO2_emitted_df,
            Forest.FOREST_DETAIL_SURFACE_DF: self.forest_model.forest_surface_df,
            Forest.FOREST_SURFACE_DF: self.forest_model.forest_surface_df[
                [GlossaryCore.Years, "forest_surface_evol", "global_forest_surface"]
            ],
            Forest.CO2_EMITTED_FOREST_DF: self.forest_model.CO2_emitted_df[
                [GlossaryCore.Years, "emitted_CO2_evol_cumulative"]
            ],
        }

        # -- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        in_dict = self.get_sosdisc_inputs()
        self.forest_model.compute(in_dict)

        # gradient for deforestation rate
        d_deforestation_surface_d_deforestation_surface = (
            self.forest_model.d_deforestation_surface_d_deforestation_surface()
        )
        d_cum_deforestation_d_deforestation_surface = self.forest_model.d_cum(
            d_deforestation_surface_d_deforestation_surface
        )
        d_forest_surface_d_invest = self.forest_model.d_forestation_surface_d_invest()
        d_cun_forest_surface_d_invest = self.forest_model.d_cum(d_forest_surface_d_invest)

        # forest surface vs deforestation grad
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, "forest_surface_evol"),
            (Forest.DEFORESTATION_SURFACE, "deforested_surface"),
            d_deforestation_surface_d_deforestation_surface,
        )
        #         self.set_partial_derivative_for_other_types(
        #             (Forest.FOREST_SURFACE_DF,
        #              'forest_surface_evol_cumulative'),
        #             (Forest.DEFORESTATION_SURFACE, 'deforested_surface'),
        #             d_cum_deforestation_d_deforestation_surface)

        # forest surface vs forest invest
        self.set_partial_derivative_for_other_types(
            (Forest.FOREST_SURFACE_DF, "forest_surface_evol"),
            (Forest.REFORESTATION_INVESTMENT, "forest_investment"),
            d_forest_surface_d_invest,
        )
        #         self.set_partial_derivative_for_other_types(
        #             (Forest.FOREST_SURFACE_DF,
        #              'forest_surface_evol_cumulative'),
        #             (Forest.REFORESTATION_INVESTMENT, 'forest_investment'),
        #             d_cun_forest_surface_d_invest)

        # d_CO2 d deforestation
        d_CO2_emitted_d_deforestation_surface = self.forest_model.d_CO2_emitted(
            d_deforestation_surface_d_deforestation_surface
        )
        d_cum_CO2_emitted_d_deforestation_surface = self.forest_model.d_cum(d_CO2_emitted_d_deforestation_surface)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, "emitted_CO2_evol_cumulative"),
            (Forest.DEFORESTATION_SURFACE, "deforested_surface"),
            d_cum_CO2_emitted_d_deforestation_surface,
        )

        # d_CO2 d invest
        d_CO2_emitted_d_invest = self.forest_model.d_CO2_emitted(d_forest_surface_d_invest)
        d_cum_CO2_emitted_d_invest = self.forest_model.d_cum(d_CO2_emitted_d_invest)

        self.set_partial_derivative_for_other_types(
            (Forest.CO2_EMITTED_FOREST_DF, "emitted_CO2_evol_cumulative"),
            (Forest.REFORESTATION_INVESTMENT, "forest_investment"),
            d_cum_CO2_emitted_d_invest,
        )

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [ForestDiscipline.FOREST_CHARTS]

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter("Charts filter", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        """
        For the outputs, making a graph for tco vs year for each range and for specific
        value of ToT with a shift of five year between then
        """
        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == "charts":
                    chart_list = chart_filter.selected_values

        if ForestDiscipline.FOREST_CHARTS in chart_list:

            forest_surface_df = self.get_sosdisc_outputs(Forest.FOREST_DETAIL_SURFACE_DF)
            years = forest_surface_df[GlossaryCore.Years].values.tolist()
            # values are *1000 to convert from Gha to Mha
            surface_evol_by_year = forest_surface_df["forest_surface_evol"].values * 1000
            surface_evol_cum = forest_surface_df["forest_surface_evol_cumulative"].values * 1000
            deforested_surface_by_year = forest_surface_df["deforested_surface"].values * 1000
            deforested_surface_cum = forest_surface_df["deforested_surface_cumulative"].values * 1000
            forested_surface_by_year = forest_surface_df["forested_surface"].values * 1000
            forested_surface_cum = forest_surface_df["forested_surface_cumulative"].values * 1000

            # forest evolution year by year chart
            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "Forest surface evolution [Mha / year]",
                chart_name="Forest surface evolution",
                stacked_bar=True,
            )

            deforested_series = InstanciatedSeries(years, deforested_surface_by_year.tolist(), "Deforestation", "bar")
            forested_series = InstanciatedSeries(years, forested_surface_by_year.tolist(), "Reforestation", "bar")
            total_series = InstanciatedSeries(
                years, surface_evol_by_year.tolist(), "Surface evolution", InstanciatedSeries.LINES_DISPLAY
            )

            new_chart.add_series(deforested_series)
            new_chart.add_series(total_series)
            new_chart.add_series(forested_series)

            instanciated_charts.append(new_chart)

            # forest cumulative evolution chart
            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "Cumulative forest surface evolution [Mha]",
                chart_name="Cumulative forest surface evolution",
                stacked_bar=True,
            )

            deforested_series = InstanciatedSeries(years, deforested_surface_cum.tolist(), "Deforested surface", "bar")
            forested_series = InstanciatedSeries(years, forested_surface_cum.tolist(), "Forested surface", "bar")
            total_series = InstanciatedSeries(
                years, surface_evol_cum.tolist(), "Surface evolution", InstanciatedSeries.LINES_DISPLAY
            )
            new_chart.add_series(deforested_series)
            new_chart.add_series(total_series)
            new_chart.add_series(forested_series)

            instanciated_charts.append(new_chart)

            # CO2 graph

            CO2_emissions_df = self.get_sosdisc_outputs(GlossaryCore.CO2EmissionsDetailDfValue)
            CO2_emitted_year_by_year = CO2_emissions_df["emitted_CO2"]
            CO2_captured_year_by_year = CO2_emissions_df["captured_CO2"]
            CO2_total_year_by_year = CO2_emissions_df["emitted_CO2_evol"]
            CO2_emitted_cum = CO2_emissions_df["emitted_CO2_cumulative"]
            CO2_captured_cum = CO2_emissions_df["captured_CO2_cumulative"]
            CO2_total_cum = CO2_emissions_df["emitted_CO2_evol_cumulative"]

            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "CO2 emission & capture [GtCO2 / year]",
                chart_name="Yearly forest delta CO2 emissions",
                stacked_bar=True,
            )

            CO2_emitted_series = InstanciatedSeries(
                years, CO2_emitted_year_by_year.tolist(), "CO2 emissions", InstanciatedSeries.BAR_DISPLAY
            )
            CO2_captured_series = InstanciatedSeries(
                years, CO2_captured_year_by_year.tolist(), "CO2 capture", InstanciatedSeries.BAR_DISPLAY
            )
            CO2_total_series = InstanciatedSeries(
                years, CO2_total_year_by_year.tolist(), "CO2 evolution", InstanciatedSeries.LINES_DISPLAY
            )

            new_chart.add_series(CO2_emitted_series)
            new_chart.add_series(CO2_total_series)
            new_chart.add_series(CO2_captured_series)
            instanciated_charts.append(new_chart)

            # in Gt
            new_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years,
                "CO2 emission & capture [GtCO2]",
                chart_name="Forest CO2 emissions",
                stacked_bar=True,
            )
            CO2_emitted_series = InstanciatedSeries(
                years, CO2_emitted_cum.tolist(), "CO2 emissions", InstanciatedSeries.BAR_DISPLAY
            )
            CO2_captured_series = InstanciatedSeries(
                years, CO2_captured_cum.tolist(), "CO2 capture", InstanciatedSeries.BAR_DISPLAY
            )
            CO2_total_series = InstanciatedSeries(
                years, CO2_total_cum.tolist(), "CO2 evolution", InstanciatedSeries.LINES_DISPLAY, custom_data=["width"]
            )

            new_chart.add_series(CO2_emitted_series)
            new_chart.add_series(CO2_total_series)
            new_chart.add_series(CO2_captured_series)
            instanciated_charts.append(new_chart)
        return instanciated_charts
