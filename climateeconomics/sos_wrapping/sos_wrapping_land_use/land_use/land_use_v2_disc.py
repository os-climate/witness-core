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
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class LandUseV2Discipline(SoSWrapp):
    ''' Discipline intended to host land use pyworld3 with land use for food input from agriculture pyworld3
    '''

    # ontology information
    _ontology_data = {
        'label': 'Land Use V2 Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-globe-europe fa-fw',
        'version': '',
    }
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050
    initial_unmanaged_forest_surface = 4 - 1.25

    DESC_IN = {GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
               LandUseV2.LAND_DEMAND_DF: {'type': 'dataframe', 'unit': 'Gha',
                                                  'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_land_use',
                                          "dynamic_dataframe_columns": True, },
               LandUseV2.TOTAL_FOOD_LAND_SURFACE: {'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                                                   'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                            'total surface (Gha)': ('float', None, False),
                                                                            }
                                                   },
               LandUseV2.FOREST_SURFACE_DF: {
                   'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS,
                   'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                            'forest_constraint_evolution': ('float', None, False),
                                            'global_forest_surface': ('float', None, False), }
               },
               LandUseV2.LAND_DEMAND_CONSTRAINT_REF: {
                   'type': 'float', 'unit': 'GHa', 'default': 0.1, 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE}, }

    DESC_OUT = {
        LandUseV2.LAND_DEMAND_CONSTRAINT: {
            'type': 'array', 'unit': 'Gha', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_FUNCTIONS},
        LandUseV2.LAND_SURFACE_DETAIL_DF: {'type': 'dataframe', 'unit': 'Gha'},
        LandUseV2.LAND_SURFACE_FOR_FOOD_DF: {
            'type': 'dataframe', 'unit': 'Gha', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_WITNESS}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.land_use_model = LandUseV2(param)

    def run(self):

        # -- get inputs
        inputs = list(self.DESC_IN.keys())
        inputs_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        # -- compute
        land_demand_df = inputs_dict['land_demand_df']
        total_food_land_surface = inputs_dict.pop('total_food_land_surface')
        total_forest_surface_df = deepcopy(inputs_dict['forest_surface_df'])
        total_forest_surface_df.index = land_demand_df[GlossaryCore.Years]
        self.land_use_model.compute(
            land_demand_df, total_food_land_surface, total_forest_surface_df)

        outputs_dict = {
            LandUseV2.LAND_DEMAND_CONSTRAINT: self.land_use_model.land_demand_constraint,
            LandUseV2.LAND_SURFACE_DETAIL_DF: self.land_use_model.land_surface_df,
            # Surface for food used by crop energy techno as input kept output and col name for now but
            # could be changed later on when a single version of agriculture mix is selected
            LandUseV2.LAND_SURFACE_FOR_FOOD_DF: self.land_use_model.land_surface_df[[
                'Food Surface (Gha)']].rename(columns={'Food Surface (Gha)': 'Agriculture total (Gha)'}),
        }

        # -- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        land_demand_constraint wrt forest_surface_df
        land_demand_constraint wrt food_surface_df
        land_demand_constraint wrt land_demand_df
        """
        # -- get inputs
        inputs = list(self.DESC_IN.keys())
        inputs_dict = self.get_sosdisc_inputs(inputs, in_dict=True)
        years = np.arange(inputs_dict[GlossaryCore.YearStart], inputs_dict[GlossaryCore.YearEnd] + 1)
        land_demand_df = inputs_dict['land_demand_df']
        agri_techno = []
        forest_techno = []
        land_demand_columns = list(land_demand_df)
        for techno in LandUseV2.AGRICULTURE_TECHNO:
            if techno in land_demand_columns:
                agri_techno.append(techno)
        for techno in LandUseV2.FOREST_TECHNO:
            if techno in land_demand_columns:
                forest_techno.append(techno)
        # land_surface_for_food_df wrt food_surface_df
        self.set_partial_derivative_for_other_types(
            (LandUseV2.LAND_SURFACE_FOR_FOOD_DF, 'Agriculture total (Gha)'),
            (LandUseV2.TOTAL_FOOD_LAND_SURFACE, 'total surface (Gha)'),
            np.identity(len(years)))
        # land_demand_constraint wrt forest_surface_df
        self.set_partial_derivative_for_other_types(
            (LandUseV2.LAND_DEMAND_CONSTRAINT,), (LandUseV2.FOREST_SURFACE_DF, 'global_forest_surface'),
            - np.identity(len(years)) / inputs_dict[LandUseV2.LAND_DEMAND_CONSTRAINT_REF])
        # land_demand_constraint wrt food_surface_df
        self.set_partial_derivative_for_other_types(
            (LandUseV2.LAND_DEMAND_CONSTRAINT,), (LandUseV2.TOTAL_FOOD_LAND_SURFACE, 'total surface (Gha)'),
            - np.identity(len(years)) / inputs_dict[LandUseV2.LAND_DEMAND_CONSTRAINT_REF])
        # land_demand_constraint wrt land_demand_df
        for techno in agri_techno + forest_techno:
            self.set_partial_derivative_for_other_types(
                (LandUseV2.LAND_DEMAND_CONSTRAINT,), (LandUseV2.LAND_DEMAND_DF, techno),
                - np.identity(len(years)) / inputs_dict[LandUseV2.LAND_DEMAND_CONSTRAINT_REF])

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Land Demand Constraint', 'Detailed Land Usage [Gha]', 'Surface Type in 2020 [Gha]']

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
        inputs_dict = self.get_sosdisc_inputs()
        outputs_dict = self.get_sosdisc_outputs()
        years = list(np.arange(inputs_dict[GlossaryCore.YearStart], inputs_dict[GlossaryCore.YearEnd] + 1))
        total_food_land_surface = inputs_dict['total_food_land_surface']
        total_forest_surface_df = inputs_dict['forest_surface_df']
        land_surface_detailed = outputs_dict[LandUseV2.LAND_SURFACE_DETAIL_DF]
        land_demand_constraint = outputs_dict[LandUseV2.LAND_DEMAND_CONSTRAINT]
        # Surfaces available
        total_land_available = list(land_surface_detailed['Available Agriculture Surface (Gha)'].values +
                                    land_surface_detailed['Available Forest Surface (Gha)'].values +
                                    land_surface_detailed['Available Shrub Surface (Gha)'].values)
        # Habitable land in 2020 covered by agriculture and forest. When agri and forest surface go above
        # this value over the years, shrub surface decreases
        habitable_land_from_forest_agri_2020 = land_surface_detailed['Available Agriculture Surface (Gha)'][0] + \
                                               land_surface_detailed['Available Forest Surface (Gha)'][0]
        # data from ourworld in data are identical for all years to year 2020 5.1 + 3.9 + 1.2 = 10.2 Gha
        available_forest_agri_shrub = total_land_available[0]
        urban_land = 0.15
        fresh_water = 0.15
        habitable_land = available_forest_agri_shrub + urban_land + fresh_water
        glaciers = 1.5
        barren_land = 2.8
        land = habitable_land + glaciers + barren_land

        # shrub surface cannot be <0
        shrub_surface = np.maximum(np.zeros(len(years)), total_land_available[0] * np.ones(len(years)) -
                                   (land_surface_detailed['Total Forest Surface (Gha)'] +
                                    land_surface_detailed['Total Agriculture Surface (Gha)']).values)

        if 'Land Demand Constraint' in chart_list:
            if 'Land Demand Constraint' in chart_list:
                if land_surface_detailed is not None and land_demand_constraint is not None:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    for column in list(land_surface_detailed.columns):
                        if column in ['Total Forest Surface (Gha)', 'Total Agriculture Surface (Gha)']:
                            legend = column.replace(' (Gha)', '')
                            color = {'Total Forest Surface (Gha)': qualitative.Dark2[4],
                                   'Total Agriculture Surface (Gha)': qualitative.Dark2[6]}
                            fig.add_trace(go.Bar(
                                x=years,
                                y=list(land_surface_detailed[column].values),
                                marker_color=color[column],
                                opacity=0.7,
                                name=legend,
                            ), secondary_y=False)
                    fig.add_trace(go.Bar(
                        x=years,
                        y=list(shrub_surface),
                        marker_color=qualitative.Dark2[2],
                        opacity=0.7,
                        name='Total Shrub Surface',
                    ), secondary_y=False)
                    fig.add_trace(go.Scatter(x=years, y=list(np.ones(len(years)) * total_land_available),
                                             line=dict(color=qualitative.Dark2[7]),
                                             name='Total Land Available'), secondary_y=False)
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=list(np.maximum(0.0, -land_demand_constraint)),
                        name="Land Demand Constraint (capped below 0)",
                        line=dict(color=qualitative.Set1[0]),
                    ), secondary_y=True)
                    fig.update_layout(
                        barmode='stack',)

                    fig.update_yaxes(title_text="Land Surfaces [Gha]", secondary_y=False)
                    fig.update_yaxes(title_text="(-1) * Land Demand Constraint", secondary_y=True,
                                     color=qualitative.Set1[0], range=[0, 1.1 * max(-land_demand_constraint)])
                    chart_name = 'Land Demand Constraint'
                    new_chart = InstantiatedPlotlyNativeChart(
                        fig=fig, chart_name=chart_name)

                    instanciated_charts.append(new_chart)

        if 'Detailed Land Usage [Gha]' in chart_list:
            if land_surface_detailed is not None:
                series_to_add = []
                # Total surface usage
                for column in list(land_surface_detailed.columns):
                    if column not in [GlossaryCore.Years, 'Total Forest Surface (Gha)', 'Total Agriculture Surface (Gha)',
                                      'Available Agriculture Surface (Gha)', 'Available Forest Surface (Gha)',
                                      'Available Shrub Surface (Gha)']:
                        legend = column.replace(' (Gha)', '')
                        new_series = InstanciatedSeries(
                            years, (land_surface_detailed[column]).values.tolist(), legend, InstanciatedSeries.BAR_DISPLAY)
                        series_to_add.append(new_series)

                new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Detailed Land Usage [Gha]',
                                                     chart_name='Detailed Land Usage [Gha]', stacked_bar=True)
                for serie in series_to_add:
                    new_chart.add_series(serie)

                instanciated_charts.append(new_chart)

        if 'Surface Type in 2020 [Gha]' in chart_list:
            # ------------------------------------------------------------
            # GLOBAL LAND USE -> Display surfaces (Ocean, Land, Forest..)
            if land_surface_detailed is not None:
                # ------------------
                # Sunburst figure for global land use. Source
                # https://ourworldindata.org/land-use
                fig = go.Figure()
                for year in years:
                    # data from the model
                    agriculture_land = land_surface_detailed.loc[land_surface_detailed[GlossaryCore.Years] == year]['Total Agriculture Surface (Gha)'].values[0]  # 5.1 Gha in 2020
                    forest_land = land_surface_detailed.loc[land_surface_detailed[GlossaryCore.Years] == year]['Total Forest Surface (Gha)'].values[0]  # 3.9 Gha in 2020
                    if agriculture_land + forest_land > available_forest_agri_shrub:
                        shrub = 0.
                        agriculture_land = available_forest_agri_shrub * agriculture_land / (agriculture_land + forest_land)
                        forest_land = available_forest_agri_shrub - agriculture_land
                    else:
                        shrub = available_forest_agri_shrub - agriculture_land - forest_land  # 1.2 Gha in 2020
                    # Create figure
                    fig_i = go.Sunburst(
                        labels=["Land", "Ocean", "Habitable land", "Glaciers", "Barren land",
                                "Agriculture", "Forest", "Shrub", "Urban", "Fresh water"],
                        parents=["Earth", "Earth", "Land", "Land", "Land", "Habitable land",
                                 "Habitable land", "Habitable land", "Habitable land", "Habitable land"],
                        values=[land, 36.1, habitable_land, glaciers, barren_land,
                                agriculture_land, forest_land, shrub, urban_land, fresh_water],
                        marker=dict(colors=["#CD912A", "#1456C5", "#DBBF6A", "#D3D3D0",
                                            "#E7C841", "#7CC873", "#1EA02F", "#5C8C56", "#B1B4AF", "#18CDFA"]),
                        branchvalues="total",
                        rotation=90,
                    )
                    fig.add_trace(fig_i)
                # Create and add slider only for the available slider_values
                chart_name = 'Global land use (Gha)'
                steps = []
                for i, val in enumerate(years):
                    step = dict(
                        method="update",
                        args=[{"visible": [False] * len(fig.data)},
                              {"title": f"{chart_name} for year {val}"}],
                        label=str(val)
                    )
                    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
                    fig.data[i].visible = False
                    steps.append(step)

                sliders = [dict(
                    active=0,  # by default activates the first slider_value available
                    currentvalue={"prefix": "Year "},  # assumes slider_value label='year' to be adapted accordingly
                    pad={"t": 50},
                    steps=steps
                )]
                fig.update_layout(
                    autosize=True,
                    margin=dict(t=80, l=0, r=0, b=0)
                )
                fig.update_layout(sliders=sliders)
                fig.data[0].visible = True

                # Create native plotly chart
                land_use_chart = InstantiatedPlotlyNativeChart(
                    fig=fig, chart_name=chart_name)

                instanciated_charts.append(land_use_chart)

        return instanciated_charts
