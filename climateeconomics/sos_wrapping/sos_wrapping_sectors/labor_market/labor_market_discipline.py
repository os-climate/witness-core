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

from climateeconomics.core.core_sectorization.labor_market_sectorisation import (
    LaborMarketModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class LaborMarketDiscipline(ClimateEcoDiscipline):
    ''' Discipline intended to agregate resource parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Labor Market Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-briefcase',
        'version': '',
    }

    DESC_IN = {GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
               GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
               GlossaryCore.SectorListValue: GlossaryCore.SectorList,
               # Employment rate param
               'employment_a_param': {'type': 'float', 'default': 0.6335, 'user_level': 3, 'unit': '-'},
               'employment_power_param': {'type': 'float', 'default': 0.0156, 'user_level': 3, 'unit': '-'},
               'employment_rate_base_value': {'type': 'float', 'default': 0.659, 'user_level': 3, 'unit': '-'},
               GlossaryCore.WorkingAgePopulationDfValue: {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS,
                                             'dataframe_descriptor': {GlossaryCore.Years: ('float', None, False),
                                                                      GlossaryCore.Population1570: ('float', None, False),}
                                             },
               GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
              }
    DESC_OUT = {
        GlossaryCore.WorkforceDfValue: {'type': GlossaryCore.WorkforceDf['type'],
                                        'unit': GlossaryCore.WorkforceDf['unit'], 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                         'namespace': GlossaryCore.NS_WITNESS},
        'employment_df': {'type': 'dataframe', 'unit': '-'}
    }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.labor_model = LaborMarketModel(inputs_dict)

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        if self.get_data_in() is not None:
            if GlossaryCore.SectorListValue in self.get_data_in():
                sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
                df_descriptor = {GlossaryCore.Years: ('float', None, False)}
                df_descriptor.update({col: ('float', None, True)
                                 for col in sector_list})
                
                dynamic_inputs['workforce_share_per_sector'] = {'type': 'dataframe', 'unit': '%',
                                                'dataframe_descriptor': df_descriptor,
                                                'dataframe_edition_locked': False}
              
            self.add_inputs(dynamic_inputs)


    def run(self):

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        
        # -- configure class with inputs
        self.labor_model.configure_parameters(inputs_dict)

        # -- compute
        workforce_df, employment_df  = self.labor_model.compute(inputs_dict)

        outputs_dict = {GlossaryCore.WorkforceDfValue: workforce_df,
                        'employment_df': employment_df}

        
        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        net_output and invest wrt sector net_output 
        """
        sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
        # Gradient wrt working age population
        grad_workforcetotal = self.labor_model.compute_dworkforcetotal_dworkagepop()
        self.set_partial_derivative_for_other_types((GlossaryCore.WorkforceDfValue, GlossaryCore.Workforce),
                                                        (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
                                                        grad_workforcetotal)
        for sector in sector_list:
            grad_workforcesector = self.labor_model.compute_dworkforcesector_dworkagepop(sector)
            self.set_partial_derivative_for_other_types((GlossaryCore.WorkforceDfValue, sector),
                                                        (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
                                                        grad_workforcesector)
            

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['workforce per sector', 'total workforce', 'employment rate','workforce share per sector']

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        workforce_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.WorkforceDfValue))
        employment_df = deepcopy(self.get_sosdisc_outputs('employment_df'))
        sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'employment rate' in chart_list:

            years = list(employment_df.index)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = 0, 1

            chart_name = 'Employment rate'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'employment rate',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(employment_df[GlossaryCore.EmploymentRate])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.EmploymentRate, 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'total workforce' in chart_list:

            working_age_pop_df = self.get_sosdisc_inputs(
                GlossaryCore.WorkingAgePopulationDfValue)
            years = list(workforce_df[GlossaryCore.Years].values)

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                working_age_pop_df[GlossaryCore.Population1570])

            chart_name = 'Workforce'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Number of people [million]',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(workforce_df[GlossaryCore.Workforce])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Workforce', 'lines', visible_line)
            ordonate_data_bis = list(working_age_pop_df[GlossaryCore.Population1570])
            new_chart.series.append(new_series)
            new_series = InstanciatedSeries(
                years, ordonate_data_bis, 'Working-age population', 'lines', visible_line)
            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'workforce per sector' in chart_list:
            chart_name = 'Workforce per economic sector'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Workforce per sector [million of people]',
                                                 [year_start - 5, year_end + 5],
                                                 chart_name=chart_name)

            for sector in sector_list:
                sector_workforce = workforce_df[sector].values
                visible_line = True
                ordonate_data = list(sector_workforce)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} workforce', 'lines', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'workforce share per sector' in chart_list:
            share_workforce = self.get_sosdisc_inputs('workforce_share_per_sector')
            chart_name = 'Workforce distribution per sector'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'share of total workforce [%]',
                                                 [year_start - 5, year_end + 5], stacked_bar=True,
                                                 chart_name=chart_name)

            for sector in sector_list:
                share = share_workforce[sector].values
                visible_line = True
                ordonate_data = list(share)
                new_series = InstanciatedSeries(years, ordonate_data,
                                                f'{sector} share of total workforce', 'bar', visible_line)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
