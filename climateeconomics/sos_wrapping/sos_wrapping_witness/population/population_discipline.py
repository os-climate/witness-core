'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/02-2023/11/03 Copyright 2023 Capgemini

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
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import sostrades_core.tools.post_processing.post_processing_tools as ppt
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.population_model import Population
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class PopulationDiscipline(ClimateEcoDiscipline):
    "     Temperature evolution"

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Population Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-users fa-fw',
        'version': '',
    }
    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
    global_data_dir = join(Path(__file__).parents[3], 'data')
    default_death_rate_params_df = pd.read_csv(
        join(global_data_dir, 'death_rate_params_v2.csv'))
    # Provided by WHO. (2014). Quantitative risk assessment of the effects of climate
    # change on selected causes of death, 2030s and 2050s. Geneva:
    # World Health Organization.
    default_climate_mortality_param_df = pd.read_csv(
        join(global_data_dir, 'climate_additional_deaths_V2.csv'))
    # ADD DICTIONARY OF VALUES FOR DEATH RATE

    desc_in_default_diet_mortality_param = GlossaryCore.DietMortalityParamDf
    desc_in_default_diet_mortality_param['default'] = pd.read_csv(join(global_data_dir, 'diet_mortality_param.csv'))

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.PopulationStart: GlossaryCore.PopulationStartDf,
        GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
        GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
        'climate_mortality_param_df': {'type': 'dataframe', 'default': default_climate_mortality_param_df, 'user_level': 3, 'unit': '-',
                                       'dataframe_descriptor': {'param': ('string', None, False),
                                                                'beta': ('float', None, False),}
                                       },
        'calibration_temperature_increase': {'type': 'float', 'default': 2.5, 'user_level': 3 , 'unit': '°C'},
        'theta': {'type': 'float', 'default': 2, 'user_level': 3, 'unit': '-'},
        'death_rate_param': {'type': 'dataframe', 'default': default_death_rate_params_df, 'user_level': 3, 'unit': '-',
                             'dataframe_descriptor': {'param': ('string', None, False),
                                                      'death_rate_upper': ('float', None, False),
                                                      'death_rate_lower': ('float', None, False),
                                                      'death_rate_delta': ('float', None, False),
                                                      'death_rate_phi': ('float', None, False),
                                                      'death_rate_nu': ('float', None, False),}
                             },
        'birth_rate_upper': {'type': 'float', 'default': 1.12545946e-01, 'user_level': 3, 'unit': '-'},
        # 2.2e-2
        'birth_rate_lower': {'type': 'float', 'default': 2.02192894e-02, 'user_level': 3, 'unit': '-'},
        # 1.92403581e-04
        'birth_rate_delta': {'type': 'float', 'default': 6.19058508e-04, 'user_level': 3, 'unit': '-'},
        # 4.03359157e+03
        'birth_rate_phi': {'type': 'float', 'default': 4.03360000e+03, 'user_level': 3, 'unit': '-'},
        # 3.93860555e-01
        'birth_rate_nu': {'type': 'float', 'default': 1.75808789e-01, 'user_level': 3, 'unit': '-'},
        'lower_knowledge': {'type': 'float', 'default': 10, 'user_level': 3, 'unit': '%'},
        'upper_knowledge': {'type': 'float', 'default': 100, 'user_level': 3, 'unit': '%'},
        'delta_knowledge': {'type': 'float', 'default': 0.0293357, 'user_level': 3, 'unit': '-'},
        'phi_knowledge': {'type': 'float', 'default': 149.7919, 'user_level': 3, 'unit': '-'},
        'nu_knowledge': {'type': 'float', 'default': 1.144062855, 'user_level': 3, 'unit': '-'},
        'constant_birthrate_know': {'type': 'float', 'default': 1.99999838e-02, 'user_level': 3, 'unit': '-'},
        'alpha_birthrate_know': {'type': 'float', 'default': 1.02007061e-01, 'user_level': 3, 'unit': '-'},
        'beta_birthrate_know': {'type': 'float', 'default': 8.01923418e-01, 'user_level': 3, 'unit': '-'},
        'share_know_birthrate': {'type': 'float', 'default': 7.89207064e-01, 'user_level': 3, 'unit': '-'},
        ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN['var_name']: ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        GlossaryCore.CaloriesPerCapitaValue: GlossaryCore.CaloriesPerCapita,
        GlossaryCore.DietMortalityParamDf['var_name']: desc_in_default_diet_mortality_param,
        'theta_diet': {'type': 'float', 'default': 5.0, 'user_level': 3, 'unit': '-'},
        'kcal_pc_ref': {'type': 'float', 'default': 2000.0, 'user_level': 3, 'unit': 'kcal'},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.PandemicParamDfValue: GlossaryCore.PandemicParamDf,
    }

    DESC_OUT = {
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.WorkingAgePopulationDfValue: {'type': 'dataframe', 'unit': 'millions of people', 'visibility': 'Shared',
                                      'namespace': GlossaryCore.NS_WITNESS},
        'population_detail_df': {'type': 'dataframe', 'unit': 'people'},
        'birth_rate_df': {'type': 'dataframe', 'unit': '-'},
        'death_rate_dict': {'type': 'dict', 'subtype_descriptor':{'dict':'dataframe'}, 'unit': '-'},
        'death_dict': {'type': 'dict', 'subtype_descriptor':{'dict':'dataframe'}, 'unit': 'people' },
        'birth_df': {'type': 'dataframe', 'unit': 'people'},
        'life_expectancy_df': {'type': 'dataframe', 'unit': 'age'}
    }

    _maturity = 'Research'
        

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.model = Population(in_dict)

    def update_default_values(self):
        """
        Update all default dataframes with years
        """
        if self.get_data_in() is not None:
            if GlossaryCore.YearStart in self.get_data_in():
                year_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
                if year_start is not None:
                    self.update_default_value(GlossaryCore.PopulationStart, 'in', DatabaseWitnessCore.PopulationYearStart.get_df_at_year(year_start))

    def setup_sos_disciplines(self):  # type: (...) -> None
        self.update_default_values()
        if GlossaryCore.YearStart in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                [GlossaryCore.YearStart, GlossaryCore.YearEnd])
            if year_start is not None and year_end is not None:
                years = np.arange(year_start, year_end + 1)
                default_calories_pc_df = pd.DataFrame({GlossaryCore.Years: years,
                                                       'kcal_pc': 2400.})
                self.set_dynamic_default_values({GlossaryCore.CaloriesPerCapitaValue: default_calories_pc_df})

    def run(self):
        ''' model execution '''
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        

        # model execution
        population_detail_df, birth_rate_df, death_rate_dict, birth_df, death_dict, life_expectancy_df, working_age_population_df = self.model.compute(
            in_dict)

        population_df = population_detail_df[[GlossaryCore.Years, 'total']]
        population_df = population_df.rename(columns={"total": GlossaryCore.PopulationValue})

        # Convert population in billion of people
        population_df[GlossaryCore.PopulationValue] = population_df[GlossaryCore.PopulationValue] / \
            self.model.million
        # disable pylint warning, known issue for pylint >2.4, pylint cannot get some variable type even if it has been set
        population_detail_df[GlossaryCore.Population1570] = working_age_population_df[GlossaryCore.Population1570] # pylint: disable=unsubscriptable-object
        working_age_population_df[GlossaryCore.Population1570] = working_age_population_df[GlossaryCore.Population1570] / self.model.million # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
        # store output data
        out_dict = {GlossaryCore.PopulationDfValue: population_df,
                    GlossaryCore.WorkingAgePopulationDfValue: working_age_population_df, # pylint: disable=unsubscriptable-object
                    "population_detail_df": population_detail_df,
                    "birth_rate_df": birth_rate_df,
                    "death_rate_dict": death_rate_dict,
                    "birth_df": birth_df,
                    "death_dict": death_dict,
                    "life_expectancy_df": life_expectancy_df}

        
        self.store_sos_outputs_values(out_dict)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        """

        d_pop_d_output, d_working_pop_d_output = self.model.compute_d_pop_d_output()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            d_pop_d_output / self.model.million)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            (GlossaryCore.EconomicsDfValue, GlossaryCore.OutputNetOfDamage),
            d_working_pop_d_output / self.model.million)

        d_pop_d_temp, d_working_pop_d_temp = self.model.compute_d_pop_d_temp()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            d_pop_d_temp / self.model.million)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            d_working_pop_d_temp / self.model.million)
        
        d_pop_d_kcal_pc, d_working_pop_d_kcal_pc = self.model.compute_d_pop_d_kcal_pc()
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
            (GlossaryCore.CaloriesPerCapitaValue, 'kcal_pc'),
            d_pop_d_kcal_pc / self.model.million)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.WorkingAgePopulationDfValue, GlossaryCore.Population1570),
            (GlossaryCore.CaloriesPerCapitaValue, 'kcal_pc'),
            d_working_pop_d_kcal_pc / self.model.million)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['World population', 'Population detailed', 'Population detailed year start', 'Population detailed mid year', '15-49 age range birth rate',
                      'knowledge', 'death rate per age range', 'Number of birth and death per year',
                      'Cumulative climate deaths', 'Number of climate death per year',
                      'Cumulative malnutrition deaths', 'Number of malnutrition death per year', 'Malnutrition death rate per age range',
                      'Life expectancy evolution', 'working-age population over years']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))
        year_start, year_end = self.get_sosdisc_inputs(
            [GlossaryCore.YearStart, GlossaryCore.YearEnd])
        years = list(np.arange(year_start, year_end + 1, 5))
        chart_filters.append(ChartFilter(
            'Years for population', years, [year_start, year_end], GlossaryCore.Years))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []
        years_list = None
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values
                if chart_filter.filter_key == GlossaryCore.Years:
                    years_list = chart_filter.selected_values
        if years_list is None:
            raise Exception("no years selected in filters")
        pop_df = deepcopy(
            self.get_sosdisc_outputs('population_detail_df'))
        birth_rate_df = deepcopy(
            self.get_sosdisc_outputs('birth_rate_df'))
        birth_df = deepcopy(
            self.get_sosdisc_outputs('birth_df'))
        death_rate_dict = deepcopy(
            self.get_sosdisc_outputs('death_rate_dict'))
        death_dict = deepcopy(
            self.get_sosdisc_outputs('death_dict'))
        life_expectancy_df = deepcopy(
            self.get_sosdisc_outputs('life_expectancy_df'))

        years = list(pop_df[GlossaryCore.Years].values)
        if 'World population' in chart_list:

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                pop_df['total'])

            chart_name = 'World population over years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.PopulationValue,
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(pop_df['total'])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.PopulationValue, 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'working-age population over years' in chart_list:

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                pop_df[GlossaryCore.Population1570])

            chart_name = 'working-age population over years'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '15-70 age range population',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(pop_df[GlossaryCore.Population1570])

            new_series = InstanciatedSeries(
                years, ordonate_data, GlossaryCore.PopulationValue, 'lines', visible_line)

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if '15-49 age range birth rate' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                birth_rate_df['birth_rate'])

            chart_name = '15-49 age range birth rate'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' birth rate',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(birth_rate_df['birth_rate'])

            new_series = InstanciatedSeries(
                years, ordonate_data, '15-49 birth rate', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'knowledge' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                birth_rate_df['knowledge'])

            chart_name = 'Knowledge yearly evolution'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'knowledge',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(birth_rate_df['knowledge'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'knowledge', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'death rate per age range' in chart_list:

            headers = list(death_rate_dict['total'].columns.values)
            to_plot = headers[:]

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_values = {}
            max_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    death_rate_dict['total'][key])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Death rate per age range'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' death rate',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(death_rate_dict['total'][key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, f'death rate for age range {key}', 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Malnutrition death rate per age range' in chart_list:

            headers = list(death_rate_dict['diet'].columns.values)
            to_plot = headers[:]

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_values = {}
            max_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    death_rate_dict['diet'][key])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'malnutrition death rate per age range'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' death rate',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)
            for key in to_plot:
                visible_line = True
                ordonate_data = list(death_rate_dict['diet'][key])

                new_series = InstanciatedSeries(
                    years, ordonate_data, f'death rate imputable to malnutrition for age range {key}', 'lines', visible_line)

                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Number of birth and death per year' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_values = {}
            max_values = {}

            min_values['number_of_birth'], max_values['number_of_birth'] = self.get_greataxisrange(
                birth_df['number_of_birth'])
            min_values['number_of_death'], max_values['number_of_death'] = self.get_greataxisrange(
                death_dict['total']['total'])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Number of birth and death per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Number of birth and death',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(birth_df['number_of_birth'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'Number of birth per year', 'lines', visible_line)

            new_chart.series.append(new_series)
            ordonate_data = list(death_dict['total']['total'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'Number of death per year', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Number of climate death per year' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                death_dict['climate']['total'])

            chart_name = 'Human cost of global warming per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Number of death',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(death_dict['climate']['total'])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Number of death due to climate change per year', 'lines', visible_line)

            note = {'Climate deaths': 'Undernutrition, diseases and heat waves'}
            new_chart.annotation_upper_left = note
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Cumulative climate deaths' in chart_list:

            instanciated_charts = graph_model_cumulative_climate_deaths(death_dict, instanciated_charts)

            
        if 'Number of malnutrition death per year' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                death_dict['diet']['total'])

            chart_name = 'Human cost of malnutrition per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Number of death',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True

            ordonate_data = list(death_dict['diet']['total'])
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Number of death due to malnutrition per year', 'lines', visible_line)

            note = {'Malnutrition': 'Undernutrition or overnutrion'}
            new_chart.annotation_upper_left = note
            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Cumulative malnutrition deaths' in chart_list:

            headers = list(death_dict['diet'].columns.values)
            to_plot = headers[:]

            year_start = years[0]
            year_end = years[len(years) - 1]

            min_values = {}
            max_values = {}
            for key in to_plot:
                min_values[key], max_values[key] = self.get_greataxisrange(
                    death_dict['diet'][key])

            min_value = min(min_values.values())
            max_value = max(max_values.values())

            chart_name = 'Cumulative malnutrition deaths'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' cumulative malnutrition deaths',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(death_dict['diet']['cum_total'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'cumulative malnutrition deaths', 'bar')

            new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        if 'Life expectancy evolution' in chart_list:


            year_start = years[0]
            year_end = years[len(years) - 1]

            min_value, max_value = self.get_greataxisrange(
                life_expectancy_df['life_expectancy'])

            chart_name = 'Life expectancy at birth per year'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Life expectancy at birth',
                                                 [year_start - 5, year_end + 5],
                                                 [min_value, max_value],
                                                 chart_name)

            visible_line = True
            ordonate_data = list(life_expectancy_df['life_expectancy'])

            new_series = InstanciatedSeries(
                years, ordonate_data, 'Life expectancy', 'lines', visible_line)

            new_chart.series.append(new_series)
            instanciated_charts.append(new_chart)

        if 'Population detailed' in chart_list:

            pop_column = list(np.arange(0, 101))

            year_start = years[0]

            for year in years_list:

                chart_name = f'Population by age at year {year}'

                new_chart = TwoAxesInstanciatedChart('age', ' number of people',
                                                     chart_name=chart_name)

                ordonate_data = list(pop_df.iloc[year - year_start, 1:-2])

                new_series = InstanciatedSeries(
                    pop_column, ordonate_data, '', 'bar')

                new_chart.series.append(new_series)
                note = {'Age 100': 'regroups everyone at 100 and above'}
                new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

        if 'Population detailed year start' in chart_list:

            pop_column = list(np.arange(0, 101))

            year_start = years[0]

            chart_name = f'Population by age at year {year_start}'

            new_chart = TwoAxesInstanciatedChart('age', ' number of people',
                                                 chart_name=chart_name)

            ordonate_data = list(pop_df.iloc[0, 1:-2])

            new_series = InstanciatedSeries(
                pop_column, ordonate_data, '', 'bar')

            new_chart.series.append(new_series)
            note = {'Age 100': 'regroups everyone at 100 and above'}
            new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

        if 'Population detailed mid year' in chart_list:

            pop_column = list(np.arange(0, 101))

            year_start = years[0]
            year_end = years[len(years) - 1]
            # Take year in the middle of the period
            year = round((year_end - year_start) / 2) + year_start

            chart_name = f'Population by age at year {year}'

            new_chart = TwoAxesInstanciatedChart('age', ' number of people',
                                                 chart_name=chart_name)

            ordonate_data = list(pop_df.iloc[year - year_start, 1:-2])

            new_series = InstanciatedSeries(
                pop_column, ordonate_data, '', 'bar')

            new_chart.series.append(new_series)
            note = {'Age 100': 'regroups everyone at 100 and above'}
            new_chart.annotation_upper_left = note
            instanciated_charts.append(new_chart)

        return instanciated_charts


# externalize graph methods out of the class so that they can be reused in an external dashboard for instance
def graph_model_cumulative_climate_deaths(death_dict, instanciated_charts):
    years = list(death_dict['climate']['cum_total'].index)
    headers = list(death_dict['climate'].columns.values)
    to_plot = headers[:]

    year_start = years[0]
    year_end = years[len(years) - 1]

    min_values = {}
    max_values = {}
    for key in to_plot:
        min_values[key], max_values[key] = ppt.get_greataxisrange(
            death_dict['climate'][key])

    min_value = min(min_values.values())
    max_value = max(max_values.values())

    chart_name = 'Cumulative climate deaths'

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' cumulative climatic deaths',
                                            [year_start - 5, year_end + 5],
                                            [min_value, max_value],
                                            chart_name)

    visible_line = True
    ordonate_data = list(death_dict['climate']['cum_total'])

    new_series = InstanciatedSeries(
        years, ordonate_data, 'cumulative climatic deaths', 'bar')

    new_chart.series.append(new_series)

    instanciated_charts.append(new_chart)

    return instanciated_charts


def graph_model_world_population(pop_df, instanciated_charts):

    years = list(pop_df[GlossaryCore.Years].values)

    year_start = years[0]
    year_end = years[len(years) - 1]

    min_value, max_value = ppt.get_greataxisrange(
        pop_df['total'])

    chart_name = 'World population over years'

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.PopulationValue,
                                            [year_start - 5, year_end + 5],
                                            [min_value, max_value],
                                            chart_name)

    visible_line = True

    ordonate_data = list(pop_df['total'])

    new_series = InstanciatedSeries(
        years, ordonate_data, GlossaryCore.PopulationValue, 'lines', visible_line)

    new_chart.series.append(new_series)

    instanciated_charts.append(new_chart)

    return instanciated_charts


def graph_model_world_pop_and_cumulative_deaths(pop_df, death_dict, instanciated_charts):
    headers = list(death_dict['climate'].columns.values)
    to_plot = headers[:]

    years = list(pop_df[GlossaryCore.Years].values)
    year_start = years[0]
    year_end = years[len(years) - 1]

    # find min and max values for axis from pop and death data
    min_values = {}
    max_values = {}
    for key in to_plot:
        min_values[key], max_values[key] = ppt.get_greataxisrange(
            death_dict['climate'][key])

    min_value_pop, max_value_pop = ppt.get_greataxisrange(pop_df['total'])
    min_value = min(min(min_values.values()), min_value_pop)
    max_value = max(max(max_values.values()), max_value_pop)

    chart_name = 'World population and cumulative climate deaths '

    new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, ' Number of people',
                                            [year_start - 5, year_end + 5],
                                            [min_value, max_value],
                                            chart_name)

    visible_line = True
    ordonate_data = list(death_dict['climate']['cum_total'])
    new_series = InstanciatedSeries(
        years, ordonate_data, 'Cumulative climate deaths', 'bar')
    new_chart.series.append(new_series)

    ordonate_data = list(pop_df['total'])
    new_series = InstanciatedSeries(
        years, ordonate_data, GlossaryCore.PopulationValue, 'lines', visible_line)
    new_chart.series.append(new_series)

    instanciated_charts.append(new_chart)

    return instanciated_charts
