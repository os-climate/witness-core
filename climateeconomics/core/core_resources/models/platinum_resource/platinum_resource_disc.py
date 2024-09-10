'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2024/06/24 Copyright 2023 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_resources.models.platinum_resource.platinum_resource_model import (
    PlatinumResourceModel,
)
from climateeconomics.core.core_resources.resource_model.resource_disc import (
    ResourceDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class PlatinumResourceDiscipline(ResourceDiscipline):
    ''' Discipline intended to get Platinum parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Platinum Resource Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-ring',
        'version': '',
    }
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = GlossaryCore.YearEndDefault
    default_regression_start = 1995
    default_regression_stop = 2005
    default_years = np.arange(default_year_start, default_year_end + 1, 1)

    default_world_consumption_dict = {'exhaust_treatment_system': 0.00008064,
                                      'jewelry': 0.00005824,
                                      'chemical_catalysts': 0.00002016,
                                      'glass_production': 0.00001792,
                                      'electronics': 0.00000672,
                                      'other': 0.00004032,
                                      'total': 0.00022404}

    default_stock_start = 0.0
    default_recycled_rate = 0.5
    default_lifespan = 3
    default_resource_max_price = 5 * 32825887
    resource_name = PlatinumResourceModel.resource_name

    prod_unit = 'Mt'
    stock_unit = 'Mt'
    price_unit = '$/t'

    # Get default data for resource
    default_resource_data = pd.read_csv(
        join(dirname(__file__), f'../resources_data/{resource_name}_data.csv'))
    default_resource_production_data = pd.read_csv(join(
        dirname(__file__), f'../resources_data/{resource_name}_production_data.csv'))
    default_resource_price_data = pd.read_csv(
        join(dirname(__file__), f'../resources_data/{resource_name}_price_data.csv'))
    default_resource_consumed_data = pd.read_csv(
        join(dirname(__file__), f'../resources_data/{resource_name}_consumed_data.csv'))

    DESC_IN = {'resource_data': {'type': 'dataframe', 'unit': '-', 'default': default_resource_data,
                                 'user_level': 2, 'namespace': 'ns_platinum_resource',
                                 'dataframe_descriptor':
                                     {
                                         'platinum_type': ('string', None, True),
                                         'Price': ('float', None, True),
                                         'Price_unit': ('string', None, True),
                                         'Reserve': ('float', None, True),
                                         'Reserve_unit': ('string', None, True),
                                         'Region': ('string', None, True),
                                     }
               },
               'resource_production_data': {'type': 'dataframe', 'unit': 'Mt', 'optional': True,
                                            'default': default_resource_production_data, 'user_level': 2, 'namespace': 'ns_platinum_resource',
                                            'dataframe_descriptor': {
                                                GlossaryCore.Years: ('float', None, False),
                                                'platinum': ('float', None, True), }
                                            },
               'resource_price_data': {'type': 'dataframe', 'unit': 'USD/t', 'default': default_resource_price_data, 'user_level': 2,
                                       'dataframe_descriptor': {'resource_type': ('string', None, False),
                                                                'price': ('float', None, False),
                                                                'unit': ('string', None, False)},
                                       'namespace': 'ns_platinum_resource'},
               'resource_consumed_data': {'type': 'dataframe', 'unit': 'Mt', 'default': default_resource_consumed_data,
                                          'user_level': 2, 'namespace': 'ns_platinum_resource',
                                          'dataframe_descriptor': {
                                              'platinum_consumption': ('float', None, True), }
                                          },
               'production_start': {'type': 'int', 'default': default_regression_start, 'unit': '-',
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_platinum_resource'},
               'regression_stop': {'type': 'int', 'default': default_regression_stop, 'unit': '-',
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_platinum_resource'},
               'world_consumption': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'}, 'unit': '-', 'default': default_world_consumption_dict, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                 'user_level': 2, 'namespace': 'ns_platinum_resource'},
               'stock_start': {'type': 'float', 'default': default_stock_start, 'unit': 'Mt'},
               'recycled_rate': {'type': 'float', 'default': default_recycled_rate, 'unit': '-'},
               'lifespan': {'type': 'int', 'default': default_lifespan, 'unit': '-'},
               'resource_max_price': {'type': 'float', 'default': default_resource_max_price, 'user_level': 2, 'unit': '$/t'},
               }

    DESC_IN.update(ResourceDiscipline.DESC_IN)

    DESC_OUT = {
        'resource_stock': {'type': 'dataframe', 'unit': stock_unit, },
        'resource_price': {'type': 'dataframe', 'unit': price_unit, },
        'use_stock': {'type': 'dataframe', 'unit': stock_unit, },
        'predictable_production': {'type': 'dataframe', 'unit': prod_unit, },
        'recycled_production': {
            'type': 'dataframe', 'unit': prod_unit}
    }
    DESC_OUT.update(ResourceDiscipline.DESC_OUT)

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.resource_model = PlatinumResourceModel(self.resource_name)
        self.resource_model.configure_parameters(inputs_dict)

    def get_stock_charts(self, stock_df, use_stock_df):

        sub_resource_list = [col for col in stock_df.columns if col != GlossaryCore.Years]
        stock_chart = TwoAxesInstanciatedChart('Years', 'maximum stocks [t]',
                                               chart_name=f'{self.resource_name} stocks through the years',
                                               stacked_bar=True)
        if len(sub_resource_list) > 1:
            use_stock_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} use [t]',
                                                       chart_name=f'{self.resource_name} use per subtypes through the years',
                                                       stacked_bar=True)
        use_stock_cumulated_chart = TwoAxesInstanciatedChart('Years',
                                                             f'{self.resource_name} use per Subtypes [t]',
                                                             chart_name=f'{self.resource_name} use through the years',
                                                             stacked_bar=True)

        for sub_resource_type in sub_resource_list:
            stock_serie = InstanciatedSeries(
                list(stock_df[GlossaryCore.Years]), (stock_df[sub_resource_type] * 1000 * 1000).values.tolist(), sub_resource_type, InstanciatedSeries.LINES_DISPLAY)
            stock_chart.add_series(stock_serie)

            use_stock_serie = InstanciatedSeries(
                list(use_stock_df[GlossaryCore.Years]), (use_stock_df[sub_resource_type] * 1000 * 1000).values.tolist(), sub_resource_type, InstanciatedSeries.BAR_DISPLAY)
            if len(sub_resource_list) > 1:
                use_stock_chart.add_series(use_stock_serie)
            use_stock_cumulated_chart.add_series(use_stock_serie)

        list_of_charts = [stock_chart, use_stock_cumulated_chart]
        if len(sub_resource_list) > 1:
            list_of_charts.insert(1, use_stock_chart)
        return list_of_charts

    def get_production_charts(self, production_df, past_production_df, year_start, production_start):
        sub_resource_list = [
            col for col in production_df.columns if col != GlossaryCore.Years]

        past_production_cut = past_production_df.loc[past_production_df[GlossaryCore.Years]
                                                     >= production_start]
        production_cut = production_df.loc[production_df[GlossaryCore.Years]
                                           <= year_start]
        if len(sub_resource_list) > 1:
            production_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} production per subtypes [t]',
                                                        chart_name=f'{self.resource_name} production per subtypes through the years',
                                                        stacked_bar=True)
        production_cumulated_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} production [t]',
                                                              chart_name=f'{self.resource_name} production through the years',
                                                              stacked_bar=True)

        model_production_cumulated_chart = TwoAxesInstanciatedChart('Years',
                                                                    f'Comparison between pyworld3 and real {self.resource_name} production [t]',
                                                                    chart_name=f'{self.resource_name} production through the years',
                                                                    stacked_bar=True)
        past_production_chart = TwoAxesInstanciatedChart('Years',
                                                         f'{self.resource_name} past production [t]',
                                                         chart_name=f'{self.resource_name} past production through the years',
                                                         stacked_bar=True)

        for sub_resource_type in sub_resource_list:
            production_serie = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (production_df[sub_resource_type] * 1000 * 1000).values.tolist(
                ), sub_resource_type,
                InstanciatedSeries.BAR_DISPLAY)
            if len(sub_resource_list) > 1:
                production_chart.add_series(production_serie)
            production_cumulated_chart.add_series(production_serie)
            production_cut_series = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (production_cut[sub_resource_type] * 1000 * 1000).values.tolist(
                ), sub_resource_type + ' predicted production',
                InstanciatedSeries.BAR_DISPLAY)
            past_production_series = InstanciatedSeries(
                list(past_production_df[GlossaryCore.Years]), (past_production_df[sub_resource_type] * 1000 * 1000).values.tolist(
                ), sub_resource_type,
                InstanciatedSeries.LINES_DISPLAY)
            past_production_cut_series = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (past_production_cut[sub_resource_type] * 1000 * 1000).values.tolist(
                ), sub_resource_type + ' real production',
                InstanciatedSeries.LINES_DISPLAY)
            past_production_chart.add_series(past_production_series)
            model_production_cumulated_chart.add_series(
                past_production_cut_series)
            model_production_cumulated_chart.add_series(
                production_cut_series)

        list_of_charts = [past_production_chart,
                          model_production_cumulated_chart, production_cumulated_chart]
        if len(sub_resource_list) > 1:
            list_of_charts.insert(0, production_chart)
        return list_of_charts

    def get_recycling_charts(self, recycling_df, use_stock_df):
        recycling_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} recycling and used stock [t]',
                                                   chart_name=f'{self.resource_name} recycled quantity compared to used quantity through the years',
                                                   stacked_bar=False)

        sub_resource_list = [
            col for col in recycling_df.columns if col != GlossaryCore.Years]
        for sub_resource_type in sub_resource_list:
            recycling_serie = InstanciatedSeries(
                list(recycling_df[GlossaryCore.Years]), (recycling_df[sub_resource_type] * 1000 * 1000).values.tolist(), f'{self.resource_name} recycled quantity', InstanciatedSeries.LINES_DISPLAY)
            used_stock_serie = InstanciatedSeries(
                list(use_stock_df[GlossaryCore.Years]), (use_stock_df[sub_resource_type] * 1000 * 1000).values.tolist(), f'{self.resource_name} extracted quantity', InstanciatedSeries.LINES_DISPLAY)

        recycling_chart.add_series(recycling_serie)
        recycling_chart.add_series(used_stock_serie)
        return [recycling_chart]
