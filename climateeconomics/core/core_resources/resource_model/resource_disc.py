'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/06/24 Copyright 2023 Capgemini

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
import logging

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class ResourceDiscipline(SoSWrapp):
    ''' Resource Discipline
    General implementation of the resource discipline, to be inherited by each specific resource
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Resource Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050
    default_years = np.arange(default_year_start, default_year_end + 1, 1)

    prod_unit = 'Mt'
    stock_unit = 'Mt'
    price_unit = '$/Mt'

    resource_name = 'Fill with the resource name'

    DESC_IN = {'resources_demand': {'type': 'dataframe', 'unit': 'Mt',
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_resource',
                                    'dataframe_descriptor': {
                                        GlossaryCore.Years: ('float', None, False),
                                        'natural_gas_resource': ('float', None, True),
                                    'uranium_resource': ('float', None, True),
                                    'coal_resource': ('float', None, True),
                                    'oil_resource': ('float', None, True),
                                    'copper_resource': ('float', None, True),
                                    'platinum_resource': ('float', None, True),},

                                    },
               GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,

               }

    DESC_OUT = {}
    # MMBTU:(Metric Million British Thermal Unit)

    _maturity = 'Research'

    def __init__(self, sos_name, logger:logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        self.resource_model = None

    def setup_sos_disciplines(self):
        pass

    def run(self):
        '''Generic run for all resources
        '''

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.resource_model.configure_parameters_update(inputs_dict)

        self.resource_model.compute()

       

        outputs_dict = {
            'resource_stock': self.resource_model.resource_stock,
            'resource_price': self.resource_model.resource_price,
            'use_stock': self.resource_model.use_stock,
            'predictable_production': self.resource_model.predictable_production,
            'recycled_production': self.resource_model.recycled_production
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)
        

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        gradient of coupling variable to compute:
        price and stock resource with resource_demand_df
        """
        inputs_dict = self.get_sosdisc_inputs()
        output_dict = self.get_sosdisc_outputs()
        resources_demand = inputs_dict['resources_demand']
        sub_resource_list = self.resource_model.sub_resource_list

        grad_stock, grad_price, grad_use, grad_recycling = self.resource_model.get_derivative_resource()
        # # ------------------------------------------------
        # # Stock resource gradient
        for sub_resource_type in sub_resource_list:
            self.set_partial_derivative_for_other_types(
                ('resource_stock', sub_resource_type), ('resources_demand',
                                                        self.resource_name),
                grad_stock[sub_resource_type])
        # # ------------------------------------------------
        # # Price resource gradient
        self.set_partial_derivative_for_other_types(
            ('resource_price', 'price'), ('resources_demand', self.resource_name), grad_price)
        # # ------------------------------------------------
        # # Use resource gradient
        for sub_resource_type in sub_resource_list:
            self.set_partial_derivative_for_other_types(
                ('use_stock', sub_resource_type),
                ('resources_demand', self.resource_name), grad_use[sub_resource_type])
        # # ------------------------------------------------
        # # Recycled resource gradient
        for sub_resource_type in sub_resource_list:
            self.set_partial_derivative_for_other_types(
                ('recycled_production', sub_resource_type),
                ('resources_demand', self.resource_name), grad_recycling[sub_resource_type])
              
        # # ------------------------------------------------
        # # Prod resource gradient did not depend on demand

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Stock', 'Price', 'Production', 'Recycling']

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = ['Stock', 'Price', 'Production', 'Recycling']
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        inputs_dict = self.get_sosdisc_inputs()
        outputs_dict = self.get_sosdisc_outputs()
        year_start = inputs_dict[GlossaryCore.YearStart]
        year_end = inputs_dict[GlossaryCore.YearEnd]
        production_start = inputs_dict['production_start']
        number_of_subtypes = 0
        if 'Stock' in chart_list:
            stock_df = outputs_dict['resource_stock']
            use_stock_df = outputs_dict['use_stock']
            stock_charts = self.get_stock_charts(stock_df, use_stock_df)
            instanciated_charts.extend(stock_charts)
        if 'Price' in chart_list:
            price_df = outputs_dict['resource_price']
            price_charts = self.get_price_charts(price_df)
            instanciated_charts.extend(price_charts)
        if 'Production' in chart_list:
            production_df = outputs_dict['predictable_production']
            past_production_df = inputs_dict['resource_production_data']
            production_charts = self.get_production_charts(
                production_df, past_production_df, year_start, production_start)
            instanciated_charts.extend(production_charts)
            sub_resource_list = [col for col in list(
                production_df.columns) if col != GlossaryCore.Years]
            number_of_subtypes = len(sub_resource_list)

        if 'Recycling' in chart_list and number_of_subtypes < 2:
            recycling_df = outputs_dict['recycled_production']
            #production_df = outputs_dict['predictable_production']
            use_stock_df = outputs_dict['use_stock']
            # use_stock_df = use_stock_df.loc[use_stock_df[GlossaryCore.Years] >= year_start]
            # use_stock_df = use_stock_df.loc[use_stock_df[GlossaryCore.Years] <= year_end]
            recycling_charts = self.get_recycling_charts(
                recycling_df, use_stock_df)
            instanciated_charts.extend(recycling_charts)

        return instanciated_charts

    def get_stock_charts(self, stock_df, use_stock_df):

        sub_resource_list = [col for col in stock_df.columns if col != GlossaryCore.Years]
        stock_chart = TwoAxesInstanciatedChart('Years', f'maximum stocks [{self.stock_unit}]',
                                               chart_name=f'{self.resource_name} stocks through the years',
                                               stacked_bar=True)
        if len(sub_resource_list) > 1:
            use_stock_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} use [{self.stock_unit}]',
                                                       chart_name=f'{self.resource_name} use per subtypes through the years',
                                                       stacked_bar=True)
        use_stock_cumulated_chart = TwoAxesInstanciatedChart('Years',
                                                             f'{self.resource_name} use per Subtypes [{self.stock_unit}]',
                                                             chart_name=f'{self.resource_name} use through the years',
                                                             stacked_bar=True)

        for sub_resource_type in sub_resource_list:
            stock_serie = InstanciatedSeries(
                list(stock_df[GlossaryCore.Years]), (stock_df[sub_resource_type]).values.tolist(), sub_resource_type, InstanciatedSeries.LINES_DISPLAY)
            stock_chart.add_series(stock_serie)

            use_stock_serie = InstanciatedSeries(
                list(use_stock_df[GlossaryCore.Years]), (use_stock_df[sub_resource_type]).values.tolist(), sub_resource_type, InstanciatedSeries.BAR_DISPLAY)
            if len(sub_resource_list) > 1:
                use_stock_chart.add_series(use_stock_serie)
            use_stock_cumulated_chart.add_series(use_stock_serie)

        list_of_charts = [stock_chart, use_stock_cumulated_chart]
        if len(sub_resource_list) > 1:
            list_of_charts.insert(1, use_stock_chart)
        return list_of_charts

    def get_price_charts(self, price_df):
        price_chart = TwoAxesInstanciatedChart('Years', f'price [{self.price_unit}]',
                                               chart_name=f'{self.resource_name} price through the years',
                                               stacked_bar=True)
        price_serie = InstanciatedSeries(
            list(price_df[GlossaryCore.Years]), (price_df['price']).values.tolist(), f'{self.resource_name} price', InstanciatedSeries.LINES_DISPLAY)

        price_chart.add_series(price_serie)
        return [price_chart, ]

    def get_production_charts(self, production_df, past_production_df, year_start, production_start):
        sub_resource_list = [
            col for col in production_df.columns if col != GlossaryCore.Years]

        past_production_cut = past_production_df.loc[past_production_df[GlossaryCore.Years]
                                                     >= production_start]
        production_cut = production_df.loc[production_df[GlossaryCore.Years]
                                           <= year_start]
        if len(sub_resource_list) > 1:
            production_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} production per subtypes [{self.prod_unit}]',
                                                        chart_name=f'{self.resource_name} production per subtypes through the years',
                                                        stacked_bar=True)
        production_cumulated_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} production [{self.prod_unit}]',
                                                              chart_name=f'{self.resource_name} production through the years',
                                                              stacked_bar=True)

        model_production_cumulated_chart = TwoAxesInstanciatedChart('Years',
                                                                    f'Comparison between pyworld3 and real {self.resource_name} production [{self.prod_unit}]',
                                                                    chart_name=f'{self.resource_name} production through the years',
                                                                    stacked_bar=True)
        past_production_chart = TwoAxesInstanciatedChart('Years',
                                                         f'{self.resource_name} past production [{self.prod_unit}]',
                                                         chart_name=f'{self.resource_name} past production through the years',
                                                         stacked_bar=True)

        for sub_resource_type in sub_resource_list:
            production_serie = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (production_df[sub_resource_type]).values.tolist(
                ), sub_resource_type,
                InstanciatedSeries.BAR_DISPLAY)
            if len(sub_resource_list) > 1:
                production_chart.add_series(production_serie)
            production_cumulated_chart.add_series(production_serie)
            production_cut_series = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (production_cut[sub_resource_type]).values.tolist(
                ), sub_resource_type + ' predicted production',
                InstanciatedSeries.BAR_DISPLAY)
            past_production_series = InstanciatedSeries(
                list(past_production_df[GlossaryCore.Years]), (past_production_df[sub_resource_type]).values.tolist(
                ), sub_resource_type,
                InstanciatedSeries.LINES_DISPLAY)
            past_production_cut_series = InstanciatedSeries(
                list(production_df[GlossaryCore.Years]), (past_production_cut[sub_resource_type]).values.tolist(
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
        recycling_chart = TwoAxesInstanciatedChart('Years', f'{self.resource_name} recycling and used stock [{self.stock_unit}]',
                                                   chart_name=f'{self.resource_name} recycled quantity compared to used quantity through the years',
                                                   stacked_bar=False)

        sub_resource_list = [
            col for col in recycling_df.columns if col != GlossaryCore.Years]
        for sub_resource_type in sub_resource_list:
            recycling_serie = InstanciatedSeries(
                list(recycling_df[GlossaryCore.Years]), (recycling_df[sub_resource_type]).values.tolist(), f'{self.resource_name} recycled quantity', InstanciatedSeries.LINES_DISPLAY)
            used_stock_serie = InstanciatedSeries(
                list(use_stock_df[GlossaryCore.Years]), use_stock_df[sub_resource_type].values.tolist(), f'{self.resource_name} extracted quantity', InstanciatedSeries.LINES_DISPLAY)

        recycling_chart.add_series(recycling_serie)
        recycling_chart.add_series(used_stock_serie)
        return [recycling_chart]
