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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
# from climateeconomics.core.core_land_use.land_use import LandUse,\
# OrderOfMagnitude
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from climateeconomics.core.core_resources.resources_model import ResourceModel
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


class CoalDiscipline(SoSDiscipline):
    ''' Discipline intended to get coal parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Coal Resource Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-cloud-meatball fa_fw',
        'version': '',
    }
    default_year_start = 2020
    default_year_end = 2050
    default_production_start = 2001
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    default_production_start = 2001

    resource_name = ResourceGlossary.Coal['name']

    DESC_IN = {ResourceModel.DEMAND: {'type': 'dataframe', 'unit': 'Mt',
                                      'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource'},
               'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'production_start': {'type': 'int', 'default': default_production_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'}
               }

    DESC_OUT = {
        ResourceModel.RESOURCE_STOCK: {
            'type': 'dataframe', 'unit': 'million_tonnes', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'},
        ResourceModel.RESOURCE_PRICE: {
            'type': 'dataframe', 'unit': 'USD/USton', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'},
        ResourceModel.USE_STOCK: {'type': 'dataframe', 'unit': 'million_tonnes', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'},
        ResourceModel.PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'},
        ResourceModel.PAST_PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes',
                                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'coal_resource'}
    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.coal_model = ResourceModel(param)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inputs_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        coal_demand = pd.DataFrame(
            {'years': inputs_dict['All_Demand']['years'].values})
        coal_demand[self.resource_name] = inputs_dict['All_Demand'][self.resource_name]
        self.coal_model.compute(coal_demand, self.resource_name, 2001)

        outputs_dict = {
            ResourceModel.RESOURCE_STOCK: self.coal_model.resource_stock,
            ResourceModel.RESOURCE_PRICE: self.coal_model.resource_price,
            ResourceModel.USE_STOCK: self.coal_model.use_stock,
            ResourceModel.PRODUCTION: self.coal_model.predictible_production,
            ResourceModel.PAST_PRODUCTION: self.coal_model.past_production
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['all']

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute: 
        price and stock resource with resource_demand_df
        """
        # get coal input demand
        inputs_dict = self.get_sosdisc_inputs()
        output_dict = self.get_sosdisc_outputs()
        coal_resource = self.resource_name
        coal_demand = pd.DataFrame(
            {'years': inputs_dict['All_Demand']['years'].values})
        coal_demand[self.resource_name] = inputs_dict['All_Demand'][coal_resource]

        grad_stock, grad_price, grad_use = self.coal_model.get_derivative_resource(
            coal_resource)
        # # ------------------------------------------------
        # # Stock resource gradient
        for coal_type in output_dict['resource_stock']:
            self.set_partial_derivative_for_other_types(
                (ResourceModel.RESOURCE_STOCK, coal_type), (f'{ResourceModel.DEMAND}', self.resource_name), grad_stock[coal_type])
        # # # ------------------------------------------------
        # # # Price resource gradient
        self.set_partial_derivative_for_other_types(
            (ResourceModel.RESOURCE_PRICE, 'price'), (f'{ResourceModel.DEMAND}', self.resource_name), grad_price)
        # # # ------------------------------------------------
        # # # Use resource gradient
        for coal_type in output_dict['resource_stock']:
            self.set_partial_derivative_for_other_types(
                (ResourceModel.USE_STOCK, coal_type), (f'{ResourceModel.DEMAND}', self.resource_name), grad_use[coal_type])
        # # ------------------------------------------------
        # # Prod resource gradient did not depend on demand

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'all' in chart_list:
            production_start = self.get_sosdisc_inputs(
                ResourceModel.PRODUCTION_START)
            years_start = self.get_sosdisc_inputs(ResourceModel.YEAR_START)
            stock_df = self.get_sosdisc_outputs(
                ResourceModel.RESOURCE_STOCK)
            years = stock_df.index.values.tolist()

            price_df = self.get_sosdisc_outputs(
                ResourceModel.RESOURCE_PRICE)
            use_stock_df = self.get_sosdisc_outputs(
                ResourceModel.USE_STOCK)
            production_df = self.get_sosdisc_outputs(
                ResourceModel.PRODUCTION)
            past_production_df = self.get_sosdisc_outputs(
                ResourceModel.PAST_PRODUCTION)
            past_production_cut = past_production_df.loc[past_production_df['years']
                                                         >= production_start]
            production_cut = production_df.loc[production_df.index
                                               <= years_start]
            production_years = production_df.index.values.tolist()
            past_production_year = past_production_df['years'].values.tolist()

            # two charts for stock evolution and price evolution
            stock_chart = TwoAxesInstanciatedChart('years', 'maximum stocks [Mt]',
                                                   chart_name='Coal stocks through the years', stacked_bar=False)
            price_chart = TwoAxesInstanciatedChart('years', 'price [USD/USton]',
                                                   chart_name='Coal price through the years', stacked_bar=False)
            use_stock_cumulated_chart = TwoAxesInstanciatedChart('years', 'Coal use [Mt]',
                                                                 chart_name='Coal use through the years',
                                                                 stacked_bar=True)
            use_stock_chart = TwoAxesInstanciatedChart('years', 'Coal use per subtypes [Mt]',
                                                       chart_name='Coal use per subtypes through the years', stacked_bar=False)
            production_chart = TwoAxesInstanciatedChart('years',
                                                        'Coal production per subtypes [Mt]',

                                                        chart_name='Coal production per subtypes through the years',
                                                        stacked_bar=False)
            production_cumulated_chart = TwoAxesInstanciatedChart('years',
                                                                  'Coal production [Mt]',
                                                                  chart_name='Coal production through the years',
                                                                  stacked_bar=True)
            model_production_cumulated_chart = TwoAxesInstanciatedChart('years',
                                                                        'Comparison between model and real Coal production [Mt]',
                                                                        chart_name='Coal production through the years',
                                                                        stacked_bar=False)
            past_production_chart = TwoAxesInstanciatedChart('years',
                                                             'Coal past production [Mt]',
                                                             chart_name='Coal past production through the years',
                                                             stacked_bar=False)
            for stock_kind in stock_df:
                stock_serie = InstanciatedSeries(
                    years, (stock_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.LINES_DISPLAY)
                stock_chart.add_series(stock_serie)

                production_serie = InstanciatedSeries(
                    production_years, (production_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.BAR_DISPLAY)
                production_cut_series = InstanciatedSeries(
                    production_years, (production_cut[stock_kind]).values.tolist(), stock_kind + ' predicted production', InstanciatedSeries.BAR_DISPLAY)
                past_production_series = InstanciatedSeries(
                    past_production_year, (past_production_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.LINES_DISPLAY)
                past_production_cut_series = InstanciatedSeries(
                    production_years, (past_production_cut[stock_kind]).values.tolist(), stock_kind + ' real production', InstanciatedSeries.LINES_DISPLAY)

                production_chart.add_series(production_serie)
                past_production_chart.add_series(past_production_series)
                production_cumulated_chart.add_series(production_serie)
                model_production_cumulated_chart.add_series(
                    past_production_cut_series)
                model_production_cumulated_chart.add_series(
                    production_cut_series)
                use_stock_serie = InstanciatedSeries(
                    years, (use_stock_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.BAR_DISPLAY)
                use_stock_chart.add_series(use_stock_serie)
                use_stock_cumulated_chart.add_series(use_stock_serie)

            price_serie = InstanciatedSeries(
                years, (price_df['price']).values.tolist(), 'coal price', InstanciatedSeries.LINES_DISPLAY)
            price_chart.add_series(price_serie)

            instanciated_charts.append(stock_chart)
            instanciated_charts.append(price_chart)
            instanciated_charts.append(use_stock_chart)
            instanciated_charts.append(use_stock_cumulated_chart)
            instanciated_charts.append(production_chart)
            instanciated_charts.append(production_cumulated_chart)
            instanciated_charts.append(model_production_cumulated_chart)
            instanciated_charts.append(past_production_chart)

        return instanciated_charts
