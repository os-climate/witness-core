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
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from climateeconomics.core.core_resources.resources_model import ResourceModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


class GasDiscipline(SoSDiscipline):
    ''' Discipline intended to get natural gas parameters
    '''
    default_year_start = 2020
    default_year_end = 2050
    default_production_start = 2010
    default_years = np.arange(default_year_start, default_year_end + 1, 1)

    DESC_IN = {ResourceModel.DEMAND: {'type': 'dataframe', 'unit': 'Mt',
                                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource'},
               'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'production_start':{'type': 'int', 'default': default_production_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'}
               }

    DESC_OUT = {
        ResourceModel.RESOURCE_STOCK: {
            'type': 'dataframe', 'unit': 'billion_cubic_metres', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'},
        ResourceModel.RESOURCE_PRICE:{
            'type': 'dataframe', 'unit': 'USD/MMBTU', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'},
        ResourceModel.USE_STOCK: {'type': 'dataframe', 'unit': 'billion cubic metre', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'},
        ResourceModel.PRODUCTION:{'type':'dataframe', 'unit': 'billion cubic metre', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'},
        ResourceModel.PAST_PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'natural_gas_resource'}
    }
    #MMBTU:(Metric Million British Thermal Unit)

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.gas_model = ResourceModel(param)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute

        gas_demand=pd.DataFrame(
            {'years': inp_dict['All_Demand']['years'].values})
        gas_demand ['natural_gas_resource'] = inp_dict['All_Demand']['natural_gas_resource']
        self.gas_model.compute(gas_demand,'natural_gas_resource',2010)

        years = np.arange(inp_dict['year_start'], inp_dict['year_end']+1)

        outputs_dict = {
            ResourceModel.RESOURCE_STOCK: self.gas_model.resource_stock,
            ResourceModel.RESOURCE_PRICE: self.gas_model.resource_price,
            ResourceModel.USE_STOCK: self.gas_model.use_stock,
            ResourceModel.PRODUCTION:self.gas_model.predictible_production,
            ResourceModel.PAST_PRODUCTION: self.gas_model.past_production
        }

        #-- store outputs
        self.store_sos_outputs_values(outputs_dict)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

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
        # get gas input demand
        inputs_dict = self.get_sosdisc_inputs()
        output_dict = self.get_sosdisc_outputs()
        gas_resource = 'natural_gas_resource'
        gas_demand = pd.DataFrame(
            {'years': inputs_dict['All_Demand']['years'].values})
        gas_demand['gas_resource'] = inputs_dict['All_Demand'][gas_resource]

        grad_stock, grad_price, grad_use = self.gas_model.get_derivative_resource(gas_resource)
        # # ------------------------------------------------
        # # Stock resource gradient
        for gas_type in output_dict['resource_stock']:
            self.set_partial_derivative_for_other_types(
                (ResourceModel.RESOURCE_STOCK, gas_type), (f'{ResourceModel.DEMAND}', gas_resource), grad_stock[gas_type])
        # # ------------------------------------------------
        # # Price resource gradient
        self.set_partial_derivative_for_other_types(
                (ResourceModel.RESOURCE_PRICE, 'price'), (f'{ResourceModel.DEMAND}', gas_resource), grad_price)
        # # ------------------------------------------------
        # # Use resource gradient
        for gas_type in output_dict['resource_stock']:
            self.set_partial_derivative_for_other_types(
                (ResourceModel.USE_STOCK, gas_type), (f'{ResourceModel.DEMAND}', gas_resource), grad_use[gas_type])
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
            production_start=self.get_sosdisc_inputs(ResourceModel.PRODUCTION_START)
            years_start=self.get_sosdisc_inputs(ResourceModel.YEAR_START)
            stock_df = self.get_sosdisc_outputs(
                ResourceModel.RESOURCE_STOCK)
            years = stock_df.index.values.tolist()
            price_df=self.get_sosdisc_outputs(
               ResourceModel.RESOURCE_PRICE)
            use_stock_df=self.get_sosdisc_outputs(
               ResourceModel.USE_STOCK)
            production_df = self.get_sosdisc_outputs(
                ResourceModel.PRODUCTION)
            past_production_df =self.get_sosdisc_outputs(ResourceModel.PAST_PRODUCTION)
            past_production_cut= past_production_df.loc[past_production_df['years']
                                              >= production_start]
            production_cut=production_df.loc[production_df.index
                                              <= years_start]
            production_years= production_df.index.values.tolist()
            past_production_year=past_production_df['years'].values.tolist()
            #two charts for stock evolution and price evolution

            stock_chart = TwoAxesInstanciatedChart('years','maximum stocks [bcm]',
                                                 chart_name='Gas stocks through the years', stacked_bar=False)
            price_chart = TwoAxesInstanciatedChart('Years', 'price [USD/MCF]',
                                                   chart_name='Gas price through the years', stacked_bar=False)

            production_chart = TwoAxesInstanciatedChart('years', 'gas production per subtypes [bcm]',
                                                                 chart_name='Gas production per subtypes through the years',
                                                                 stacked_bar=False)
            production_cumulated_chart=TwoAxesInstanciatedChart('Years', 'Gas production [bcm]',
                                                                 chart_name='Gas production through the years',stacked_bar=True)

            model_production_cumulated_chart = TwoAxesInstanciatedChart('years',
                                                                  'Comparison between model and real Gas production [Mt]',
                                                                  chart_name='Gas production through the years',
                                                                  stacked_bar=False)
            past_production_chart = TwoAxesInstanciatedChart('years',
                                                                  'Gas past production [Mt]',
                                                                  chart_name='Gas past production through the years',
                                                                  stacked_bar=False)
            use_stock_chart = TwoAxesInstanciatedChart('years','Gas use [bcm]',
                                                                 chart_name='Gas use per subtypes through the years',
                                                                 stacked_bar=False)
            use_stock_cumulated_chart = TwoAxesInstanciatedChart('years', 'Gas use per Subtypes [bcm]',
                                                                 chart_name='Gas use through the years',
                                                                 stacked_bar=True)

            for stock_kind in stock_df:
                stock_serie = InstanciatedSeries(
                    years, (stock_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.LINES_DISPLAY)
                stock_chart.add_series(stock_serie)

                production_serie = InstanciatedSeries(
                    production_years, (production_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.BAR_DISPLAY)
                production_chart.add_series(production_serie)

                production_cumulated_chart.add_series(production_serie)

                use_stock_serie = InstanciatedSeries(
                    years, (use_stock_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.BAR_DISPLAY)
                use_stock_chart.add_series(use_stock_serie)
                production_cut_series=InstanciatedSeries(
                    production_years, (production_cut[stock_kind]).values.tolist(), stock_kind +' predicted production', InstanciatedSeries.BAR_DISPLAY)
                past_production_series = InstanciatedSeries(
                    past_production_year, (past_production_df[stock_kind]).values.tolist(), stock_kind, InstanciatedSeries.LINES_DISPLAY)
                past_production_cut_series=InstanciatedSeries(
                    production_years, (past_production_cut[stock_kind]).values.tolist(), stock_kind+' real production', InstanciatedSeries.LINES_DISPLAY)
                past_production_chart.add_series(past_production_series)
                model_production_cumulated_chart.add_series(past_production_cut_series)
                model_production_cumulated_chart.add_series(production_cut_series)
                use_stock_cumulated_chart.add_series(use_stock_serie)

            price_serie = InstanciatedSeries(
                years, (price_df['price']).values.tolist(), 'gas price', InstanciatedSeries.LINES_DISPLAY)


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
