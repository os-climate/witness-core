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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart
from climateeconomics.core.core_resources.oil_availability_and_price_model.oil_availability_and_price_prediction_model import dataset, reserves_model, price_model
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class OilDiscipline(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Oil price and avaibility model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-oil-can fa-fw',
        'version': '',
    }

    default_year_start = 2020
    default_year_end = 2100
    
    DESC_IN = {'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'}}

    DESC_OUT = {
        "Predicted_prices":{'type': 'dataframe', 'unit': '[USD/barrel]'},
        "Reserves_prediction": {'type': 'dataframe', 'unit': '[billion.barrel]'},
        "Historical_stats":{'type': 'dataframe', 'unit': '[billion.barrel]'},
        "Production_proportions_2019":{'type': 'dataframe', 'unit': '[billion.barrel]'},
        "End_of_reserves":{'type': 'dataframe', 'unit': '[year]'},
        "Supply_Shortage":{'type': 'array', 'unit': '[year]'},
        "Historical_prices":{'type': 'dataframe', 'unit': '[USD/barrel]'},
        "Production_change":{'type': 'array', 'unit': '[billion.barrel/year]'},
        "Price_change":{'type': 'array', 'unit': '[USD/barrel/year]'},
        "Price_correction_funtion_training_data":{'type': 'dataframe', 'unit': '[billion.barrel/year,USD/barrel/year]'},
        "Truncated_historical_prices":{'type': 'dataframe', 'unit': '[USD/barrel]'},
        "Price_baseline_regression": {'type': 'dataframe', 'unit': '[year, USD/barrel]'},
        "Price_baseline_regression_predictions": {'type': 'dataframe', 'unit': '[year, USD/barrel]'}
    }

    

    def init_execution(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        self.dt = dataset()
        self.res = reserves_model(self.dt, inp_dict)
        self.pri = price_model(self.dt, self.res, inp_dict)

    def run(self):

        #-- get inputs
        inputs = list(self.DESC_IN.keys())
        inp_dict = self.get_sosdisc_inputs(inputs, in_dict=True)

        #-- compute
        self.dt = dataset()
        self.res = reserves_model(self.dt, inp_dict)
        self.pri = price_model(self.dt, self.res, inp_dict)
        
        outputs_dict = {
            "Predicted_prices": self.pri.prices_pred,
            "Historical_stats":self.dt.df,
            "Reserves_prediction": self.res.preds,
            "Production_proportions_2019": self.res.production_proportions_2019,
            "End_of_reserves":self.res.eor,
            "Supply_Shortage":self.res.shortage_world_reserves,
            "Historical_prices":self.pri.pr,
            "Production_change":self.pri.drop,
            "Price_change":self.pri.pr_increase,
            "Price_correction_funtion_training_data":self.pri.dr,
            "Truncated_historical_prices":self.pri.price,
            "Price_baseline_regression": self.pri.baseline_reg,
            "Price_baseline_regression_predictions": self.pri.baseline_reg_pred
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

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values


        if 'all' in chart_list:

            prices = self.get_sosdisc_outputs("Predicted_prices")
            reserves = self.get_sosdisc_outputs("Reserves_prediction")
            df = self.get_sosdisc_outputs("Historical_stats")
            end_of_reserves = self.get_sosdisc_outputs("End_of_reserves")
            supply_shortage = self.get_sosdisc_outputs("Supply_Shortage")
            price = self.get_sosdisc_outputs("Truncated_historical_prices")
            price_prod_change = self.get_sosdisc_outputs("Price_correction_funtion_training_data")
            price_baseline_reg = self.get_sosdisc_outputs("Price_baseline_regression")
            price_baseline_reg_pred = self.get_sosdisc_outputs("Price_baseline_regression_predictions")

            # predicted price evolution _ equivalent to self.pri.plot_price_preds()

            price_chart = TwoAxesInstanciatedChart('years','price [USD/bl]', chart_name='Oil price through the years', stacked_bar=False)
            price_serie = InstanciatedSeries(prices["year"].values.tolist(), prices["price"].values.tolist(), "Oil price", InstanciatedSeries.LINES_DISPLAY)
            price_chart.add_series(price_serie)

            # predicted world reserves _ equivalent to self.res.world_reserves() first plot

            world_reserves_chart = TwoAxesInstanciatedChart('years','World crude oil reserves [bbl]', chart_name='Crude oil world reserves predictions', stacked_bar=False)
            world_reserves_serie = InstanciatedSeries(reserves["Year"].values.tolist(), reserves["World"].values.tolist(), "World crude oil reserves [bbl]", InstanciatedSeries.LINES_DISPLAY)
            world_reserves_chart.add_series(world_reserves_serie)

            # predicted reserves by country_ equivalent to self.res.world_reserves() second plot

            country_reserves_chart = TwoAxesInstanciatedChart('years','Crude oil reserves by country [bbl]', chart_name='Crude oil reserves predictions by country', stacked_bar=False)
            for country in df.Country.unique():
                country_reserves_serie = InstanciatedSeries(reserves["Year"].values.tolist(), reserves[country].values.tolist(), f"{country} crude oil reserves [bbl]", InstanciatedSeries.LINES_DISPLAY)
                country_reserves_chart.add_series(country_reserves_serie)

            # shortage along world crude oil reserves predictions _ equivalent to self.res.plot_world_reserves_vs_country_shortage()

            # using native plotly
            fig = go .Figure()
            fig.add_trace(go.Scatter(x = end_of_reserves["end_year"].values.tolist(), y= supply_shortage.values.tolist(), text = end_of_reserves["country"].values.tolist(),textposition="top center", mode = 'markers+text', name='countries shortage'))
            fig.add_trace(go.Scatter(x= reserves["Year"].values.tolist(), y = reserves["World"].values.tolist(), mode = 'lines', name = 'World crude oil reserves'))
            
            # edit the layout

            fig.update_layout(xaxis_title = 'Year', yaxis_title = 'Crude oil reserves (bbl)')

            chart_name = 'Shortage in crude oil supply by country along world reserves predictions'
            
            native_shortage_chart = InstantiatedPlotlyNativeChart(fig, chart_name)

            # baseline price evolution _ equivalent to self.pri.plot_baseline()
            

            baseline_price_chart = TwoAxesInstanciatedChart('years','price [USD/bl]', chart_name='Baseline price regression', stacked_bar=False)

            # plot training points
            baseline_training_serie = InstanciatedSeries(price_baseline_reg['year_training'].values.tolist(), price_baseline_reg['price_training'].values.tolist(), "Training points", InstanciatedSeries.SCATTER_DISPLAY)
            baseline_trained_serie = InstanciatedSeries(price_baseline_reg['year_training'].values.tolist(), price_baseline_reg['price_trained'].values.tolist(), "Baseline price predictions on training points", InstanciatedSeries.LINES_DISPLAY)
            baseline_pred_serie = InstanciatedSeries(price_baseline_reg_pred['year_pred'].values.tolist(), price_baseline_reg_pred['price_pred'].values.tolist(), "Baseline price predictions", InstanciatedSeries.LINES_DISPLAY)
            baseline_price_chart.add_series(baseline_training_serie)
            baseline_price_chart.add_series(baseline_trained_serie)
            baseline_price_chart.add_series(baseline_pred_serie)

            # price/production change function _ equivalent to self.pri.plot_price_vs_prod_change()

            price_vs_prod_change_chart = TwoAxesInstanciatedChart('production change [bbl]','price change [USD/bl]', chart_name='Function for crude oil price correction vs production changes', stacked_bar=False)
            price_vs_prod_change_serie = InstanciatedSeries(price_prod_change["production_change"].values.tolist(), price_prod_change["price_change"].values.tolist(), "Training points", InstanciatedSeries.LINES_DISPLAY)
            price_vs_prod_change_pred_serie = InstanciatedSeries(np.arange(-1, 0.5, .1).tolist(), self.pri.pr_reg.predict(np.arange(-1, 0.5, .1).reshape(-1, 1)).tolist(), "predictions", InstanciatedSeries.SCATTER_DISPLAY)
            price_vs_prod_change_chart.add_series(price_vs_prod_change_serie)
            price_vs_prod_change_chart.add_series(price_vs_prod_change_pred_serie)

            # add all chart to instanciated chart
            instanciated_charts.append(price_chart)
            instanciated_charts.append(world_reserves_chart)
            instanciated_charts.append(country_reserves_chart)
            instanciated_charts.append(native_shortage_chart)
            instanciated_charts.append(baseline_price_chart)
            instanciated_charts.append(price_vs_prod_change_chart)

        return instanciated_charts