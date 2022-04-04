# from sos_trades_core.api import SoSDiscipline, InstanciatedSeries, TwoAxesInstanciatedChart, ChartFilter
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter

from climateeconomics.core.core_resources.new_resources_v0.copper_model import CopperModel

import numpy as np


class CopperDisc(SoSDiscipline):

    _ontology_data = {
        'label': 'Copper Resource Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-reel',
        'version': '',
    }
    _maturity = 'Fake'

    DESC_IN = { 'copper_demand': {'type': 'dataframe', 'unit': 'Mt'},
                'year_start': {'type' : 'int', 'default': CopperModel.YEAR_START, 'unit': '[-]'},
                'year_end': {'type': 'int', 'default': CopperModel.YEAR_END, 'unit': '[-]'},
                'annual_extraction' : {'type' : 'float_list', 'unit' : 'Mt', 'default' : [26] * 81}}

    DESC_OUT = { CopperModel.COPPER_RESERVE: {'type': 'dataframe', 'unit': 'million_tonnes'},
                 CopperModel.COPPER_STOCK: {'type': 'dataframe', 'unit': 'million_tonnes'},
                 CopperModel.PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes'},
                 CopperModel.COPPER_PRICE : {'type': 'dataframe', 'unit': 'USD'}}

    
    

        
    def run(self):
        period_of_exploitation = np.arange(self.DESC_IN['year_start']['default'], self.DESC_IN['year_end']['default'] + 1, 1)

        # call models
        copper_demand, annual_extraction = self.get_sosdisc_inputs(['copper_demand', 'annual_extraction'])

        self.copper_model = CopperModel(copper_demand, annual_extraction)
        self.copper_model.compute(copper_demand, period_of_exploitation)

        dict_values = { CopperModel.COPPER_RESERVE : self.copper_model.copper_reserve,
                        CopperModel.COPPER_STOCK : self.copper_model.copper_stock,
                        CopperModel.PRODUCTION : self.copper_model.copper_prod,
                        CopperModel.COPPER_PRICE : self.copper_model.copper_prod_price}

        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)


    
    


    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['all']

        chart_filters.append(ChartFilter(
            'Charts filter', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts_list = chart_filter.selected_values

        if 'all' in charts_list:

            period_of_exploitation = np.arange(self.DESC_IN['year_start']['default'], self.DESC_IN['year_end']['default'] + 1, 1).tolist()

            production = self.get_sosdisc_outputs(CopperModel.PRODUCTION)
            stock = self.get_sosdisc_outputs(CopperModel.COPPER_STOCK)
            reserve = self.get_sosdisc_outputs(CopperModel.COPPER_RESERVE)
            price = self.get_sosdisc_outputs(CopperModel.COPPER_PRICE)

            production_list = production['World Production'].values
            cumulated_production_list = production['Cumulated World Production'].values
            stock_list = stock['Stock'].values
            reserve_list = reserve['Reserve'].values
            price_evolution_list = price['Price/Mt'].values
            extraction_list = production['Extraction'].values

    


            chart_production = TwoAxesInstanciatedChart('Years [y]', 
                                                 'Production [Mt]',  chart_name="Copper Production")

            chart_stock = TwoAxesInstanciatedChart('Years [y]', 
                                                   'Stock [Mt]', chart_name="Copper Use")

            chart_price_evolution = TwoAxesInstanciatedChart('Years [y]', 
                                                             'Price/Mt [USD]', chart_name="Copper Price Evolution")
            
            chart_copper_situation = TwoAxesInstanciatedChart('Years [y]', 
                                                              'Copper [Mt]', chart_name="Copper Repartition", stacked_bar=True)



            production_series = InstanciatedSeries(period_of_exploitation, production_list.tolist(), "Copper Production",'lines')
            stock_series = InstanciatedSeries(period_of_exploitation, stock_list.tolist(), "Copper Stock", 'lines')
            reserve_series = InstanciatedSeries(period_of_exploitation, reserve_list.tolist(), "Copper Reserve", 'lines')
            price_evolution_series = InstanciatedSeries(period_of_exploitation, price_evolution_list.tolist(), "Copper Price Evolution", 'lines')
            extraction_series = InstanciatedSeries(period_of_exploitation, extraction_list.tolist(), "Copper Extraction", 'lines')

            bar_cumulated_production_serie = InstanciatedSeries(period_of_exploitation, cumulated_production_list.tolist(), "Cumulated Copper Production", InstanciatedSeries.BAR_DISPLAY)
            bar_stock_series = InstanciatedSeries(period_of_exploitation, stock_list.tolist(), "Copper Stock", InstanciatedSeries.BAR_DISPLAY)
            bar_reserve_series = InstanciatedSeries(period_of_exploitation, reserve_list.tolist(), "Copper Reserve", InstanciatedSeries.BAR_DISPLAY)

            chart_production.series.append(production_series)
            chart_stock.series.append(stock_series)
            chart_stock.series.append(reserve_series)
            chart_stock.series.append(production_series)
            chart_stock.series.append(extraction_series)
            chart_price_evolution.series.append(price_evolution_series)

            chart_copper_situation.series.append(bar_cumulated_production_serie)
            chart_copper_situation.series.append(bar_stock_series)
            chart_copper_situation.series.append(bar_reserve_series)

            instanciated_charts.append(chart_production)
            instanciated_charts.append(chart_stock)
            instanciated_charts.append(chart_price_evolution)
            instanciated_charts.append(chart_copper_situation)

        return instanciated_charts
