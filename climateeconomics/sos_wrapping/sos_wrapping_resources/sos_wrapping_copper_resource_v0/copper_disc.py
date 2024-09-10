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
import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_resources.new_resources_v0.copper_model import (
    CopperModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)

# from sostrades_core.api import SoSDiscipline, InstanciatedSeries, TwoAxesInstanciatedChart, ChartFilter
from climateeconomics.glossarycore import GlossaryCore


class CopperDisc(SoSWrapp):
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

    DESC_IN = {'copper_demand': {'type': 'dataframe', 'unit': 'Mt',
                                 'dataframe_descriptor': {'Year': ('float', None, False),
                                                          'Demand': ('float', None, False),
                                                          'unit': ('string', None, False), }
                                 },
               GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,

               'annual_extraction': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': 'Mt',
                                     'default': [26] * 81},
               'initial_copper_reserve': {'type': 'float', 'unit': 'Mt', 'default': 3500, 'user_level': 2},

               'initial_copper_stock': {'type': 'float', 'unit': 'Mt', 'default': 880, 'user_level': 2}}

    DESC_OUT = {CopperModel.COPPER_RESERVE: {'type': 'dataframe', 'unit': 'million_tonnes'},
                CopperModel.COPPER_STOCK: {'type': 'dataframe', 'unit': 'million_tonnes'},
                CopperModel.PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes'},
                CopperModel.COPPER_PRICE: {'type': 'dataframe', 'unit': 'USD'}}

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.copper_model = CopperModel(param)

    def run(self):

        copper_demand, year_start, year_end = self.get_sosdisc_inputs(
            ['copper_demand', GlossaryCore.YearStart, GlossaryCore.YearEnd])
        period_of_exploitation = np.arange(year_start, year_end + 1, 1)

        # call models

        self.copper_model.compute(copper_demand, period_of_exploitation)

        dict_values = {CopperModel.COPPER_RESERVE: self.copper_model.copper_reserve,
                       CopperModel.COPPER_STOCK: self.copper_model.copper_stock,
                       CopperModel.PRODUCTION: self.copper_model.copper_prod,
                       CopperModel.COPPER_PRICE: self.copper_model.copper_prod_price}

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
        charts_list = []
        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts_list = chart_filter.selected_values

        if 'all' in charts_list:
            period_of_exploitation = np.arange(
                self.DESC_IN[GlossaryCore.YearStart]['default'], self.DESC_IN[GlossaryCore.YearEnd]['default'] + 1, 1).tolist()

            production = self.get_sosdisc_outputs(CopperModel.PRODUCTION)
            stock = self.get_sosdisc_outputs(CopperModel.COPPER_STOCK)
            reserve = self.get_sosdisc_outputs(CopperModel.COPPER_RESERVE)
            price = self.get_sosdisc_outputs(CopperModel.COPPER_PRICE)

            production_list = production['World Production'].values
            cumulated_production_list = production['Cumulated World Production'].values
            ratio_list = production['Ratio'].values
            stock_list = stock['Stock'].values
            reserve_list = reserve['Reserve'].values
            price_evolution_list = price['Price/t'].values
            extraction_list = production['Extraction'].values

            chart_production = TwoAxesInstanciatedChart('Years [y]',
                                                        'Production [Mt]', chart_name="Copper Production")

            chart_stock = TwoAxesInstanciatedChart('Years [y]',
                                                   'Stock [Mt]', chart_name="Copper Use")

            chart_price_evolution = TwoAxesInstanciatedChart('Years [y]',
                                                             'Price/t [USD/t]', chart_name="Copper Price Evolution")

            chart_copper_situation = TwoAxesInstanciatedChart('Years [y]',
                                                              'Copper [Mt]', chart_name="Copper Repartition",
                                                              stacked_bar=True)

            chart_ratio = TwoAxesInstanciatedChart('Years [y]',
                                                   'Ratio [-]', chart_name="Copper Ratio extraction/demand")

            production_series = InstanciatedSeries(
                period_of_exploitation, production_list.tolist(), "Copper Production", 'lines')
            stock_series = InstanciatedSeries(
                period_of_exploitation, stock_list.tolist(), "Copper Stock", 'lines')
            reserve_series = InstanciatedSeries(
                period_of_exploitation, reserve_list.tolist(), "Copper Reserve", 'lines')
            price_evolution_series = InstanciatedSeries(
                period_of_exploitation, price_evolution_list.tolist(), "Copper Price Evolution", 'lines')
            extraction_series = InstanciatedSeries(
                period_of_exploitation, extraction_list.tolist(), "Copper Extraction", 'lines')
            ratio_series = InstanciatedSeries(
                period_of_exploitation, ratio_list.tolist(), "Copper Ratio", 'lines')

            bar_cumulated_production_serie = InstanciatedSeries(period_of_exploitation,
                                                                cumulated_production_list.tolist(
                                                                ), "Cumulated Copper Consumption",
                                                                InstanciatedSeries.BAR_DISPLAY)
            bar_stock_series = InstanciatedSeries(period_of_exploitation, stock_list.tolist(
            ), "Copper Stock", InstanciatedSeries.BAR_DISPLAY)
            bar_reserve_series = InstanciatedSeries(period_of_exploitation, reserve_list.tolist(
            ), "Copper Reserve", InstanciatedSeries.BAR_DISPLAY)

            chart_production.series.append(production_series)
            chart_ratio.series.append(ratio_series)
            chart_stock.series.append(stock_series)
            chart_stock.series.append(reserve_series)
            chart_stock.series.append(production_series)
            chart_stock.series.append(extraction_series)
            chart_price_evolution.series.append(price_evolution_series)

            chart_copper_situation.series.append(
                bar_cumulated_production_serie)
            chart_copper_situation.series.append(bar_stock_series)
            chart_copper_situation.series.append(bar_reserve_series)

            instanciated_charts.append(chart_production)
            instanciated_charts.append(chart_stock)
            instanciated_charts.append(chart_price_evolution)
            instanciated_charts.append(chart_copper_situation)
            instanciated_charts.append(chart_ratio)

        return instanciated_charts
