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
from climateeconomics.core.core_resources.all_resources_model import AllResourceModel
from climateeconomics.core.core_resources.resources_model import ResourceModel
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries,\
    TwoAxesInstanciatedChart
import numpy as np
import pandas as pd


class AllResourceDiscipline(SoSDiscipline):
    ''' Discipline intended to agregate resource parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'All Resources Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-globe fa-fw',
        'version': '',
    }
    default_year_start = 2020
    default_year_end = 2050
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    years_default = np.arange(2020, 2051)
    ratio_available_resource_default = pd.DataFrame(
        {'years': np.arange(2020, 2050 + 1)})
    for resource in AllResourceModel.RESOURCE_LIST:
        ratio_available_resource_default[resource] = np.linspace(
            1.0, 1.0, len(ratio_available_resource_default.index))

    DESC_IN = {'year_start': {'type': 'int', 'default': default_year_start, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'year_end': {'type': 'int', 'default': default_year_end, 'unit': '[-]', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_public'},
               'resource_list': {'type': 'string_list', 'default': AllResourceModel.RESOURCE_LIST, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource', 'editable': False, 'structuring': True},
               AllResourceModel.NON_MODELED_RESOURCE_PRICE: {'type': 'dataframe', 'unit': '$/t', 'namespace': 'ns_resource'},
               ResourceModel.DEMAND: {'type': 'dataframe', 'unit': 'Mt',
                                      'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource'}}

    DESC_OUT = {
        AllResourceModel.ALL_RESOURCE_STOCK: {
            'type': 'dataframe', 'unit': 'million_tonnes'},
        AllResourceModel.ALL_RESOURCE_PRICE: {
            'type': 'dataframe', 'unit': 'USD/USton'},
        AllResourceModel.All_RESOURCE_USE: {'type': 'dataframe', 'unit': 'million_tonnes'},
        AllResourceModel.ALL_RESOURCE_PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes'},
        AllResourceModel.RATIO_USABLE_DEMAND: {'type': 'dataframe', 'default': ratio_available_resource_default, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource'},
        AllResourceModel.ALL_RESOURCE_DEMAND: {'type': 'dataframe', 'unit': '-',
                                               'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_resource'}

    }

    def init_execution(self):
        inputs = list(self.DESC_IN.keys())
        param = self.get_sosdisc_inputs(inputs, in_dict=True)
        self.all_resource_model = AllResourceModel(param)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        #dynamic_outputs = {}

        if 'resource_list' in self._data_in:
            resource_list = self.get_sosdisc_inputs('resource_list')
            for resource in resource_list:
                dynamic_inputs[f'{resource}.resource_price'] = {
                    'type': 'dataframe'}
                dynamic_inputs[f'{resource}.resource_stock'] = {
                    'type': 'dataframe'}
                dynamic_inputs[f'{resource}.use_stock'] = {'type': 'dataframe'}
                dynamic_inputs[f'{resource}.predictible_production'] = {
                    'type': 'dataframe'}
            self.add_inputs(dynamic_inputs)
        # self.add_outputs(dynamic_outputs)

    def run(self):

        #-- get inputs
        inp_dict = self.get_sosdisc_inputs()

        #-- compute

        self.all_resource_model.compute(inp_dict)

        years = np.arange(inp_dict['year_start'], inp_dict['year_end'] + 1)

        outputs_dict = {
            AllResourceModel.ALL_RESOURCE_STOCK: self.all_resource_model.all_resource_stock,
            AllResourceModel.ALL_RESOURCE_PRICE: self.all_resource_model.all_resource_price,
            AllResourceModel.All_RESOURCE_USE: self.all_resource_model.all_resource_use,
            AllResourceModel.ALL_RESOURCE_PRODUCTION: self.all_resource_model.all_resource_production,
            AllResourceModel.RATIO_USABLE_DEMAND: self.all_resource_model.all_resource_ratio_usable_demand,
            AllResourceModel.ALL_RESOURCE_DEMAND: self.all_resource_model.resource_demand
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
        inputs_dict = self.get_sosdisc_inputs()
        resource_list = self.get_sosdisc_inputs('resource_list')
        output_dict = self.get_sosdisc_outputs()
        for resource_type in resource_list:
            grad_price, grad_use, grad_stock = self.all_resource_model.get_derivative_all_resource(
                inputs_dict, resource_type)
            grad_use_ratio_on_use, grad_use_ratio_on_demand = self.all_resource_model.get_derivative_ratio(
                inputs_dict, resource_type, grad_use, output_dict)

            for types in inputs_dict[f'{resource_type}.use_stock']:
                if types != 'years':
                    self.set_partial_derivative_for_other_types(
                        (AllResourceModel.All_RESOURCE_USE, resource_type), (f'{resource_type}.use_stock', types), grad_use)
                    self.set_partial_derivative_for_other_types(
                        (AllResourceModel.RATIO_USABLE_DEMAND, resource_type), (f'{resource_type}.use_stock', types), grad_use_ratio_on_use)

            self.set_partial_derivative_for_other_types((AllResourceModel.RATIO_USABLE_DEMAND, resource_type), (
                ResourceModel.DEMAND, resource_type), grad_use_ratio_on_demand)

            for types in inputs_dict[f'{resource_type}.resource_stock']:
                if types != 'years':
                    self.set_partial_derivative_for_other_types(
                        (AllResourceModel.ALL_RESOURCE_STOCK, resource_type), (f'{resource_type}.resource_stock', types), grad_stock)

            self.set_partial_derivative_for_other_types(
                (AllResourceModel.ALL_RESOURCE_PRICE, resource_type), (f'{resource_type}.resource_price', 'price'), grad_price)
        data_frame_other_resource_price = inputs_dict['non_modeled_resource_price']

        for resource_type in data_frame_other_resource_price:
            if resource_type not in resource_list and resource_type != 'years':
                self.set_partial_derivative_for_other_types((AllResourceModel.ALL_RESOURCE_PRICE, resource_type), (
                    AllResourceModel.NON_MODELED_RESOURCE_PRICE, resource_type), np.identity(len(data_frame_other_resource_price)))

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'all' in chart_list:

            stock_df = self.get_sosdisc_outputs(
                AllResourceModel.ALL_RESOURCE_STOCK)
            years = stock_df.index.values.tolist()
            price_df = self.get_sosdisc_outputs(
                AllResourceModel.ALL_RESOURCE_PRICE)
            use_stock_df = self.get_sosdisc_outputs(
                AllResourceModel.All_RESOURCE_USE)
            production_df = self.get_sosdisc_outputs(
                AllResourceModel.ALL_RESOURCE_PRODUCTION)
            ratio_use_df = self.get_sosdisc_outputs(
                AllResourceModel.RATIO_USABLE_DEMAND)
            demand_df = self.get_sosdisc_outputs(
                AllResourceModel.ALL_RESOURCE_DEMAND)

            # two charts for stock evolution and price evolution
            stock_chart = TwoAxesInstanciatedChart('years', 'stocks (Mt)',
                                                   chart_name='Resources stocks through the years', stacked_bar=False)
            price_chart = TwoAxesInstanciatedChart('years', 'price ($/t)',
                                                   chart_name='Resource price through the years', stacked_bar=False)
            use_stock_chart = TwoAxesInstanciatedChart('years', 'resource use (Mt) ',
                                                       chart_name='Resource use through the years', stacked_bar=False)
            production_chart = TwoAxesInstanciatedChart('years',
                                                        'resource production (Mt)',
                                                        chart_name='Resource production through the years',
                                                        stacked_bar=False)
            ratio_use_demand_chart = TwoAxesInstanciatedChart(
                'years', 'ratio usable stock / demand ', chart_name='ratio usable stock and prod on demand through the years', stacked_bar=False)
            resource_demand_chart = TwoAxesInstanciatedChart(
                'years', 'demand (Mt)', chart_name='resource demand through the years', stacked_bar=False)
            for resource_kind in stock_df:
                if resource_kind != 'years':
                    stock_serie = InstanciatedSeries(
                        years, (stock_df[resource_kind]).values.tolist(), resource_kind, InstanciatedSeries.LINES_DISPLAY)
                    stock_chart.add_series(stock_serie)

                    production_serie = InstanciatedSeries(
                        years, (production_df[resource_kind]).values.tolist(), resource_kind, InstanciatedSeries.BAR_DISPLAY)
                    production_chart.add_series(production_serie)

                    use_stock_serie = InstanciatedSeries(
                        years, (use_stock_df[resource_kind]).values.tolist(), resource_kind, InstanciatedSeries.BAR_DISPLAY)
                    use_stock_chart.add_series(use_stock_serie)
                    ratio_use_serie = InstanciatedSeries(
                        years, (ratio_use_df[resource_kind]).values.tolist(), resource_kind, InstanciatedSeries.LINES_DISPLAY)
                    ratio_use_demand_chart.add_series(ratio_use_serie)
                    demand_serie = InstanciatedSeries(
                        years, (demand_df[resource_kind]).values.tolist(), resource_kind, InstanciatedSeries.LINES_DISPLAY)
                    resource_demand_chart.add_series(demand_serie)
            for resource_types in price_df:
                if resource_types != 'years':
                    price_serie = InstanciatedSeries(years, (price_df[resource_types]).values.tolist(
                    ), resource_types, InstanciatedSeries.LINES_DISPLAY)
                    price_chart.add_series(price_serie)

            instanciated_charts.append(stock_chart)
            instanciated_charts.append(price_chart)
            instanciated_charts.append(use_stock_chart)
            instanciated_charts.append(production_chart)
            instanciated_charts.append(ratio_use_demand_chart)
            instanciated_charts.append(resource_demand_chart)

        return instanciated_charts
