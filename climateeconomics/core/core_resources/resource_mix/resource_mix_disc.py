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
import numpy as np
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc import (
    CoalResourceDiscipline,
)
from climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc import (
    CopperResourceDiscipline,
)
from climateeconomics.core.core_resources.models.natural_gas_resource.natural_gas_resource_disc import (
    NaturalGasResourceDiscipline,
)
from climateeconomics.core.core_resources.models.oil_resource.oil_resource_disc import (
    OilResourceDiscipline,
)
from climateeconomics.core.core_resources.models.platinum_resource.platinum_resource_disc import (
    PlatinumResourceDiscipline,
)
from climateeconomics.core.core_resources.models.uranium_resource.uranium_resource_disc import (
    UraniumResourceDiscipline,
)
from climateeconomics.core.core_resources.resource_mix.resource_mix import (
    ResourceMixModel,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class ResourceMixDiscipline(SoSWrapp):
    ''' Discipline intended to agregate resource parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Resource Mix Model',
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
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    years_default = np.arange(GlossaryCore.YearStartDefault, 2051)
    ratio_available_resource_default = pd.DataFrame(
        {GlossaryCore.Years: np.arange(GlossaryCore.YearStartDefault, 2050 + 1)})
    for resource in ResourceMixModel.RESOURCE_LIST:
        ratio_available_resource_default[resource] = np.linspace(
            1.0, 1.0, len(ratio_available_resource_default.index))
    default_conversion_dict = {
        UraniumResourceDiscipline.resource_name:
            {'price': (1 / 0.001102) * 0.907185,
             'production': 10 ** -6, 'stock': 10 ** -6, 
             'global_demand': 1.0},
        CoalResourceDiscipline.resource_name:
            {'price': 0.907185, 'production': 1.0, 'stock': 1.0,
            'global_demand': 1.0},
        NaturalGasResourceDiscipline.resource_name:
            {'price': 1.379 * 35310700 * 10 ** -6,
             'production': 1 / 1.379, 'stock': 1 / 1.379,
             'global_demand': 1.0},
        OilResourceDiscipline.resource_name:
            {'price': 7.33, 'production': 1.0, 'stock': 1.0,
            'global_demand': 1.0},
        CopperResourceDiscipline.resource_name:
            {'price': 1.0, 'production': 1.0, 'stock': 1.0,
            'global_demand': 24.987 / 0.000213421},
        PlatinumResourceDiscipline.resource_name:
            {'price': 1.0, 'production': 1.0, 'stock': 1.0,
             'global_demand': 1.0},

    }

    DESC_IN = {GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
               GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
               'resource_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                                 'unit': '-',
                                 'default': ResourceMixModel.RESOURCE_LIST,
                                 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_resource',
                                 'editable': False, 'structuring': True},
               ResourceMixModel.NON_MODELED_RESOURCE_PRICE: {'type': 'dataframe', 'unit': '$/t',
                                                             'namespace': 'ns_resource',
                                                             'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                                                                      GlossaryCore.CO2: ('float', None, True),
                                                                                      'uranium_resource': ('float', None, True),
                                                                                      'biomass_dry': ('float', None, True),
                                                                                      'biomass_dry_resource': ('float', None, True),
                                                                                      'wet_biomass': ('float', None, True),
                                                                                      'wood': ('float', None, True),
                                                                                      'carbon': ('float', None, True),
                                                                                      'oil_resource': ('float', None, True),
                                                                                      'NaturalOil': ('float', None, True),
                                                                                      'Methanol': ('float', None, True),
                                                                                      'Sodium_Hydroxyde': ('float', None, True),
                                                                                      'Potassium_Hydroxyde': ('float', None, True),
                                                                                      'oxygen': ('float', None, True),
                                                                                      'calcium': ('float', None, True),
                                                                                      'potassium': ('float', None, True),
                                                                                      'amine': ('float', None, True),
                                                                                      'sea_water': ('float', None, True),
                                                                                      'water': ('float', None, True),
                                                                                      'water_resource': (
                                                                                      'float', None, True),
                                                                                      'sea_water_resource': (
                                                                                      'float', None, True),
                                                                                      'CO2_resource': (
                                                                                      'float', None, True),
                                                                                      'wet_biomass_resource': (
                                                                                      'float', None, True),
                                                                                      'natural_oil_resource': (
                                                                                      'float', None, True),
                                                                                      'methanol_resource': (
                                                                                      'float', None, True),
                                                                                      'sodium_hydroxide_resource': (
                                                                                      'float', None, True),
                                                                                      'wood_resource': (
                                                                                      'float', None, True),
                                                                                      'carbon_resource': (
                                                                                      'float', None, True),
                                                                                      'managed_wood_resource': (
                                                                                      'float', None, True),
                                                                                      'oxygen_resource': (
                                                                                      'float', None, True),
                                                                                      'dioxygen_resource': (
                                                                                      'float', None, True),
                                                                                      'crude_oil_resource': (
                                                                                      'float', None, True),
                                                                                      'solid_fuel_resource': (
                                                                                      'float', None, True),
                                                                                      'calcium_resource': (
                                                                                      'float', None, True),
                                                                                      'calcium_oxyde_resource': (
                                                                                      'float', None, True),
                                                                                      'potassium_resource': (
                                                                                      'float', None, True),
                                                                                      'potassium_hydroxide_resource': (
                                                                                      'float', None, True),
                                                                                      'amine_resource': (
                                                                                      'float', None, True),
                                                                                      'ethanol_amine_resource': (
                                                                                      'float', None, True),
                                                                                      'mono_ethanol_amine_resource': (
                                                                                      'float', None, True),
                                                                                      'glycerol_resource': (
                                                                                      'float', None, True),
                                                                                      'platinum_resource': (
                                                                                      'float', None, True),
                                                                                      }
                                                             },
               'resources_demand': {'type': 'dataframe', 'unit': 'Mt',
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_resource',
                                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                                             'natural_gas_resource': ('float', None, True),
                                                             'uranium_resource': ('float', None, True),
                                                             'coal_resource': ('float', None, True),
                                                             'oil_resource': ('float', None, True),
                                                             'copper_resource': ('float', None, True),
                                                             'platinum_resource': ('float', None, True),}
                                    },
               'resources_demand_woratio': {'type': 'dataframe', 'unit': 'Mt',
                                            'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_resource',
                                            'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                                                     'natural_gas_resource': ('float', None, True),
                                                                     'uranium_resource': ('float', None, True),
                                                                     'coal_resource': ('float', None, True),
                                                                     'oil_resource': ('float', None, True),
                                                                     'copper_resource': ('float', None, True),
                                                                     'platinum_resource': ('float', None, True),
                                                                     }
                                            },
               'conversion_dict': {'type': 'dict', 'subtype_descriptor': {'dict': {'dict': 'float'}}, 'unit': '[-]', 'default': default_conversion_dict,
                                   'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_resource'}
               }

    DESC_OUT = {
        ResourceMixModel.ALL_RESOURCE_STOCK: {
            'type': 'dataframe', 'unit': 'million_tonnes'},
        GlossaryCore.ResourcesPriceValue: GlossaryCore.ResourcesPrice,
        ResourceMixModel.All_RESOURCE_USE: {'type': 'dataframe', 'unit': 'million_tonnes'},
        ResourceMixModel.ALL_RESOURCE_PRODUCTION: {'type': 'dataframe', 'unit': 'million_tonnes'},
        ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION:  {'type': 'dataframe', 'unit': 'million_tonnes'} ,
        ResourceMixModel.RATIO_USABLE_DEMAND: {'type': 'dataframe', 'default': ratio_available_resource_default,
                                               'visibility': SoSWrapp.SHARED_VISIBILITY, 'unit': '%',
                                               'namespace': 'ns_resource',},
        ResourceMixModel.ALL_RESOURCE_DEMAND: {'type': 'dataframe', 'unit': '-',
                                               'visibility': SoSWrapp.SHARED_VISIBILITY,
                                               'namespace': 'ns_resource'},
        ResourceMixModel.ALL_RESOURCE_CO2_EMISSIONS: {
            'type': 'dataframe', 'unit': 'kgCO2/kg', 'visibility': SoSWrapp.SHARED_VISIBILITY,
            'namespace': 'ns_resource'},
    }

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.all_resource_model = ResourceMixModel(inputs_dict)
        self.all_resource_model.configure_parameters(inputs_dict)

    def setup_sos_disciplines(self):
        dynamic_inputs = {}
        # dynamic_outputs = {}

        if 'resource_list' in self.get_data_in():
            resource_list = self.get_sosdisc_inputs('resource_list')
            for resource in resource_list:
                dynamic_inputs[f'{resource}.resource_price'] = {
                    'type': 'dataframe', 'unit': ResourceMixModel.RESOURCE_PRICE_UNIT[resource],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                             'price': ('float', None, True)}}
                dynamic_inputs[f'{resource}.resource_stock'] = {
                    'type': 'dataframe', 'unit': ResourceMixModel.RESOURCE_STOCK_UNIT[resource],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                             'heavy': ('float', None, True),
                                             'uranium_40': ('float', None, True),
                                             'uranium_80': ('float', None, True),
                                             'uranium_130': ('float', None, True),
                                             'uranium_260': ('float', None, True),
                                             'Conventional': ('float', None, True),
                                             'tight': ('float', None, True),
                                             'Coalbed_methane': ('float', None, True),
                                             'shale': ('float', None, True),
                                             'other': ('float', None, True),
                                             'sub_bituminous_and_lignite': ('float', None, True),
                                             'bituminous_and_anthracite': ('float', None, True),
                                             'copper': ('float', None, True),
                                             'medium': ('float', None, True),
                                             'unassigned_production': ('float', None, True),
                                             'light': ('float', None, True),}}
                dynamic_inputs[f'{resource}.use_stock'] = {
                    'type': 'dataframe', 'unit': ResourceMixModel.RESOURCE_STOCK_UNIT[resource],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                             'heavy': ('float', None, True),
                                             'uranium_40': ('float', None, True),
                                             'uranium_80': ('float', None, True),
                                             'uranium_130': ('float', None, True),
                                             'uranium_260': ('float', None, True),
                                             'Conventional': ('float', None, True),
                                             'tight': ('float', None, True),
                                             'Coalbed_methane': ('float', None, True),
                                             'shale': ('float', None, True),
                                             'other': ('float', None, True),
                                             'sub_bituminous_and_lignite': ('float', None, True),
                                             'bituminous_and_anthracite': ('float', None, True),
                                             'copper': ('float', None, True),
                                             'medium': ('float', None, True),
                                             'unassigned_production': ('float', None, True),
                                             'light': ('float', None, True),
                                             }}
                dynamic_inputs[f'{resource}.recycled_production'] = {
                    'type': 'dataframe', 'unit': ResourceMixModel.RESOURCE_STOCK_UNIT[resource] ,
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                             'heavy': ('float', None, True),
                                             'uranium_40': ('float', None, True),
                                             'uranium_80': ('float', None, True),
                                             'uranium_130': ('float', None, True),
                                             'uranium_260': ('float', None, True),
                                             'Conventional': ('float', None, True),
                                             'tight': ('float', None, True),
                                             'Coalbed_methane': ('float', None, True),
                                             'shale': ('float', None, True),
                                             'other': ('float', None, True),
                                             'sub_bituminous_and_lignite': ('float', None, True),
                                             'bituminous_and_anthracite': ('float', None, True),
                                             'copper': ('float', None, True),
                                             'medium': ('float', None, True),
                                             'unassigned_production': ('float', None, True),
                                             'light': ('float', None, True),
                                             }}
                dynamic_inputs[f'{resource}.predictable_production'] = {
                    'type': 'dataframe', 'unit': ResourceMixModel.RESOURCE_PROD_UNIT[resource],
                    'dataframe_descriptor': {GlossaryCore.Years: ('float', None, True),
                                             'heavy': ('float', None, True),
                                             'uranium_40': ('float', None, True),
                                             'uranium_80': ('float', None, True),
                                             'uranium_130': ('float', None, True),
                                             'uranium_260': ('float', None, True),
                                             'Conventional': ('float', None, True),
                                             'tight': ('float', None, True),
                                             'Coalbed_methane': ('float', None, True),
                                             'shale': ('float', None, True),
                                             'other': ('float', None, True),
                                             'sub_bituminous_and_lignite': ('float', None, True),
                                             'bituminous_and_anthracite': ('float', None, True),
                                             'copper': ('float', None, True),
                                             'medium': ('float', None, True),
                                             'unassigned_production': ('float', None, True),
                                             'light': ('float', None, True),
                                             }}
                
            self.add_inputs(dynamic_inputs)


    def run(self):

        # -- get inputs
        inputs_dict = self.get_sosdisc_inputs()
        # -- configure class with inputs
        self.all_resource_model.configure_parameters_update(inputs_dict)

        # -- compute
        self.all_resource_model.compute(inputs_dict)

        years = np.arange(inputs_dict[GlossaryCore.YearStart],
                          inputs_dict[GlossaryCore.YearEnd] + 1)

        outputs_dict = {
            ResourceMixModel.ALL_RESOURCE_STOCK: self.all_resource_model.all_resource_stock.reset_index(),
            GlossaryCore.ResourcesPriceValue: self.all_resource_model.all_resource_price.reset_index(),
            ResourceMixModel.All_RESOURCE_USE: self.all_resource_model.all_resource_use.reset_index(),
            ResourceMixModel.ALL_RESOURCE_PRODUCTION: self.all_resource_model.all_resource_production.reset_index(),
            ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION: self.all_resource_model.all_resource_recycled_production.reset_index(),
            ResourceMixModel.RATIO_USABLE_DEMAND: self.all_resource_model.all_resource_ratio_usable_demand.reset_index(),
            ResourceMixModel.ALL_RESOURCE_DEMAND: self.all_resource_model.resource_demand.reset_index(),
            ResourceMixModel.ALL_RESOURCE_CO2_EMISSIONS: self.all_resource_model.all_resource_co2_emissions.reset_index(),
        }

        
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
            grad_price, grad_use, grad_stock, grad_recycling,  grad_demand = self.all_resource_model.get_derivative_all_resource(
                inputs_dict, resource_type)
            grad_ratio_on_stock, grad_ratio_on_demand, grad_ratio_on_recycling = self.all_resource_model.get_derivative_ratio(
                inputs_dict, resource_type, output_dict)

            for types in inputs_dict[f'{resource_type}.use_stock']:
                if types != GlossaryCore.Years:
                    self.set_partial_derivative_for_other_types(
                        (ResourceMixModel.All_RESOURCE_USE,
                         resource_type), (f'{resource_type}.use_stock', types),
                        grad_use)

            self.set_partial_derivative_for_other_types((ResourceMixModel.RATIO_USABLE_DEMAND, resource_type), (
                'resources_demand_woratio', resource_type), grad_ratio_on_demand)

            for types in inputs_dict[f'{resource_type}.resource_stock']:
                if types != GlossaryCore.Years:
                    self.set_partial_derivative_for_other_types(
                        (ResourceMixModel.ALL_RESOURCE_STOCK, resource_type),
                        (f'{resource_type}.resource_stock', types), grad_stock)
                    self.set_partial_derivative_for_other_types(
                        (ResourceMixModel.RATIO_USABLE_DEMAND, resource_type),
                        (f'{resource_type}.resource_stock', types), grad_ratio_on_stock * grad_stock)
            
            for types in inputs_dict[f'{resource_type}.recycled_production']:
                if types != GlossaryCore.Years:
                    self.set_partial_derivative_for_other_types(
                        (ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION, resource_type),
                        (f'{resource_type}.recycled_production', types), grad_recycling)
                    self.set_partial_derivative_for_other_types(
                        (ResourceMixModel.RATIO_USABLE_DEMAND, resource_type),
                        (f'{resource_type}.recycled_production', types), grad_ratio_on_recycling * grad_recycling)
            
            
            self.set_partial_derivative_for_other_types(
                (ResourceMixModel.ALL_RESOURCE_DEMAND, resource_type),
                ('resources_demand', resource_type), grad_demand)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.ResourcesPriceValue,
                 resource_type), (f'{resource_type}.resource_price', 'price'),
                grad_price)
            
        data_frame_other_resource_price = inputs_dict['non_modeled_resource_price']

        for resource_type in data_frame_other_resource_price:
            if resource_type not in resource_list and resource_type != GlossaryCore.Years:
                self.set_partial_derivative_for_other_types((GlossaryCore.ResourcesPriceValue, resource_type), (
                    ResourceMixModel.NON_MODELED_RESOURCE_PRICE, resource_type),
                    np.identity(len(data_frame_other_resource_price)))

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = ['all']
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if 'all' in chart_list:

            stock_df = self.get_sosdisc_outputs(
                ResourceMixModel.ALL_RESOURCE_STOCK)
            years = stock_df[GlossaryCore.Years].values.tolist()
            price_df = self.get_sosdisc_outputs(
                GlossaryCore.ResourcesPriceValue)
            use_stock_df = self.get_sosdisc_outputs(
                ResourceMixModel.All_RESOURCE_USE)
            production_df = self.get_sosdisc_outputs(
                ResourceMixModel.ALL_RESOURCE_PRODUCTION)
            ratio_use_df = self.get_sosdisc_outputs(
                ResourceMixModel.RATIO_USABLE_DEMAND)
            demand_df = self.get_sosdisc_outputs(
                ResourceMixModel.ALL_RESOURCE_DEMAND)
            recycling_df = self.get_sosdisc_outputs(
                ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION)

            # two charts for stock evolution and price evolution
            stock_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'stocks (Mt)',
                                                   chart_name='Resources stocks through the years', stacked_bar=True)
            price_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'price ($/t)',
                                                   chart_name='Resource price through the years', stacked_bar=True)
            use_stock_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'resource use (Mt) ',
                                                       chart_name='Resource use through the years', stacked_bar=True)
            production_chart = TwoAxesInstanciatedChart(GlossaryCore.Years,
                                                        'resource production (Mt)',
                                                        chart_name='Resource production through the years',
                                                        stacked_bar=True)
            ratio_use_demand_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, 'ratio usable stock / demand ',
                chart_name='ratio usable stock and prod on demand through the years', stacked_bar=True)
            resource_demand_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, 'demand (Mt)', chart_name='resource demand through the years', stacked_bar=True)
            recycling_chart = TwoAxesInstanciatedChart(
                GlossaryCore.Years, 'recycled production (Mt)', chart_name='recycled production through the years', stacked_bar=True)
            for resource_kind in stock_df:
                if resource_kind != GlossaryCore.Years:
                    stock_serie = InstanciatedSeries(
                        years, (stock_df[resource_kind]
                                ).values.tolist(), resource_kind,
                        InstanciatedSeries.LINES_DISPLAY)
                    stock_chart.add_series(stock_serie)

                    production_serie = InstanciatedSeries(
                        years, (production_df[resource_kind]
                                ).values.tolist(), resource_kind,
                        InstanciatedSeries.BAR_DISPLAY)
                    production_chart.add_series(production_serie)

                    use_stock_serie = InstanciatedSeries(
                        years, (use_stock_df[resource_kind]
                                ).values.tolist(), resource_kind,
                        InstanciatedSeries.BAR_DISPLAY)
                    use_stock_chart.add_series(use_stock_serie)
                    ratio_use_serie = InstanciatedSeries(
                        years, (ratio_use_df[resource_kind]
                                ).values.tolist(), resource_kind,
                        InstanciatedSeries.LINES_DISPLAY)
                    ratio_use_demand_chart.add_series(ratio_use_serie)
                    demand_serie = InstanciatedSeries(
                        years, (demand_df[resource_kind]
                                ).values.tolist(), resource_kind,
                        InstanciatedSeries.LINES_DISPLAY)
                    resource_demand_chart.add_series(demand_serie)

                    recycled_production_serie = InstanciatedSeries(
                        years, (recycling_df[resource_kind]).values.tolist(), resource_kind,
                        InstanciatedSeries.BAR_DISPLAY)
                    recycling_chart.add_series(recycled_production_serie)

            for resource_types in price_df:
                if resource_types != GlossaryCore.Years:
                    price_serie = InstanciatedSeries(years, (price_df[resource_types]).values.tolist(
                    ), resource_types, InstanciatedSeries.LINES_DISPLAY)
                    price_chart.add_series(price_serie)

            instanciated_charts.append(stock_chart)
            instanciated_charts.append(price_chart)
            instanciated_charts.append(use_stock_chart)
            instanciated_charts.append(production_chart)
            instanciated_charts.append(ratio_use_demand_chart)
            instanciated_charts.append(resource_demand_chart)
            instanciated_charts.append(recycling_chart)

        return instanciated_charts
