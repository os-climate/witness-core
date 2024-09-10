'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023-2024/06/24 Copyright 2023 Capgemini

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

from climateeconomics.core.core_resources.models.oil_resource.oil_resource_model import (
    OilResourceModel,
)
from climateeconomics.core.core_resources.resource_model.resource_disc import (
    ResourceDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class OilResourceDiscipline(ResourceDiscipline):
    ''' Discipline intended to get oil parameters
    '''

    # ontology information
    _ontology_data = {
        'label': 'Oil Resource Model',
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
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = 2050
    default_production_start = 1990
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    default_stock_start = 0.0
    default_recycled_rate = 0.0
    default_lifespan = 0
    resource_name = OilResourceModel.resource_name

    prod_unit = 'Mt'
    stock_unit = 'Mt'
    price_unit = '$/bbl'

    #Get default data for resource
    default_resource_data=pd.read_csv(join(dirname(__file__), f'../resources_data/{resource_name}_data.csv'))
    default_resource_production_data = pd.read_csv(join(dirname(__file__), f'../resources_data/{resource_name}_production_data.csv'))
    default_resource_price_data = pd.read_csv(join(dirname(__file__), f'../resources_data/{resource_name}_price_data.csv'))
    default_resource_consumed_data = pd.read_csv(join(dirname(__file__), f'../resources_data/{resource_name}_consumed_data.csv'))


    DESC_IN = {'resource_data': {'type': 'dataframe', 'unit': '[-]', 'default': default_resource_data,
                                 'user_level': 2, 'namespace': 'ns_oil_resource'},
               'resource_production_data': {'type': 'dataframe', 'unit': '[million_barrels]', 'optional': True,
                                            'default': default_resource_production_data, 'user_level': 2, 'namespace': 'ns_oil_resource'},
               'resource_price_data': {'type': 'dataframe', 'unit': 'USD/barrel', 'default': default_resource_price_data, 'user_level': 2,
                                       'dataframe_descriptor': {'resource_type': ('string', None, False),
                                                                'price': ('float', None, False),
                                                                'unit': ('string', None, False)},
                                       'namespace': 'ns_oil_resource'},
               'resource_consumed_data': {'type': 'dataframe', 'unit': '[million_barrels]', 'default': default_resource_consumed_data,
                                            'user_level': 2, 'namespace': 'ns_oil_resource'},
               'production_start': {'type': 'int', 'default': default_production_start, 'unit': '[-]',
                                    'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_oil_resource'},
               'stock_start': {'type': 'float', 'default': default_stock_start, 'unit': '[Mt]'},
               'recycled_rate': {'type': 'float', 'default': default_recycled_rate, 'unit': '[-]'},
               'lifespan': {'type': 'int', 'default': default_lifespan, 'unit': '[-]'},
               }

    DESC_IN.update(ResourceDiscipline.DESC_IN)

    DESC_OUT = {
        'resource_stock': {'type': 'dataframe', 'unit': 'billion cubic metres', },
        'resource_price': {'type': 'dataframe', 'unit': 'USD/MMBTU', },
        'use_stock': {'type': 'dataframe', 'unit': 'billion cubic metre', },
        'predictable_production': {'type': 'dataframe', 'unit': 'billion cubic metre',},
        'recycled_production' : {
            'type': 'dataframe', 'unit': 'billion cubic metres'}
    }
    DESC_OUT.update(ResourceDiscipline.DESC_OUT)

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.resource_model = OilResourceModel(self.resource_name)
        self.resource_model.configure_parameters(inputs_dict)