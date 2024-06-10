"""
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
"""

from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

from climateeconomics.core.core_resources.models.copper_resource.copper_resource_model import (
    CopperResourceModel,
)
from climateeconomics.core.core_resources.resource_model.resource_disc import (
    ResourceDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class CopperResourceDiscipline(ResourceDiscipline):
    """Discipline intended to get copper parameters"""

    # ontology information
    _ontology_data = {
        "label": "Copper Resource Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fa-solid fa-pallet",
        "version": "",
    }
    default_year_start = GlossaryCore.YearStartDefault
    default_year_end = GlossaryCore.YearEndDefault
    default_production_start = 1974
    default_years = np.arange(default_year_start, default_year_end + 1, 1)
    default_stock_start = 780.0
    default_recycled_rate = 0.5
    default_lifespan = 30
    default_sectorisation_dict = {
        "power_generation": 0.000213421 / 24.987,
    }  # 9.86175
    default_resource_max_price = 50000  # roughly 5 times the current price (10057 $/t)
    resource_name = CopperResourceModel.resource_name

    prod_unit = "Mt"
    stock_unit = "Mt"
    price_unit = "$/t"

    # Get default data for resource
    default_resource_data = pd.read_csv(join(dirname(__file__), f"../resources_data/{resource_name}_data.csv"))
    default_resource_production_data = pd.read_csv(
        join(dirname(__file__), f"../resources_data/{resource_name}_production_data.csv")
    )
    default_resource_price_data = pd.read_csv(
        join(dirname(__file__), f"../resources_data/{resource_name}_price_data.csv")
    )
    default_resource_consumed_data = pd.read_csv(
        join(dirname(__file__), f"../resources_data/{resource_name}_consumed_data.csv")
    )

    DESC_IN = {
        "resource_data": {
            "type": "dataframe",
            "unit": "-",
            "default": default_resource_data,
            "user_level": 2,
            "namespace": "ns_copper_resource",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                "copper_type": ("string", None, False),
                "Price": ("float", None, True),
                "Price_unit": ("string", None, True),
                "Reserve": ("float", None, True),
                "Reserve_unit": ("string", None, True),
                "Region": ("string", None, True),
            },
        },
        "resource_production_data": {
            "type": "dataframe",
            "unit": "Mt",
            "optional": True,
            "default": default_resource_production_data,
            "user_level": 2,
            "namespace": "ns_copper_resource",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                "copper": ("float", None, False),
            },
        },
        "resource_price_data": {
            "type": "dataframe",
            "unit": "$/t",
            "default": default_resource_price_data,
            "user_level": 2,
            "dataframe_descriptor": {
                "resource_type": ("string", None, False),
                "price": ("float", None, False),
                "unit": ("string", None, False),
            },
            "namespace": "ns_copper_resource",
        },
        "resource_consumed_data": {
            "type": "dataframe",
            "unit": "Mt",
            "optional": True,
            "default": default_resource_consumed_data,
            "user_level": 2,
            "namespace": "ns_copper_resource",
            "dataframe_descriptor": {
                GlossaryCore.Years: ("float", None, False),
                "copper_consumption": ("float", None, True),
            },
        },
        "production_start": {
            "type": "float",
            "default": default_production_start,
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_copper_resource",
        },
        "stock_start": {
            "type": "float",
            "default": default_stock_start,
            "user_level": 2,
            "unit": "Mt",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_copper_resource",
        },
        "recycled_rate": {
            "type": "float",
            "default": default_recycled_rate,
            "user_level": 2,
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_copper_resource",
        },
        "lifespan": {
            "type": "int",
            "default": default_lifespan,
            "user_level": 2,
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_copper_resource",
        },
        "sectorisation": {
            "type": "dict",
            "subtype_descriptor": {"dict": "float"},
            "unit": "-",
            "default": default_sectorisation_dict,
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "user_level": 2,
            "namespace": "ns_copper_resource",
        },
        "resource_max_price": {"type": "float", "default": default_resource_max_price, "user_level": 2, "unit": "$/t"},
    }

    DESC_IN.update(ResourceDiscipline.DESC_IN)

    DESC_OUT = {
        "resource_stock": {"type": "dataframe", "unit": "Mt"},
        "resource_price": {"type": "dataframe", "unit": "$/t"},
        "use_stock": {"type": "dataframe", "unit": "Mt"},
        "predictable_production": {"type": "dataframe", "unit": "Mt"},
        "recycled_production": {"type": "dataframe", "unit": "Mt"},
    }

    DESC_OUT.update(ResourceDiscipline.DESC_OUT)

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.resource_model = CopperResourceModel(self.resource_name)
        self.resource_model.configure_parameters(inputs_dict)
