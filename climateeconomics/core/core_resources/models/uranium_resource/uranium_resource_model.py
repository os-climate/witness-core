"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)

from climateeconomics.core.core_resources.resource_model.resource_model import (
    ResourceModel,
)
from climateeconomics.core.tools.Hubbert_Curve import compute_Hubbert_regression
from climateeconomics.glossarycore import GlossaryCore


class OrderOfMagnitude:
    KILO = "k"
    # USD_PER_USton = 'USD/USton'
    # MILLION_TONNES='million_tonnes'

    magnitude_factor = {
        KILO: 10
        ** 3
        # USD_PER_USton:1/0.907
        # MILLION_TONNES: 10**6
    }


class UraniumResourceModel(ResourceModel):
    """
    Resource pyworld3
    General implementation of a resource pyworld3, to be inherited by specific models for each type of resource
    """

    resource_name = ResourceGlossary.Uranium["name"]

    # Units conversion
    oil_barrel_to_tonnes = 6.84
    bcm_to_Mt = 1 / 1.379
    kU_to_Mt = 10**-6

    conversion_factor = 1 / kU_to_Mt

    def convert_demand(self, demand):
        self.resource_demand = demand
        self.resource_demand[self.resource_name] = demand[self.resource_name] * self.conversion_factor

    def configure_parameters(self, inputs_dict):
        self.regression_start = inputs_dict["regression_start"]
        ResourceModel.configure_parameters(self, inputs_dict)

    def init_dataframes(self):
        """
        Init dataframes with years
        """
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.predictable_production = pd.DataFrame(
            {GlossaryCore.Years: np.arange(self.production_start, self.year_end + 1, 1)}
        )
        self.recycled_production = pd.DataFrame({GlossaryCore.Years: self.years})
        self.total_consumption = pd.DataFrame({GlossaryCore.Years: self.years})
        self.resource_stock = pd.DataFrame({GlossaryCore.Years: self.years})
        self.resource_price = pd.DataFrame({GlossaryCore.Years: self.years})
        self.use_stock = pd.DataFrame({GlossaryCore.Years: self.years})

        """
        Set the index as the years
        """
        self.predictable_production.index = self.predictable_production[GlossaryCore.Years]
        self.recycled_production.index = self.recycled_production[GlossaryCore.Years]
        self.total_consumption.index = self.total_consumption[GlossaryCore.Years]
        self.resource_stock.index = self.resource_stock[GlossaryCore.Years]
        self.resource_price.index = self.resource_price[GlossaryCore.Years]
        self.use_stock.index = self.use_stock[GlossaryCore.Years]

    def configure_parameters_update(self, inputs_dict):
        self.regression_start = inputs_dict["regression_start"]
        ResourceModel.configure_parameters_update(self, inputs_dict)

    def compute_predictable_production(self):
        """
        Special production function for uranium pyworld3
        """
        for resource_type in self.sub_resource_list:
            if resource_type == "uranium_40":
                self.predictable_production[resource_type] = compute_Hubbert_regression(
                    self.resource_production_data, self.production_years, self.regression_start, resource_type
                )
            elif resource_type != GlossaryCore.Years:
                self.predictable_production[resource_type] = np.linspace(0, 0, len(self.predictable_production.index))

        current_year = GlossaryCore.YearStartDefault
        production_sample = self.predictable_production.loc[
            self.predictable_production[GlossaryCore.Years] >= self.production_start
        ]
        for idx in self.predictable_production.index:
            year = self.predictable_production.loc[idx, GlossaryCore.Years]
            if year > current_year:
                prod_u40_current_year = self.predictable_production.loc[
                    self.predictable_production[GlossaryCore.Years] == current_year, "uranium_40"
                ].values[0]
                prod_u40_regression_start = self.predictable_production.loc[
                    self.predictable_production[GlossaryCore.Years] == self.regression_start, "uranium_40"
                ].values[0]
                self.predictable_production.loc[idx, "uranium_80"] = (
                    production_sample.loc[idx - (current_year - self.production_start), "uranium_40"]
                    * (1243900 - 744500)
                    / (prod_u40_current_year - prod_u40_regression_start + 744500)
                )
                self.predictable_production.loc[idx, "uranium_130"] = (
                    production_sample.loc[idx - (current_year - self.production_start), "uranium_40"]
                    * (3791700 - 1243900)
                    / (prod_u40_current_year - prod_u40_regression_start + 744500)
                )
                self.predictable_production.loc[idx, "uranium_260"] = (
                    production_sample.loc[idx - (current_year - self.production_start), "uranium_40"]
                    * (4723700 - 3791700)
                    / (prod_u40_current_year - prod_u40_regression_start + 744500)
                )
