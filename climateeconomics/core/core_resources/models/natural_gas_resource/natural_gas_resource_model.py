"""
Copyright 2022 Airbus SAS
Modifications on 2024/06/07-2024/06/24 Copyright 2024 Capgemini

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

from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)

from climateeconomics.core.core_resources.resource_model.resource_model import (
    ResourceModel,
)


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


class NaturalGasResourceModel(ResourceModel):
    """
    NaturalGas Resource pyworld3
    Overloads the generic resource pyworld3
    """

    resource_name = ResourceGlossary.NaturalGas["name"]

    # Units conversion
    oil_barrel_to_tonnes = 6.84
    bcm_to_Mt = 1 / 1.379
    kU_to_Mt = 10**-6

    conversion_factor = 1 / bcm_to_Mt

    def convert_demand(self, demand):
        self.resource_demand = demand
        self.resource_demand[self.resource_name] = demand[self.resource_name] * self.conversion_factor
