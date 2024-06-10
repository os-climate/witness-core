"""
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2023/11/06 Copyright 2023 Capgemini

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

# -*- coding: utf-8 -*-
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_emissions.agriculture_emissions.agriculture_emissions_discipline import (
    AgricultureEmissionsDiscipline,
)


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        "label": "Diet Process",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):

        ns_scatter = self.ee.study_name

        ns_dict = {
            GlossaryCore.NS_WITNESS: ns_scatter,
            GlossaryCore.NS_ENERGY_MIX: ns_scatter,
            GlossaryCore.NS_REFERENCE: f"{ns_scatter}.NormalizationReferences",
            "ns_agriculture": ns_scatter,
            GlossaryCore.NS_CCS: ns_scatter,
            "ns_energy": ns_scatter,
            "ns_forest": ns_scatter,
            "ns_invest": f"{self.ee.study_name}.InvestmentDistribution",
        }

        builder_list = []

        chain_builders_landuse = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "land_use_v2_process"
        )
        builder_list.extend(chain_builders_landuse)

        chain_builders_agriculture = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "agriculture_mix_process"
        )
        builder_list.extend(chain_builders_agriculture)

        chain_builders_population = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "population_process"
        )
        builder_list.extend(chain_builders_population)

        ns_dict = {
            "ns_land_use": f"{self.ee.study_name}.EnergyMix",
            GlossaryCore.NS_FUNCTIONS: f"{self.ee.study_name}.EnergyMix",
            "ns_resource": f"{self.ee.study_name}.EnergyMix",
            GlossaryCore.NS_REFERENCE: f"{self.ee.study_name}.NormalizationReferences",
            "ns_invest": f"{self.ee.study_name}.InvestmentDistribution",
        }

        self.ee.ns_manager.add_ns_def(ns_dict)

        """
        Add emissions disciplines
        """
        mods_dict = {
            AgricultureEmissionsDiscipline.name: "climateeconomics.sos_wrapping.sos_wrapping_emissions.agriculture_emissions.agriculture_emissions_discipline.AgricultureEmissionsDiscipline",
        }
        non_use_capital_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)
        builder_list.extend(non_use_capital_list)

        return builder_list
