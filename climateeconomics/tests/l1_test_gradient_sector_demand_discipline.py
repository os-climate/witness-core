"""
Copyright 2023 Capgemini

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

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class SectorsDemandDiscipline(AbstractJacobianUnittest):

    def setUp(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        self.years = np.arange(self.year_start, self.year_end + 1)

        self.sector_list = GlossaryCore.SectorsPossibleValues

        self.population_df = DatabaseWitnessCore.WorldPopulationForecast.value

        gdp_forecast = DatabaseWitnessCore.WorldGDPForecastSSP3.value[GlossaryCore.GrossOutput].values
        population_2021 = DatabaseWitnessCore.WorldPopulationForecast.value[GlossaryCore.PopulationValue].values[1]

        share_gdp_agriculture_2021 = DatabaseWitnessCore.ShareGlobalGDPAgriculture2021.value / 100.0
        share_gdp_industry_2021 = DatabaseWitnessCore.ShareGlobalGDPIndustry2021.value / 100.0
        share_gdp_services_2021 = DatabaseWitnessCore.ShareGlobalGDPServices2021.value / 100.0

        # has to be in $/person : T$ x constant  / (Mperson) = M$/person = 1 000 000 $/person
        demand_agriculture_per_person_population_2021 = (
            gdp_forecast[1] * share_gdp_agriculture_2021 / population_2021 * 1e6
        )
        demand_industry_per_person_population_2021 = gdp_forecast[1] * share_gdp_industry_2021 / population_2021 * 1e6
        demand_services_per_person_population_2021 = gdp_forecast[1] * share_gdp_services_2021 / population_2021 * 1e6

        self.demand_per_capita_agriculture = pd.DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.SectorDemandPerCapitaDfValue: demand_agriculture_per_person_population_2021,
            }
        )

        self.demand_per_capita_industry = pd.DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.SectorDemandPerCapitaDfValue: demand_industry_per_person_population_2021,
            }
        )

        self.demand_per_capita_services = pd.DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.SectorDemandPerCapitaDfValue: demand_services_per_person_population_2021,
            }
        )

    def analytic_grad_entry(self):
        return [self.test_analytic_grad]

    def test_analytic_grad(self):
        name = "Test"
        model_name = "demand_discipline"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}",
            GlossaryCore.NS_ENERGY_MIX: f"{name}",
            "ns_coal_resource": f"{name}",
            "ns_resource": f"{name}",
            GlossaryCore.NS_SECTORS: f"{name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_sectors.demand.demand_discipline.DemandDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {
            f"{name}.{model_name}.{GlossaryCore.SectorListValue}": self.sector_list,
            f"{name}.{GlossaryCore.PopulationDfValue}": self.population_df,
            f"{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.SectorDemandPerCapitaDfValue}": self.demand_per_capita_agriculture,
            f"{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.SectorDemandPerCapitaDfValue}": self.demand_per_capita_industry,
            f"{name}.{GlossaryCore.SectorServices}.{GlossaryCore.SectorDemandPerCapitaDfValue}": self.demand_per_capita_services,
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        inputs_checked = [f"{name}.{GlossaryCore.PopulationDfValue}"]
        inputs_checked += [
            f"{name}.{sector}.{GlossaryCore.SectorDemandPerCapitaDfValue}" for sector in self.sector_list
        ]

        output_checked = [f"{name}.{sector}.{GlossaryCore.SectorGDPDemandDfValue}" for sector in self.sector_list]

        disc_techno = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_sectors_demand_discipline.pkl",
            discipline=disc_techno,
            step=1e-15,
            derr_approx="complex_step",
            local_data=disc_techno.local_data,
            inputs=inputs_checked,
            outputs=output_checked,
        )
