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

from climateeconomics.glossarycore import GlossaryCore


class SectorsRedistributionInvestsDiscipline(AbstractJacobianUnittest):

    def setUp(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2030
        self.years = np.arange(self.year_start, self.year_end + 1)
        n_years = len(self.years)

        self.sector_list = GlossaryCore.SectorsPossibleValues

        time = self.years - self.year_start
        self.economics_df = pd.DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.GrossOutput: 0,
                GlossaryCore.OutputNetOfDamage: 90 * 1.015**time,
                GlossaryCore.PerCapitaConsumption: 0,
            }
        )

        self.share_invests_agriculture = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.ShareInvestment: np.linspace(2, 2.02, n_years)}
        )

        self.share_invests_industry = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.ShareInvestment: np.linspace(3, 5, n_years)}
        )

        self.share_invests_services = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.ShareInvestment: np.linspace(4, 2.7, n_years)}
        )

    def analytic_grad_entry(self):
        return [self.test_analytic_grad]

    def test_analytic_grad(self):
        name = "Test"
        model_name = "sectors_redistribution_invests.SectorsRedistributionInvestsDiscipline"
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

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_invests.sectors_redistribution_invest_discipline.SectorsRedistributionInvestsDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {
            f"{name}.{GlossaryCore.SectorListValue}": self.sector_list,
            f"{name}.{GlossaryCore.EconomicsDfValue}": self.economics_df,
            f"{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorInvestmentDfValue}": self.share_invests_agriculture,
            f"{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ShareSectorInvestmentDfValue}": self.share_invests_industry,
            f"{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorInvestmentDfValue}": self.share_invests_services,
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        inputs_checked = [f"{name}.{GlossaryCore.EconomicsDfValue}"]
        inputs_checked += [
            f"{name}.{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}" for sector in self.sector_list
        ]

        output_checked = [f"{name}.{sector}.{GlossaryCore.InvestmentDfValue}" for sector in self.sector_list]

        disc_techno = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_sectors_redistribution_invest_discipline.pkl",
            discipline=disc_techno,
            step=1e-15,
            derr_approx="complex_step",
            local_data=disc_techno.local_data,
            inputs=inputs_checked,
            outputs=output_checked,
        )
