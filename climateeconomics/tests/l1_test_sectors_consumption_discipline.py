'''
Copyright 2024 Capgemini

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

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class SectorsConsumptionDiscipline(AbstractJacobianUnittest):
    def setUp(self):
        self.name = "Test"
        self.ee = ExecutionEngine(self.name)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.nb_per = len(self.years)

        self.sector_list = GlossaryCore.SectorsPossibleValues
        self.all_sectors_energy_supply = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.SectorServices: 15.,
            GlossaryCore.SectorAgriculture: 3.,
            GlossaryCore.SectorIndustry: 20.,
        })
        self.total_invest = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.InvestmentsValue: 5 * 1.02 ** np.arange(self.nb_per)})

        self.energy_investment_wo_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: 3.5})

        self.invest_indus = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, self.nb_per) * 1 / 3})

        self.invest_services = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, self.nb_per) * 1 / 6})

        self.invest_agriculture = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, self.nb_per) * 1 / 2})

        # Test With a GDP and capital that grows at 2%
        gdp_year_start = 130.187
        capital_year_start = 376.6387
        gdp_serie = np.zeros(self.nb_per)
        capital_serie = np.zeros(self.nb_per)
        gdp_serie[0] = gdp_year_start
        capital_serie[0] = capital_year_start
        for year in np.arange(1, self.nb_per):
            gdp_serie[year] = gdp_serie[year - 1] * 1.02
            capital_serie[year] = capital_serie[year - 1] * 1.02
        # for each sector share of total gdp 2020
        gdp_agri = gdp_serie * 6.775773 / 100
        gdp_indus = gdp_serie * 28.4336 / 100
        gdp_service = gdp_serie * 64.79 / 100
        self.prod_agri = pd.DataFrame({GlossaryCore.Years: self.years,
                                       GlossaryCore.GrossOutput: gdp_agri,
                                       GlossaryCore.OutputNetOfDamage: gdp_agri * 0.995})
        self.prod_indus = pd.DataFrame({GlossaryCore.Years: self.years,
                                        GlossaryCore.GrossOutput: gdp_indus,
                                        GlossaryCore.OutputNetOfDamage: gdp_indus * 0.995})
        self.prod_service = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.GrossOutput: gdp_service,
                                          GlossaryCore.OutputNetOfDamage: gdp_service * 0.995})

    def analytic_grad_entry(self):
        return [self.test_analytic_grad]

    def test_analytic_grad(self):
        name = "Test"
        model_name = "consumption_discipline"
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

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_sectors.consumption.consumption_discipline.ConsumptionDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.{model_name}.{GlossaryCore.SectorListValue}': self.sector_list,
                       f'{name}.{GlossaryCore.AllSectorsShareEnergyDfValue}': self.all_sectors_energy_supply,
                       f'{name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,

                       f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}': self.invest_agriculture,
                       f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.invest_indus,
                       f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}': self.invest_services,

                       f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}': self.prod_agri,
                       f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}': self.prod_indus,
                       f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ProductionDfValue}': self.prod_service,

        }
        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        inputs_checked = [f"{name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}", f"{name}.{GlossaryCore.AllSectorsShareEnergyDfValue}", ]
        inputs_checked += [f"{name}.{sector}.{GlossaryCore.InvestmentDfValue}" for sector in self.sector_list ]
        inputs_checked += [f"{name}.{sector}.{GlossaryCore.ProductionDfValue}" for sector in self.sector_list ]

        output_checked = [f"{name}.{GlossaryCore.SectorizedConsumptionDfValue}"]

        disc_techno = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_sectors_consumption_discipline.pkl",
            discipline=disc_techno,
            step=1e-15,
            derr_approx="complex_step",
            local_data=disc_techno.local_data,
            inputs=inputs_checked,
            outputs=output_checked,
        )
