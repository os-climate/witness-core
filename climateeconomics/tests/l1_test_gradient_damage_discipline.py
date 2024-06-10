"""
Copyright 2022 Airbus SAS
Modifications on 2023/05/04-2023/11/03 Copyright 2023 Capgemini

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
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class DamageJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [self.test_damage_analytic_grad]

    def test_damage_analytic_grad(self):
        self.model_name = "Test"
        ns_dict = {
            GlossaryCore.NS_WITNESS: f"{self.name}",
            "ns_public": f"{self.name}",
            "ns_dashboard": f"{self.name}",
            f"ns_{GlossaryCore.SectorIndustry.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorIndustry.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_gdp": self.name,
            GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS: f"{self.name}",
            GlossaryCore.NS_SECTORS_POST_PROC_GDP: f"{self.name}",
            GlossaryCore.NS_REGIONALIZED_POST_PROC: f"{self.name}",
            GlossaryCore.NS_ENERGY_MIX: f"{self.name}",
            GlossaryCore.NS_REFERENCE: f"{self.name}",
            GlossaryCore.NS_HOUSEHOLDS_EMISSIONS: f"{self.name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline"
        )
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), "data")

        temperature_df_all = read_csv(join(data_dir, "temperature_data_onestep.csv"))

        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= GlossaryCore.YearStartDefault][
            [GlossaryCore.Years, GlossaryCore.TempAtmo]
        ]

        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        damage_df = pd.DataFrame(
            {
                GlossaryCore.Years: temperature_df_y[GlossaryCore.Years],
                GlossaryCore.Damages: np.linspace(40, 60, len(temperature_df_y)),
                GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(temperature_df_y)),
            }
        )
        temperature_df_y.index = years

        extra_co2_t_since_preindustrial = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(years)),
            }
        )

        inputs_dict = {
            f"{self.name}.{self.model_name}.tipping_point": True,
            f"{self.name}.{GlossaryCore.DamageDfValue}": damage_df,
            f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}": extra_co2_t_since_preindustrial,
            f"{self.name}.{GlossaryCore.CO2TaxesValue}": pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(years))}
            ),
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df_y,
            f"{self.name}.{self.model_name}.damage_constraint_factor": np.concatenate(
                (np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15)))
            ),
        }

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_damage_discipline.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            step=1e-15,
            inputs=[
                f"{self.name}.{GlossaryCore.TemperatureDfValue}",
                f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}",
                f"{self.name}.{GlossaryCore.DamageDfValue}",
            ],
            outputs=[
                f"{self.name}.{GlossaryCore.DamageFractionDfValue}",
                f"{self.name}.{GlossaryCore.CO2DamagePrice}",
                f"{self.name}.{self.model_name}.{GlossaryCore.ExtraCO2tDamagePrice}",
            ],
            derr_approx="complex_step",
        )

    def test_damage_analytic_grad_wo_damage_on_climate(self):
        """
        Test analytic gradient with damage on climate deactivated
        """

        self.model_name = "Test"
        ns_dict = {
            GlossaryCore.NS_WITNESS: f"{self.name}",
            "ns_public": f"{self.name}",
            "ns_dashboard": f"{self.name}",
            f"ns_{GlossaryCore.SectorIndustry.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorIndustry.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_gdp": self.name,
            GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS: f"{self.name}",
            GlossaryCore.NS_SECTORS_POST_PROC_GDP: f"{self.name}",
            GlossaryCore.NS_REGIONALIZED_POST_PROC: f"{self.name}",
            GlossaryCore.NS_ENERGY_MIX: f"{self.name}",
            GlossaryCore.NS_REFERENCE: f"{self.name}",
            GlossaryCore.NS_HOUSEHOLDS_EMISSIONS: f"{self.name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline"
        )
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), "data")

        temperature_df_all = read_csv(join(data_dir, "temperature_data_onestep.csv"))

        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= GlossaryCore.YearStartDefault][
            [GlossaryCore.Years, GlossaryCore.TempAtmo]
        ]

        damage_df = pd.DataFrame(
            {
                GlossaryCore.Years: temperature_df_y[GlossaryCore.Years],
                GlossaryCore.Damages: np.linspace(40, 60, len(temperature_df_y)),
                GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(temperature_df_y)),
            }
        )

        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        damage_df.index = years
        temperature_df_y.index = years

        extra_co2_t_since_preindustrial = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(years)),
            }
        )

        inputs_dict = {
            f"{self.name}.{self.model_name}.tipping_point": True,
            f"{self.name}.{GlossaryCore.DamageDfValue}": damage_df,
            f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}": extra_co2_t_since_preindustrial,
            f"{self.name}.{GlossaryCore.CO2TaxesValue}": pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(years))}
            ),
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df_y,
            f"{self.name}.{self.model_name}.damage_constraint_factor": np.concatenate(
                (np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15)))
            ),
            f"{self.name}.assumptions_dict": {
                "compute_gdp": True,
                "compute_climate_impact_on_gdp": False,
                "activate_climate_effect_population": False,
                "activate_pandemic_effects": False,
            },
        }

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_damage_discipline_wo_damage_on_climate.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            step=1e-15,
            inputs=[
                f"{self.name}.{GlossaryCore.TemperatureDfValue}",
                f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}",
                f"{self.name}.{GlossaryCore.DamageDfValue}",
            ],
            outputs=[
                f"{self.name}.{GlossaryCore.DamageFractionDfValue}",
                f"{self.name}.{GlossaryCore.CO2DamagePrice}",
                f"{self.name}.{self.model_name}.{GlossaryCore.ExtraCO2tDamagePrice}",
            ],
            derr_approx="complex_step",
        )

    def test_damage_analytic_grad_dev_formula(self):
        self.model_name = "Test"
        ns_dict = {
            GlossaryCore.NS_WITNESS: f"{self.name}",
            "ns_public": f"{self.name}",
            "ns_dashboard": f"{self.name}",
            f"ns_{GlossaryCore.SectorIndustry.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_emissions": self.name,
            f"ns_{GlossaryCore.SectorIndustry.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorServices.lower()}_gdp": self.name,
            f"ns_{GlossaryCore.SectorAgriculture.lower()}_gdp": self.name,
            GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS: f"{self.name}",
            GlossaryCore.NS_SECTORS_POST_PROC_GDP: f"{self.name}",
            GlossaryCore.NS_REGIONALIZED_POST_PROC: f"{self.name}",
            GlossaryCore.NS_ENERGY_MIX: f"{self.name}",
            GlossaryCore.NS_REFERENCE: f"{self.name}",
            GlossaryCore.NS_HOUSEHOLDS_EMISSIONS: f"{self.name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline"
        )
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), "data")

        temperature_df_all = read_csv(join(data_dir, "temperature_data_onestep.csv"))

        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= GlossaryCore.YearStartDefault][
            [GlossaryCore.Years, GlossaryCore.TempAtmo]
        ]

        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        damage_df = pd.DataFrame(
            {
                GlossaryCore.Years: temperature_df_y[GlossaryCore.Years],
                GlossaryCore.Damages: np.linspace(40, 60, len(temperature_df_y)),
                GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(temperature_df_y)),
            }
        )
        temperature_df_y.index = years

        extra_co2_t_since_preindustrial = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(years)),
            }
        )

        inputs_dict = {
            f"{self.name}.{self.model_name}.tipping_point": True,
            f"{self.name}.co2_damage_price_dev_formula": True,
            f"{self.name}.{GlossaryCore.DamageDfValue}": damage_df,
            f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}": extra_co2_t_since_preindustrial,
            f"{self.name}.{GlossaryCore.CO2TaxesValue}": pd.DataFrame(
                {GlossaryCore.Years: years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(years))}
            ),
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df_y,
            f"{self.name}.{self.model_name}.damage_constraint_factor": np.concatenate(
                (np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15)))
            ),
        }

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_damage_discipline_dev_formula.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            step=1e-15,
            inputs=[
                f"{self.name}.{GlossaryCore.TemperatureDfValue}",
                f"{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}",
                f"{self.name}.{GlossaryCore.DamageDfValue}",
            ],
            outputs=[
                f"{self.name}.{GlossaryCore.DamageFractionDfValue}",
                f"{self.name}.{GlossaryCore.CO2DamagePrice}",
                f"{self.name}.{self.model_name}.{GlossaryCore.ExtraCO2tDamagePrice}",
            ],
            derr_approx="complex_step",
        )
