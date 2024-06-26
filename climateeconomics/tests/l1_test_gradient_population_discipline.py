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
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


class PopulationJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    def setUp(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)
        self.model_name = GlossaryCore.PopulationValue
        ns_dict = {GlossaryCore.NS_WITNESS: f"{self.name}", "ns_public": f"{self.name}"}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline.PopulationDiscipline"
        )
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), "data")
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2035
        years = np.arange(self.year_start, self.year_end + 1)
        nb_per = self.year_end + 1 - self.year_start

        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        self.economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        self.economics_df_y.index = years
        self.temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        self.temperature_df.index = years

    def analytic_grad_entry(self):
        return [
            self.test_population_discipline_analytic_grad_output,
            self.test_population_discipline_analytic_grad_temperature,
            self.test_population_discipline_analytic_grad_big_gdp,
            self.test_population_discipline_analytic_big_pop,
            self.test_population_discipline_analytic_grad_big_temp,
            self.test_population_discipline_analytic_small_pop,
            self.test_population_discipline_analytic_grad_temp_negative,
            self.test_population_discipline_analytic_3000_calories_pc,
            self.test_population_discipline_deactivate_climate_effect(),
            self.test_population_discipline_activate_pandemic_effect(),
        ]

    def test_population_discipline_analytic_grad_output(self):
        """
        Test gradient population wrt economics_df
        """
        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_output.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}"],
            outputs=[f"{self.name}.{GlossaryCore.PopulationDfValue}"],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_working_population_discipline_analytic_grad_output(self):
        """
        Test gradient population wrt economics_df
        """

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_working_population_discipline_output.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}"],
            outputs=[f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}"],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_working_population_discipline_analytic_grad_temp(self):
        """
        Test gradient population wrt economics_df
        """
        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_working_population_discipline_temp.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}"],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_grad_temperature(self):
        """
        Test gradient population wrt temperature_df
        """

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_temp.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[f"{self.name}.{GlossaryCore.PopulationDfValue}"],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_grad_temp_negative(self):
        """
        Test gradient population with negative temperature
        """

        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)
        nb_per = year_end + 1 - year_start

        gdp_year_start = 130.5
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.05)
            temp_serie.append(temp_serie[year - 1] - 0.85)
        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        calories_pc = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2400, 2400, len(years))})

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_temp_neg.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_grad_big_gdp(self):
        """
        Test gradient population with big GDP
        """

        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)
        nb_per = year_end + 1 - year_start

        gdp_year_start = 130.5
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1000)
            temp_serie.append(temp_serie[year - 1] * 1.02)
        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        calories_pc = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2400, 2400, len(years))})

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_big_gdp.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_grad_big_temp(self):
        """
        Test gradient population with big temp but not so big
        """

        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)
        nb_per = year_end + 1 - year_start

        gdp_year_start = 130.5
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.05)
            temp_serie.append(temp_serie[year - 1] + 8.05)
        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        calories_pc = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2400, 2400, len(years))})

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_augmente_temp.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_small_pop(self):
        """
        Test gradient population with small population
        """

        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)
        nb_per = year_end + 1 - year_start

        gdp_year_start = 130.5
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.02)
        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        data_dir = join(dirname(__file__), "data")
        pop_init_df = pd.read_csv(join(data_dir, "population_by_age_2020_small.csv"))

        calories_pc = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2400, 2400, len(years))})

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_small_pop.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_big_pop(self):
        """
        Test gradient population with big population
        """

        data_dir = join(dirname(__file__), "data")
        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)
        nb_per = year_end + 1 - year_start

        gdp_year_start = 130.5
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.05)
            temp_serie.append(temp_serie[year - 1] * 1.02)
        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        pop_init_df = pd.read_csv(join(data_dir, "population_by_age_2020_large.csv"))

        calories_pc = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2400, 2400, len(years))})

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_big_pop.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_analytic_3000_calories_pc(self):
        """
        Test gradient population with a huge increase in calories intake
        """
        year_start = GlossaryCore.YearStartDefault
        year_end = 2035
        years = np.arange(year_start, year_end + 1)

        calories_pc_df = pd.DataFrame({GlossaryCore.Years: years, "kcal_pc": np.linspace(2000, 3000, len(years))})
        calories_pc_df.index = years

        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
            f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}": calories_pc_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.override_dump_jacobian = True
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_3000_kcal.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.CaloriesPerCapitaValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_deactivate_climate_effect(self):
        """
        Test gradient population wrt economics_df
        """

        assumptions_dict = ClimateEcoDiscipline.assumptions_dict_default
        assumptions_dict["activate_climate_effect_population"] = False
        assumptions_dict["activate_pandemic_effects"] = False
        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
            f"{self.name}.assumptions_dict": assumptions_dict,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_output_wo_climate_effect.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )

    def test_population_discipline_activate_pandemic_effect(self):
        """
        Test gradient population wrt economics_df
        """
        assumptions_dict = ClimateEcoDiscipline.assumptions_dict_default
        assumptions_dict["activate_pandemic_effects"] = True
        values_dict = {
            f"{self.name}.{GlossaryCore.EconomicsDfValue}": self.economics_df_y,
            f"{self.name}.{GlossaryCore.YearStart}": self.year_start,
            f"{self.name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{self.name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
            f"{self.name}.assumptions_dict": assumptions_dict,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_population_discipline_output_w_pandemic_effect.pkl",
            discipline=disc_techno,
            local_data=disc_techno.local_data,
            inputs=[f"{self.name}.{GlossaryCore.EconomicsDfValue}", f"{self.name}.{GlossaryCore.TemperatureDfValue}"],
            outputs=[
                f"{self.name}.{GlossaryCore.PopulationDfValue}",
                f"{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}",
            ],
            step=1e-15,
            derr_approx="complex_step",
        )
