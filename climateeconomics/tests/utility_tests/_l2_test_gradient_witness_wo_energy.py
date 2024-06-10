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

# -*- coding: utf-8 -*-

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import (
    DataStudy as datacase_witness,
)


class WitnessWOEnergyTestCase(AbstractJacobianUnittest):
    """
    Design variables test class
    """

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

        repo = "climateeconomics.sos_processes.iam"
        chain_builders = self.ee.factory.get_builder_from_process(repo, "witness_wo_energy")

        ns_dict = {
            GlossaryCore.NS_FUNCTIONS: f"{self.ee.study_name}",
            "ns_optim": f"{self.ee.study_name}",
            "ns_public": f"{self.ee.study_name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.factory.set_builders_to_coupling_builder(chain_builders)

        self.ee.configure()

        dc_witness = datacase_witness()
        dc_witness.study_name = self.name
        values_dict = {}
        for dict_item in dc_witness.setup_usecase():
            values_dict.update(dict_item)
        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefault

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.economics_df = pd.DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.GrossOutput: np.linspace(121, 91, len(self.years)),
            }
        )

        # energy simple outputs
        values_dict[f"{self.name}.{GlossaryCore.EnergyProductionValue}"] = pd.DataFrame(
            {
                GlossaryCore.Years: np.arange(year_start, year_end + 1),
                GlossaryCore.TotalProductionValue: [
                    173340.0,
                    175835.14104537413,
                    178097.05882947898,
                    180128.3289974944,
                    181931.21296785408,
                    183507.70680985705,
                    184859.5853427277,
                    185988.4420204021,
                    186895.72509200562,
                    187582.77045928058,
                    188050.83159109022,
                    188301.10680186382,
                    188334.76415497687,
                    188152.9642128537,
                    187543.06616462852,
                    186486.60856058312,
                    185017.06862288466,
                    183163.96287214552,
                    180953.53218984886,
                    178409.2853603249,
                    175552.43494429247,
                    172402.25017291363,
                    168976.34513454058,
                    165534.4855684096,
                    162074.3319254275,
                    158593.48370087313,
                    152553.37310256364,
                    146681.55977419773,
                    140974.73442018288,
                    135429.97909609717,
                    130044.72412732094,
                    124816.7095041681,
                    119743.95020124805,
                    114824.7049417392,
                    110057.44799483508,
                    105440.84365481937,
                    100973.72310268266,
                    96655.06339629693,
                    92483.9683735579,
                    88677.97888038075,
                    85261.33946942173,
                    82206.15193411146,
                    79488.30919541811,
                    77086.83638298033,
                    74983.36660113289,
                    73161.71871823145,
                    71607.55344106231,
                    70308.09018514975,
                    69002.55789431673,
                    67692.33371552263,
                    66378.9027711045,
                    65105.81967484658,
                    63874.07986497068,
                    62684.67067184421,
                    61538.56490911869,
                    60436.7144760655,
                    59380.043946723694,
                    58369.444123591704,
                    57405.76553572868,
                    56489.81186330803,
                    55622.333272958436,
                    54804.01965072657,
                    54035.49372230678,
                    53317.30405341118,
                    52645.28687158386,
                    52014.28999037328,
                    51419.25915135939,
                    50855.20221462671,
                    50317.16140414808,
                    49800.192211142865,
                    49299.34787702992,
                    48809.66859577396,
                    48326.174724068165,
                    47849.76091675441,
                    47381.314121470496,
                    46921.709029482176,
                    46470.78467222658,
                    46028.32298311648,
                    45594.046971628406,
                    45167.61915183676,
                    44748.64024847417,
                ],
            }
        )
        values_dict[f"{self.name}.{GlossaryCore.CO2EmissionsGtValue}"] = pd.DataFrame(
            {
                GlossaryCore.Years: np.arange(year_start, year_end + 1),
                GlossaryCore.TotalCO2Emissions: [
                    38.1348,
                    38.589443619528566,
                    38.903155176349536,
                    39.08108850827246,
                    39.128263622937205,
                    39.04958757607771,
                    38.849872141337016,
                    38.53384864298817,
                    38.10618027301267,
                    37.571472170758405,
                    36.93427950657557,
                    36.19911377976067,
                    35.370447515071966,
                    34.452717520280885,
                    33.43322167642287,
                    32.319229644139696,
                    31.121681247877788,
                    29.850313960340827,
                    28.513848733048967,
                    27.120137328163246,
                    25.676280319842594,
                    24.18872247212881,
                    22.66333047212881,
                    21.13793847212881,
                    19.61254647212881,
                    18.08715447212881,
                    16.10711885260024,
                    14.268015295779271,
                    12.564689963856345,
                    10.992122849191599,
                    9.545406896051096,
                    8.219730330791792,
                    7.010361829140637,
                    5.912638199116134,
                    4.921954301370396,
                    4.033754965553236,
                    3.243528692368138,
                    2.546802957056833,
                    1.93914095184792,
                    1.4332447957059333,
                    1.0218448279891137,
                    0.6940012242510194,
                    0.43997651178797753,
                    0.2510497390798442,
                    0.11936914396556286,
                    0.037834152286211495,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
        values_dict[f"{self.name}.{GlossaryCore.EnergyPriceValue}"] = pd.DataFrame(
            {
                GlossaryCore.Years: np.arange(year_start, year_end + 1),
                GlossaryCore.EnergyPriceValue: [
                    110.0,
                    114.16666666666667,
                    118.33333333333333,
                    122.5,
                    126.66666666666667,
                    130.83333333333334,
                    135.0,
                    139.16666666666669,
                    143.33333333333334,
                    147.49999999999997,
                    151.66666666666669,
                    155.83333333333334,
                    159.99999999999997,
                    164.16666666666666,
                    173.00000000000003,
                    183.0,
                    193.0,
                    203.0,
                    213.0,
                    223.00000000000003,
                    233.0,
                    243.00000000000003,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                    253.0,
                ],
            }
        )

        values_dict[f"{self.name}.linearization_mode"] = "adjoint"
        values_dict[f"{self.name}.warm_start"] = False
        values_dict[f"{self.name}.chain_linearize"] = False
        values_dict[f"{self.name}.tolerance"] = 1.0e-12
        values_dict[f"{self.name}.tolerance_linear_solver_MDO"] = 1.0e-14

        values_dict[f"{self.name}.sub_mda_class"] = "MDAGaussSeidel"
        values_dict[f"{self.name}.{GlossaryCore.CO2TaxesValue}"] = pd.DataFrame(
            {GlossaryCore.Years: np.arange(year_start, year_end + 1), GlossaryCore.CO2Tax: 50.0}
        )
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.display_treeview_nodes()
        # self.ee.execute()

    #         self.ee.root_process.coupling_structure.graph.export_initial_graph(
    #             "initial2.pdf")

    def analytic_grad_entry(self):
        return [
            self.test_01_check_gradient_obj_witness_wo_energy_NR,
            self.test_02_check_gradient_redidus_witness_wo_energy,
            self.test_03_check_gradient_obj_energy_outputs_witness_wo_energy,
            self.test_04_check_gradient_residus_energy_outputs_witness_wo_energy,
        ]

    def test_01_check_gradient_obj_witness_wo_energy_NR(self):

        # design
        input_full_names = [f"{self.name}.share_energy_investment", f"{self.name}.{GlossaryCore.CO2TaxesValue}"]
        # objectif
        output_full_names = [
            f"{self.name}.{GlossaryCore.WelfareObjective}",
            f"{self.name}.temperature_objective",
            f"{self.name}.CO2_objective",
            f"{self.name}.ppm_objective",
            f"{self.name}.CO2_tax_minus_CO2_damage_constraint_df",
        ]

        disc_closed_loop = self.ee.root_process
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_obj_witness_wo_energy_NR.pkl",
            local_data={},
            discipline=disc_closed_loop,
            inputs=input_full_names,
            outputs=output_full_names,
            derr_approx="complex_step",
            step=1.0e-15,
            threshold=1e-5,
            parallel=True,
        )

    def test_02_check_gradient_redidus_witness_wo_energy(self):

        # design
        input_full_names = [f"{self.name}.{GlossaryCore.CO2TaxesValue}"]
        # objectif
        output_full_names = [f"{self.name}.{GlossaryCore.UtilityDfValue}"]

        disc_closed_loop = self.ee.root_process
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_residus_witness_wo_energy.pkl",
            local_data={},
            discipline=disc_closed_loop,
            inputs=input_full_names,
            outputs=output_full_names,
            output_column=GlossaryCore.DiscountedUtility,
            derr_approx="complex_step",
            step=1.0e-15,
            threshold=1e-5,
            parallel=True,
        )

    def test_03_check_gradient_obj_energy_outputs_witness_wo_energy(self):

        # design
        input_full_names = [f"{self.name}.{GlossaryCore.CO2EmissionsGtValue}"]
        # objectif
        output_full_names = [
            f"{self.name}.{GlossaryCore.WelfareObjective}",
            f"{self.name}.temperature_objective",
            f"{self.name}.CO2_objective",
            f"{self.name}.ppm_objective",
            f"{self.name}.CO2_tax_minus_CO2_damage_constraint_df",
        ]

        disc_closed_loop = self.ee.root_process
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_obj_energy_outputs_witness_wo_energy.pkl",
            local_data={},
            discipline=disc_closed_loop,
            inputs=input_full_names,
            outputs=output_full_names,
            derr_approx="complex_step",
            step=1.0e-15,
            threshold=1e-5,
            parallel=True,
        )

    def test_04_check_gradient_residus_energy_outputs_witness_wo_energy(self):

        # design
        input_full_names = [
            f"{self.name}.{GlossaryCore.EnergyProductionValue}",
            f"{self.name}.{GlossaryCore.CO2EmissionsGtValue}",
        ]
        # objectif
        output_full_names = [f"{self.name}.{GlossaryCore.EnergyInvestmentsValue}"]

        disc_closed_loop = self.ee.root_process
        self.check_jacobian(
            location=dirname(__file__),
            filename=f"jacobian_residus_energy_outputs_witness_wo_energy.pkl",
            local_data={},
            discipline=disc_closed_loop,
            inputs=input_full_names,
            outputs=output_full_names,
            derr_approx="complex_step",
            step=1.0e-15,
            threshold=1e-5,
            parallel=True,
        )


if "__main__" == __name__:
    cls = WitnessWOEnergyTestCase()
    cls.setUp()
    cls.test_04_check_gradient_residus_energy_outputs_witness_wo_energy()
