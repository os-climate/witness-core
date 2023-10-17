'''
Copyright 2022 Airbus SAS

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
import os.path
import unittest
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import read_csv
from scipy.interpolate import interp1d

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline import SectorDiscipline
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class SectorDisciplineJacobianTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2023
        self.time_step = 1
        self.years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_per = round((self.year_end - self.year_start) / self.time_step + 1)
        # -------------------------
        # input
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        total_workforce_df = read_csv(join(data_dir, 'workingage_population_df.csv'))
        total_workforce_df = total_workforce_df[total_workforce_df[GlossaryCore.Years] <= self.year_end]
        # multiply ageworking pop by employment rate and by % in services
        workforce = total_workforce_df[GlossaryCore.Population1570] * 0.659 * 0.509
        self.workforce_df = pd.DataFrame({GlossaryCore.Years: self.years, SectorDiscipline.sector_name: workforce})

        # Energy_supply
        brut_net = 1 / 1.45
        share_indus = 0.37
        # prepare energy df
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182,
                       220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        # Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(self.year_start, self.year_end + 1))
        energy_supply_values = energy_supply * brut_net * share_indus
        energy_supply_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.TotalProductionValue: energy_supply_values})
        energy_supply_df.index = self.years
        self.energy_supply_df = energy_supply_df
        # energy_supply_df.loc[2020, GlossaryCore.TotalProductionValue] = 91.936

        # Investment growth at 2%
        init_value = 25
        invest_serie = []
        invest_serie.append(init_value)
        for year in np.arange(1, self.nb_per):
            invest_serie.append(invest_serie[year - 1] * 1.002)
        self.total_invest = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.InvestmentsValue: invest_serie})

        # damage
        self.damage_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.Damages: np.zeros(self.nb_per), GlossaryCore.DamageFractionOutput: np.zeros(self.nb_per),
             GlossaryCore.BaseCarbonPrice: np.zeros(self.nb_per)})
        self.damage_df.index = self.years
        self.damage_df[GlossaryCore.DamageFractionOutput] = 1e-2

    def analytic_grad_entry(self):
        return [
            self.test_analytic_grad,
            self.test_gradient_withotudamagetoproductivity
        ]

    def test_analytic_grad(self):
        self.model_name = SectorDiscipline.sector_name
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_macro': f'{self.name}',
                   'ns_sectors': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.prod_function_fitting': False,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'capital_start'}": 6.92448579,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'output_alpha'}": 0.99,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'depreciation_capital'}": 0.058,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=os.path.abspath(dirname(__file__)), filename=f'jacobian_sector_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkforceDfValue}',
                                    f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[
                                f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.ProductionDfValue}',
                                f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.CapitalDfValue}',
                                f'{self.name}.{SectorDiscipline.sector_name}.emax_enet_constraint'])

    def test_gradient_withotudamagetoproductivity(self):
        self.model_name = SectorDiscipline.sector_name
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_macro': f'{self.name}',
                   'ns_sectors': f'{self.name}'
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.damage_to_productivity': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.prod_function_fitting': False,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'capital_start'}": 6.92448579,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'output_alpha'}": 0.99,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'depreciation_capital'}": 0.058,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_sector_discipline_withoutdamage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkforceDfValue}',
                                    f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.ProductionDfValue}',
                                     f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{SectorDiscipline.sector_name}.emax_enet_constraint'])
