'''
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
'''
import os.path
from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class SectorDisciplineJacobianTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.time_step = 1
        self.years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_per = round((self.year_end - self.year_start) / self.time_step + 1)
        # -------------------------
        # input
        data_dir = join(dirname(__file__), 'data')

        total_workforce_df = read_csv(join(data_dir, 'workingage_population_df.csv'))
        total_workforce_df = total_workforce_df[total_workforce_df[GlossaryCore.Years] <= self.year_end]
        # multiply ageworking pop by employment rate and by % in services
        workforce = total_workforce_df[GlossaryCore.Population1570] * 0.659 * 0.509
        self.workforce_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.SectorIndustry: workforce})

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(23, 76, len(self.years))
        })

        self.total_invest = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.InvestmentsValue: 5 * 1.02 ** np.arange(len(self.years))})

        self.damage_fraction_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: 1e-2,
            GlossaryCore.BaseCarbonPrice: 0.
        })

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        weighted_average_percentage_per_sector_df = pd.read_csv(
            join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        subsector_share_dict = {
            **{GlossaryCore.Years: self.years, },
            **dict(zip(weighted_average_percentage_per_sector_df.columns[1:],
                       weighted_average_percentage_per_sector_df.values[0, 1:]))
        }
        self.section_gdp_df = pd.DataFrame(subsector_share_dict)

        self.energy_carbon_intensity = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyCarbonIntensityDfValue: 100.0
        })

    def analytic_grad_entry(self):
        return [
            self.test_analytic_grad,
            self.test_gradient_withotudamagetoproductivity
        ]

    def test_analytic_grad(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline.IndustrialDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        section_list = GlossaryCore.SectionsIndustry

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.prod_function_fitting': False,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'capital_start'}": 100.92448579,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'output_alpha'}": 0.99,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.energy_carbon_intensity,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'depreciation_capital'}": 0.058,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': True,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           'invest_co2_tax_in_renewables': True
                       }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{GlossaryCore.SectorIndustry}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass


        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=os.path.abspath(dirname(__file__)), filename='jacobian_sector_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkforceDfValue}',
                                    f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}',
                                    ],
                            outputs=[
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.CapitalDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyWastedObjective}',
                                ])

    def test_gradient_withotudamagetoproductivity(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline.IndustrialDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        section_list = GlossaryCore.SectionsIndustry

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.prod_function_fitting': False,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'capital_start'}": 6.92448579,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'output_alpha'}": 0.99,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.energy_carbon_intensity,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'depreciation_capital'}": 0.058,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkforceDfValue}',
                                    f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}',
                                    ],
                            outputs=[
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.CapitalDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyWastedObjective}',
                            ])

    def test_gradient_without_climate_impact_on_gdp(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline.IndustrialDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        section_list = GlossaryCore.SectionsIndustry

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.frac_damage_prod': 0.3,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.prod_function_fitting': False,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'capital_start'}": 6.92448579,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'output_alpha'}": 0.99,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.energy_carbon_intensity,
                       f"{self.name}.{GlossaryCore.SectorIndustry}.{'depreciation_capital'}": 0.058,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': False,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           'invest_co2_tax_in_renewables': True
                       }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage_on_gdp.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkforceDfValue}',
                                    f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}',
                                    ],
                            outputs=[
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.CapitalDfValue}',
                                f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyWastedObjective}',
                            ])
