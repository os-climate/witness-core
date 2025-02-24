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
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.nb_per = self.year_end - self.year_start + 1
        # -------------------------
        # multiply ageworking pop by employment rate and by % in services
        self.workforce_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.SectorIndustry: np.linspace(1800., 2200., len(self.years)),
                                          GlossaryCore.SectorServices: np.linspace(1800., 2200., len(self.years)),
                                          GlossaryCore.SectorAgriculture: np.linspace(1800., 2200., len(self.years))})

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(23, 76, len(self.years))
        })

        self.total_invest = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.InvestmentsValue: 5 * 1.02 ** np.arange(len(self.years))})

        self.damage_fraction_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: 1e-2,
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


    def analytic_grad_entry(self):
        return [
        ]

    def generate_checked_inputs(self):
        return [
            f'{self.name}.{GlossaryCore.WorkforceDfValue}',
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
            f'{self.name}.{self.model_name}.{GlossaryCore.EnergyProductionValue}',
            f'{self.name}.{self.model_name}.{GlossaryCore.InvestmentDfValue}',
        ]

    def generate_checked_outputs(self):
        return [
            f'{self.name}.{self.model_name}.{GlossaryCore.DamageDfValue}',
            f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
            f'{self.name}.{self.model_name}.{GlossaryCore.ProductionDfValue}',
            f'{self.name}.{self.model_name}.{GlossaryCore.ConstraintUpperBoundUsableCapital}',
        ]

    def test_analytic_grad_industry(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
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
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': True,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
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

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=os.path.abspath(dirname(__file__)), filename='jacobian_sector_discipline_industry.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

    def test_gradient_withotudamagetoproductivity_industry(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
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
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage_industry.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

    def test_gradient_without_climate_impact_on_gdp_industry(self):
        self.model_name = GlossaryCore.SectorIndustry
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
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
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': False,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage_on_gdp_industry.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

    def test_analytic_grad_services(self):
        self.model_name = GlossaryCore.SectorServices
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        section_list = GlossaryCore.SectionsServices

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': True,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{GlossaryCore.SectorServices}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=os.path.abspath(dirname(__file__)), filename='jacobian_sector_discipline_services.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

    def test_gradient_withotudamagetoproductivity_services(self):
        self.model_name = GlossaryCore.SectorServices
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        section_list = GlossaryCore.SectionsServices

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage_services.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

    def test_gradient_without_climate_impact_on_gdp_services(self):
        self.model_name = GlossaryCore.SectorServices
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        #mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline.ServicesDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        section_list = GlossaryCore.SectionsServices

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                       f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': False,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sector_discipline_withoutdamage_on_gdp_services.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.generate_checked_inputs(),
                            outputs=self.generate_checked_outputs())

