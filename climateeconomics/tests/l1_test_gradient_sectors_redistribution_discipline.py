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
import unittest
import numpy as np
import pandas as pd
from pandas import DataFrame
from os.path import join, dirname

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class SectorsRedistributionDisciplineJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2022
        self.years = np.arange(self.year_start, self.year_end + 1)

        self.sector_list = GlossaryCore.SectorsPossibleValues

        self.energy_production_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                  GlossaryCore.TotalProductionValue: np.linspace(25000, 50000,
                                                                                                 len(self.years))})

        self.share_energy_agriculture = pd.DataFrame({GlossaryCore.Years: self.years,
                                                      GlossaryCore.ShareSectorEnergy: np.linspace(12, 20,
                                                                                                  len(self.years))})

        self.share_energy_industry = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareSectorEnergy: np.linspace(39, 59,
                                                                                               len(self.years))})

        shares_energy_services = 100. - self.share_energy_industry[GlossaryCore.ShareSectorEnergy] - \
                                 self.share_energy_agriculture[GlossaryCore.ShareSectorEnergy]
        self.share_energy_services = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareSectorEnergy: shares_energy_services})

        self.investments_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                            GlossaryCore.InvestmentsValue: np.linspace(40, 65, len(self.years))})

        self.share_invest_agriculture = pd.DataFrame({GlossaryCore.Years: self.years,
                                                      GlossaryCore.ShareInvestment: np.linspace(12, 20,
                                                                                                len(self.years))})

        self.share_invest_industry = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareInvestment: np.linspace(39, 59,
                                                                                             len(self.years))})

        shares_invest_services = 100. - self.share_invest_industry[GlossaryCore.ShareInvestment] - \
                                 self.share_invest_agriculture[GlossaryCore.ShareInvestment]
        self.share_invest_services = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareInvestment: shares_invest_services})

    def analytic_grad_entry(self):
        return [
            self.test_analytic_grad
        ]

    def test_analytic_grad(self):
        name = 'Test'
        model_name = 'sectors_redistribution.SectorsRedistributionDiscipline'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness': f'{name}',
                   'ns_functions': f'{name}',
                   'ns_energy_mix': f'{name}',
                   'ns_coal_resource': f'{name}',
                   'ns_resource': f'{name}'}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution.sectors_redistribution_discipline.SectorsRedistributionDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.{GlossaryCore.InvestmentDfValue}': self.investments_df,
                       f'{name}.{GlossaryCore.EnergyProductionValue}': self.energy_production_df,
                       f'{name}.{GlossaryCore.SectorListValue}': self.sector_list,
                       f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorInvestmentDfValue}': self.share_invest_agriculture,
                       f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorEnergyDfValue}': self.share_energy_agriculture,
                       f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ShareSectorInvestmentDfValue}': self.share_invest_industry,
                       f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ShareSectorEnergyDfValue}': self.share_energy_industry,
                       f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorInvestmentDfValue}': self.share_invest_services,
                       f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorEnergyDfValue}': self.share_energy_services,

                       }
        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        inputs_checked = [f'{name}.{GlossaryCore.InvestmentDfValue}', f'{name}.{GlossaryCore.EnergyProductionValue}']
        inputs_checked += [f'{name}.{sector}.{GlossaryCore.ShareSectorEnergyDfValue}' for sector in self.sector_list]
        inputs_checked += [f'{name}.{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}' for sector in self.sector_list]
        
        output_checked = []
        output_checked += [f'{name}.{sector}.{GlossaryCore.InvestmentDfValue}' for sector in self.sector_list]
        output_checked += [f'{name}.{sector}.{GlossaryCore.EnergyProductionValue}' for sector in self.sector_list]
        
        disc_techno = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_sectors_redistribution_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data= disc_techno.local_data,
                            inputs=inputs_checked,
                            outputs=output_checked
                            )
        
   
