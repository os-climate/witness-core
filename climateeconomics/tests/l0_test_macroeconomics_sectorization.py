'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/03 Copyright 2023 Capgemini

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
from pandas import DataFrame
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class MacroeconomicsTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        nb_per = round(self.year_end - self.year_start + 1)
        self.nb_per = nb_per
        self.years = np.arange(self.year_start, self.year_end + 1)

        invests = np.linspace(40, 65, nb_per)
        self.invests = DataFrame({GlossaryCore.Years: self.years,
                                  GlossaryCore.InvestmentsValue: invests})

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
        self.prod_agri = DataFrame({GlossaryCore.Years: self. years,
                                    GlossaryCore.GrossOutput: gdp_agri,
                                    GlossaryCore.OutputNetOfDamage: gdp_agri * 0.995})
        self.prod_indus = DataFrame({GlossaryCore.Years: self. years,
                                     GlossaryCore.GrossOutput: gdp_indus,
                                     GlossaryCore.OutputNetOfDamage: gdp_indus * 0.995})
        self.prod_service = DataFrame({GlossaryCore.Years: self. years,
                                       GlossaryCore.GrossOutput: gdp_service,
                                       GlossaryCore.OutputNetOfDamage: gdp_service * 0.995})
        cap_agri = capital_serie * 0.018385
        cap_indus = capital_serie * 0.234987
        cap_service = capital_serie * 0.74662
        self.cap_agri_df = DataFrame({GlossaryCore.Years: self. years, GlossaryCore.Capital: cap_agri, GlossaryCore.UsableCapital: cap_agri * 0.8})
        self.cap_indus_df = DataFrame({GlossaryCore.Years: self. years, GlossaryCore.Capital: cap_indus, GlossaryCore.UsableCapital: cap_indus * 0.8})
        self.cap_service_df = DataFrame({GlossaryCore.Years: self. years, GlossaryCore.Capital: cap_service, GlossaryCore.UsableCapital: cap_service * 0.8})

        self.sectors_list = GlossaryCore.SectorsPossibleValues
        self.share_max_invest = 10.
        self.energy_investment_wo_tax = DataFrame({GlossaryCore.Years: self.years, GlossaryCore.EnergyInvestmentsWoTaxValue: invests})
        self.max_invest_constraint_ref = 10.

        self.damage_agri = DataFrame({GlossaryCore.Years: self.years,
                                      GlossaryCore.Damages: np.linspace(20, 48, self.nb_per),
                                      GlossaryCore.DamagesFromClimate: np.linspace(10, 24, self.nb_per),
                                      GlossaryCore.DamagesFromProductivityLoss: np.linspace(10, 24, self.nb_per),
                                      GlossaryCore.EstimatedDamagesFromClimate: np.linspace(10, 24, self.nb_per),
                                      GlossaryCore.EstimatedDamagesFromProductivityLoss: np.linspace(10, 24, self.nb_per),
                                      GlossaryCore.EstimatedDamages: np.linspace(20, 48, self.nb_per), })
        self.damage_indus = DataFrame({GlossaryCore.Years: self.years,
                                       GlossaryCore.Damages: np.linspace(15, 34, self.nb_per),
                                       GlossaryCore.DamagesFromClimate: np.linspace(5, 10, self.nb_per),
                                       GlossaryCore.DamagesFromProductivityLoss: np.linspace(10, 24, self.nb_per),
                                       GlossaryCore.EstimatedDamagesFromClimate: np.linspace(5, 10, self.nb_per),
                                       GlossaryCore.EstimatedDamagesFromProductivityLoss: np.linspace(10, 24, self.nb_per),
                                       GlossaryCore.EstimatedDamages: np.linspace(15, 34, self.nb_per), })
        self.damage_service = DataFrame({GlossaryCore.Years: self.years,
                                         GlossaryCore.Damages: np.linspace(4, 15, self.nb_per),
                                         GlossaryCore.DamagesFromClimate: np.linspace(1, 6, self.nb_per),
                                         GlossaryCore.DamagesFromProductivityLoss: np.linspace(3, 9, self.nb_per),
                                         GlossaryCore.EstimatedDamagesFromClimate: np.linspace(1, 6, self.nb_per),
                                         GlossaryCore.EstimatedDamagesFromProductivityLoss: np.linspace(3, 9, self.nb_per),
                                         GlossaryCore.EstimatedDamages: np.linspace(4, 15, self.nb_per),
                                         })

    def test_macroeconomics_discipline(self):
        '''
        Check discipline setup and run
        '''
        name = 'Test'
        model_name = 'Macreconomics'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS:  f'{name}',
                   GlossaryCore.NS_MACRO: f'{name}.{model_name}',
                   GlossaryCore.NS_SECTORS: f'{name}'}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()
        inputs_dict = {
            f'{name}.{GlossaryCore.DamageToProductivity}': True,
            f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}': self.invests,
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}': self.invests,
            f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}': self.invests,
            f'{name}.{GlossaryCore.SectorListValue}': self.sectors_list,
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}': self.prod_agri,
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.CapitalDfValue}': self.cap_agri_df,
            f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}': self.prod_indus,
            f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.CapitalDfValue}': self.cap_indus_df,
            f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ProductionDfValue}': self.prod_service,
            f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.CapitalDfValue}': self.cap_service_df,
            f'{name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
            f'{name}.{model_name}.{GlossaryCore.ShareMaxInvestName}': self.share_max_invest,
            f'{name}.{model_name}.{GlossaryCore.MaxInvestConstraintRefName}': self.max_invest_constraint_ref,
            f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDfValue}': self.damage_indus[GlossaryCore.DamageDf['dataframe_descriptor'].keys()],
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDfValue}': self.damage_agri[GlossaryCore.DamageDf['dataframe_descriptor'].keys()],
            f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.DamageDfValue}': self.damage_service[GlossaryCore.DamageDf['dataframe_descriptor'].keys()],
            f'{name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DamageDetailedDfValue}': self.damage_indus[GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys()],
            f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DamageDetailedDfValue}': self.damage_agri[GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys()],
            f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.DamageDetailedDfValue}': self.damage_service[GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys()],
            f'{name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': True,
                'activate_climate_effect_population': True,
                'activate_pandemic_effects': True,
                },
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.root_process.proxy_disciplines[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            # graph.to_plotly().show()
            pass
