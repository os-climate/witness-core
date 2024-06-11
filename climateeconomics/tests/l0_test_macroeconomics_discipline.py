'''
Copyright 2022 Airbus SAS
Modifications on 2023/08/17-2023/11/09 Copyright 2023 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas import DataFrame

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class MacroDiscTest(unittest.TestCase):
    '''
    Economic Manufacturer static pyworld3 test case
    '''

    def setUp(self):
        '''
        Set up function
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.energy_supply_df_all = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years))
        })

        self.co2_emissions_gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(1035, 0, len(self.years)),
        })

    def test_execute(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   'ns_energy_study': f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        self.years = years

        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefault
        time_step = 1
        nb_per = round(
            (year_end - year_start) / time_step + 1)
        self.nb_per = nb_per

        energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: np.asarray([3.5] * nb_per)})  # in G$

        share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: np.asarray([27. - 2.6] * nb_per)})

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(43, 76, len(self.years))
        })

        self.damage_fraction_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                GlossaryCore.DamageFractionOutput: np.linspace(0.01, 0.05, nb_per),
                                                GlossaryCore.BaseCarbonPrice: np.zeros(self.nb_per)})

        default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2Tax: 50.0}, index=years)
        
        # energy_capital
        nb_per = len(self.years)
        energy_capital_year_start = 16.09
        energy_capital = []
        energy_capital.append(energy_capital_year_start)
        for year in np.arange(1, nb_per):
            energy_capital.append(energy_capital[year - 1] * 1.02)
        self.energy_capital_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.Capital: energy_capital})

        # retrieve co2_emissions_gt input
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        self.working_age_pop_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Population1570: np.linspace(5490, 6061, len(self.years))
        })

        self.share_energy_per_sector_percentage = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.SectorServices: 14.56,
            GlossaryCore.SectorAgriculture: 2.13,
            GlossaryCore.SectorIndustry: 63.30,
            #GlossaryCore.SectorNonEco: 20.0, # not an economic sector
        })

        carbon_intensity_energy = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyCarbonIntensityDfValue: 10.
        })

        default_co2_efficiency = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2TaxEfficiencyValue: 40.0}, index=years)
        sectors_list = [GlossaryCore.SectorServices, GlossaryCore.SectorAgriculture, GlossaryCore.SectorIndustry]
        section_list = GlossaryCore.SectionsPossibleValues
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        weighted_average_percentage_per_sector_df = pd.read_csv(
            join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        subsector_share_dict = {
            **{GlossaryCore.Years: self.years, },
            **dict(zip(weighted_average_percentage_per_sector_df.columns[1:],
                       weighted_average_percentage_per_sector_df.values[0, 1:]))
        }
        section_gdp_df = pd.DataFrame(subsector_share_dict)
        share_residential_energy_consumption = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.ShareSectorEnergy: 10.
        })
        # out dict definition
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_pop_df,
                       f'{self.name}.{GlossaryCore.EnergyCapitalDfValue}': self.energy_capital_df,
                       f'{self.name}.{GlossaryCore.SectorListValue}': sectors_list,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': section_gdp_df,
                       f'{self.name}.{GlossaryCore.ShareResidentialEnergyDfValue}': share_residential_energy_consumption,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': True,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effects': True,
                           'invest_co2_tax_in_renewables': True
                           },
                       f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': carbon_intensity_energy
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for i,graph in enumerate(graph_list):
            #graph.to_plotly().show()
            pass

