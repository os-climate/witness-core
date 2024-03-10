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
from os.path import join, dirname

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv
from scipy.interpolate import interp1d

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

    def test_execute(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   'ns_energy_study': f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault +1, 1)
        self.years = years

        year_start = GlossaryCore.YeartStartDefault
        year_end = GlossaryCore.YeartEndDefault
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

        # Our world in data Direct primary energy conso data until 2019, then for 2020 drop in 6% according to IEA
        # then IEA data*0.91 (WEO 2020 stated) until 2040 then invented. 0.91 =
        # ratio net brut in 2020
        # Energy production divided by 1e3 (scaling factor production)
        # source for IEA energy outlook: IEA 2022; World energy outlook 2020, https://www.iea.org/reports/world-energy-outlook-2020, License: CC BY 4.0.
        brut_net = 1/1.45
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(year_start, year_end+1))
        energy_supply_values = energy_supply * brut_net 
        energy_supply_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.TotalProductionValue: energy_supply_values})
        energy_supply_df.index = self.years
        energy_supply_df.loc[2021, GlossaryCore.TotalProductionValue] = 116.1036348

        self.damage_fraction_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                GlossaryCore.DamageFractionOutput: np.linspace(0.01, 0.05, nb_per),
                                                GlossaryCore.BaseCarbonPrice: np.zeros(self.nb_per)})
        self.damage_fraction_df.index = self.years

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
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        population_df = read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        working_age_pop_df = read_csv(
            join(data_dir, 'workingage_population_df.csv'))
        working_age_pop_df.index = years
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault][[
            GlossaryCore.Years, 'total_CO2_emitted']]
        energy_supply_df_y[GlossaryCore.Years] = energy_supply_df_all[GlossaryCore.Years]
        co2_emissions_gt = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        co2_emissions_gt.index = years
        default_co2_efficiency = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2TaxEfficiencyValue: 40.0}, index=years)
        sectors_list = [GlossaryCore.SectorServices, GlossaryCore.SectorAgriculture, GlossaryCore.SectorIndustry]
        section_list = GlossaryCore.SectionsPossibleValues
        section_gdp_df = pd.read_csv(join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        # out dict definition
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': working_age_pop_df, 
                       f'{self.name}.{GlossaryCore.EnergyCapitalDfValue}': self.energy_capital_df,
                       f'{self.name}.{GlossaryCore.SectorListValue}': sectors_list,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': section_gdp_df,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': True,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effect_population': True,
                           'invest_co2_tax_in_renewables': True
                           },
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False,
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

