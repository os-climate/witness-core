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
import unittest
from os.path import join, dirname

import numpy as np
import pandas as pd
from pandas import read_csv
from scipy.interpolate import interp1d

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline import SectorDiscipline
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class ServicesDiscTest(unittest.TestCase):
    '''
    Economic Manufacturer static pyworld3 test case
    '''

    def setUp(self):
        '''
        Set up function
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        SectorDiscipline.sector_name = GlossaryCore.SectorIndustry
        self.model_name = SectorDiscipline.sector_name
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)
        
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault +1, 1)
        self.years = years

        year_start = GlossaryCore.YeartStartDefault
        self.year_start = year_start
        year_end = GlossaryCore.YeartEndDefault
        self.year_end = year_end
        time_step = 1
        self.time_step = time_step
        nb_per = round((year_end - year_start) / time_step + 1)
        self.nb_per = nb_per
       
        # input
        data_dir = join(dirname(__file__), 'data')
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        total_workforce_df = read_csv(join(data_dir, 'workingage_population_df.csv'))
        total_workforce_df.index = years
        #multiply ageworking pop by employment rate and by % in services
        workforce = total_workforce_df[GlossaryCore.Population1570]* 0.659 * 0.509
        self.workforce_df = pd.DataFrame({GlossaryCore.Years: years, SectorDiscipline.sector_name: workforce})

        #Energy_supply
        brut_net = 1/1.45
        share_indus = 0.37
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(year_start, year_end+1))
        energy_supply_values = energy_supply * brut_net * share_indus
        self.energy_supply_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.TotalProductionValue: energy_supply_values})
        self.energy_supply_df.index = self.years
        #energy_supply_df.loc[2020, GlossaryCore.TotalProductionValue] = 91.936

        #Investment growth at 2% 
        init_value = 25
        invest_serie = []
        invest_serie.append(init_value)
        for year in np.arange(1, nb_per):
            invest_serie.append(invest_serie[year - 1] * 1.02)
        self.total_invest = pd.DataFrame({GlossaryCore.Years: years,
                                          GlossaryCore.InvestmentsValue: invest_serie})
        
        #damage
        self.damage_fraction_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                GlossaryCore.DamageFractionOutput: np.linspace(0.02, 0.05, len(self.years)),
                                                GlossaryCore.BaseCarbonPrice: np.zeros(self.nb_per)})
        self.damage_fraction_df.index = self.years



    def test_execute(self):
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        section_gdp_df = pd.read_csv(join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        section_list = GlossaryCore.SectionsIndustry
        # out dict definition
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df, 
                       f'{self.name}.{SectorDiscipline.sector_name}.capital_start': 273.1805902, #2019 value for test
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
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': section_gdp_df,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'depreciation_capital'}": 0.058,
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
            f'{self.name}.{SectorDiscipline.sector_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

    def test_execute_forfitting(self):
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        section_gdp_df = pd.read_csv(join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        section_list = GlossaryCore.SectionsIndustry
        # out dict definition
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.InvestmentDfValue}': self.total_invest, #To check if not used
                       f'{self.name}.{SectorDiscipline.sector_name}.hist_sector_investment': self.total_invest,
                       f'{self.name}.{SectorDiscipline.sector_name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                       f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df, 
                       f'{self.name}.{SectorDiscipline.sector_name}.capital_start': 273.1805902, #2019 value for test
                       f'{self.name}.prod_function_fitting': True,
                       f'{self.name}.{SectorDiscipline.sector_name}.energy_eff_max_range_ref' : 15,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_start'}": 1.31162,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'capital_start'}": 6.92448579,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'productivity_gr_start'}": 0.0027844,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'decline_rate_tfp'}": 0.098585,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_k'}": 0.1,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_cst'}": 0.490463,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_xzero'}": 1993,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'energy_eff_max'}": 2.35832,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'output_alpha'}": 0.99,
                       f'{self.name}.{GlossaryCore.SectionList}': section_list,
                       f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': section_gdp_df,
                       f"{self.name}.{SectorDiscipline.sector_name}.{'depreciation_capital'}": 0.058,
                       f'{self.name}.assumptions_dict': {
                           'compute_gdp': True,
                           'compute_climate_impact_on_gdp': False,
                           'activate_climate_effect_population': True,
                           'activate_pandemic_effect_population': True,
                           'invest_co2_tax_in_renewables': True
                       }
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{SectorDiscipline.sector_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        # for graph in graph_list:
        #     graph.to_plotly().show()

