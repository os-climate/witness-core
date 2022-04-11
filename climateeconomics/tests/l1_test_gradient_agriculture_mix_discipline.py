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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pandas import read_csv
import numpy as np
import pandas as pd
import scipy.interpolate as sc
from numpy import asarray, arange, array

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class AgricultureMixJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    def setUp(self):
        self.test_name = 'Test'
        self.ee = ExecutionEngine(self.test_name)

    def analytic_grad_entry(self):
        return [
            self.test_agriculture_discipline_mix_analytic_grad
        ]
                    
    def test_agriculture_discipline_mix_analytic_grad(self):
        '''
        Check discipline setup and run
        '''

        disc_name = 'AgricultureMix'
        ns_dict = {'ns_agriculture': f'{self.test_name}',
                   'ns_energy_study': f'{self.test_name}',
                   'ns_public': f'{self.test_name}',
                   'ns_witness': f'{self.test_name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_mix_disc.AgricultureMixDiscipline'
        builder = self.ee.factory.get_builder_from_module(disc_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        
        self.year_start = 2020
        self.year_end = 2100
        self.years = np.arange(self.year_start, self.year_end + 1)
        year_range = self.year_end - self.year_start + 1
        self.technology_list = ['Crop', 'Forest']

        # ----------------------------------------------------
        # Crop related inputs
        population = np.array(np.linspace(7800, 9500, year_range))
        temperature = np.array(np.linspace(1.05, 4, year_range))

        self.temperature_df = pd.DataFrame(
            {"years": self.years, "temp_atmo": temperature})
        self.temperature_df.index = self.years

        self.population_df = pd.DataFrame(
            {"years": self.years, "population": population})
        self.population_df.index = self.years

        red_meat_percentage = np.linspace(6, 1, year_range)
        white_meat_percentage = np.linspace(14, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
                            'years': self.years,
                            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                'years': self.years,
                                'white_meat_percentage': white_meat_percentage})

        self.diet_df = pd.DataFrame({'red meat': [11.02],
                                'white meat': [31.11],
                                'milk': [79.27],
                                'eggs': [9.68],
                                'rice and maize': [97.76],
                                'potatoes': [32.93],
                                'fruits and vegetables': [217.62],
                                })
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_crop = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_crop = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(1.15, 1.45, year_range)
        self.techno_production_crop = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(17.03, 17.03, year_range)
        techno_prices_wotaxes = np.linspace(17.03, 17.03, year_range)
        self.techno_prices_crop = pd.DataFrame(
            {'years': self.years, 'Crop': techno_prices, 'Crop_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_crop = pd.DataFrame(
            {'years': self.years, 'Crop': CO2_emissions})

        land_use_required_energy = np.linspace(0.06, 0.13, year_range)
        self.land_use_required_crop = pd.DataFrame(
            {'years': self.years, 'Crop (Gha)': land_use_required_energy})
        # ----------------------------------------------------
        # Forest related inputs
        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {"years": self.years, "deforested_surface": deforestation_surface})
        forest_invest = np.linspace(5, 8, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": self.years, "forest_investment": forest_invest})
        mw_invest = np.linspace(1, 4, year_range)
        uw_invest = np.linspace(0, 1, year_range)
        self.mw_invest_df = pd.DataFrame(
            {"years": self.years, "investment": mw_invest})
        self.uw_invest_df = pd.DataFrame(
            {"years": self.years, "investment": uw_invest})

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_forest = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_forest = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(15, 16, year_range)
        self.techno_production_forest = pd.DataFrame(
            {'years': self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(9, 9, year_range)
        techno_prices_wotaxes = np.linspace(9, 9, year_range)
        self.techno_prices_forest = pd.DataFrame(
            {'years': self.years, 'Forest': techno_prices, 'Forest_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_forest = pd.DataFrame(
            {'years': self.years, 'Forest': CO2_emissions})

        land_use_required = np.linspace(4, 3.5, year_range)
        self.land_use_required_forest = pd.DataFrame(
            {'years': self.years, 'Forest (Gha)': land_use_required})
        # -----------------------------------------------------
        # Agriculture mix related inputs
        self.margin = pd.DataFrame(
            {'years': self.years, 'margin': np.ones(len(self.years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {'years': self.years, 'transport': np.ones(len(self.years)) * 7.6})

        self.energy_carbon_emissions = pd.DataFrame(
            {'years': self.years, 'biomass_dry': - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, 'electricity': 0.0, 'methane': 0.123 / 15.4, 'syngas': 0.0, 'hydrogen.gaseous_hydrogen': 0.0, 'crude oil': 0.02533})
        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01,  34.05,   39.08,  44.69,   50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        self.co2_taxes = pd.DataFrame(
            {'years': self.years, 'CO2_tax': func(self.years)})
        # -----------------------------------------------------
        # Investments related inputs
        invest_level_crop = np.linspace(0.5, 0.2, year_range)
        invest_level_forest = np.linspace(0.5, 0.7, year_range)
        self.invest_level_crop_df = pd.DataFrame({'years': self.years,
                                                        'invest': invest_level_crop})
        self.invest_level_forest_df = pd.DataFrame({'years': self.years,
                                                        'invest': invest_level_forest})

                                                        
        inputs_dict = {f'{self.test_name}.year_start': self.year_start,
                       f'{self.test_name}.year_end': self.year_end,
                       f'{self.test_name}.{disc_name}.Crop.techno_consumption': self.techno_consumption_crop,
                       f'{self.test_name}.{disc_name}.Crop.techno_consumption_woratio': self.techno_consumption_woratio_crop,
                       f'{self.test_name}.{disc_name}.Crop.techno_production': self.techno_production_crop,
                       f'{self.test_name}.{disc_name}.Crop.techno_prices': self.techno_prices_crop,
                       f'{self.test_name}.{disc_name}.Crop.CO2_emissions': self.CO2_emissions_crop,
                       f'{self.test_name}.{disc_name}.Crop.land_use_required': self.land_use_required_crop,
                       f'{self.test_name}.{disc_name}.Forest.techno_consumption': self.techno_consumption_forest,
                       f'{self.test_name}.{disc_name}.Forest.techno_consumption_woratio': self.techno_consumption_woratio_forest,
                       f'{self.test_name}.{disc_name}.Forest.techno_production': self.techno_production_forest,
                       f'{self.test_name}.{disc_name}.Forest.techno_prices': self.techno_prices_forest,
                       f'{self.test_name}.{disc_name}.Forest.CO2_emissions': self.CO2_emissions_forest,
                       f'{self.test_name}.{disc_name}.Forest.land_use_required': self.land_use_required_forest,
                       f'{self.test_name}.technologies_list': self.technology_list,
                       f'{self.test_name}.CO2_taxes': self.co2_taxes
                      }
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_agriculture_mix_discipline.pkl', discipline=disc_techno,
                            step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.test_name}.{disc_name}.Crop.techno_consumption', 
                                    f'{self.test_name}.{disc_name}.Crop.techno_consumption_woratio',
                                    f'{self.test_name}.{disc_name}.Crop.techno_production',
                                    f'{self.test_name}.{disc_name}.Crop.techno_prices',
                                    f'{self.test_name}.{disc_name}.Crop.CO2_emissions',
                                    f'{self.test_name}.{disc_name}.Crop.land_use_required',
                                    f'{self.test_name}.{disc_name}.Forest.techno_consumption', 
                                    f'{self.test_name}.{disc_name}.Forest.techno_consumption_woratio',
                                    f'{self.test_name}.{disc_name}.Forest.techno_production',
                                    f'{self.test_name}.{disc_name}.Forest.techno_prices',
                                    f'{self.test_name}.{disc_name}.Forest.CO2_emissions',
                                    f'{self.test_name}.{disc_name}.Forest.land_use_required',
                                    ],
                            outputs=[f'{self.test_name}.{disc_name}.CO2_emissions',
                                    f'{self.test_name}.{disc_name}.CO2_per_use',
                                    f'{self.test_name}.{disc_name}.energy_prices',
                                    f'{self.test_name}.{disc_name}.energy_consumption',
                                    f'{self.test_name}.{disc_name}.energy_consumption_woratio',
                                    f'{self.test_name}.{disc_name}.energy_production',
                                    f'{self.test_name}.{disc_name}.land_use_required'
                            ])
