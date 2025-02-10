'''
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
'''
from os.path import dirname

import numpy as np
import pandas as pd
import scipy.interpolate as sc
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class AgricultureMixJacobianDiscTest(AbstractJacobianUnittest):
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

        disc_name = 'Agriculture'
        ns_dict = {'ns_agriculture': f'{self.test_name}',
                   'ns_energy_study': f'{self.test_name}',
                   'ns_public': f'{self.test_name}',
                   GlossaryCore.NS_WITNESS: f'{self.test_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_mix_disc.AgricultureMixDiscipline'
        builder = self.ee.factory.get_builder_from_module(disc_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        
        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        self.years = np.arange(self.year_start, self.year_end + 1)
        year_range = self.year_end - self.year_start + 1
        self.technology_list = ['Crop', GlossaryCore.Forestry]

        # ----------------------------------------------------
        # Crop related inputs
        population = np.array(np.linspace(7800, 9500, year_range))
        temperature = np.array(np.linspace(1.05, 4, year_range))

        self.temperature_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.TempAtmo: temperature})

        self.population_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.PopulationValue: population})

        red_meat_percentage = np.linspace(6, 1, year_range)
        white_meat_percentage = np.linspace(14, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
                            GlossaryCore.Years: self.years,
                            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                GlossaryCore.Years: self.years,
                                'white_meat_percentage': white_meat_percentage})

        self.diet_df = pd.DataFrame({GlossaryCore.RedMeat: [11.02],
                                GlossaryCore.WhiteMeat: [31.11],
                                GlossaryCore.Milk: [79.27],
                                GlossaryCore.Eggs: [9.68],
                                GlossaryCore.RiceAndMaize: [97.76],
                                'potatoes': [32.93],
                                GlossaryCore.FruitsAndVegetables: [217.62],
                                })
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(1.15, 1.45, year_range)
        self.techno_production_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(17.03, 17.03, year_range)
        techno_prices_wotaxes = np.linspace(17.03, 17.03, year_range)
        self.techno_prices_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop': techno_prices, 'Crop_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop': CO2_emissions})

        land_use_required_energy = np.linspace(0.06, 0.13, year_range)
        self.land_use_required_crop = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Crop (Gha)': land_use_required_energy})
        # ----------------------------------------------------
        # Forest related inputs
        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, "deforested_surface": deforestation_surface})
        reforestation_invest = np.linspace(5.0, 8.0, len(self.years))
        self.reforestation_investment_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, "reforestation_investment": reforestation_invest})
        mw_invest = np.linspace(1, 4, year_range)
        uw_invest = np.linspace(0, 1, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.InvestmentsValue: mw_invest})
        self.uw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.InvestmentsValue: uw_invest})

        techno_consumption = np.linspace(0, 0, year_range)
        self.techno_consumption_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption})

        techno_consumption_woratio = np.linspace(0, 0, year_range)
        self.techno_consumption_woratio_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_consumption_woratio})

        techno_production = np.linspace(15, 16, year_range)
        self.techno_production_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'biomass_dry (TWh)': techno_production})

        techno_prices = np.linspace(9, 9, year_range)
        techno_prices_wotaxes = np.linspace(9, 9, year_range)
        self.techno_prices_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.Forestry: techno_prices, 'Forestry_wotaxes': techno_prices_wotaxes})

        CO2_emissions = np.linspace(0, 0, year_range)
        self.CO2_emissions_forestry = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.Forestry: CO2_emissions})

        CO2_emitted_land = pd.DataFrame()
        emission_forest = np.linspace(0.04, 0.04, len(self.years))
        cum_emission = np.cumsum(emission_forest) + 3.21
        CO2_emitted_land[GlossaryCore.Years] = self.years
        CO2_emitted_land['emitted_CO2_evol_cumulative'] = cum_emission

        CH4_emitted_land = pd.DataFrame()
        CH4_emitted_land[GlossaryCore.Years] = self.years
        CH4_emitted_land['emitted_CH4_evol_cumulative'] = cum_emission/100

        N2O_emitted_land = pd.DataFrame()
        N2O_emitted_land[GlossaryCore.Years] = self.years
        N2O_emitted_land['emitted_N2O_evol_cumulative'] = cum_emission / 500


        land_use_required = np.linspace(4, 3.5, year_range)
        self.land_use_required_forest = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'Forestry (Gha)': land_use_required})
        # -----------------------------------------------------
        # Agriculture mix related inputs
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'margin': np.ones(len(self.years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'transport': np.ones(len(self.years)) * 7.6})

        self.stream_co2_emissions = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryEnergy.biomass_dry: - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, GlossaryEnergy.electricity: 0.0, GlossaryEnergy.methane: 0.123 / 15.4, 'syngas': 0.0, f"{GlossaryEnergy.hydrogen}.{GlossaryEnergy.gaseous_hydrogen}": 0.0, 'crude oil': 0.02533})
        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01,  34.05,   39.08,  44.69,   50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        self.co2_taxes = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: func(self.years)})

        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years),
            GlossaryCore.NonUseCapital: 0.,
        })
                                                        
        inputs_dict = {f'{self.test_name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.test_name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.test_name}.{disc_name}.Crop.techno_consumption': self.techno_consumption_crop,
                       f'{self.test_name}.{disc_name}.Crop.{GlossaryCore.TechnoEnergyDemandsValue}': self.techno_consumption_woratio_crop,
                       f'{self.test_name}.{disc_name}.Crop.techno_production': self.techno_production_crop,
                       f'{self.test_name}.{disc_name}.Crop.techno_prices': self.techno_prices_crop,
                       f'{self.test_name}.{disc_name}.Crop.CO2_emissions': self.CO2_emissions_crop,
                       f'{self.test_name}.{disc_name}.Crop.land_use_required': self.land_use_required_crop,
                       f'{self.test_name}.{disc_name}.Crop.{GlossaryEnergy.TechnoCapitalValue}': techno_capital,
                       f'{self.test_name}.{disc_name}.Forest.{GlossaryEnergy.TechnoCapitalValue}': techno_capital,
                       f'{self.test_name}.{disc_name}.Forest.techno_consumption': self.techno_consumption_forest,
                       f'{self.test_name}.{disc_name}.Forest.{GlossaryCore.TechnoEnergyDemandsValue}': self.techno_consumption_woratio_forest,
                       f'{self.test_name}.{disc_name}.Forest.techno_production': self.techno_production_forest,
                       f'{self.test_name}.{disc_name}.Forest.techno_prices': self.techno_prices_forest,
                       f'{self.test_name}.{disc_name}.Forest.CO2_emissions': self.CO2_emissions_forest,
                       f'{self.test_name}.{disc_name}.Forest.land_use_required': self.land_use_required_forest,
                       f'{self.test_name}.{GlossaryCore.techno_list}': self.technology_list,
                       f'{self.test_name}.{GlossaryCore.CO2TaxesValue}': self.co2_taxes
                      }
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_agriculture_mix_discipline.pkl', discipline=disc_techno,
                            step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.test_name}.{disc_name}.Crop.techno_consumption',
                                    f'{self.test_name}.{disc_name}.Crop.{GlossaryCore.TechnoEnergyDemandsValue}',
                                    f'{self.test_name}.{disc_name}.Crop.techno_production',
                                    f'{self.test_name}.{disc_name}.Crop.techno_prices',
                                    f'{self.test_name}.{disc_name}.Crop.CO2_emissions',
                                    f'{self.test_name}.{disc_name}.Crop.land_use_required',
                                    f'{self.test_name}.{disc_name}.Forest.techno_consumption',
                                    f'{self.test_name}.{disc_name}.Forest.{GlossaryCore.TechnoEnergyDemandsValue}',
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
                                    f'{self.test_name}.{disc_name}.{GlossaryCore.StreamProductionValue}',
                                    f'{self.test_name}.{disc_name}.land_use_required',
                            ])
