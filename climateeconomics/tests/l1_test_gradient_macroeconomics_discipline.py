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
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv
from scipy.interpolate import interp1d

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class MacroEconomicsJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = 2020
        self.year_end = 2050
        self.time_step = 1
        self.years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        # -------------------------
        # csv data
        # energy production
        self.data_dir = join(dirname(__file__), 'data')
        brut_net = 1/1.45
        #prepare energy df
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842, 206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(self.year_start, self.year_end+1))
        energy_supply_values = energy_supply * brut_net
        energy_supply_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.TotalProductionValue: energy_supply_values})
        energy_supply_df.index = self.years
        energy_supply_df.loc[2021, GlossaryCore.TotalProductionValue] = 116.1036348
        self.energy_supply_df = energy_supply_df

        # -------------------------
        # csv data
        # co2 emissions
        energy_supply_csv = read_csv(join(self.data_dir, 'energy_supply_data_onestep.csv'))
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        # energy production divided by 1e3 (scaling factor production)
        self.co2_emissions_gt = energy_supply_df.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        self.co2_emissions_gt.index = self.years
        for i in np.arange(2021, self.year_end+1):
            emission_vefore =  self.co2_emissions_gt.loc[i-1, GlossaryCore.TotalCO2Emissions]
            self.co2_emissions_gt.loc[i,GlossaryCore.TotalCO2Emissions] = emission_vefore*(1.02)
        self.default_co2_efficiency = pd.DataFrame({GlossaryCore.Years: self.years, 'CO2_tax_efficiency': 40.0}, index=self.years)
        # -------------------------
        # csv data
        # damage
        damage_csv = read_csv(join(self.data_dir, 'damage_data_onestep.csv'))
        # adapt lenght to the year range
        damage_df_start = damage_csv.loc[damage_csv[GlossaryCore.Years] >= self.year_start]
        damage_df_end = damage_csv.loc[damage_csv[GlossaryCore.Years] <= self.year_end]
        damage_df = pd.merge(damage_df_start, damage_df_end)
        self.damage_df = damage_df[[GlossaryCore.Years, GlossaryCore.DamageFractionOutput]]
        self.damage_df.index = self.years
        # -------------------------
        # csv data
        # population
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        population_csv = read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df_start = population_csv.loc[population_csv[GlossaryCore.Years] >= self.year_start]
        population_df_end = population_csv.loc[population_csv[GlossaryCore.Years] <= self.year_end]
        self.population_df = pd.merge(population_df_start, population_df_end)
        self.population_df.index = self.years

        self.energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: [3.5] * self.nb_per})

        self.share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: [27.0 - 2.6] * self.nb_per})

        # default CO2 tax
        self.default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: 50.0}, index = self.years)
        self.default_CO2_tax.loc[2020, GlossaryCore.CO2Tax] = 5000.0
        self.default_CO2_tax.loc[2021, GlossaryCore.CO2Tax] = 120.0

        #Population workforce
        self.working_age_population_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.Population1570: 6300}, index=self.years)

        # energy_capital
        nb_per = len(self.years)
        energy_capital_year_start = 16.09
        energy_capital = []
        energy_capital.append(energy_capital_year_start)
        for year in np.arange(1, nb_per):
            energy_capital.append(energy_capital[year - 1] * 1.02)
        self.energy_capital = pd.DataFrame({GlossaryCore.Years: self.years, 'energy_capital': energy_capital})
        self.sectors_list = [GlossaryCore.SectorServices, GlossaryCore.SectorAgriculture, GlossaryCore.SectorIndustry]

    def analytic_grad_entry(self):
        return [
            self.test_macro_economics_analytic_grad,
            self.test_macro_economics_analytic_grad_damageproductivity,
            self.test_macro_economics_analytic_grad_max_damage,
            self.test_macro_economics_analytic_grad_gigantic_invest,
            self.test_macro_economics_very_high_emissions,
            self.test_macro_economics_negativeco2_emissions,
            self.test_macro_economics_negativeco2_tax
        ]

    def test_macro_economics_analytic_grad(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}' }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                            ])

    def test_macro_economics_analytic_grad_damageproductivity(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_damageproductivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data= disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_analytic_grad_max_damage(self):

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.damage_df[GlossaryCore.DamageFractionOutput] = 0.9

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_max_damage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_analytic_grad_gigantic_invest(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: [50.] * self.nb_per,})

        share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: [20.0] * self.nb_per})

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_macroeconomics_discipline_grad_gigantic_invest.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_very_high_emissions(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_csv = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_high_CO2.csv'))
        # adapt lenght to the year range
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        energy_supply_df[GlossaryCore.Years] = energy_supply_df[GlossaryCore.Years]
        co2_emissions_gt = energy_supply_df.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        co2_emissions_gt.index = self.years
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_very_high_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_negativeco2_emissions(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        #- retrieve co2_emissions_gt input
        energy_supply_csv = read_csv(
            join(self.data_dir, 'energy_supply_data_onestep_negative_CO2.csv'))
        # adapt lenght to the year range
        energy_supply_start = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] >= self.year_start]
        energy_supply_end = energy_supply_csv.loc[energy_supply_csv[GlossaryCore.Years] <= self.year_end]
        energy_supply_df = pd.merge(energy_supply_start, energy_supply_end)
        energy_supply_df[GlossaryCore.Years] = energy_supply_df[GlossaryCore.Years]
        co2_emissions_gt = energy_supply_df.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})
        co2_emissions_gt.index = self.years

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_negative_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_negativeco2_tax(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref':f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, -50, len(self.years))}, index=self.years)
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}' : self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_negative_co2_tax.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                             inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                     f'{self.name}.{GlossaryCore.DamageDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                     f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                     f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                     f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                     f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                     f'{self.name}.energy_capital'
                                     ],
                             outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                      f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                      f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                      f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
        ])

    def test_macro_economics_without_compute_gdp_analytic_grad(self):
        """
        Test of analytic gradients when compute_gdp is deactivated
        """

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': False,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'invest_co2_tax_in_renewables': True
                            },
                       f'{self.name}.gross_output_in': pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.GrossOutput: .02}, index=self.years),
                       }


        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_without_compute_gdp.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])

    def test_macro_economics_without_compute_gdp_w_damage_to_productivity_analytic_grad(self):
        """
        Test of analytic gradients when compute_gdp is deactivated
        """

        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': False,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'invest_co2_tax_in_renewables': True,
                            },
                       f'{self.name}.gross_output_in': pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.GrossOutput: .02}, index=self.years),
                       }


        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_without_compute_gdp_w_damage_to_productivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])

    def test_macro_economics_analytic_grad_deactive_co2_tax_investment(self):
        """
        Test of analytic gradient when invest_co2_tax_in_renawables is set to False
        """
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': True,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'invest_co2_tax_in_renewables': False,
                            },
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_without_invest_co2_tax_in_renewables.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])

    def test_gigantic_energy_production_no_damage_productivity(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': False,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_gigantic_energy_production_no_damage_productivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])

    def test_gigantic_energy_production_damage_productivity(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.frac_damage_prod': 0.03,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_gigantic_energy_production_damage_productivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])

    def test_gigantic_energy_production_wo_compute_gdp(self):
        self.model_name = 'Macroeconomics'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': self.time_step,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.{self.model_name}.damage_to_productivity': True,
                       f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
                       f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
                       f'{self.name}.energy_capital': self.energy_capital,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': False,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'invest_co2_tax_in_renewables': True
                            },
                       f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_macroeconomics_discipline_gigantic_energy_production_wo_compute_gdp.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                                    f'{self.name}.{GlossaryCore.DamageDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                                    f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                    f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                                    f'{self.name}.energy_capital'
                                    ],
                            outputs=[#f'{self.name}.{self.model_name}.{GlossaryCore.CapitalDfValue}',
                                     f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                     f'{self.name}.{GlossaryCore.EnergyInvestmentsValue}',
                                     f'{self.name}.{GlossaryCore.ConstraintLowerBoundUsableCapital}',
                                     ])


if '__main__' == __name__:
    cls = MacroEconomicsJacobianDiscTest()
    cls.setUp()
    cls.test_macro_economics_analytic_grad()
