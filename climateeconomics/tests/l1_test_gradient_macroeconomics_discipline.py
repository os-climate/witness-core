'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/04-2023/11/09 Copyright 2023 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class MacroEconomicsJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = 'Test'

        self.ee = ExecutionEngine(self.name)

        self.model_name = 'Macroeconomics'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.nb_per = self.year_end - self.year_start + 1

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(43, 76, len(self.years))
        })

        self.default_co2_efficiency = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'CO2_tax_efficiency': 40.0
        })

        self.damage_fraction_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: np.linspace(0.001, 0.01, len(self.years))
        })

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        self.energy_investment_wo_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: [3.5] * self.nb_per})

        self.share_non_energy_investment = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: np.linspace(27.0 - 2.6, 27.0 - 2.6, self.nb_per)})

        # default CO2 tax
        self.default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: 50.0}, index=self.years)
        self.default_CO2_tax.loc[GlossaryCore.YearStartDefault, GlossaryCore.CO2Tax] = 5000.0
        self.default_CO2_tax.loc[2021, GlossaryCore.CO2Tax] = 120.0

        self.working_age_population_df = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.Population1570: 6300})

        self.energy_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 16.09 * (1.02 ** np.arange(len(self.years)))
        })

        self.sectors_list = [GlossaryCore.SectorServices, GlossaryCore.SectorAgriculture, GlossaryCore.SectorIndustry]

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        weighted_average_percentage_per_sector_df = pd.read_csv(
            join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        subsector_share_dict = {
            **{GlossaryCore.Years: self.years, },
            **dict(zip(weighted_average_percentage_per_sector_df.columns[1:],
                       weighted_average_percentage_per_sector_df.values[0, 1:]))
        }
        self.gdp_section_df = pd.DataFrame(subsector_share_dict)

        self.energy_supply_df_all = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years))
        })

        self.co2_emissions_gt = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.TotalCO2Emissions: np.linspace(34, 55, len(self.years))})

        self.carbon_intensity_energy = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyCarbonIntensityDfValue: 10.
        })

        default_value_energy_consumption_dict = DatabaseWitnessCore.EnergyConsumptionPercentageSectionsDict.value
        default_non_energy_emissions_dict = DatabaseWitnessCore.SectionsNonEnergyEmissionsDict.value
        self.default_value_dict_consumption = {}
        self.default_value_dict_emissions = {}
        for sector, default_value_consumption in default_value_energy_consumption_dict.items():
            section_gdp_percentage_dict = {
                **{GlossaryCore.Years: self.years, },
                **dict(zip(default_value_consumption.columns[1:],
                           default_value_consumption.values[0, 1:]))
            }
            self.default_value_dict_consumption[sector] = pd.DataFrame(section_gdp_percentage_dict)

        for sector, default_value_emissions in default_non_energy_emissions_dict.items():
            section_gdp_percentage_dict = {
                **{GlossaryCore.Years: self.years, },
                **dict(zip(default_value_emissions.columns[1:],
                           default_value_emissions.values[0, 1:]))
            }
            self.default_value_dict_emissions[sector] = pd.DataFrame(section_gdp_percentage_dict)
        self.model_name = 'Macroeconomics'
        self.inputs_dict = {
            f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.name}.init_rate_time_pref': 0.015,
            f'{self.name}.conso_elasticity': 1.45,
            f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': False,
            f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
            f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
            f'{self.name}.{GlossaryCore.EnergyProductionValue}': self.energy_supply_df,
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
            f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
            f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
            f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
            f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_population_df,
            f'{self.name}.{GlossaryCore.EnergyCapitalDfValue}': self.energy_capital,
            f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list,
            f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': self.gdp_section_df,
            f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.carbon_intensity_energy,
            f'{self.name}.{self.model_name}.{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Services': self.default_value_dict_consumption['Services'],
            f'{self.name}.{self.model_name}.{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Agriculture': self.default_value_dict_consumption['Agriculture'],
            f'{self.name}.{self.model_name}.{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Industry': self.default_value_dict_consumption['Industry'],
        }

        self.checked_inputs = [f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}',
                               f'{self.name}.{GlossaryCore.EnergyProductionValue}',
                               f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
                               f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                               f'{self.name}.{GlossaryCore.PopulationDfValue}',
                               f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}',
                               f'{self.name}.{GlossaryCore.EnergyCapitalDfValue}',
                               ]

        self.checked_outputs = [
            f'{self.name}.{self.model_name}.{GlossaryCore.WorkforceDfValue}',
            #f'{self.name}.{GlossaryCore.TempOutput}',
            f'{self.name}.{GlossaryCore.DamageDfValue}',
            f'{self.name}.{GlossaryCore.EconomicsDfValue}',
            f'{self.name}.{GlossaryCore.UsableCapitalObjectiveName}',
            f'{self.name}.{GlossaryCore.ConstraintUpperBoundUsableCapital}',
            f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.SectionEnergyConsumptionDfValue}',
            f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.SectionEnergyConsumptionDfValue}',
            f'{self.name}.{GlossaryCore.SectorServices}.{GlossaryCore.SectionGdpDfValue}',
            f'{self.name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.SectionGdpDfValue}',
            f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}',
        ]

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
        inputs_dict = self.inputs_dict

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__), filename='jacobian_macroeconomics_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_analytic_grad_damageproductivity(self):
        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': True,
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_grad_damageproductivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_analytic_grad_max_damage(self):
        self.damage_fraction_df[GlossaryCore.DamageFractionOutput] = 0.9

        inputs_dict = self.inputs_dict
        inputs_dict.update({
            f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': False,
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
        })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_grad_max_damage.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_analytic_grad_gigantic_invest(self):
        energy_investment_wo_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: [50.] * self.nb_per, })

        share_non_energy_investment = pd.DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: [20.0] * self.nb_per})
        inputs_dict = self.inputs_dict

        inputs_dict.update({f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': energy_investment_wo_tax,
                       f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': share_non_energy_investment,
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_grad_gigantic_invest.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_very_high_emissions(self):
        co2_emissions_gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(1035, 0, len(self.years)),
        })

        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_very_high_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_negativeco2_emissions(self):
        co2_emissions_gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(45, -0.66, len(self.years)),
        })
        inputs_dict = self.inputs_dict
        inputs_dict.update({
            f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
        })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_negative_emissions.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_negativeco2_tax(self):

        self.default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, -50, len(self.years))})
        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax})

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_negative_co2_tax.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_macro_economics_without_compute_gdp_analytic_grad(self):
        """
        Test of analytic gradients when compute_gdp is deactivated
        """

        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.carbon_intensity_energy,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': False,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'activate_pandemic_effects': True,
                            },
                       f'{self.name}.gross_output_in': pd.DataFrame(
                           {GlossaryCore.Years: self.years, GlossaryCore.GrossOutput: .02}),
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_without_compute_gdp.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_gigantic_energy_production_no_damage_productivity(self):
        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': False,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_gigantic_energy_production_no_damage_productivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_gigantic_energy_production_damage_productivity(self):
        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': True,
                       f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_gigantic_energy_production_damage_productivity.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def test_gigantic_energy_production_wo_compute_gdp(self):


        energy_supply = pd.DataFrame.copy(self.energy_supply_df)
        energy_prod = energy_supply[GlossaryCore.TotalProductionValue] * 1.035 ** np.arange(self.nb_per)
        energy_prod[20:] = energy_prod[20:] / 10.
        energy_supply[GlossaryCore.TotalProductionValue] = energy_prod
        inputs_dict = self.inputs_dict
        inputs_dict.update({f'{self.name}.{GlossaryCore.EnergyProductionValue}': energy_supply,
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': False,
                            'compute_climate_impact_on_gdp': True,
                            'activate_climate_effect_population': True,
                            'activate_pandemic_effects': True,
                            },
                       })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_macroeconomics_discipline_gigantic_energy_production_wo_compute_gdp.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)

    def _test_problematic_optim_point(self):

        import os
        import pickle
        with open(os.path.join("data", "uc1optim.pkl"), "rb") as f:
            dict_input_optimized_point = pickle.load(f)
        
        def find_var_in_dict(varname: str):
            try:
                varname_in_dict_optimized = list(filter(lambda x: varname in x, dict_input_optimized_point.keys()))[0]
                var_value = dict_input_optimized_point[varname_in_dict_optimized]
                return var_value
            except IndexError :
                print(varname)

        for checked_input in list(self.inputs_dict.keys()) + self.checked_inputs:
            checked_inputvarname = checked_input.split('.')[-1]
            var_value = find_var_in_dict(checked_inputvarname)

            varname_in_input_dicts = list(filter(lambda x: checked_inputvarname in x, self.inputs_dict.keys()))[0]

            self.inputs_dict.update({varname_in_input_dicts: var_value})

        self.inputs_dict.update({
            f'{self.name}.assumptions_dict': find_var_in_dict('assumptions_dict'),
            f'{self.name}.{GlossaryCore.YearEnd}': find_var_in_dict(GlossaryCore.YearEnd),
        })

        self.ee.load_study_from_input_dict(self.inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_at_opt_point_uc1.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=self.checked_inputs,
                            outputs=self.checked_outputs)


if '__main__' == __name__:
    cls = MacroEconomicsJacobianDiscTest()
    cls.setUp()
    cls.test_macro_economics_analytic_grad()
