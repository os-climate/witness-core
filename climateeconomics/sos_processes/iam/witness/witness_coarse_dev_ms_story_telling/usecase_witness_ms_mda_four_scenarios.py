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
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda_four_scenarios_tp35 import (
    Study as StudyMSmdaTippingPoint35,
)


class Study(StudyMSmdaTippingPoint35):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(file_path=__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.check_outputs = True

    def setup_usecase(self, study_folder_path=None):

        values_dict = super().setup_usecase()
        tipping_point_variable = 'Damage.tp_a3'
        values_dict.update({
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE2}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE2B}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7}.{tipping_point_variable}': 6.081,
        })
        return values_dict

    def specific_check_outputs(self):
        """Some outputs are retrieved and their range is checked"""
        list_scenario = {self.USECASE2, self.USECASE2B, self.USECASE4, self.USECASE7}
        dm = self.execution_engine.dm
        all_temp_increase = dm.get_all_namespaces_from_var_name('temperature_df')
        ref_value_temp_increase = {
            self.USECASE2: 4.23,
            self.USECASE2B: 4.07,
            self.USECASE4: 3.34,
            self.USECASE7: 2.41
        }
        ref_temperature_2020 = 1.3

        all_co2_taxes = dm.get_all_namespaces_from_var_name('CO2_taxes')
        ref_value_co2_tax = {
            self.USECASE2: 0,
            self.USECASE2B: 0,
            self.USECASE4: 0,
            self.USECASE7: 1192
        }
        all_gdps = dm.get_all_namespaces_from_var_name(GlossaryCore.EconomicsDfValue)
        ref_value_world_gdp_net_of_damage = {
            self.USECASE2: 421,
            self.USECASE2B: 126,
            self.USECASE4: 198,
            self.USECASE7: 251
        }
        ref_gdp_2020 = 129.9

        all_co2_emissions = dm.get_all_namespaces_from_var_name(GlossaryCore.TotalGWPEmissionsDfValue)
        ref_value_co2_emissions = {
            self.USECASE2: 139,
            self.USECASE2B: 73,
            self.USECASE4: 51,
            self.USECASE7: -2.8 # todo : refit NZE scenario
        }  # todo : replace by GWP in future

        all_net_energy_productions = dm.get_all_namespaces_from_var_name(f'EnergyMix.{GlossaryCore.EnergyProductionDetailedValue}')
        ref_value_net_energy_production = {
            self.USECASE2: 338*1e3,
            self.USECASE2B: 182*1e3,
            self.USECASE4: 223*1e3,
            self.USECASE7: 248*1e3
        }
        ref_net_prod_2020 = 116*1e3

        all_populations = dm.get_all_namespaces_from_var_name(GlossaryCore.PopulationDfValue)
        ref_value_populations = {
            self.USECASE2: 9.17 * 1e3,
            self.USECASE2B: 8.64 * 1e3,
            self.USECASE4: 8.84 * 1e3,
            self.USECASE7: 8.96 * 1e3
        }
        ref_population_2020 = 7.79 * 1e3

        all_energy_investment_without_tax = dm.get_all_namespaces_from_var_name(GlossaryCore.EnergyInvestmentsWoTaxValue)
        ref_value_energy_investment_without_tax = {
            self.USECASE2: 2.11,
            self.USECASE2B: 0.65,
            self.USECASE4: 1.41,
            self.USECASE7: 3.0
        }
        ref_co2_2020 = 45.88
        tolerance_high_ref_2020 = 1.015
        tolerance_low_ref_2020 = 0.985
        error_msg = ''
        for scenario in list_scenario:
            # Checking that the temperature value in 2100 is in an acceptable range for each usecase
            for scenario_temp_increase in all_temp_increase:
                if scenario in scenario_temp_increase:
                    temp_increase = dm.get_value(scenario_temp_increase)
                    value_temp_increase = temp_increase.loc[temp_increase['years'] == 2100]['temp_atmo'].values[0]
                    # temperature year 2020
                    value_temp_2020 = temp_increase.loc[temp_increase['years'] == 2020]['temp_atmo'].values[0]
                    error_msg += self.should_be_greater(value_temp_increase, ref_value_temp_increase[scenario] * 0.8, f"{scenario_temp_increase}[2100]")
                    error_msg += self.should_be_lower(value_temp_increase, ref_value_temp_increase[scenario] * 1.2, f"{scenario_temp_increase}[2100]")
                    error_msg += self.should_be_greater(value_temp_2020, ref_temperature_2020 * tolerance_low_ref_2020, f"{scenario_temp_increase}[2020]")
                    error_msg += self.should_be_lower(value_temp_2020,  ref_temperature_2020 * tolerance_high_ref_2020, f"{scenario_temp_increase}[2020]")

            # Checking that the CO2 tax value in 2100 is in an acceptable range for each usecase
            for scenario_co2_tax in all_co2_taxes:
                if scenario in scenario_co2_tax:
                    co2_tax = dm.get_value(scenario_co2_tax)
                    value_co2_tax = co2_tax.loc[co2_tax['years'] == 2100]['CO2_tax'].values[0]
                    error_msg += self.should_be_greater(value_co2_tax, ref_value_co2_tax[scenario] * 0.8, f"{scenario_co2_tax}[2100]")
                    error_msg += self.should_be_lower(value_co2_tax, ref_value_co2_tax[scenario] * 1.2, f"{scenario_co2_tax}[2100]")

            for scenario_gdp in all_gdps:
                if scenario in scenario_gdp:
                    gdp_df = dm.get_value(scenario_gdp)
                    value_gdp = gdp_df.loc[gdp_df['years'] == 2100][GlossaryCore.OutputNetOfDamage].values[0]
                    value_gdp_2020 = gdp_df.loc[gdp_df['years'] == 2020][GlossaryCore.OutputNetOfDamage].values[0]
                    error_msg += self.should_be_greater(value_gdp, ref_value_world_gdp_net_of_damage[scenario] * 0.8, f"{scenario_gdp}[2100]")
                    error_msg += self.should_be_lower(value_gdp, ref_value_world_gdp_net_of_damage[scenario] * 1.2, f"{scenario_gdp}[2100]")
                    # assert on value of start, tolerance at 1.5%
                    error_msg += self.should_be_greater(value_gdp_2020, ref_gdp_2020 * tolerance_low_ref_2020, f"{scenario_gdp}[2020]")
                    error_msg += self.should_be_lower(value_gdp_2020, ref_gdp_2020 * tolerance_high_ref_2020, f"{scenario_gdp}[2020]")
            for scenario_emissions in all_co2_emissions:
                if scenario in scenario_emissions:
                    emissions_df = dm.get_value(scenario_emissions)
                    value_emissions = emissions_df.loc[emissions_df['years'] == 2100]['CO2_20'].values[0] # todo : replace by GWP in futur
                    value_emissions_2020 = emissions_df.loc[emissions_df['years'] == 2020]['CO2_20'].values[0] # todo : replace by GWP in futur
                    if self.USECASE7 in scenario:
                        error_msg += self.should_be_lower(value_emissions, ref_value_co2_emissions[scenario] * 0.8, f"{scenario_emissions}[2100]")
                        error_msg += self.should_be_greater(value_emissions, ref_value_co2_emissions[scenario] * 1.2, f"{scenario_emissions}[2100]")
                    else:
                        error_msg += self.should_be_greater(value_emissions, ref_value_co2_emissions[scenario] * 0.8, f"{scenario_emissions}[2100]")
                        error_msg += self.should_be_lower(value_emissions, ref_value_co2_emissions[scenario] * 1.2, f"{scenario_emissions}[2100]")
                    error_msg += self.should_be_greater(value_emissions_2020, ref_co2_2020 * tolerance_low_ref_2020, f"{scenario_emissions}[2020]")
                    error_msg += self.should_be_lower(value_emissions_2020, ref_co2_2020 * tolerance_high_ref_2020, f"{scenario_emissions}[2020]")

            for scenario_net_energy_production in all_net_energy_productions:
                if scenario in scenario_net_energy_production:
                    net_energy_production_df = dm.get_value(scenario_net_energy_production)
                    value_net_energy_production = \
                    net_energy_production_df.loc[net_energy_production_df['years'] == 2100]['Total production (uncut)'].values[0]
                    value_net_energy_production_2020 = \
                    net_energy_production_df.loc[net_energy_production_df['years'] == 2020]['Total production (uncut)'].values[0]
                    error_msg += self.should_be_greater(value_net_energy_production, ref_value_net_energy_production[scenario] * 0.8, f"{scenario_net_energy_production}[2100]")
                    error_msg += self.should_be_lower(value_net_energy_production, ref_value_net_energy_production[scenario] * 1.2, f"{scenario_net_energy_production}[2100]")
                    error_msg += self.should_be_greater(value_net_energy_production_2020, ref_net_prod_2020 * tolerance_low_ref_2020, f"{scenario_net_energy_production}[2020]")
                    error_msg += self.should_be_lower(value_net_energy_production_2020, ref_net_prod_2020 * tolerance_high_ref_2020, f"{scenario_net_energy_production}[2020]")


            for scenario_population in all_populations:
                if scenario in scenario_population:
                    population_df = dm.get_value(scenario_population)
                    value_population = population_df.loc[population_df['years'] == 2100][
                        GlossaryCore.PopulationValue].values[0]
                    value_population_2020 = population_df.loc[population_df['years'] == 2020][
                        GlossaryCore.PopulationValue].values[0]
                    error_msg += self.should_be_greater(value_population, ref_value_populations[scenario] * 0.8, f"{scenario_population}[2100]")
                    error_msg += self.should_be_lower(value_population, ref_value_populations[scenario] * 1.2, f"{scenario_population}[2100]")
                    error_msg += self.should_be_greater(value_population_2020, ref_population_2020 * tolerance_low_ref_2020, f"{scenario_population}[2020]")
                    error_msg += self.should_be_lower(value_population_2020, ref_population_2020 * tolerance_high_ref_2020, f"{scenario_population}[2020]")

            for scenario_energy_investment_without_tax in all_energy_investment_without_tax:
                if scenario in scenario_energy_investment_without_tax:
                    energy_investment_without_tax_df = dm.get_value(scenario_energy_investment_without_tax)
                    value_energy_investment_without_tax = \
                    energy_investment_without_tax_df.loc[energy_investment_without_tax_df['years'] == 2100][
                        GlossaryCore.EnergyInvestmentsWoTaxValue].values[0]
                    error_msg += self.should_be_greater(value_energy_investment_without_tax, ref_value_energy_investment_without_tax[scenario] * 0.8, scenario_energy_investment_without_tax)
                    error_msg += self.should_be_lower(value_energy_investment_without_tax, ref_value_energy_investment_without_tax[scenario] * 1.2, scenario_energy_investment_without_tax)

        if error_msg:
            raise AssertionError(f'Check range de valeurs invalide pour {self.study_name}:\n {error_msg}')


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()