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
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import arange, asarray
from pandas import DataFrame

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import \
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import Study as datacase_agriculture_mix
from climateeconomics.sos_processes.iam.witness.land_use_v2_process.usecase import Study as datacase_landuse
from climateeconomics.sos_processes.iam.witness.resources_process.usecase import Study as datacase_resource
from climateeconomics.sos_processes.iam.witness.sectorization_process.usecase import Study as usecase_sectorization
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_DELTA = FunctionManager.AGGR_TYPE_DELTA
AGGR_TYPE_LIN_TO_QUAD = FunctionManager.AGGR_TYPE_LIN_TO_QUAD


class DataStudy():
    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1,
                 agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT):
        self.study_name = 'default_name'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.techno_dict = agri_techno_list
        self.study_name_wo_extra_name = self.study_name
        self.dspace = {}
        self.dspace['dspace_size'] = 0

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []
        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)

        forest_invest = np.linspace(5.0, 8.0, len(years))
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        # private values economics operator pyworld3
        witness_input = {}
        witness_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        witness_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        witness_input[f"{self.study_name}.{GlossaryCore.TimeStep}"] = self.time_step

        witness_input[f"{self.study_name}.{'Damage'}.{'tipping_point'}"] = True
        witness_input[f"{self.study_name}.{'Macroeconomics'}.{GlossaryCore.DamageToProductivity}"] = True
        witness_input[f"{self.study_name}.{GlossaryCore.FractionDamageToProductivityValue}"] = 0.30
        witness_input[f"{self.study_name}.{'init_rate_time_pref'}"] = .015
        witness_input[f"{self.study_name}.{'conso_elasticity'}"] = 1.45
        witness_input[f"{self.study_name}.{GlossaryCore.InitialGrossOutput['var_name']}"] = 130.187
        # Relax constraint for 15 first years
        witness_input[f"{self.study_name}.{'Damage.damage_constraint_factor'}"] = np.concatenate(
            (np.linspace(1.0, 1.0, 20), np.asarray([1] * (len(years) - 20))))
        #         witness_input[f"{self.study_name}.{}#                      '.Damage.damage_constraint_factor'}" = np.asarray([1] * len(years))
        witness_input[f"{self.study_name}.{'InvestmentDistribution'}.forest_investment"] = self.forest_invest_df
        # get population from csv file
        # get file from the data folder 3 folder up.
        global_data_dir = join(Path(__file__).parents[3], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        witness_input[f"{self.study_name}.{'population_df'}"] = population_df
        working_age_population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.Population1570: 6300}, index=years)
        witness_input[f"{self.study_name}.{GlossaryCore.WorkingAgePopulationDfValue}"] = working_age_population_df

        energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: asarray([1.65] * nb_per)},
            index=years)

        share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: asarray([27. - 1.65] * nb_per)},
            index=years)

        witness_input[f'{self.study_name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}'] = energy_investment_wo_tax
        witness_input[f'{self.study_name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}'] = share_non_energy_investment

        data = arange(1.0, nb_per + 1.0, 1)

        df_eco = DataFrame({GlossaryCore.Years: years,
                            GlossaryCore.GrossOutput: data,
                            GlossaryCore.PerCapitaConsumption: data,
                            GlossaryCore.OutputNetOfDamage: data},
                           index=arange(self.year_start, self.year_end + 1, self.time_step))

        witness_input[f"{self.study_name}.{GlossaryCore.EconomicsDfValue}"] = df_eco

        nrj_invest = arange(1000, nb_per + 1000, 1)

        df_energy_investment = DataFrame({GlossaryCore.Years: years,
                                          GlossaryCore.EnergyInvestmentsValue: nrj_invest},
                                         index=arange(self.year_start, self.year_end + 1, self.time_step))
        df_energy_investment_before_year_start = DataFrame({'past_years': [2017, 2018, 2019],
                                                            'energy_investment_before_year_start': [1924, 1927, 1935]},
                                                           index=[2017, 2018, 2019])

        witness_input[f"{self.study_name}.{'agri_capital_techno_list'}"] = []

        CO2_emitted_land = pd.DataFrame()
        # GtCO2
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest)
        CO2_emitted_land['Crop'] = np.zeros(len(years))
        CO2_emitted_land['Forest'] = cum_emission

        witness_input[f"{self.study_name}.{'CO2_land_emissions'}"] = CO2_emitted_land

        self.CO2_tax = np.asarray([50.] * len(years))

        witness_input[f"{self.study_name}.{GlossaryCore.EnergyInvestmentsValue}"] = df_energy_investment
        witness_input[
            f"{self.study_name}.{'energy_investment_before_year_start'}"] = df_energy_investment_before_year_start

        intermediate_point = 30
        # CO2 taxes related inputs
        CO2_tax_efficiency = np.concatenate(
            (np.linspace(30, intermediate_point, 15), np.asarray([intermediate_point] * (len(years) - 15))))
        # CO2_tax_efficiency = 30.0
        default_co2_efficiency = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.CO2TaxEfficiencyValue: CO2_tax_efficiency})

        forest_invest = np.linspace(5.0, 8.0, len(years))
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        # -- load data from resource
        dc_resource = datacase_resource(
            self.year_start, self.year_end, main_study=False)
        dc_resource.study_name = self.study_name

        # -- load data from land use
        dc_landuse = datacase_landuse(
            self.year_start, self.year_end, self.time_step, name='.Land_Use_V2', extra_name='.EnergyMix')
        dc_landuse.study_name = self.study_name

        # -- load data from agriculture
        dc_agriculture_mix = datacase_agriculture_mix(
            self.year_start, self.year_end, self.time_step, agri_techno_list=self.techno_dict)
        dc_agriculture_mix.additional_ns = '.InvestmentDistribution'
        dc_agriculture_mix.study_name = self.study_name

        # -- load data from sectorization process
        uc_sectorization = usecase_sectorization()
        uc_sectorization.study_name = self.study_name
        data_sect = uc_sectorization.setup_usecase()
        setup_data_list = setup_data_list + data_sect

        resource_input_list = dc_resource.setup_usecase()
        setup_data_list = setup_data_list + resource_input_list

        land_use_list = dc_landuse.setup_usecase()
        setup_data_list = setup_data_list + land_use_list

        agriculture_list = dc_agriculture_mix.setup_usecase()
        setup_data_list = setup_data_list + agriculture_list
        self.dspace_size = dc_agriculture_mix.dspace.pop('dspace_size')
        self.dspace.update(dc_agriculture_mix.dspace)
        nb_poles = 8

        # WITNESS
        # setup objectives
        energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: asarray([10.] * nb_per)},
            index=years)

        share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: asarray([27. - 1.65] * nb_per)},
            index=years)

        witness_input[f'{self.study_name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}'] = energy_investment_wo_tax
        witness_input[f'{self.study_name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}'] = share_non_energy_investment
        witness_input[f'{self.study_name}.Macroeconomics.{GlossaryCore.CO2TaxEfficiencyValue}'] = default_co2_efficiency

        witness_input[f'{self.study_name}.beta'] = 1.0
        witness_input[f'{self.study_name}.gamma'] = 0.5
        witness_input[f'{self.study_name}.init_discounted_utility'] = 4000.0

        witness_input[f'{self.study_name}.init_rate_time_pref'] = 0.0

        witness_input[f'{self.study_name}.temperature_change_ref'] = 1.0

        # 

        GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.TotalCO2Emissions: np.linspace(37., 10., len(years)),
                                                   GlossaryCore.TotalN2OEmissions: np.linspace(1.7e-3, 5.e-4,
                                                                                               len(years)),
                                                   GlossaryCore.TotalCH4Emissions: np.linspace(0.17, 0.01, len(years))})
        witness_input[f'{self.study_name}.GHG_total_energy_emissions'] = GHG_total_energy_emissions
        setup_data_list.append(witness_input)

        return setup_data_list

    def setup_objectives(self):
        data = {
            'variable': [
                GlossaryCore.NegativeWelfareObjective,
                GlossaryCore.LastYearDiscountedUtilityObjective,
                'gwp100_objective',
                'non_use_capital_objective'
            ],
            'parent': [
                'utility_objective',
                'utility_objective',
                'GWP_long_term_obj',
                'non_use_capital_objective'
            ],
            'ftype': [OBJECTIVE, OBJECTIVE, OBJECTIVE, OBJECTIVE],
            'weight': [1.0, 1.0, 0.0, 0.0],
            AGGR_TYPE: [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM],
            'namespace': [GlossaryCore.NS_FUNCTIONS, GlossaryCore.NS_FUNCTIONS, GlossaryCore.NS_FUNCTIONS,
                          GlossaryCore.NS_WITNESS]
        }

        func_df = DataFrame(data)

        return func_df

    def setup_constraints(self):
        data = [
            {
                'variable': 'rockstrom_limit_constraint',
                'parent': 'CO2 ppm',
                'ftype': INEQ_CONSTRAINT,
                'weight': 0.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
            {
                'variable': 'minimum_ppm_constraint',
                'parent': 'CO2 ppm',
                'ftype': INEQ_CONSTRAINT,
                'weight': -1.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
            {
                'variable': 'calories_per_day_constraint',
                'parent': 'agriculture_constraints',
                'ftype': INEQ_CONSTRAINT,
                'weight': -1.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
            {
                'variable': GlossaryCore.ConstraintLowerBoundUsableCapital,
                'parent': 'invests_constraints',
                'ftype': INEQ_CONSTRAINT,
                'weight': -1.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
            {
                'variable': 'non_use_capital_cons',
                'parent': 'invests_constraints',
                'ftype': INEQ_CONSTRAINT,
                'weight': -1.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
            {
                'variable': 'forest_lost_capital_cons',
                'parent': 'agriculture_constraint',
                'ftype': INEQ_CONSTRAINT,
                'weight': -1.0,
                AGGR_TYPE: AGGR_TYPE_SMAX,
                'namespace': GlossaryCore.NS_FUNCTIONS,
            },
        ]

        # Append the data to the DataFrame
        df_list = [pd.DataFrame(df) for df in data]
        func_df = pd.concat(df_list, ignore_index=True)

        return func_df
