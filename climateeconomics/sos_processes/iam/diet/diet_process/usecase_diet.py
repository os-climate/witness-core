'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import Study as datacase_agriculture_mix
from climateeconomics.sos_processes.iam.witness.land_use_v2_process.usecase import Study as datacase_landuse
from climateeconomics.sos_processes.iam.witness.resources_process.usecase import Study as datacase_resource
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

class Study(ClimateEconomicsStudyManager):
    def __init__(self, year_start=2020, year_end=2100, time_step=1, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

        #self.study_name = 'default_name'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.study_name_wo_extra_name = self.study_name
        self.dspace = {}
        self.dspace['dspace_size'] = 0

    def setup_usecase(self):
        setup_data_list = []
        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)

        forest_invest = np.linspace(5.0, 8.0, len(years))
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        # private values economics operator model
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
        #         witness_input[f"{self.study_name}.{}                       '.Damage.damage_constraint_factor'}" = np.asarray([1] * len(years))
        witness_input[f"{self.study_name}.{'InvestmentDistribution'}.forest_investment"] = self.forest_invest_df
        # get population from csv file
        # get file from the data folder 3 folder up.
        global_data_dir = join(Path(__file__).parents[4], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = years
        witness_input[f"{self.study_name}.{GlossaryCore.PopulationDfValue}"] = population_df
        working_age_population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.Population1570: 6300}, index=years)
        witness_input[f"{self.study_name}.{GlossaryCore.WorkingAgePopulationDfValue}"] = working_age_population_df

        self.share_energy_investment_array = asarray([1.65] * nb_per)

        total_invest = asarray([27.0] * nb_per)
        total_invest = DataFrame(
            {GlossaryCore.Years: years, 'share_investment': total_invest})
        witness_input[f"{self.study_name}.{GlossaryCore.InvestmentShareGDPValue}"] = total_invest
        share_energy_investment = DataFrame(
            {GlossaryCore.Years: years, 'share_investment': self.share_energy_investment_array}, index=years)
        witness_input[f"{self.study_name}.{'share_energy_investment'}"] = share_energy_investment
        gdp = [130.187]*len(years)

        df_eco = DataFrame({GlossaryCore.Years: years,
                            GlossaryCore.OutputNetOfDamage: gdp},
                           index=arange(self.year_start, self.year_end + 1, self.time_step))

        witness_input[f"{self.study_name}.{GlossaryCore.EconomicsDfValue}"] = df_eco

        nrj_invest = arange(1000, nb_per + 1000, 1)

        df_energy_investment = DataFrame({GlossaryCore.Years: years,
                                          GlossaryCore.EnergyInvestmentsValue: nrj_invest},
                                         index=arange(self.year_start, self.year_end + 1, self.time_step))
        df_energy_investment_before_year_start = DataFrame({'past_years': [2017, 2018, 2019],
                                                            'energy_investment_before_year_start': [1924, 1927,
                                                                                                    1935]},
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
        witness_input[f"{self.study_name}.{'energy_investment_before_year_start'}"] = df_energy_investment_before_year_start

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
            self.year_start, self.year_end)
        dc_resource.study_name = self.study_name

        # -- load data from land use
        dc_landuse = datacase_landuse(
            self.year_start, self.year_end, self.time_step, name='.Land_Use_V2', extra_name='.EnergyMix')
        dc_landuse.study_name = self.study_name

        # -- load data from agriculture
        dc_agriculture_mix = datacase_agriculture_mix(
            self.year_start, self.year_end, self.time_step)
        dc_agriculture_mix.additional_ns = '.InvestmentDistribution'
        dc_agriculture_mix.study_name = self.study_name

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
        self.share_energy_investment_array = asarray([1.65] * len(years))

        share_energy_investment = DataFrame(
            {GlossaryCore.Years: years, 'share_investment': self.share_energy_investment_array}, index=years)
        witness_input[f"{self.study_name}.{'share_energy_investment'}"] = share_energy_investment
        witness_input[f'{self.study_name}.Macroeconomics.{GlossaryCore.CO2TaxEfficiencyValue}'] = default_co2_efficiency

        witness_input[f'{self.study_name}.beta'] = 1.0
        witness_input[f'{self.study_name}.gamma'] = 0.5
        witness_input[f'{self.study_name}.init_discounted_utility'] = 4000.0

        witness_input[f'{self.study_name}.init_rate_time_pref'] = 0.0
        witness_input[f'{self.study_name}.temperature_change_ref'] = 1.0
        witness_input[f'{self.study_name}.is_dev'] = True

        GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.TotalCO2Emissions: np.linspace(37., 10., len(years)),
                                                   GlossaryCore.TotalN2OEmissions: np.linspace(1.7e-3, 5.e-4, len(years)),
                                                   GlossaryCore.TotalCH4Emissions: np.linspace(0.17, 0.01, len(years))})
        witness_input[f'{self.study_name}.GHG_total_energy_emissions'] = GHG_total_energy_emissions

        setup_data_list.append(witness_input)

        self.func_df = pd.concat([self.setup_constraint_land_use(), self.setup_objectives()])

        return setup_data_list

    def setup_objectives(self):
        func_df = DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['co2_eq_100', 'co2_eq_20'])
        list_parent.extend([
                            'CO2_obj','CO2_obj'])
        list_ns.extend(['ns_functions', 'ns_functions'])
        list_ftype.extend(
            [OBJECTIVE, OBJECTIVE])
        list_weight.extend([2.0, 2.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM, AGGR_TYPE_SUM])

        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        return func_df

    def setup_constraints(self):
        func_df = pd.DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []

        """
        list_var.append('non_use_capital_cons')
        list_parent.append('invests_constraints')
        list_ns.extend(['ns_functions'])
        list_ftype.append(INEQ_CONSTRAINT)
        list_weight.append(-1.0)
        list_aggr_type.append(
            AGGR_TYPE_SMAX)

        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        """

        return func_df



    def setup_constraint_land_use(self):
        func_df = DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['land_demand_constraint', 'calories_per_day_constraint'])
        list_parent.extend(['agriculture_constraint', 'agriculture_constraint'])
        list_ftype.extend([INEQ_CONSTRAINT, INEQ_CONSTRAINT])
        list_weight.extend([-1.0, -3.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM, AGGR_TYPE_SUM])
        list_ns.extend(['ns_functions', 'ns_functions'])
        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        return func_df


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
