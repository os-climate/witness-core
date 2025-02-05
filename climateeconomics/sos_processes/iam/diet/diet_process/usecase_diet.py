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
from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import (
    Study as datacase_agriculture_mix,
)
from climateeconomics.sos_processes.iam.witness.land_use_v2_process.usecase import (
    Study as datacase_landuse,
)
from climateeconomics.sos_processes.iam.witness.resources_process.usecase import (
    Study as datacase_resource,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_DELTA = FunctionManager.AGGR_TYPE_DELTA
AGGR_TYPE_LIN_TO_QUAD = FunctionManager.AGGR_TYPE_LIN_TO_QUAD

class Study(ClimateEconomicsStudyManager):
    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

        #self.study_name = 'default_name'
        self.year_start = year_start
        self.year_end = year_end
        self.study_name_wo_extra_name = self.study_name
        self.dspace = {}
        self.dspace['dspace_size'] = 0

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []
        nb_per = self.year_end - self.year_start + 1
        years = arange(self.year_start, self.year_end + 1)

        reforestation_invest = np.linspace(5.0, 8.0, len(years))
        self.reforestation_investment_df = pd.DataFrame(
            {GlossaryCore.Years: years, "reforestation_investment": reforestation_invest})

        # private values economics operator model
        witness_input = {}
        witness_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        witness_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        

        # Relax constraint for 15 first years
        witness_input[f"{self.study_name}.{'InvestmentDistribution'}.reforestation_investment"] = self.reforestation_investment_df
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

        gdp = [130.187]*len(years)

        df_eco = DataFrame({GlossaryCore.Years: years,
                            GlossaryCore.OutputNetOfDamage: gdp,
                            GlossaryCore.GrossOutput: gdp,
                            GlossaryCore.PerCapitaConsumption: 0.,
                            },
                           index=arange(self.year_start, self.year_end + 1))

        witness_input[f"{self.study_name}.{GlossaryCore.EconomicsDfValue}"] = df_eco

        nrj_invest = arange(1000, nb_per + 1000, 1)

        CO2_emitted_land = pd.DataFrame()
        # GtCO2
        emission_forest = np.linspace(0.04, 0.04, len(years))
        cum_emission = np.cumsum(emission_forest)
        CO2_emitted_land['Crop'] = np.zeros(len(years))
        CO2_emitted_land[GlossaryCore.Forestry] = cum_emission

        witness_input[f"{self.study_name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}"] = CO2_emitted_land

        self.CO2_tax = np.asarray([50.] * len(years))


        reforestation_invest = np.linspace(5.0, 8.0, len(years))
        self.reforestation_investment_df = pd.DataFrame(
            {GlossaryCore.Years: years, "reforestation_investment": reforestation_invest})
        intermediate_point = 30
        # CO2 taxes related inputs
        CO2_tax_efficiency = np.concatenate(
            (np.linspace(30, intermediate_point, 15), np.asarray([intermediate_point] * (len(years) - 15))))
        # CO2_tax_efficiency = 30.0
        # -- load data from resource
        dc_resource = datacase_resource(self.year_start, self.year_end, main_study=False)
        dc_resource.study_name = self.study_name

        # -- load data from land use
        dc_landuse = datacase_landuse(
            self.year_start, self.year_end, name='.Land_Use_V2', extra_name='.EnergyMix')
        dc_landuse.study_name = self.study_name

        # -- load data from agriculture
        dc_agriculture_mix = datacase_agriculture_mix(
            self.year_start, self.year_end)
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
        # WITNESS
        # setup objectives
        self.share_energy_investment_array = asarray([1.65] * len(years))


        setup_data_list.append(witness_input)

        self.func_df = self.setup_objectives()

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
        list_ns.extend([GlossaryCore.NS_FUNCTIONS, GlossaryCore.NS_FUNCTIONS])
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


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
