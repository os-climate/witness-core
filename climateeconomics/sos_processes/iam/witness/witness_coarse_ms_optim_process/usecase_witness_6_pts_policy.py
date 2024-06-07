'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023 Copyright 2023 Capgemini

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

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_optim_process.usecase_witness_optim_invest_distrib import (
    Study as witness_optim_usecase,
)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):
        witness_ms_usecase = witness_optim_usecase(
            execution_engine=self.execution_engine)

        self.scatter_scenario = 'optimization scenarios'
        # Set public values at a specific namespace
        witness_ms_usecase.study_name = f'{self.study_name}.{self.scatter_scenario}'

        values_dict = {}
        scenario_list = []
        alpha_list = np.linspace(0, 125, 6, endpoint=True)
        for alpha_i in alpha_list:
            scenario_i = f'scenario_policy={alpha_i}%'
            scenario_i = scenario_i.replace('.', ',')
            scenario_list.append(scenario_i)
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.co2_damage_price_percentage'] = alpha_i
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.ccs_price_percentage'] = alpha_i

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = 6

        len_scenarios = len(scenario_list)
        scenario_df = pd.DataFrame({'selected_scenario': [True] * len_scenarios ,'scenario_name': scenario_list})

        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df

        for scenario in scenario_list:
            scenarioUseCase = witness_optim_usecase(
                bspline=self.bspline, execution_engine=self.execution_engine)
            scenarioUseCase.optim_name = f'{scenario}.{scenarioUseCase.optim_name}'
            scenarioUseCase.study_name = witness_ms_usecase.study_name
            scenarioData = scenarioUseCase.setup_usecase()
            default_func_df = scenarioUseCase.func_df
            # no CO2 obejctive in this formulation
            default_func_df.loc[default_func_df['variable'] == 'CO2_objective', 'weight'] = 0.0

            for dict_data in scenarioData:
                values_dict.update(dict_data)
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario}.{witness_ms_usecase.optim_name}' \
                        f'.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.FunctionsManager.function_df'] = default_func_df
            values_dict[
                    f'{self.study_name}.{self.scatter_scenario}.{scenario}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.alpha'] = 1.
        year_start = scenarioUseCase.year_start
        year_end = scenarioUseCase.year_end
        years = np.arange(year_start, year_end + 1)

        values_dict[f'{self.study_name}.{self.scatter_scenario}.NormalizationReferences.liquid_hydrogen_percentage'] = np.concatenate((np.ones(5)/1e-4,np.ones(len(years)-5)/4), axis=None)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()


#    for graph in graph_list:
#        graph.to_plotly().show()
