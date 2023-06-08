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

from sostrades_core.study_manager.study_manager import StudyManager
from climateeconomics.sos_processes.iam.witness.climate_process.usecase import Study as climate_usecase
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(StudyManager):

    def __init__(self, bspline=False, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self):
        climate_usecase_inst = climate_usecase(
            execution_engine=self.execution_engine)

        self.scatter_scenario = 'Scenarios'
        # Set public values at a specific namespace
        climate_usecase_inst.study_name = f'{self.study_name}.{self.scatter_scenario}'

        values_dict = {}
        scenario_list = []
        forcing_model_list = ['DICE', 'Myhre', 'Etminan', 'Meinshausen']
        for forcing_model in forcing_model_list:
            scenario_i = f'scenario_{forcing_model}'
            scenario_i = scenario_i.replace('.', ',')
            scenario_list.append(scenario_i)
            values_dict[
                f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.Temperature.forcing_model'] = forcing_model

        values_dict[f'{self.study_name}.scenario_list'] = scenario_list

        for scenario in scenario_list:
            scenarioUseCase = climate_usecase(
                execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{climate_usecase_inst.study_name}.{scenario}'
            scenarioData = scenarioUseCase.setup_usecase()

            for dict_data in scenarioData:
                values_dict.update(dict_data)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()

    ppf = PostProcessingFactory()
    filters = ppf.get_post_processing_filters_by_namespace(
        uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing')
    graph_list = ppf.get_post_processing_by_namespace(uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing',
                                                      filters, as_json=False)

    for graph in graph_list:
        graph.to_plotly().show()
