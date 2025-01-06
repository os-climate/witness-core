"""
Copyright 2024 Capgemini
Modifications on 2023/04/19-2024/06/24 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Union

import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.population_process.usecase import (
    Study as PopulationStudy,
)
from climateeconomics.sos_wrapping.data_comparison_disc import DataComparisonDiscipline


class Study(StudyManager):
    def __init__(
        self,
        data: Union[None, dict] = None,
        execution_engine=None,
        year_start=GlossaryCore.YearStartDefault,
        year_end=GlossaryCore.YearEndDefault,
        run_usecase=False,
    ):
        super().__init__(
            __file__, execution_engine=execution_engine, run_usecase=run_usecase
        )
        self.year_start = year_start
        self.year_end = year_end
        self.data = data

    def setup_usecase(self, study_folder_path=None):
        witness_study = PopulationStudy()
        inputs_dict_witness = witness_study.setup_usecase()

        ns_study = self.ee.study_name

        # For Comparison Discipline
        config_df = DataComparisonDiscipline.create_config_df(
            [
                {
                    "source_ns": "ns_witness",
                    "variable_name": GlossaryCore.PopulationDfValue,
                    "column_name": GlossaryCore.PopulationValue,
                    "local_var": "local_var1",
                    "local_column": "local_col1",
                    "common_column": GlossaryCore.Years,
                    "weight": 1.0,
                    "interpolation_method": "none",
                    "error_metric": "mse",
                },
                # {
                #     "source_ns": "ns_crop",
                #     "variable_name": GlossaryCore.CropFoodEmissionsName,
                #     "column_name": "CO2",
                #     "local_var": "local_var2",
                #     "local_column": "local_col1",
                #     "common_column": GlossaryCore.Years,
                #     "weight": 1.0,
                #     "interpolation_method": "none",
                #     "error_metric": "mse",
                # },
            ]
        )

        local_var1 = pd.DataFrame(
            {
                GlossaryCore.Years: [2020, 2021, 2022, 2030, 2100],
                "local_col1": [x * 1e3 for x in [6.0, 7.0, 8.0, 8.0, 8.0]],
            }
        )

        inputs_dict = inputs_dict_witness[0] | {
            f"{ns_study}.Data Comparison.config_df": config_df,
            f"{ns_study}.Data Comparison.local_var1": local_var1,
        }

        return inputs_dict


if "__main__" == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()

    ppf = PostProcessingFactory()
    all_post_processings = ppf.get_all_post_processings(
        uc_cls.execution_engine, False, as_json=False, for_test=False
    )

    graphs = [
        fig.to_plotly().show()
        for k, post_proc_list in all_post_processings.items()
        if "Comparison" in k
        for chart in post_proc_list
        for fig in chart.post_processings
    ]
