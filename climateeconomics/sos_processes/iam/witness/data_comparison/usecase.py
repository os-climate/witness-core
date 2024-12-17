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

import numpy as np
import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
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
        if self.data is not None:
            return self.data
        ns_study = self.ee.study_name
        model_name = "Crop2"
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        crop_productivity_reduction = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.CropProductivityReductionName: 1.2,  # fake
            }
        )

        damage_fraction = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.DamageFractionOutput: 0.43,  # 2020 value
            }
        )

        investments = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                **{
                    food_type: DatabaseWitnessCore.SectorAgricultureInvest2021.value
                               * GlossaryCore.crop_calibration_data[
                                   "invest_food_type_share_start"
                               ][food_type]
                               / 100.0
                               * 1000.0
                    for food_type in GlossaryCore.DefaultFoodTypesV2
                },  # convert to G$
            }
        )
        workforce_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.SectorAgriculture: 935.0,  # millions of people (2020 value)
            }
        )

        population_2021 = 7_954_448_391
        population_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.PopulationValue: np.linspace(
                    population_2021 / 1e6, 7870 * 1.2, year_range
                ),
            }
        )

        enegy_agri = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.TotalProductionValue: 2591.0 / 1000.0,  # PWh, 2020 value
            }
        )

        # For Comparison Discipline
        config_df = DataComparisonDiscipline.create_config_df(
            [
                {
                    "source_ns": "ns_crop",
                    "variable_name": GlossaryCore.FoodTypeProductionName,
                    "column_name": "red meat",
                    "local_var": "local_var1",
                    "local_column": "local_col1",
                    "common_column": GlossaryCore.Years,
                    "weight": 1.0,
                    "interpolation_method": "none",
                    "error_metric": "mse"
                },
                {
                    "source_ns": "ns_crop",
                    "variable_name": GlossaryCore.CropFoodEmissionsName,
                    "column_name": "CO2",
                    "local_var": "local_var2",
                    "local_column": "local_col1",
                    "common_column": GlossaryCore.Years,
                    "weight": 1.0,
                    "interpolation_method": "none",
                    "error_metric": "mse"
                },
            ]
        )

        local_var1 = pd.DataFrame(
            {GlossaryCore.Years: [2020, 2021, 2022, 2030], "local_col1": [105.0, 106.0, 107.0, 108.0]})

        local_var2 = pd.DataFrame(
            {GlossaryCore.Years: [2020, 2021, 2022, 2030], "local_col1": [1.8, 1.7, 1.9, 1.8]})

        inputs_dict = {
            f"{ns_study}.{GlossaryCore.YearStart}": self.year_start,
            f"{ns_study}.{GlossaryCore.YearEnd}": self.year_end,
            f"{ns_study}.{GlossaryCore.CropProductivityReductionName}": crop_productivity_reduction,
            f"{ns_study}.{GlossaryCore.WorkforceDfValue}": workforce_df,
            f"{ns_study}.{GlossaryCore.PopulationDfValue}": population_df,
            f"{ns_study}.{GlossaryCore.DamageFractionDfValue}": damage_fraction,
            f"{ns_study}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}": enegy_agri,
            f"{ns_study}.{model_name}.{GlossaryCore.FoodTypesInvestName}": investments,
            f"{ns_study}.Data Comparison.config_df": config_df,
            f"{ns_study}.Data Comparison.local_var1": local_var1,
            f"{ns_study}.Data Comparison.local_var2": local_var2,
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
