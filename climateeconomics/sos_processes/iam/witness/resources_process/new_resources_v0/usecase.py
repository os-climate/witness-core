"""
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
"""

import random as rd

import numpy as np
import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

from climateeconomics.glossarycore import GlossaryCore


class Study(StudyManager):

    def __init__(self):
        super().__init__(__file__)

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []

        year, year_end = GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault

        copper_demand = pd.DataFrame(columns=["Year", "Demand"])

        period_of_exploitation = np.arange(year, year_end + 1, 1)

        copper_demand["Year"] = period_of_exploitation
        copper_demand.index = copper_demand["Year"].values
        copper_demand["Demand"] = [0] * len(period_of_exploitation)
        annual_extraction = []

        while year < year_end + 1:
            copper_demand.at[year, "Demand"] = rd.gauss(26, 0.5) * 1.056467 ** (year - 2020)
            annual_extraction += [26 * 1.056467 ** (year - 2020)]
            year += 1

        config_data = {
            f"{self.study_name}.CopperModel.copper_demand": copper_demand,
            f"{self.study_name}.CopperModel.annual_extraction": annual_extraction,
        }
        setup_data_list.append(config_data)
        return setup_data_list


if "__main__" == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(disc)
        graph_list = ppf.get_post_processing_by_discipline(disc, filters, as_json=False)

        for graph in graph_list:
            graph.to_plotly().show()
