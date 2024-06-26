"""
Copyright 2022 Airbus SAS
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

from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import arange
from pandas import read_csv
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

from climateeconomics.glossarycore import GlossaryCore

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
OBJECTIVE = FunctionManagerDisc.OBJECTIVE
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM


class Study(StudyManager):

    def __init__(
        self,
        year_start=GlossaryCore.YearStartDefault,
        year_end=GlossaryCore.YearEndDefault,
        time_step=1,
        execution_engine=None,
    ):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = "usecase"
        self.landuse_name = ".Population"
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []
        nb_per = round((self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)
        global_data_dir = join(Path(__file__).parents[4], "data")
        # private values economics operator pyworld3
        population_input = {}
        population_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        population_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        population_input[f"{self.study_name}.{GlossaryCore.TimeStep}"] = self.time_step

        gdp_year_start = 130.187
        gdp_serie = []
        gdp_serie.append(gdp_year_start)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)

        economics_df_y = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df_all = read_csv(join(global_data_dir, "temperature_data_onestep.csv"))

        population_input[f"{self.study_name}.{GlossaryCore.EconomicsDfValue}"] = economics_df_y
        population_input[f"{self.study_name}.{GlossaryCore.TemperatureDfValue}"] = temperature_df_all

        setup_data_list.append(population_input)

        return setup_data_list


if "__main__" == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(disc)
        graph_list = ppf.get_post_processing_by_discipline(disc, filters, as_json=False)

        for graph in graph_list:
            graph.to_plotly().show()
