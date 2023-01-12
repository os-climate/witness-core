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
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.base_functions.specific_check import specific_check_years


class ClimateEconomicsStudyManager(StudyManager):
    '''
    Class that overloads study manager to define a specific check for climate economics usecases
    '''

    def specific_check(self):
        """
        Check that the column years of the input dataframes are in [year_start, year_end]
        """
        specific_check_years(self.execution_engine.dm)

    def setup_process(self):
        self.execution_engine.root_builder_ist.setup_process()
