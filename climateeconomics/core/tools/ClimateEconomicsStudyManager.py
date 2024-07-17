'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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


class ClimateEconomicsStudyManager(StudyManager):
    '''
    Class that overloads study manager to define a specific check for climate economics usecases
    '''
    def should_be_lower(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value > ref_value:
            msg = f"{varname>140} should be lower than {ref_value} but is not. Value = {actual_value}"
        return msg

    def should_be_greater(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value < ref_value:
            msg = f"\n{varname:>140} should be greater than {ref_value} but is not. Value = {actual_value}"
        return msg


