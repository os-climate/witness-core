'''
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


from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import DataStudy as datacase_witness
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as usecase_witness

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager

import cProfile
from io import StringIO
import pstats



class Study(ClimateEconomicsStudyManager):

    def __init__(self, run_usecase = False, execution_engine=None):
        super().__init__(__file__,  run_usecase = run_usecase, execution_engine=execution_engine)


    def setup_usecase(self):
        

        witness_uc = usecase_witness()
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()
        # Create a dictionary with a key-value pair indicating that damage activation should be False for this study
        updated_data = {f'{self.study_name}.Damage.activate_damage': False}
        data_witness.append(updated_data)
        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
