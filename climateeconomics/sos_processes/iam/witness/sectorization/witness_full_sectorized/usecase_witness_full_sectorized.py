'''
Copyright 2023 Capgemini

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
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import (
    Study as datacase_energy,
)

from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization.usecase_witness_coarse_sectorization import (
    Study as StudyCoarse,
)


class Study(StudyCoarse):

    def __init__(self):
        super().__init__(file_path=__file__, techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT)

    def setup_process(self):
        datacase_energy.setup_process(self)


    def setup_usecase(self, study_folder_path=None):
        setup_data = super().setup_usecase()
        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 100,
            f'{self.study_name}.tolerance': 1.0e-12,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.inner_mda_name': 'MDAGaussSeidel',
            f'{self.study_name}.cache_type': 'SimpleCache'}

        setup_data.update(numerical_values_dict)



        return setup_data


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes(True)
    #uc_cls.run()
