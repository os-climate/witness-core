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
import pandas as pd

from energy_models.database_witness_energy import DatabaseWitnessEnergy
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
            f'{self.study_name}.max_mda_iter': 400,
            f'{self.study_name}.tolerance': 1.0e-12,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.inner_mda_name': 'MDAGaussSeidel',
            f'{self.study_name}.cache_type': 'SimpleCache'}

        setup_data.update(numerical_values_dict)



        new_invests = {}
        import numpy as np
        techno_indispo = []
        years_try = np.flip(list(range(2018, self.year_start + 1)))
        for year in years_try:
            techno_indispo = []
            for energy, energy_technos in self.dict_technos.items():
                for techno in energy_technos:
                    key = f"{energy}.{techno}"
                    if key not in new_invests:
                        try:
                            new_invests[key] = DatabaseWitnessEnergy.get_techno_invest(techno_name=techno, year=year)
                        except:
                            techno_indispo.append(key)

        for key in techno_indispo:
            new_invests[key] = 1e-3

        for key, value in new_invests.items():
            if value < 1e-3:
                new_invests[key] = 1e-3
        invest_mix_df = pd.DataFrame({
            GlossaryEnergy.Years: np.arange(self.year_start, self.year_end + 1),
            ** new_invests
        })

        setup_data.update(
            **self.set_value_at_namespace("invest_mix", invest_mix_df, GlossaryEnergy.invest_mix_df['namespace'])
        )
        """
        """
        return setup_data


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
