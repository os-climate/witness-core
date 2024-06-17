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
import os.path

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda_four_scenarios_tp35 import Study as StudyMSmdaTippingPoint35
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda_four_scenarios import Study as StudyToExport, __file__ as module_file
from sostrades_core.datasets.dataset_mapping import DatasetsMapping

class Study(StudyMSmdaTippingPoint35):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(file_path=__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.check_outputs = False

    def setup_usecase(self, study_folder_path=None):
        return {}

if '__main__' == __name__:
    uc_export = StudyToExport()
    uc_export.load_data()
    uc_cls = Study(run_usecase=True)

    # # build the datasets mapping TODO: remove when wildcard exists
    # ns_values = set()
    # for data in uc_export.execution_engine.dm.data_dict.values():
    #     ns_values.add(data['ns_reference'].value)
    # dmap_list = []
    # for i, ns in enumerate(sorted(ns_values)):
    #     dmap_list.append(f"\"v0|{ns.replace('usecase_witness_ms_mda_four_scenarios','<study_ph>')}|*\": "
    #                      f"[\"MVP0_local_datasets_connector|dataset_{i}|*\"]")
    # print(",\n".join(dmap_list))

    datasets_mapping_path = os.path.realpath(os.path.join(os.path.dirname(module_file),
                                                          "four_scenarios_local_datasets_mapping.json"))
    datasets_mapping = DatasetsMapping.from_json_file(datasets_mapping_path)
    uc_export.execution_engine.dm.export_data_in_datasets(datasets_mapping)
    uc_cls.load_data(from_datasets_mapping=datasets_mapping)
    # uc_cls.run()
