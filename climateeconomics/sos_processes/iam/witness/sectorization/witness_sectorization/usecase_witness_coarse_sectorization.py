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
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import (
    Study as datacase_energy,
)
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sect_wo_energy.datacase_witness_wo_energy import (
    DataStudy as datacase_witness,
)

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
DEFAULT_COARSE_TECHNO_DICT = {GlossaryCore.clean_energy: {'type': 'energy', 'value': [GlossaryCore.CleanEnergySimpleTechno]},
                              GlossaryEnergy.fossil: {'type': 'energy', 'value': [GlossaryEnergy.FossilSimpleTechno]},
                              GlossaryEnergy.carbon_capture: {'type': 'CCUS', 'value': [f'{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}',
                                                                           f'{GlossaryEnergy.flue_gas_capture}.{GlossaryEnergy.FlueGasTechno}']},
                              GlossaryEnergy.carbon_storage: {'type': 'CCUS', 'value': [GlossaryEnergy.CarbonStorageTechno]}}
DEFAULT_ENERGY_LIST = [key for key, value in DEFAULT_COARSE_TECHNO_DICT.items(
) if value['type'] == 'energy']
DEFAULT_CCS_LIST = [key for key, value in DEFAULT_COARSE_TECHNO_DICT.items(
) if value['type'] == 'CCUS']


class Study(StudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, bspline=True, run_usecase=True,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.energy_list = DEFAULT_ENERGY_LIST
        self.ccs_list = DEFAULT_CCS_LIST
        self.dc_energy = datacase_energy(
            self.year_start, self.year_end, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict, main_study=False)
        self.sub_study_path_dict = self.dc_energy.sub_study_path_dict

    def setup_process(self):
        datacase_energy.setup_process(self)


    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []

        # -- load data from energy pyworld3
        # -- Start with energy to have it at first position in the list...
        self.dc_energy.study_name = self.study_name
        self.energy_mda_usecase = self.dc_energy

        # -- load data from witness
        dc_witness = datacase_witness(
            self.year_start, self.year_end)
        dc_witness.study_name = self.study_name

        witness_input_list = dc_witness.setup_usecase()
        setup_data_list = setup_data_list + witness_input_list

        energy_input_list = self.dc_energy.setup_usecase()
        setup_data_list = setup_data_list + energy_input_list

        self.dict_technos = self.dc_energy.dict_technos
        dspace_energy = self.dc_energy.dspace

        self.merge_design_spaces([dspace_energy, dc_witness.dspace])


        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 50,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.inner_mda_name': 'MDAGSNewton',
            f'{self.study_name}.cache_type': 'SimpleCache'}

        setup_data_list.append(numerical_values_dict)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
    #             for graph in graph_list:
    #                 graph.to_plotly().show()
