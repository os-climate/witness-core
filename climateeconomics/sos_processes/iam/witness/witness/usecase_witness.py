'''
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
'''

import cProfile
import pstats
from io import StringIO
from os.path import join, dirname

import pandas as pd
from pandas import DataFrame, concat

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import \
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import \
    DataStudy as datacase_witness
from climateeconomics.sos_processes.iam.witness_wo_energy_dev.datacase_witness_wo_energy import \
    DataStudy as datacase_witness_dev
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import Study as datacase_energy
from energy_models.tools.jsonhandling import convert_to_editable_json, preprocess_data_and_save_json
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX

def generate_json_by_discipline(data, json_name):
    json_data = convert_to_editable_json(data)
    json_data_updt = create_fake_regions(json_data, ['US', 'UE'])
    json_data_updt['id'] = json_name.split('.')[-1]
    output_path = join(dirname(__file__), 'data', f'{json_name}.json')
    preprocess_data_and_save_json(json_data_updt, output_path)


def create_fake_regions(data, regions_list): 
    """
    Add regions 
    """
    data_updt = {}
    for reg in regions_list:
        data_updt[reg] = data 
    return data_updt


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, bspline=True, run_usecase=False,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[
                     2], techno_dict=DEFAULT_TECHNO_DICT, agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 process_level='val'):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.agri_techno_list = agri_techno_list
        self.process_level = process_level
        self.dc_energy = datacase_energy(
            self.year_start, self.year_end, self.time_step, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict)
        self.sub_study_path_dict = self.dc_energy.sub_study_path_dict

    def setup_constraint_land_use(self):

        func_df = pd.DataFrame({
            'variable': ['land_demand_constraint'],
            'parent': ['agriculture_constraint'],
            'ftype': [INEQ_CONSTRAINT],
            'weight': [-1.0],
            AGGR_TYPE: [AGGR_TYPE_SUM],
            'namespace': ['ns_functions']
        })

        return func_df

    def setup_usecase(self):
        setup_data_list = []

        # -- load data from energy pyworld3
        # -- Start with energy to have it at first position in the list...

        self.dc_energy.study_name = self.study_name
        self.energy_mda_usecase = self.dc_energy
        # -- load data from witness
        if self.process_level == 'val':
            dc_witness = datacase_witness(
                self.year_start, self.year_end, self.time_step)
        else:
            dc_witness = datacase_witness_dev(
                self.year_start, self.year_end, self.time_step, agri_techno_list=self.agri_techno_list)

        dc_witness.study_name = self.study_name
        witness_input_list = dc_witness.setup_usecase()
        setup_data_list = setup_data_list + witness_input_list

        energy_input_list = self.dc_energy.setup_usecase()
        setup_data_list = setup_data_list + energy_input_list

        dspace_energy = self.dc_energy.dspace

        self.merge_design_spaces([dspace_energy, dc_witness.dspace])

        # constraint land use
        land_use_df_constraint = self.setup_constraint_land_use()

        # WITNESS
        # setup objectives
        self.func_df = concat(
            [dc_witness.setup_objectives(), dc_witness.setup_constraints(), self.dc_energy.setup_constraints(),
             self.dc_energy.setup_objectives(), land_use_df_constraint])

        self.energy_list = self.dc_energy.energy_list
        self.ccs_list = self.dc_energy.ccs_list
        self.dict_technos = self.dc_energy.dict_technos

        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 2,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.sub_mda_class': 'GSPureNewtonMDA',
            f'{self.study_name}.cache_type': 'SimpleCache', }
        
        # f'{self.study_name}.gauss_seidel_execution': True}

        setup_data_list.append(numerical_values_dict)

        return setup_data_list

    def run(self, logger_level=None,
            dump_study=False,
            for_test=False):

        profil = cProfile.Profile()
        profil.enable()
        ClimateEconomicsStudyManager.run(
            self, logger_level=logger_level, dump_study=dump_study, for_test=for_test)
        profil.disable()

        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(500)
        result = result.getvalue()
        print(result)


if '__main__' == __name__:

#     uc_cls = Study(run_usecase=True)
#     uc_cls.load_data()
#
#
#     data = uc_cls.ee.dm.get_discipline(list(uc_cls.ee.dm.disciplines_dict.keys())[3]).get_data_in()
#     dm = uc_cls.ee.dm
#
#     #dict_data = prepare_data(dm)
#     #for k,v in dict_data.items():
#     #    generate_json_by_discipline(v, k)
#     connection_string = ''
#     database_name = 'regionalization'
#     container_name = 'regionalizationv0'
#     json_path = join(dirname(__file__), 'data', 'AgricultureMix.Crop.json')
#     import os
#     """    for f in os.listdir(join(dirname(__file__), 'data')):
#         insert_json_to_mongodb_bis(join(dirname(__file__), 'data' ,f), container_name, database_name, connection_string)
#     """
#
#
#     uc_cls.run()
#
#
#     ppf = PostProcessingFactory()
#     filters = ppf.get_post_processing_filters_by_namespace(
#         uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing')
#     graph_list = ppf.get_post_processing_by_namespace(uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing',
#                                                       filters, as_json=False)
#
# #    for graph in graph_list:
# #        graph.to_plotly().show()

    # TODO : careful, this usecase is quite long to test ! 800 sec
    uc_cls = Study()
    uc_cls.test()

