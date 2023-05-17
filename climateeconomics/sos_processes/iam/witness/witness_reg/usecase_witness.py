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

from os.path import join, dirname
from pandas import DataFrame, concat
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from sostrades_core.study_manager.study_manager import StudyManager
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import \
    DataStudy as datacase_witness
from climateeconomics.sos_processes.iam.witness_wo_energy_dev.datacase_witness_wo_energy import \
    DataStudy as datacase_witness_dev
from climateeconomics.sos_processes.iam.witness_wo_energy_thesis.datacase_witness_wo_energy_solow import \
    DataStudy as datacase_witness_thesis
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import Study as datacase_energy

from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import \
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.tools.jsonhandling import convert_to_editable_json, preprocess_data_and_save_json, insert_json_to_mongodb_bis

import cProfile
from io import StringIO
import pstats

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


def prepare_data(data):
    """
    Prepare data by getting only values
    """

    dict_data = {}
    for disc_id in list(uc_cls.ee.dm.disciplines_dict.keys()):
        disc = dm.get_discipline(disc_id)
        data_in = disc.get_data_in()
        dict_data[disc.sos_name] = {}
        for k,v in data_in.items():
            if not v['numerical']:
                dict_data[disc.sos_name][k] = v['value']
    return dict_data


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
        func_df = DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['land_demand_constraint'])
        list_parent.extend(['agriculture_constraint'])
        list_ftype.extend([INEQ_CONSTRAINT])
        list_weight.extend([-1.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM])
        list_ns.extend(['ns_functions'])
        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

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

        elif self.process_level == 'thesis':
            dc_witness = datacase_witness_thesis(
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
            f'{self.study_name}.max_mda_iter': 50,
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
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()


    data = uc_cls.ee.dm.get_discipline(list(uc_cls.ee.dm.disciplines_dict.keys())[3]).get_data_in()
    dm = uc_cls.ee.dm

    #dict_data = prepare_data(dm)
    #for k,v in dict_data.items():
    #    generate_json_by_discipline(v, k)
    connection_string = 'mongodb://sostradescosmosdb:oXrJPQYABLfREMu3bTmpNTjN8bbdj6huj1OB0PCj6ReUabAgK9bo4VdnSnKOzXD1iY8R5HTMw3fiACDb2FiEgg==@sostradescosmosdb.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@sostradescosmosdb@'
    database_name = 'regionalization'
    container_name = 'regionalizationv0'
    json_path = join(dirname(__file__), 'data', 'AgricultureMix.Crop.json')
    import os
    for f in os.listdir(join(dirname(__file__), 'data')):
        insert_json_to_mongodb_bis(json_path, container_name, database_name, connection_string)

    uc_cls.run()

