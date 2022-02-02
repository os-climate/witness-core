# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# Copyright (c) 2021 Airbus SAS.
# All rights reserved.

from sos_trades_core.study_manager.study_manager import StudyManager

from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as witness_usecase
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

import pandas as pd
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
OPTIM_NAME = "WITNESS_MDO"
COUPLING_NAME = "WITNESS_Eval"
EXTRA_NAME = "WITNESS"


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, bspline=False, run_usecase=False, execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[1], techno_dict=DEFAULT_TECHNO_DICT, process_level='val'):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

        self.coupling_name = COUPLING_NAME
        self.designvariable_name = "DesignVariables"
        self.func_manager_name = "FunctionsManager"
        self.extra_name = EXTRA_NAME
        self.energy_mix_name = 'EnergyMix'
        self.ccs_mix_name = 'CCUS'
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.process_level = process_level
        self.witness_uc = witness_usecase(
            self.year_start, self.year_end, self.time_step,  bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict, process_level=process_level)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict

    def setup_usecase(self):
        """ Overloaded method to initialize witness multiscenario optimization process

        @return list of dictionary: [{str: *}]
        """
        setup_data_list = []

        #-- retrieve energy input data

        self.witness_mda_usecase = self.witness_uc
        self.witness_uc.study_name = f'{self.study_name}.{self.coupling_name}.{self.extra_name}'
        self.witness_uc.study_name_wo_extra_name = f'{self.study_name}.{self.coupling_name}'
        witness_data_list = self.witness_uc.setup_usecase()
        setup_data_list = setup_data_list + witness_data_list

        dspace_df = self.witness_uc.dspace
        values_dict = {}

        values_dict[f'{self.study_name}.epsilon0'] = 1.0
        dv_arrays_dict = {}

        for energy in self.witness_uc.energy_list:
            energy_wo_dot = energy.replace('.', '_')
            if self.invest_discipline == INVEST_DISCIPLINE_OPTIONS[0]:
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{energy_wo_dot}_array_mix'] = dspace_df[f'{energy_wo_dot}_array_mix']['value']
            for technology in self.witness_uc.dict_technos[energy]:
                technology_wo_dot = technology.replace('.', '_')
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.energy_mix_name}.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'] = dspace_df[f'{energy_wo_dot}_{technology_wo_dot}_array_mix']['value']

        for ccs in self.witness_uc.ccs_list:
            ccs_wo_dot = ccs.replace('.', '_')
            if self.invest_discipline == INVEST_DISCIPLINE_OPTIONS[0]:
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.ccs_mix_name}.{ccs}.{ccs_wo_dot}_array_mix'] = dspace_df[f'{ccs_wo_dot}_array_mix']['value']
            for technology in self.witness_uc.dict_technos[ccs]:
                technology_wo_dot = technology.replace('.', '_')
                dv_arrays_dict[f'{self.witness_uc.study_name}.{self.ccs_mix_name}.{ccs}.{technology}.{ccs_wo_dot}_{technology_wo_dot}_array_mix'] = dspace_df[f'{ccs_wo_dot}_{technology_wo_dot}_array_mix']['value']

        if self.invest_discipline == INVEST_DISCIPLINE_OPTIONS[0]:
            dv_arrays_dict[f'{self.witness_uc.study_name}.ccs_percentage_array'] = dspace_df[f'ccs_percentage_array']['value']

        if self.process_level == 'val':
            dv_arrays_dict[f'{self.witness_uc.study_name}.livestock_usage_factor_array'] = dspace_df[f'livestock_usage_factor_array']['value']

        self.func_df = self.witness_uc.func_df
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.{FUNC_DF}'] = self.func_df

        values_dict[f'{self.study_name}.{self.coupling_name}.sub_mda_class'] = 'GSorNewtonMDA'
        #values_dict[f'{self.study_name}.{self.coupling_name}.warm_start'] = True
        values_dict[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 50
        values_dict[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0
        # design space

        dspace = self.witness_uc.dspace
        self.dspace_size = dspace.pop('dspace_size')

        dspace_df_columns = ['variable', 'value', 'lower_bnd',
                             'upper_bnd', 'enable_variable']
        dspace_df = pd.DataFrame(columns=dspace_df_columns)
        for key, elem in dspace.items():
            dict_var = {'variable': key}
            dict_var.update(elem)
            dspace_df = dspace_df.append(dict_var, ignore_index=True)

        self.dspace = dspace_df
        values_dict[f'{self.witness_uc.study_name}.{self.coupling_name}.energy_list'] = self.witness_uc.energy_list
        values_dict[f'{self.study_name}.design_space'] = self.dspace
        setup_data_list.append(values_dict)
        setup_data_list.append(dv_arrays_dict)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    print(
        len(uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines))
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()


#     uc_cls.execution_engine.root_process.sos_disciplines[0].coupling_structure.graph.export_initial_graph(
#         "initial.pdf")
