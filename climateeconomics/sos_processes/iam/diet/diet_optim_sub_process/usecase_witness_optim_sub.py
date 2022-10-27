# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# Copyright (c) 2021 Airbus SAS.
# All rights reserved.

from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sos_trades_core.study_manager.study_manager import StudyManager
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as witness_usecase
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

import pandas as pd
import numpy as np
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT
from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS

from climateeconomics.sos_processes.iam.diet.diet_process.usecase_diet import Study as usecase_diet
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
                 agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 process_level='dev'):
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
        self.agri_techno_list = agri_techno_list
        self.process_level = process_level
        self.witness_uc = usecase_diet(
            self.year_start, self.year_end, self.time_step)

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

        design_var_descriptor = {}
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)


        if self.process_level == 'dev':
            dv_arrays_dict[f'{self.witness_uc.study_name}.forest_investment_array_mix'] = dspace_df[f'forest_investment_array_mix']['value']
            design_var_descriptor['forest_investment_array_mix'] = {'out_name': 'forest_investment',
                                                                    'out_type': 'dataframe',
                                                                    'key': 'forest_investment',
                                                                    'index': years,
                                                                    'index_name': 'years',
                                                                    'namespace_in': 'ns_witness',
                                                                    'namespace_out': 'ns_invest'
                                                                    }
            if 'CropEnergy' in self.agri_techno_list:
                dv_arrays_dict[f'{self.witness_uc.study_name}.crop_investment_array_mix'] = dspace_df[f'crop_investment_array_mix']['value']
                design_var_descriptor['crop_investment_array_mix'] = {'out_name': 'crop_investment',
                                                                      'out_type': 'dataframe',
                                                                      'key': 'investment',
                                                                      'index': years,
                                                                      'index_name': 'years',
                                                                      'namespace_in': 'ns_witness',
                                                                      'namespace_out': 'ns_crop'
                                                                      }
            if 'ManagedWood' in self.agri_techno_list:
                dv_arrays_dict[f'{self.witness_uc.study_name}.managed_wood_investment_array_mix'] = dspace_df[f'managed_wood_investment_array_mix']['value']
                design_var_descriptor['managed_wood_investment_array_mix'] = {'out_name': 'managed_wood_investment',
                                                                              'out_type': 'dataframe',
                                                                              'key': 'investment',
                                                                              'index': years,
                                                                              'index_name': 'years',
                                                                              'namespace_in': 'ns_witness',
                                                                              'namespace_out': 'ns_forest'
                                                                              }
            dv_arrays_dict[f'{self.witness_uc.study_name}.deforestation_investment_ctrl'] = dspace_df[f'deforestation_investment_ctrl']['value']
            design_var_descriptor['deforestation_investment_ctrl'] = {'out_name': 'deforestation_investment',
                                                                           'out_type': 'dataframe',
                                                                           'key': 'investment',
                                                                           'index': years,
                                                                           'index_name': 'years',
                                                                           'namespace_in': 'ns_witness',
                                                                           'namespace_out': 'ns_forest'
                                                                           }
            dv_arrays_dict[f'{self.witness_uc.study_name}.red_meat_percentage_ctrl'] = dspace_df[f'red_meat_percentage_ctrl']['value']
            design_var_descriptor['red_meat_percentage_ctrl'] = {'out_name': 'red_meat_calories_per_day',
                                                                 'out_type': 'dataframe',
                                                                 'key': 'red_meat_calories_per_day',
                                                                 'index': years,
                                                                 'index_name': 'years',
                                                                 'namespace_in': 'ns_witness',
                                                                 'namespace_out': 'ns_crop'
                                                                 }
            dv_arrays_dict[f'{self.witness_uc.study_name}.white_meat_percentage_ctrl'] = dspace_df[f'white_meat_percentage_ctrl']['value']
            design_var_descriptor['white_meat_percentage_ctrl'] = {'out_name': 'white_meat_calories_per_day',
                                                                   'out_type': 'dataframe',
                                                                   'key': 'white_meat_calories_per_day',
                                                                   'index': years,
                                                                   'index_name': 'years',
                                                                   'namespace_in': 'ns_witness',
                                                                   'namespace_out': 'ns_crop'
                                                                   }
            dv_arrays_dict[f'{self.witness_uc.study_name}.other_percentage_ctrl'] = dspace_df[f'other_percentage_ctrl']['value']
            design_var_descriptor['other_percentage_ctrl'] = {'out_name': 'other_calories_per_day',
                                                                   'out_type': 'dataframe',
                                                                   'key': 'other_calories_per_day',
                                                                   'index': years,
                                                                   'index_name': 'years',
                                                                   'namespace_in': 'ns_witness',
                                                                   'namespace_out': 'ns_agriculture'
                                                                   }
        else:
            dv_arrays_dict[f'{self.witness_uc.study_name}.forest_investment_array_mix'] = dspace_df[f'forest_investment_array_mix']['value']
            design_var_descriptor['forest_investment_array_mix'] = {'out_name': 'forest_investment',
                                                                    'out_type': 'dataframe',
                                                                    'key': 'forest_investment',
                                                                    'index': years,
                                                                    'index_name': 'years',
                                                                    'namespace_in': 'ns_witness',
                                                                    'namespace_out': 'ns_invest'
                                                                    }
            dv_arrays_dict[f'{self.witness_uc.study_name}.deforested_surface_ctrl'] = dspace_df[f'deforested_surface_ctrl']['value']
            design_var_descriptor['deforested_surface_ctrl'] = {'out_name': 'deforestation_surface',
                                                                'out_type': 'dataframe',
                                                                'key': 'deforested_surface',
                                                                'index': years,
                                                                'index_name': 'years',
                                                                'namespace_in': 'ns_witness',
                                                                'namespace_out': 'ns_witness'}
            dv_arrays_dict[f'{self.witness_uc.study_name}.red_meat_percentage_ctrl'] = dspace_df[f'red_meat_percentage_ctrl']['value']
            design_var_descriptor['red_meat_percentage_ctrl'] = {'out_name': 'red_meat_calories_per_day',
                                                                 'out_type': 'dataframe',
                                                                 'key': 'red_meat_calories_per_day',
                                                                 'index': years,
                                                                 'index_name': 'years',
                                                                 'namespace_in': 'ns_witness',
                                                                 'namespace_out': 'ns_agriculture'
                                                                 }
            dv_arrays_dict[f'{self.witness_uc.study_name}.white_meat_percentage_ctrl'] = dspace_df[f'white_meat_percentage_ctrl']['value']
            design_var_descriptor['white_meat_percentage_ctrl'] = {'out_name': 'white_meat_calories_per_day',
                                                                   'out_type': 'dataframe',
                                                                   'key': 'white_meat_calories_per_day',
                                                                   'index': years,
                                                                   'index_name': 'years',
                                                                   'namespace_in': 'ns_witness',
                                                                   'namespace_out': 'ns_agriculture'
                                                                   }
            dv_arrays_dict[f'{self.witness_uc.study_name}.other_percentage_ctrl'] = dspace_df[f'other_percentage_ctrl']['value']
            design_var_descriptor['other_percentage_ctrl'] = {'out_name': 'other_calories_per_day',
                                                                   'out_type': 'dataframe',
                                                                   'key': 'other_calories_per_day',
                                                                   'index': years,
                                                                   'index_name': 'years',
                                                                   'namespace_in': 'ns_witness',
                                                                   'namespace_out': 'ns_agriculture'
                                                                   }
        dv_arrays_dict[f'{self.witness_uc.study_name}.share_energy_investment_ctrl'] = dspace_df[f'share_energy_investment_ctrl']['value']
        design_var_descriptor['share_energy_investment_ctrl'] = {'out_name': 'share_energy_investment',
                                                                 'out_type': 'dataframe',
                                                                 'key': 'share_investment',
                                                                 'index': years,
                                                                 'index_name': 'years',
                                                                 'namespace_in': 'ns_witness',
                                                                 'namespace_out': 'ns_witness'
                                                                 }

        self.func_df = self.witness_uc.func_df
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.{FUNC_DF}'] = self.func_df

        values_dict[f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor

        values_dict[f'{self.study_name}.{self.coupling_name}.sub_mda_class'] = 'GSPureNewtonMDA'
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
