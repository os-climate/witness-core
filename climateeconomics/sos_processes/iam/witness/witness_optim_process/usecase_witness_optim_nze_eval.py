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
from os.path import dirname, join

import numpy as np
import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    COUPLING_NAME,
    EXTRA_NAME,
    OPTIM_NAME,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_optim_sub_usecase,
)
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugin.models.design_var.design_var_disc import (
    DesignVarDiscipline,
)
from sostrades_optimization_plugin.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
EXPORT_CSV = FunctionManagerDisc.EXPORT_CSV
WRITE_XVECT = DesignVarDiscipline.WRITE_XVECT

# usecase of witness full to evaluate a design space with NZE investments
class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1, bspline=False, run_usecase=False,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.optim_name = OPTIM_NAME
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict

        self.witness_uc = witness_optim_sub_usecase(
            self.year_start, self.year_end, self.time_step, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict

    def setup_process(self):
        witness_optim_sub_usecase.setup_process(self)

    def setup_usecase(self, study_folder_path=None):
        ns = self.study_name

        values_dict = {}

        self.witness_uc.study_name = f'{ns}.{self.optim_name}'
        self.coupling_name = self.witness_uc.coupling_name
        witness_uc_data = self.witness_uc.setup_usecase()
        for dict_data in witness_uc_data:
            values_dict.update(dict_data)

        # design space WITNESS
        dspace_df = self.witness_uc.dspace
        self.func_df = self.witness_uc.func_df
        # df_xvect = pd.read_pickle('df_xvect.pkl')
        # df_xvect.columns = [df_xvect.columns[0]]+[col.split('.')[-1] for col in df_xvect.columns[1:]]
        # dspace_df_xvect=pd.DataFrame({'variable':df_xvect.columns, 'value':df_xvect.drop(0).values[0]})
        # dspace_df.update(dspace_df_xvect)

        dspace_size = self.witness_uc.dspace_size
        # optimization functions:
        optim_values_dict = {f'{ns}.epsilon0': 1,
                             f'{ns}.cache_type': 'SimpleCache',
                             f'{ns}.{self.optim_name}.design_space': dspace_df,
                             f'{ns}.{self.optim_name}.objective_name': FunctionManagerDisc.OBJECTIVE_LAGR,
                             f'{ns}.{self.optim_name}.eq_constraints': [],
                             f'{ns}.{self.optim_name}.ineq_constraints': [],

                             # optimization parameters:
                             f'{ns}.{self.optim_name}.max_iter': 1,
                             f'{ns}.{self.optim_name}.eval_mode': True,
                             f'{ns}.warm_start': True,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.warm_start': True,
                             # SLSQP, NLOPT_SLSQP
                             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
                             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
                             f'{ns}.{self.optim_name}.differentiation_method': 'user',
                             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                                      "ftol_abs": 3e-16,
                                                                      "normalize_design_space": True,
                                                                      "max_ls_step_nb": 3 * dspace_size,
                                                                      "maxcor": dspace_size,
                                                                      "pg_tol": 1e-16,
                                                                      "xtol_rel": 1e-16,
                                                                      "xtol_abs": 1e-16,
                                                                      "max_iter": 700,
                                                                      "disp": 30},

                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDO_options': {
                                 'tol': 1.0e-10,
                                 'max_iter': 10000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA_options': {
                                 'tol': 1.0e-10,
                                 'max_iter': 50000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.epsilon0': 1.0,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.tolerance': 1.0e-10,

                             f'{ns}.{self.optim_name}.parallel_options': {"parallel": False,  # True
                                                                          "n_processes": 32,
                                                                          "use_threading": False,
                                                                          "wait_time_between_fork": 0},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.sub_mda_class': 'GSPureNewtonMDA',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.max_mda_iter': 50,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.{WRITE_XVECT}': False}

        # print("Design space dimension is ", dspace_size)

        list_design_var_to_clean = ['red_meat_calories_per_day_ctrl', 'white_meat_calories_per_day_ctrl',
                                    'vegetables_and_carbs_calories_per_day_ctrl', 'milk_and_eggs_calories_per_day_ctrl',
                                    'forest_investment_array_mix', 'crop_investment_array_mix']
        diet_mortality_df = pd.read_csv(join(dirname(__file__), 'data', 'diet_mortality.csv'))

        # clean dspace
        dspace_df.drop(dspace_df.loc[dspace_df['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        # clean dspace descriptor
        dvar_descriptor = self.witness_uc.design_var_descriptor

        updated_dvar_descriptor = {k: v for k, v in dvar_descriptor.items() if k not in list_design_var_to_clean}


        dspace_file_name = 'invest_design_space_NZE.csv'
        dspace_out = pd.read_csv(join(dirname(__file__), 'data', dspace_file_name))


        dspace_df.drop(dspace_df.loc[dspace_df['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        values_dict_updt = {}
        for index, row in dspace_df.iterrows():
            variable = row["variable"]

            if variable in dspace_out["variable"].values:
                valeur_str = dspace_out[dspace_out["variable"] == variable]["value"].iloc[0]
                upper_bnd_str = dspace_out[dspace_out["variable"] == variable]["upper_bnd"].iloc[0]
                lower_bnd_str = dspace_out[dspace_out["variable"] == variable]["lower_bnd"].iloc[0]
                activated_elem_str = dspace_out[dspace_out["variable"] == variable]["activated_elem"].iloc[0]

                if ',' not in valeur_str:
                    valeur_array = np.array(eval(valeur_str.replace(' ', ',')))
                else:
                    valeur_array = np.array(eval(valeur_str))
                upper_bnd_array = np.array(eval(upper_bnd_str.replace(' ', ',')))
                lower_bnd_array = np.array(eval(lower_bnd_str.replace(' ', ',')))
                activated_elem_array = eval(activated_elem_str)

                dspace_df.at[index, "value"] = valeur_array
                dspace_df.at[index, "upper_bnd"] = upper_bnd_array
                dspace_df.at[index, "lower_bnd"] = lower_bnd_array
                dspace_df.at[index, "activated_elem"] = activated_elem_array
                values_dict_updt.update({
                    f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.EnergyMix.{variable}': valeur_array,
                    f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.CCUS.{variable}': valeur_array})
        dspace_df['enable_variable'] = True

        invest_mix_file = 'investment_mix.csv'
        invest_mix = pd.read_csv(join(dirname(__file__), 'data', invest_mix_file))
        forest_invest_file = 'forest_investment.csv'
        forest_invest = pd.read_csv(join(dirname(__file__), 'data', forest_invest_file))
        #dspace_df.to_csv('dspace_invest_cleaned_2.csv', index=False)
        crop_investment_df_NZE = DatabaseWitnessCore.CropInvestmentNZE.value
        values_dict_updt.update({f'{ns}.{self.optim_name}.design_space': dspace_df,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.designvariable_name}.design_var_descriptor': updated_dvar_descriptor,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.InvestmentDistribution.invest_mix': invest_mix,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.InvestmentDistribution.forest_investment': forest_invest,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.AgricultureMix.Crop.crop_investment': crop_investment_df_NZE,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.AgricultureMix.Forest.reforestation_cost_per_ha': 3800.,
                                 f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.WITNESS.Population.diet_mortality_param_df': diet_mortality_df,

                                 })

        values_dict.update(values_dict_updt)
        optim_values_dict.update(values_dict_updt)
        return [values_dict] + [optim_values_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines:
        if 'Forest' in disc.get_disc_full_name():
            filters = ppf.get_post_processing_filters_by_discipline(
                disc)
            graph_list = ppf.get_post_processing_by_discipline(
                disc, filters, as_json=False)

            for graph in graph_list:
                graph.to_plotly().show()
