"""
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
"""

import numpy as np
import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.design_var.design_var_disc import (
    DesignVarDiscipline,
)
from sostrades_core.execution_engine.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    COUPLING_NAME,
    EXTRA_NAME,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_optim_sub_usecase,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
EXPORT_CSV = FunctionManagerDisc.EXPORT_CSV
WRITE_XVECT = DesignVarDiscipline.WRITE_XVECT

"""
Optim subprocess (to be used on a DoE) of witness coarse with an added discipline InvestmentsProfileBuilderDisc that computes
profiles of investments (invest_mix) that are given as input to the discipline InvestmentsDistribution
Compared to the classic witness coarse process where design variables are the invest_mix, here the design variables are
the weights coef_i to be applied to the generic investment profiles df_i
Therefore, this usecase can be deduced from the standard witness coarse sub-process by:
- defining the  InvestmentsProfileBuilderDisc inputs (n_profiles, df_i)
- defining from scratch a new design space with coef_i (ie excluding the standard design variables such as investments
in fossil, renewable and CCUS and their utilization ratios)
- imposing the utilization ratios at 100%
"""


class Study(ClimateEconomicsStudyManager):

    def __init__(
        self,
        year_start=GlossaryCore.YearStartDefault,
        year_end=GlossaryCore.YearEndDefault,
        time_step=1,
        bspline=False,
        run_usecase=False,
        execution_engine=None,
        invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
        techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
    ):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict

        self.witness_uc = witness_optim_sub_usecase(
            self.year_start,
            self.year_end,
            self.time_step,
            bspline=self.bspline,
            execution_engine=execution_engine,
            invest_discipline=self.invest_discipline,
            techno_dict=techno_dict,
        )
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict

    def df_generator_linear_profile(self, i, columns_names, n_profiles, years):
        """
        args:
            i [int]= index of the profile
            columns_names [list] = list of names of variables
            n_profiles [int]=  total number of profiles
        assuming that there are nb_columns variables, assuming that there are one increasing and 1 decreasing
        profile per variable, then df_generator generates those 2 profiles for each variable
        the growing generic invest profiles are for the even values of i
        the decreasing generic invest profiles are for the uneven values of i
        Profiles are normalized between 0 and 1
        """
        if n_profiles != 2 * len(columns_names):
            raise ValueError(
                f"df_generator computes 2 generic invest profiles per column. n_profiles should be {2 * len(columns_names)} "
                f"whereas it it {n_profiles}"
            )
        # growing normalized invest profiles are for the even values of i (arbitrary choice)
        if i % 2 == 0:
            normalized_profile = np.linspace(1.0e-6, 3000.0, len(years))
        else:
            normalized_profile = np.linspace(3000.0, 1.0e-6, len(years))
        column_vect = np.zeros(len(columns_names))
        # put the normalized profile in the column of variable corresponding to index i assuming that there are
        # 2 profiles per variable
        column_vect[int(i / 2)] = 1.0
        data = np.array([normalized_profile]).T * column_vect
        df = pd.DataFrame(data, columns=columns_names)
        df.insert(0, GlossaryEnergy.Years, years, True)

        return df

    def df_generator_constant_profile(self, years, columns_names, list_val):
        df = pd.DataFrame({**{GlossaryEnergy.Years: years}, **dict(zip(columns_names, list_val))})
        return df

    def setup_process(self):
        witness_optim_sub_usecase.setup_process(self)

    def setup_usecase(self, study_folder_path=None):
        ns = self.study_name
        self.witness_uc.study_name = f"{ns}"
        self.coupling_name = self.witness_uc.coupling_name
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        values_dict = {}

        # remove the invests from the initial witness coarse design space
        witness_uc_data = self.witness_uc.setup_usecase()
        for dict_data in witness_uc_data:
            values_dict.update(dict_data)

        # No modif of the objective function
        self.func_df = self.witness_uc.func_df
        self.func_df.loc[self.func_df["variable"] == "gwp100_objective", "weight"] = 1.0

        # define the missing inputs:
        # InvestmentsProfileBuilderDisc inputs
        columns_names = [
            f"{GlossaryEnergy.renewable}.RenewableSimpleTechno",
            f"{GlossaryEnergy.fossil}.FossilSimpleTechno",
            f"{GlossaryEnergy.carbon_capture}.{GlossaryEnergy.direct_air_capture}.DirectAirCaptureTechno",
            f"{GlossaryEnergy.carbon_capture}.{GlossaryEnergy.flue_gas_capture}.FlueGasTechno",
            f"{GlossaryEnergy.carbon_storage}.CarbonStorageTechno",
        ]

        n_profiles = 2  # * len(columns_names) # 2 generic profiles for each of the variables, one growing and one decreasing profile

        # create the design var descriptor for the design variables discipline. In this usecase, all outputs of
        # investprofilebuilder are provided with poles into the design variable discipline
        self.design_var_descriptor = {}
        for var in columns_names:
            self.design_var_descriptor[f"{var}_array_mix"] = {
                "out_name": GlossaryCore.invest_mix,
                "out_type": "dataframe",
                "key": f"{var}",
                "index": years,
                "index_name": GlossaryCore.Years,
                "namespace_in": "ns_invest",
                "namespace_out": "ns_invest",
            }

        values_dict.update(
            {
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.column_names": columns_names,
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.n_profiles": n_profiles,
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.{GlossaryEnergy.EXPORT_PROFILES_AT_POLES}": True,
                f"{ns}.{self.witness_uc.coupling_name}.DesignVariables.design_var_descriptor": self.design_var_descriptor,
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.nb_poles": GlossaryCore.NB_POLES_COARSE,
            }
        )

        years_dfi = np.arange(self.year_start, self.year_end + 1, self.time_step)
        # values_dict.update({
        #    f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.df_{i}":
        #        self.df_generator(i, columns_names, n_profiles, years_dfi) for i in range(n_profiles)
        # })
        min = 1.0e-6
        max = 3000.0
        # profile min
        values_dict.update(
            {
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.df_{0}":
                # self.df_generator(i, columns_names, n_profiles, years_dfi) for i in range(n_profiles)
                self.df_generator_constant_profile(years_dfi, columns_names, [min] * len(columns_names))
            }
        )
        # profile max
        values_dict.update(
            {
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.df_{1}":
                # self.df_generator(i, columns_names, n_profiles, years_dfi) for i in range(n_profiles)
                self.df_generator_constant_profile(years_dfi, columns_names, [max] * len(columns_names))
            }
        )
        # impose values to the utilization ratios that are not design variables anymore
        list_utilization_ratio_var = [
            "fossil_FossilSimpleTechno_utilization_ratio_array",
            "renewable_RenewableSimpleTechno_utilization_ratio_array",
            "carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array",
            "carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array",
            "carbon_storage.CarbonStorageTechno_utilization_ratio_array",
        ]
        values_dict.update(
            {
                f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.{var}": np.ones(len(years)) * 100.0
                for var in list_utilization_ratio_var
            }
        )

        # Create the design space with the coef_i in
        dspace_df = pd.DataFrame({})
        for i in range(n_profiles):
            design_space_ctrl_dict = {}
            coeff_i = 1.0
            design_space_ctrl_dict["variable"] = f"coeff_{i}"
            design_space_ctrl_dict["value"] = [np.array([coeff_i])]
            design_space_ctrl_dict["lower_bnd"] = [np.array([0.0])]
            design_space_ctrl_dict["upper_bnd"] = [np.array([1.0])]
            design_space_ctrl_dict["enable_variable"] = True
            design_space_ctrl_dict["activated_elem"] = [[True]]
            dspace_df = pd.concat([dspace_df, pd.DataFrame(design_space_ctrl_dict)], ignore_index=True)
            values_dict.update(
                {
                    f"{ns}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.InvestmentsProfileBuilderDisc.coeff_{i}": coeff_i
                }
            )

        # must add to the design space the inputs to the design variable discipline to allow the configure to work
        # however these are not design variables as they are output of the investprofilebuilder discipline => trick
        for var in columns_names:
            design_space_ctrl_dict = {}
            design_space_ctrl_dict["variable"] = var + "_array_mix"
            design_space_ctrl_dict["value"] = [1000.0 * np.ones(GlossaryCore.NB_POLES_COARSE)]
            design_space_ctrl_dict["lower_bnd"] = [1.0 * np.ones(GlossaryCore.NB_POLES_COARSE)]
            design_space_ctrl_dict["upper_bnd"] = [3000.0 * np.ones(GlossaryCore.NB_POLES_COARSE)]
            design_space_ctrl_dict["enable_variable"] = True
            design_space_ctrl_dict["activated_elem"] = [[True] * GlossaryCore.NB_POLES_COARSE]
            dspace_df = pd.concat([dspace_df, pd.DataFrame(design_space_ctrl_dict)], ignore_index=True)

        # optimization functions:
        optim_values_dict = {
            f"{ns}.epsilon0": 1,
            f"{ns}.cache_type": "SimpleCache",
            f"{ns}.warm_start": False,  # so that mda is computed identically for analytical and approx gradients
            f"{ns}.design_space": dspace_df,
            f"{ns}.objective_name": FunctionManagerDisc.OBJECTIVE_LAGR,
            f"{ns}.eq_constraints": [],
            f"{ns}.ineq_constraints": [],
        }

        return [values_dict] + [optim_values_dict]


if "__main__" == __name__:
    uc_cls = Study()
    uc_cls.test()
    # comment above and uncomment below to test the post-processing
    """
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    ns = f'usecase_witness_grad_check_optim_sub.WITNESS_Eval.WITNESS'
    filters = ppf.get_post_processing_filters_by_namespace(uc_cls.ee, ns)

    graph_list = ppf.get_post_processing_by_namespace(uc_cls.ee, ns, filters, as_json=False)
    for graph in graph_list:
        graph.to_plotly().show()
    """
