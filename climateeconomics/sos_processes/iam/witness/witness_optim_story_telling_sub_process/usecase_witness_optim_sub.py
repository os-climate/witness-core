'''
Copyright 2024 Capgemini

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
import numpy as np
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)

# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2_witness_coarse_mda_gdp_model_wo_damage_wo_co2_tax import (
    Study as witness_usecase2_story_telling,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2b_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import (
    Study as usecase2b,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_4_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import (
    Study as usecase4,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import (
    Study as witness_usecase7_story_telling,
)
from tools.design_space_creator import (
    make_dspace_utilization_ratio,
)

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
OPTIM_NAME = "WITNESS_MDO"
COUPLING_NAME = "WITNESS_Eval"
EXTRA_NAME = "WITNESS"


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, run_usecase=False,
                 execution_engine=None, sub_usecase='uc2'):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end

        self.coupling_name = COUPLING_NAME
        self.designvariable_name = "DesignVariables"
        self.func_manager_name = "FunctionsManager"
        self.extra_name = EXTRA_NAME
        self.energy_mix_name = 'EnergyMix'
        GlossaryCore.ccus_type = 'CCUS'
        if sub_usecase == 'uc2':
            self.witness_uc = witness_usecase2_story_telling(
                self.year_start, self.year_end)
        elif sub_usecase == 'uc7':
            self.witness_uc = witness_usecase7_story_telling(self.year_start, self.year_end)
        elif sub_usecase == 'uc2b':
            self.witness_uc = usecase2b(self.year_start, self.year_end)
        elif sub_usecase == 'uc4':
            self.witness_uc = usecase4(self.year_start, self.year_end)
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        """ Overloaded method to initialize witness multiscenario optimization process

        @return list of dictionary: [{str: *}]
        """
        setup_data_list = []

        # -- retrieve energy input data

        self.witness_uc.study_name = f'{self.study_name}.{self.coupling_name}.{self.extra_name}'
        self.witness_uc.study_name_wo_extra_name = f'{self.study_name}.{self.coupling_name}'
        witness_data_list = self.witness_uc.setup_usecase()
        setup_data_list = setup_data_list + witness_data_list

        values_dict = {f'{self.study_name}.epsilon0': 1.0}

        dv_arrays_dict = {}

        design_var_descriptor = {}
        years = np.arange(self.year_start, self.year_end + 1)

        # create design variables and design space descriptor for variable percentage_of_gdp_energy_invest
        design_var_descriptor['percentage_gdp_invest_in_energy_array'] = {
            'out_name': GlossaryEnergy.EnergyInvestPercentageGDPName,
            'out_type': 'dataframe',
            'key': GlossaryEnergy.EnergyInvestPercentageGDPName,
            'index': years,
            'index_name': GlossaryCore.Years,
            'namespace_in': GlossaryCore.NS_FUNCTIONS,
            'namespace_out': 'ns_invest'
        }
        energy_list = ['fossil', GlossaryCore.clean_energy]
        techno_list = ['FossilSimpleTechno', GlossaryCore.CleanEnergySimpleTechno]
        for energy, technology in zip(energy_list, techno_list):
            # add design variable for utilization ratio per technology
            design_var_descriptor[f'{energy}_{technology}_utilization_ratio_array'] = {
                'out_name': f'{energy}.{technology}.{GlossaryCore.UtilisationRatioValue}',
                'out_type': 'dataframe',
                'key': GlossaryCore.UtilisationRatioValue,
                'index': years,
                'index_name': GlossaryCore.Years,
                'namespace_in': GlossaryCore.NS_ENERGY_MIX,
                'namespace_out': GlossaryCore.NS_ENERGY_MIX
            }

        self.design_var_descriptor = design_var_descriptor
        values_dict[
            f'{self.study_name}.{self.coupling_name}.{self.designvariable_name}.design_var_descriptor'] = design_var_descriptor
        # design space
        initial_pole_value = self.get_share_invest_in_eneryg_relative_to_gdp(selected_year=self.year_start)
        dspace_dict = {'variable': ['percentage_gdp_invest_in_energy_array'],
                       'value': [[initial_pole_value] + [1.] * (GlossaryCore.NB_POLES_OPTIM_KU - 1)],
                       'lower_bnd': [[2e-1] * GlossaryCore.NB_POLES_OPTIM_KU],
                       'upper_bnd': [[5.] * GlossaryCore.NB_POLES_OPTIM_KU],
                       'enable_variable': [True],
                       'activated_elem': [[False] + [True] * (GlossaryCore.NB_POLES_OPTIM_KU - 1)]
                       }

        dspace_share_invest = pd.DataFrame(dspace_dict)
        min_UR = 85.
        dspace_UR = {
            'fossil_FossilSimpleTechno_utilization_ratio_array': [min_UR, min_UR, 100., True],
            f"{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_utilization_ratio_array": [min_UR, min_UR, 100., True],
        }
        dspace_UR = make_dspace_utilization_ratio(dspace_UR, allow_year_start=True)
        self.dspace = pd.concat([dspace_share_invest, dspace_UR])
        values_dict[f'{self.study_name}.design_space'] = self.dspace
        # create func manager
        func_dict = {FunctionManagerDisc.VARIABLE: [GlossaryCore.QuantityObjectiveValue,
                                                    GlossaryCore.UsableCapitalObjectiveName,],
                     FunctionManagerDisc.PARENT: 'objectives',
                     FunctionManagerDisc.FTYPE: 'objective',
                     FunctionManagerDisc.WEIGHT: [0.0, 1.0,],
                     FunctionManagerDisc.AGGR_TYPE: 'sum',
                     FunctionManagerDisc.NAMESPACE_VARIABLE: GlossaryCore.NS_FUNCTIONS}
        func_df = pd.DataFrame(data=func_dict)
        values_dict[f'{self.study_name}.{self.coupling_name}.{self.func_manager_name}.{FUNC_DF}'] = func_df


        values_dict[f'{self.study_name}.{self.coupling_name}.sub_mda_class'] = 'GSPureNewtonMDA'
        # values_dict[f'{self.study_name}.{self.coupling_name}.warm_start'] = True
        values_dict[f'{self.study_name}.{self.coupling_name}.max_mda_iter'] = 50
        values_dict[f'{self.study_name}.{self.coupling_name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.study_name}.{self.coupling_name}.epsilon0'] = 1.0
        values_dict[
            f'{self.study_name}.{self.coupling_name}.{self.extra_name}.percentage_gdp_invest_in_energy_array'] = np.ones(GlossaryCore.NB_POLES_OPTIM_KU - 1)
        values_dict[
            f'{self.study_name}.{self.coupling_name}.{self.extra_name}.EnergyMix.fossil_FossilSimpleTechno_utilization_ratio_array'] = np.ones(GlossaryCore.NB_POLES_UTILIZATION_RATIO) * 100.
        values_dict[
            f'{self.study_name}.{self.coupling_name}.{self.extra_name}.EnergyMix.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_utilization_ratio_array'] = np.ones(GlossaryCore.NB_POLES_UTILIZATION_RATIO) * 100.

        setup_data_list.append(values_dict)
        setup_data_list.append(dv_arrays_dict)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
