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
from os.path import dirname, join, pardir

import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import (
    INVEST_DISC_NAME,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import (
    Study as usecase_witness_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda import (
    Study as uc_ms_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2_witness_coarse_mda_gdp_model_wo_damage_wo_co2_tax import (
    Study as usecase2,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_4_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import (
    Study as usecase4,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import (
    Study as usecase7,
)


class Study(ClimateEconomicsStudyManager):
    TIPPING_POINT = 'Tipping point'
    TIPPING_POINT_LIST = [6, 4.5, 3.5]
    SEP = ' '
    UNIT = 'deg C'
    # scenarios name
    # NB: this name pattern TIPPING_POINT + SEP + TIPPING_POINT_LIST[0] + UNIT is reused in post-processing_witness_coarse_mda.py
    # cannot keep the decimal in the scenario name, otherwise it adds a node in the usecase
    USECASE2 = uc_ms_mda.USECASE2
    USECASE4_TP_REF = uc_ms_mda.USECASE4 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[0]).replace('.', '_') + UNIT
    USECASE4_TP1 = uc_ms_mda.USECASE4 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[1]).replace('.', '_') + UNIT
    USECASE4_TP2 = uc_ms_mda.USECASE4 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[2]).replace('.', '_') + UNIT
    USECASE7_TP_REF = uc_ms_mda.USECASE7 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[0]).replace('.', '_') + UNIT
    USECASE7_TP1 = uc_ms_mda.USECASE7 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[1]).replace('.', '_') + UNIT
    USECASE7_TP2 = uc_ms_mda.USECASE7 + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[2]).replace('.', '_') + UNIT

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.check_outputs = False
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):

        self.scatter_scenario = 'mda_scenarios'



        scenario_dict = {
            self.USECASE2: usecase2,
            self.USECASE4_TP_REF: usecase4,
            self.USECASE4_TP1: usecase4,
            self.USECASE4_TP2: usecase4,
            self.USECASE7_TP_REF: usecase7,
            self.USECASE7_TP1: usecase7,
            self.USECASE7_TP2: usecase7,
        }

        # can select a reduced list of scenarios to compute
        scenario_list = list(scenario_dict.keys())
        values_dict = {}

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_list),
                                    'scenario_name': scenario_list})

        # setup mda
        uc_mda = usecase_witness_mda(execution_engine=self.execution_engine)
        uc_mda.study_name = self.study_name  # mda settings on root coupling
        values_dict.update(uc_mda.setup_mda())

        # setup each scenario (mda settings ignored)
        for scenario_name, studyClass in scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario_name}'
            scenarioData = scenarioUseCase.setup_usecase()
            scenarioDatadict = {}
            for data in scenarioData:
                scenarioDatadict.update(data)
            values_dict.update(scenarioDatadict)

        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_list'] = scenario_list
        # assumes max of 16 cores per computational node
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))

        # update values dict with tipping point value of the damage model
        tipping_point_variable = 'Damage.tp_a3'
        values_dict.update({
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4_TP1}.{tipping_point_variable}': self.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4_TP2}.{tipping_point_variable}': self.TIPPING_POINT_LIST[2],
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7_TP1}.{tipping_point_variable}': self.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7_TP2}.{tipping_point_variable}': self.TIPPING_POINT_LIST[2],
            })
        # Inputs were optimized manually through the sostrades GUI and saved in csv files  => recover inputs
        invest_gdp_uc4_tp1 = pd.read_csv(join(dirname(__file__), pardir, 'witness_coarse_dev_story_telling',
                                              'uc4_percentage_of_gdp_energy_invest_tp_4.csv'))
        invest_gdp_uc4_tp2 = pd.read_csv(join(dirname(__file__), pardir, 'witness_coarse_dev_story_telling',
                                              'uc4_percentage_of_gdp_energy_invest_tp_3.csv'))
        invest_gdp_uc7_tp1 = pd.read_csv(join(dirname(__file__), pardir, 'witness_coarse_dev_story_telling',
                                              'uc7_percentage_of_gdp_energy_invest_tp_4.csv'))
        invest_gdp_uc7_tp2 = pd.read_csv(join(dirname(__file__), pardir, 'witness_coarse_dev_story_telling',
                                              'uc7_percentage_of_gdp_energy_invest_tp_3.csv'))
        values_dict.update({
        f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4_TP1}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_gdp_uc4_tp1,
        f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4_TP2}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_gdp_uc4_tp2,
        f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7_TP1}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_gdp_uc7_tp1,
        f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7_TP2}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_gdp_uc7_tp2,
        })

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()