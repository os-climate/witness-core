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
from pathlib import Path

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
from sostrades_core.execution_engine.design_var.design_var_disc import (
    DesignVarDiscipline,
)
from sostrades_core.execution_engine.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)
from climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation_discipline import IEADataPreparationDiscipline
from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2

OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
EXPORT_CSV = FunctionManagerDisc.EXPORT_CSV
WRITE_XVECT = DesignVarDiscipline.WRITE_XVECT
IEA_DISC = IEADataPreparationDiscipline.IEA_NAME

DATA_DIR = Path(__file__).parents[4] / "data"

def create_df_from_csv(filename: str, data_dir=DATA_DIR, **kwargs):
    """Creates a pandas DataFrame from a given filename"""
    return pd.read_csv(str(data_dir / filename), **kwargs)

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
        diet_mortality_df = pd.read_csv(join(dirname(__file__), '../witness_optim_process/data', 'diet_mortality.csv'))

        # clean dspace
        dspace_df.drop(dspace_df.loc[dspace_df['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        # clean dspace descriptor
        dvar_descriptor = self.witness_uc.design_var_descriptor

        updated_dvar_descriptor = {k: v for k, v in dvar_descriptor.items() if k not in list_design_var_to_clean}


        dspace_file_name = 'invest_design_space_NZE.csv'
        dspace_out = pd.read_csv(join(dirname(__file__), '../witness_optim_process/data', dspace_file_name))


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
        invest_mix = pd.read_csv(join(dirname(__file__), '../witness_optim_process/data', invest_mix_file))
        forest_invest_file = 'forest_investment.csv'
        forest_invest = pd.read_csv(join(dirname(__file__), '../witness_optim_process/data', forest_invest_file))
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

        # input for IEA data
        CO2_emissions_df = create_df_from_csv("IEA_NZE_co2_emissions_Gt.csv")
        GDP_df = create_df_from_csv("IEA_NZE_output_net_of_d.csv")
        CO2_tax_df = create_df_from_csv("IEA_NZE_CO2_taxes.csv")
        population_df = create_df_from_csv("IEA_NZE_population.csv")
        temperature_df = create_df_from_csv("IEA_NZE_temp_atmo.csv")
        energy_production_df = create_df_from_csv("IEA_NZE_energy_production_brut.csv")
        nuclear_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.Nuclear.techno_production.csv")
        hydro_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.Hydropower.techno_production.csv")
        solar_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.SolarPv.techno_production.csv")
        wind_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.WindXXshore.techno_production.csv")
        coal_production_df = create_df_from_csv("IEA_NZE_EnergyMix_solid_fuel_CoalExtraction_techno_production.csv")
        fossil_gas_production_df = create_df_from_csv("IEA_NZE_EnergyMix.methane.FossilGas.techno_detailed_production.csv")
        biogas_production_df = create_df_from_csv("IEA_NZE_EnergyMix.biogas.energy_production_detailed.csv")
        crop_production_df = create_df_from_csv("IEA_NZE_crop_mix_detailed_production.csv")
        forest_production_df = create_df_from_csv("IEA_NZE_forest_techno_production.csv")
        electricity_prices_df = create_df_from_csv("IEA_NZE_electricity_Technologies_Mix_prices.csv")
        natural_gas_price_df = create_df_from_csv("IEA_NZE_EnergyMix.methane.FossilGas.techno_prices.csv")
        land_use_df = create_df_from_csv("IEA_NZE_Land_Use.land_surface_detail_df.csv")

        values_dict.update({
            f'{ns}.{GlossaryEnergy.YearStart}': self.year_start,
            f'{ns}.{GlossaryEnergy.YearEnd}': self.year_end,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.CO2EmissionsGtValue}': CO2_emissions_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.EconomicsDfValue}': GDP_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.CO2TaxesValue}': CO2_tax_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.EnergyProductionValue}': energy_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.TemperatureDfValue}': temperature_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.PopulationDfValue}': population_df,
            # energy production
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Nuclear}_techno_production': nuclear_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Hydropower}_techno_production': hydro_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.electricity}_{GlossaryEnergy.Solar}_techno_production': solar_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.electricity}_{GlossaryEnergy.WindOnshoreAndOffshore}_techno_production': wind_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.solid_fuel}_{GlossaryEnergy.CoalExtraction}_techno_production': coal_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.methane}_{GlossaryEnergy.FossilGas}_techno_production': fossil_gas_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.biogas}_{GlossaryEnergy.AnaerobicDigestion}_techno_production': biogas_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.CropEnergy}_techno_production': crop_production_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.ForestProduction}_techno_production': forest_production_df,
            f'{ns}.{IEA_DISC}.{LandUseV2.LAND_SURFACE_DETAIL_DF}': land_use_df,
            # energy prices
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.electricity}_energy_prices': electricity_prices_df,
            f'{ns}.{IEA_DISC}.{GlossaryEnergy.methane}_{GlossaryEnergy.EnergyPricesValue}': natural_gas_price_df
            })

        return [values_dict] + [optim_values_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
    '''
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    ns = f'usecase_witness_optim_nze_eval'
    filters = ppf.get_post_processing_filters_by_namespace(uc_cls.ee, ns)

    graph_list = ppf.get_post_processing_by_namespace(uc_cls.ee, ns, filters, as_json=False)
    for graph in graph_list:
        graph.to_plotly().show()
    '''