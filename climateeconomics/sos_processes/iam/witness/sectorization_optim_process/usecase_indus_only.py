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
from os.path import dirname, join

import numpy as np
import pandas as pd
from numpy import arange
from scipy.interpolate import interp1d

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization_process.usecase import (
    Study as witness_sect_usecase,
)
from sostrades_optimization_plugins.models.func_manager.func_manager import FunctionManager
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)
from sostrades_core.study_manager.study_manager import StudyManager

AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(StudyManager):

    def __init__(self, year_start=2000, year_end=GlossaryCore.YearStartDefault, time_step=1, execution_engine=None, run_usecase=False):
        super().__init__(__file__, execution_engine=execution_engine, run_usecase=run_usecase)
        self.study_name = 'usecase_indus_only'
        self.macro_name = 'Macroeconomics'
        self.obj_name = 'Objectives'
        self.coupling_name = "Sectorization_Eval"
        self.optim_name = "SectorsOpt"
        self.ns_industry = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{GlossaryCore.SectorIndustry}"
        self.ns_agriculture = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{GlossaryCore.SectorAgriculture}"
        self.ns_services = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{GlossaryCore.SectorServices}"
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.witness_sect_uc = witness_sect_usecase(self.year_start, self.year_end, self.time_step,
                                                    execution_engine=execution_engine, main_study=False)

    def setup_usecase(self, study_folder_path=None):
        ns_coupling = f"{self.study_name}.{self.optim_name}.{self.coupling_name}"
        ns_optim = f"{self.study_name}.{self.optim_name}"
        # Optim param
        OBJECTIVE = FunctionManager.OBJECTIVE

        dspace_dict = {'variable': ['output_alpha_indus_in', 'prod_gr_start_indus_in', 'decl_rate_tfp_indus_in',
                                    'prod_start_indus_in',
                                    'energy_eff_k_indus_in', 'energy_eff_cst_indus_in', 'energy_eff_xzero_indus_in',
                                    'energy_eff_max_indus_in',
                                    ],
                       'value': [[0.72], [0.001], [0.071], [0.203],
                                 [0.052], [0.1657], [2015.0], [4.194],
                                 ],
                       'lower_bnd': [[0.5], [0.001], [0.00001], [0.01],
                                     [0.0], [1.0e-5], [1900.0], [1.0],
                                     ],
                       'upper_bnd': [[0.99], [0.1], [0.1], [2.0],
                                     [0.09], [2.0], [2015.0], [8.0],
                                     ],
                       'enable_variable':  # design variables ON
                           [True] * 8,
                       'activated_elem': [[True], [True], [True], [True],
                                          [True], [True], [True], [True],
                                          ]
                       }

        dspace = pd.DataFrame(dspace_dict)

        industry = 'Industry.'
        design_var_descriptor = {
            'output_alpha_indus_in': {'out_name': industry + 'output_alpha', 'out_type': 'float', 'index': arange(1),
                                      'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'prod_gr_start_indus_in': {'out_name': industry + 'productivity_gr_start', 'out_type': 'float',
                                       'index': arange(1),
                                       'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'decl_rate_tfp_indus_in': {'out_name': industry + 'decline_rate_tfp', 'out_type': 'float',
                                       'index': arange(1),
                                       'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'prod_start_indus_in': {'out_name': industry + 'productivity_start', 'out_type': 'float',
                                    'index': arange(1),
                                    'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'energy_eff_k_indus_in': {'out_name': industry + 'energy_eff_k', 'out_type': 'float', 'index': arange(1),
                                      'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'energy_eff_cst_indus_in': {'out_name': industry + 'energy_eff_cst', 'out_type': 'float',
                                        'index': arange(1),
                                        'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'energy_eff_xzero_indus_in': {'out_name': industry + 'energy_eff_xzero', 'out_type': 'float',
                                          'index': arange(1),
                                          'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            'energy_eff_max_indus_in': {'out_name': industry + 'energy_eff_max', 'out_type': 'float',
                                        'index': arange(1),
                                        'namespace_in': 'ns_optim', 'namespace_out': GlossaryCore.NS_MACRO},
            }

        disc_dict = {}
        disc_dict[f'{ns_coupling}.DesignVariables.design_var_descriptor'] = design_var_descriptor

        # Optim inputs
        disc_dict[f"{ns_optim}.{'max_iter'}"] = 400
        disc_dict[f"{ns_optim}.{'algo'}"] = "L-BFGS-B"
        disc_dict[f"{ns_optim}.{'design_space'}"] = dspace
        disc_dict[f"{ns_optim}.{'formulation'}"] = 'DisciplinaryOpt'
        disc_dict[f"{ns_optim}.{'objective_name'}"] = 'objective_lagrangian'
        disc_dict[f"{ns_optim}.{'differentiation_method'}"] = 'complex_step'  # complex_step user
        disc_dict[f"{ns_optim}.{'fd_step'}"] = 1.e-15
        disc_dict[f"{ns_optim}.{'ineq_constraints'}"] = []
        disc_dict[f"{ns_optim}.{'eq_constraints'}"] = []
        disc_dict[f"{ns_optim}.{'algo_options'}"] = {
            "maxls_step_nb": 48,
            "maxcor": 24,
            "ftol_rel": 1e-15,
            "pg_tol": 1e-8
        }

        # design var inputs
        disc_dict[f"{ns_optim}.{'output_alpha_indus_in'}"] = np.array([0.715])
        disc_dict[f"{ns_optim}.{'prod_gr_start_indus_in'}"] = np.array([0.001])
        disc_dict[f"{ns_optim}.{'decl_rate_tfp_indus_in'}"] = np.array([0.071])
        disc_dict[f"{ns_optim}.{'prod_start_indus_in'}"] = np.array([0.203])
        disc_dict[f"{ns_optim}.{'energy_eff_k_indus_in'}"] = np.array([0.051])
        disc_dict[f"{ns_optim}.{'energy_eff_cst_indus_in'}"] = np.array([0.16])
        disc_dict[f"{ns_optim}.{'energy_eff_xzero_indus_in'}"] = np.array([2015.0])
        disc_dict[f"{ns_optim}.{'energy_eff_max_indus_in'}"] = np.array([4.19])
        disc_dict[f"{ns_coupling}.{'workforce_df'}"] = 0

        func_df = pd.DataFrame(
            columns=['variable', 'ftype', 'weight', AGGR_TYPE, 'namespace'])
        func_df['variable'] = ['Industry.gdp_error',
                               'Industry.energy_eff_error',
                               ]
        func_df['ftype'] = [OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [1, 1]
        func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM]
        func_df['namespace'] = ['ns_obj', 'ns_obj']
        func_mng_name = 'FunctionsManager'

        prefix = f'{ns_coupling}.{func_mng_name}.'
        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        disc_dict.update(values_dict)

        # Inputs for objective 
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data/sectorization_fitting')
        hist_gdp = pd.read_csv(join(data_dir, 'hist_gdp_sect.csv'))
        hist_capital = pd.read_csv(join(data_dir, 'hist_capital_sect.csv'))
        hist_energy = pd.read_csv(join(data_dir, 'hist_energy_sect.csv'))
        long_term_energy_eff = pd.read_csv(join(data_dir, 'long_term_energy_eff_sectors.csv'))
        lt_enef_agri = pd.DataFrame({GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                                     GlossaryCore.EnergyEfficiency: long_term_energy_eff[
                                         GlossaryCore.SectorAgriculture]})
        lt_enef_indus = pd.DataFrame({GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                                      GlossaryCore.EnergyEfficiency: long_term_energy_eff[GlossaryCore.SectorIndustry]})
        lt_enef_services = pd.DataFrame({GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                                         GlossaryCore.EnergyEfficiency: long_term_energy_eff[
                                             GlossaryCore.SectorServices]})

        n_years = len(long_term_energy_eff)
        workforce_df = pd.DataFrame({
            GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
            GlossaryCore.SectorIndustry: np.ones(n_years) * 1000,
            GlossaryCore.SectorServices: np.ones(n_years) * 1000,
            GlossaryCore.SectorAgriculture: np.ones(n_years) * 1000,
        })
        ns_industry_macro = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.macro_name}.{GlossaryCore.SectorIndustry}"
        ns_agriculture_macro = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.macro_name}.{GlossaryCore.SectorAgriculture}"
        ns_services_macro = f"{self.study_name}.{self.optim_name}.{self.coupling_name}.{self.macro_name}.{GlossaryCore.SectorServices}"
        sect_input = {}
        sect_input[f"{ns_coupling}.{self.obj_name}.{'historical_gdp'}"] = hist_gdp
        sect_input[f"{ns_coupling}.{self.obj_name}.{'historical_capital'}"] = hist_capital
        sect_input[f"{ns_coupling}.{self.obj_name}.{'historical_energy'}"] = hist_energy
        sect_input[f"{ns_coupling}.{self.macro_name}.{'prod_function_fitting'}"] = False
        sect_input[f"{ns_industry_macro}.{'longterm_energy_efficiency'}"] = lt_enef_indus
        sect_input[f"{ns_agriculture_macro}.{'longterm_energy_efficiency'}"] = lt_enef_agri
        sect_input[f"{ns_services_macro}.{'longterm_energy_efficiency'}"] = lt_enef_services
        sect_input[f"{ns_coupling}.{'workforce_df'}"] = workforce_df
        disc_dict.update(sect_input)

        self.witness_sect_uc.study_name = f'{ns_coupling}'
        witness_sect_uc_data = self.witness_sect_uc.setup_usecase()
        for dict_data in witness_sect_uc_data:
            disc_dict.update(dict_data)

        #################
        # add inputs that are now computed in "redistribution" disciplines
        years = np.arange(self.year_start, self.year_end + 1, 1)

        # Energy
        brut_net = 1 / 1.45
        energy_outlook = pd.DataFrame({
            'year': [2000, 2005, 2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050, 2060, 2100],
            'energy': [118.112, 134.122, 149.483879, 162.7848774, 166.4685636, 180.7072889, 189.6932084, 197.8418842,
                       206.1201182, 220.000, 250.0, 300.0]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        # Find values for 2020, 2050 and concat dfs
        energy_supply = f2(np.arange(self.year_start, self.year_end + 1))
        energy_supply_values = energy_supply * brut_net

        indus_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.2894})
        agri_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.02136})
        services_energy = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: energy_supply_values * 0.37})

        invest_indus = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40,65, len(years))*1/3})

        invest_services = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, len(years)) * 1/6})

        invest_agriculture = pd.DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.InvestmentsValue: np.linspace(40, 65, len(years))* 1/2})

        sect_input = {}
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.InvestmentDfValue}"] = invest_indus
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}"] = invest_agriculture
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.InvestmentDfValue}"] = invest_services
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.EnergyProductionValue}"] = indus_energy
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}"] = agri_energy
        sect_input[f"{ns_coupling}.{self.macro_name}.{GlossaryCore.SectorServices}.{GlossaryCore.EnergyProductionValue}"] = services_energy

        disc_dict.update(sect_input)


        return [disc_dict]



if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()