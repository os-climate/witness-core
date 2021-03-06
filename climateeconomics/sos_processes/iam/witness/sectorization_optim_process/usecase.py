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

from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.study_manager.study_manager import StudyManager

from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from climateeconomics.sos_processes.iam.witness.sectorization_process.usecase import Study as witness_sect_usecase
from gemseo.api import generate_n2_plot

AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(StudyManager):

    def __init__(self, year_start=2000, year_end=2020, time_step=1, name='', execution_engine=None, run_usecase=False):
        super().__init__(__file__, execution_engine=execution_engine, run_usecase=run_usecase)
        self.study_name = 'usecase'
        self.macro_name = '.Macroeconomics'
        self.obj_name = '.Objectives'
        self.coupling_name = ".Sectorization_Eval"
        self.optim_name = ".SectorsOpt"
        self.ns_industry = self.study_name + self.optim_name + self.coupling_name  + self.macro_name +'.Industry'
        self.ns_agriculture = self.study_name + self.optim_name + self.coupling_name + self.macro_name + '.Agriculture'
        self.ns_services = self.study_name + self.optim_name + self.coupling_name + self.macro_name + '.Services'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.witness_sect_uc = witness_sect_usecase(self.year_start, self.year_end, self.time_step, execution_engine=execution_engine)


    def setup_usecase(self):
        
        ns = self.study_name
        ns_coupling = self.study_name + self.optim_name + self.coupling_name
        ns_optim = self.study_name + self.optim_name 
        #Optim param
        INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
        OBJECTIVE = FunctionManager.OBJECTIVE

        dspace_dict = {'variable': ['output_alpha_services_in', 'prod_gr_start_services_in','decl_rate_tfp_services_in','prod_start_services_in', 
                                    'energy_eff_k_services_in', 'energy_eff_cst_services_in', 'energy_eff_xzero_services_in','energy_eff_max_services_in',
                                    'output_alpha_agri_in', 'prod_gr_start_agri_in','decl_rate_tfp_agri_in','prod_start_agri_in', 
                                    'energy_eff_k_agri_in', 'energy_eff_cst_agri_in', 'energy_eff_xzero_agri_in','energy_eff_max_agri_in',
                                    'output_alpha_indus_in', 'prod_gr_start_indus_in','decl_rate_tfp_indus_in','prod_start_indus_in', 
                                    'energy_eff_k_indus_in', 'energy_eff_cst_indus_in', 'energy_eff_xzero_indus_in','energy_eff_max_indus_in',
                                    ],
                       'value': [[0.77], [0.01], [0.03],[0.19], 
                                 [0.04], [0.98], [2004.0], [8.0], 
                                 [0.99], [0.05], [0.07],[1.08], 
                                 [0.02], [0.08], [2030.0], [5.92], 
                                 [0.71], [0.001], [0.02],[0.20], 
                                 [0.05], [0.16], [2015.0], [4.18]],
                       'lower_bnd': [[0.5], [0.001], [0.00001], [0.01],
                                     [0.0], [1e-5],[1900.0], [1.0],
                                      [0.5], [0.001], [0.00001], [0.01],
                                      [0.0], [1e-5],[1900.0], [1.0],
                                      [0.5], [0.001], [0.00001], [0.01],
                                      [0.0], [1e-5],[1900.0], [1.0]],
                       'upper_bnd': [[0.99], [0.07], [0.1],[2.0],
                                      [1.0], [2.0],[2050.0], [15.0],
                                      [0.99], [0.1], [0.1],[2.0],
                                      [1.0], [2.0],[2050.0], [8.0],
                                      [0.99], [0.1], [0.1],[2.0],
                                      [1.0], [2.0],[2015.0], [8.0]],
                       'enable_variable': [ True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True],
                       'activated_elem': [[True], [True], [True], [True],
                                           [True], [True],[True], [True],
                                           [True], [True], [True], [True],
                                           [True], [True],[True], [True],
                                           [True], [True], [True], [True],
                                           [True], [True],[True], [True]]
                       }
 
        dspace = pd.DataFrame(dspace_dict)
        
        services = 'Services.'
        agri = 'Agriculture.'
        industry = 'Industry.'
        design_var_descriptor = {'output_alpha_services_in': {'out_name': services+'output_alpha','out_type': 'float', 'index': arange(1), 
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_gr_start_services_in': {'out_name': services+'productivity_gr_start','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'decl_rate_tfp_services_in': {'out_name': services+'decline_rate_tfp','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_start_services_in': {'out_name': services+'productivity_start', 'out_type': 'float','index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_k_services_in': {'out_name': services+'energy_eff_k', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_cst_services_in': {'out_name': services+'energy_eff_cst', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_xzero_services_in': {'out_name': services+'energy_eff_xzero', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_max_services_in': {'out_name': services+'energy_eff_max', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'output_alpha_agri_in': {'out_name': agri+'output_alpha','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_gr_start_agri_in': {'out_name': agri+'productivity_gr_start','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'decl_rate_tfp_agri_in': {'out_name': agri+'decline_rate_tfp','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_start_agri_in': {'out_name': agri+'productivity_start', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_k_agri_in': {'out_name': agri+'energy_eff_k', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_cst_agri_in': {'out_name': agri+'energy_eff_cst', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_xzero_agri_in': {'out_name': agri+'energy_eff_xzero', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_max_agri_in': {'out_name': agri+'energy_eff_max', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'output_alpha_indus_in': {'out_name': industry+'output_alpha','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_gr_start_indus_in': {'out_name': industry+'productivity_gr_start','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'decl_rate_tfp_indus_in': {'out_name': industry+'decline_rate_tfp','out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'prod_start_indus_in': {'out_name': industry+'productivity_start', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_k_indus_in': {'out_name': industry+'energy_eff_k', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_cst_indus_in': {'out_name': industry+'energy_eff_cst', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_xzero_indus_in': {'out_name': industry+'energy_eff_xzero', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},
                                'energy_eff_max_indus_in': {'out_name': industry+'energy_eff_max', 'out_type': 'float', 'index': arange(1),
                                                              'namespace_in': 'ns_optim','namespace_out': 'ns_macro'},         
                             }

        disc_dict = {}
        disc_dict[f'{ns_coupling}.DesignVariables.design_var_descriptor'] = design_var_descriptor

        # Optim inputs
        disc_dict[f'{ns_optim}.max_iter'] = 300
        disc_dict[f'{ns_optim}.algo'] = "L-BFGS-B"
        disc_dict[f'{ns_optim}.design_space'] = dspace
        disc_dict[f'{ns_optim}.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{ns_optim}.objective_name'] = 'objective_lagrangian'
        disc_dict[f'{ns_optim}.differentiation_method'] = 'complex_step' #complex_step user
        disc_dict[f'{ns_optim}.fd_step'] = 1.e-15
        disc_dict[f'{ns_optim}.ineq_constraints'] = []
        disc_dict[f'{ns_optim}.eq_constraints'] = []
        disc_dict[f'{ns_optim}.algo_options'] = {
            "maxls_step_nb": 48,
            "maxcor": 24,
            "ftol_rel": 1e-15,
            "pg_tol": 1e-8
        }
        
    # design var inputs
        disc_dict[f'{ns_optim}.output_alpha_services_in'] = ([0.77])
        disc_dict[f'{ns_optim}.prod_gr_start_services_in'] = ([0.01])
        disc_dict[f'{ns_optim}.decl_rate_tfp_services_in'] = ([0.03])
        disc_dict[f'{ns_optim}.prod_start_services_in'] = ([0.19])
        disc_dict[f'{ns_optim}.energy_eff_k_services_in'] = ([0.04])
        disc_dict[f'{ns_optim}.energy_eff_cst_services_in'] = ([0.98])
        disc_dict[f'{ns_optim}.energy_eff_xzero_services_in'] = ([2004.0])
        disc_dict[f'{ns_optim}.energy_eff_max_services_in'] = ([8.0])
        
        disc_dict[f'{ns_optim}.output_alpha_agri_in'] = ([0.99])
        disc_dict[f'{ns_optim}.prod_gr_start_agri_in'] = ([0.04])
        disc_dict[f'{ns_optim}.decl_rate_tfp_agri_in'] = ([0.07])
        disc_dict[f'{ns_optim}.prod_start_agri_in'] = ([1.08])
        disc_dict[f'{ns_optim}.energy_eff_k_agri_in'] = ([0.02])
        disc_dict[f'{ns_optim}.energy_eff_cst_agri_in'] = ([0.08])
        disc_dict[f'{ns_optim}.energy_eff_xzero_agri_in'] = ([2030.0])
        disc_dict[f'{ns_optim}.energy_eff_max_agri_in'] = ([5.9])
        
        disc_dict[f'{ns_optim}.output_alpha_indus_in'] = ([0.71])
        disc_dict[f'{ns_optim}.prod_gr_start_indus_in'] = ([0.001])
        disc_dict[f'{ns_optim}.decl_rate_tfp_indus_in'] = ([0.02])
        disc_dict[f'{ns_optim}.prod_start_indus_in'] = ([0.20])
        disc_dict[f'{ns_optim}.energy_eff_k_indus_in'] = ([0.05])
        disc_dict[f'{ns_optim}.energy_eff_cst_indus_in'] = ([0.16])
        disc_dict[f'{ns_optim}.energy_eff_xzero_indus_in'] = ([2015])
        disc_dict[f'{ns_optim}.energy_eff_max_indus_in'] = ([4.18])

        
        func_df = pd.DataFrame(
            columns=['variable', 'ftype', 'weight', AGGR_TYPE, 'namespace'])
        func_df['variable'] = [ 'error_pib_total',
                               'Industry.gdp_error', 'Agriculture.gdp_error', 'Services.gdp_error',
                               'Industry.energy_eff_error', 'Agriculture.energy_eff_error', 'Services.energy_eff_error',
                               'Industry.range_energy_eff_constraint', 'Agriculture.range_energy_eff_constraint', 
                               'Services.range_energy_eff_constraint']
        func_df['ftype'] =  [OBJECTIVE,OBJECTIVE, OBJECTIVE,OBJECTIVE, OBJECTIVE,OBJECTIVE, OBJECTIVE,
                             INEQ_CONSTRAINT, INEQ_CONSTRAINT, INEQ_CONSTRAINT]
        func_df['weight'] = [1,1,1,1,1,1,1,-1, -1, -1]
        func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, 
                              AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM, 
                              AGGR_TYPE_SUM, AGGR_TYPE_SUM,  AGGR_TYPE_SUM]
        func_df['namespace'] = ['ns_obj', 'ns_obj', 'ns_obj', 'ns_obj', 'ns_obj', 'ns_obj', 'ns_obj',
                                'ns_macro', 'ns_macro', 'ns_macro']
        func_mng_name = 'FunctionsManager'

        prefix = f'{ns_coupling}.{func_mng_name}.'
        values_dict = {}
        values_dict[prefix +FunctionManagerDisc.FUNC_DF] = func_df

        disc_dict.update(values_dict)     

        # Inputs for objective 
        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data/sectorization_fitting')
        hist_gdp = pd.read_csv(join(data_dir, 'hist_gdp_sect.csv'))
        hist_capital = pd.read_csv(join(data_dir, 'hist_capital_sect.csv'))
        hist_energy = pd.read_csv(join(data_dir, 'hist_energy_sect.csv'))
        sect_input = {}
        sect_input[ns_coupling + self.obj_name +  '.historical_gdp'] = hist_gdp
        sect_input[ns_coupling + self.obj_name +  '.historical_capital'] = hist_capital
        sect_input[ns_coupling + self.obj_name +  '.historical_energy'] = hist_energy
        sect_input[ns_coupling + self.macro_name + '.prod_function_fitting'] = True
        disc_dict.update(sect_input)
        
        self.witness_sect_uc.study_name = f'{ns_coupling}'
        witness_sect_uc_data = self.witness_sect_uc.setup_usecase()
        for dict_data in witness_sect_uc_data:
            disc_dict.update(dict_data)
    
        return [disc_dict]


#---------------- SPECIFIC CODE TO ENABLE COMPLEX AS REAL INTO PLOTLY CHARTS
import json as _json
import plotly


class ComplexJsonEncoder(_json.JSONEncoder):
    def default(self, o):# pylint: disable=E0202

        if isinstance(o, np.complex):
            return o.real

        # default, if not one of the specified object. Caller's problem if this is not
        # serializable.
        return _json.JSONEncoder.default(self, o)
    
#---------------- SPECIFIC CODE TO ENABLE COMPLEX AS REAL INTO PLOTLY CHARTS

if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)

    #uc_cls.execution_engine.set_debug_mode()
#     generate_n2_plot(uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].sos_disciplines)
#     uc_cls.execution_engine.dm.export_couplings(in_csv=True, f_name='couplings.csv')
    uc_cls.run()
    
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].sos_disciplines:
        if disc.sos_name == 'Objectives':
            filters = ppf.get_post_processing_filters_by_discipline(disc)
            graph_list = ppf.get_post_processing_by_discipline(
                disc, filters, as_json=False)

            for graph in graph_list:

                # Get chart as plotly dict instead of plotly json to avoid complex type error
                d = graph.to_plotly_dict()
                # Convert dict into json using a custom encoder that manage complex type
                j = _json.dumps(d, cls=ComplexJsonEncoder)
                # Set up a new plotly object using this generated json
                p = plotly.io.from_json(j)
                # display the plotly chart
                #p.show()

                #g = graph.to_plotly()
                #g.show()
    