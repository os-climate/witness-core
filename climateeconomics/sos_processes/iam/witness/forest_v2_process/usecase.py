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
from climateeconomics.core.core_forest.forest_v2 import Forest

from pathlib import Path
from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


def update_dspace_with(dspace_dict, name, value, lower, upper):
    ''' type(value) has to be ndarray
    '''
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)
    dspace_dict['variable'].append(name)
    dspace_dict['value'].append(value.tolist())
    dspace_dict['lower_bnd'].append(lower)
    dspace_dict['upper_bnd'].append(upper)
    dspace_dict['dspace_size'] += len(value)


def update_dspace_dict_with(dspace_dict, name, value, lower, upper, activated_elem=None, enable_variable=True):
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)

    if activated_elem is None:
        activated_elem = [True] * len(value)
    dspace_dict[name] = {'value': value,
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable, 'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='.Land.Forest', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.forest_name = name
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):

        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        deforestation_limit = 1000

        initial_emissions = 3.21

        CO2_per_ha = 4000

        self.deforestation_surface = np.linspace(10, 5, year_range)
        deforestation_surface_df = pd.DataFrame(
            {"years": years, "deforested_surface": self.deforestation_surface})

        forest_invest = np.linspace(5, 8, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})
        reforestation_cost_per_ha = 3800

        wood_density = 600.0  # kg/m3
        residues_density = 200.0  # kg/m3
        residue_density_m3_per_ha = 46.5
        # average of 360 and 600 divided by 5
        wood_density_m3_per_ha = 96
        construction_delay = 3
        wood_residue_price_percent_dif = 0.34
        wood_percentage_for_energy = 0.48
        residue_percentage_for_energy = 0.48

        density_per_ha = residue_density_m3_per_ha + \
            wood_density_m3_per_ha

        wood_percentage = wood_density_m3_per_ha / density_per_ha
        residue_percentage = residue_density_m3_per_ha / density_per_ha

        mean_density = wood_percentage * wood_density + \
            residue_percentage * residues_density
        years_between_harvest = 20

        recycle_part = 0.52  # 52%
        wood_techno_dict = {'maturity': 5,
                            'wood_residues_moisture': 0.35,  # 35% moisture content
                            'wood_residue_colorific_value': 4.356,
                            'Opex_percentage': 0.045,
                            'managed_wood_price_per_ha': 14872,  # 13047,
                            'unmanaged_wood_price_per_ha': 11500,  # 10483,
                            'Price_per_ha_unit': 'euro/ha',
                            'full_load_hours': 8760.0,
                            'euro_dollar': 1.1447,  # in 2019, date of the paper
                            'percentage_production': 0.52,
                            'residue_density_percentage': residue_percentage,
                            'non_residue_density_percentage': wood_percentage,
                            'density_per_ha': density_per_ha,
                            'wood_percentage_for_energy': wood_percentage_for_energy,
                            'residue_percentage_for_energy': residue_percentage_for_energy,
                            'density': mean_density,
                            'wood_density': wood_density,
                            'residues_density': residues_density,
                            'density_per_ha_unit': 'm^3/ha',
                            'techno_evo_eff': 'no',  # yes or no
                            'years_between_harvest': years_between_harvest,
                            'wood_residue_price_percent_dif': wood_residue_price_percent_dif,
                            'recycle_part': recycle_part,
                            'construction_delay': construction_delay,
                            'WACC': 0.07
                            }
        invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0), 'investment': [1.135081, 1.135081, 1.135081]})
        mw_initial_production = 1.25 * 0.92 * \
            density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)  # in Twh

        uw_initial_production = 1.25 * 0.08 * \
            density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)

        mw_invest = np.linspace(1, 4, year_range)
        mw_invest_df = pd.DataFrame(
            {"years": years, "investment": mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        transport_df = pd.DataFrame(
            {"years": years, "transport": transport})
        margin = np.linspace(1.1, 1.1, year_range)
        margin_df = pd.DataFrame(
            {"years": years, "margin": margin})
        initial_unsused_forest_surface = 4 - 1.25

        # values of model
        forest_input = {}
        forest_input[self.study_name + '.year_start'] = self.year_start
        forest_input[self.study_name + '.year_end'] = self.year_end

        forest_input[self.study_name + self.forest_name +
                     '.limit_deforestation_surface'] = deforestation_limit

        forest_input[self.study_name + self.forest_name +
                     '.CO2_per_ha'] = CO2_per_ha

        forest_input[self.study_name + self.forest_name +
                     '.initial_emissions'] = initial_emissions
        forest_input[self.study_name + self.forest_name +
                     '.reforestation_cost_per_ha'] = reforestation_cost_per_ha

        forest_input[self.study_name +
                     '.deforestation_surface'] = deforestation_surface_df
        forest_input[self.study_name +
                     '.forest_investment'] = self.forest_invest_df

        forest_input[self.study_name + self.forest_name +
                     '.wood_techno_dict'] = wood_techno_dict
        forest_input[self.study_name + self.forest_name +
                     '.managed_wood_initial_prod'] = mw_initial_production
        forest_input[self.study_name + self.forest_name +
                     '.managed_wood_initial_surface'] = 1.25 * 0.92
        forest_input[self.study_name + self.forest_name +
                     '.managed_wood_invest_before_year_start'] = invest_before_year_start
        forest_input[self.study_name +
                     '.managed_wood_investment'] = mw_invest_df
        forest_input[self.study_name + self.forest_name +
                     '.unmanaged_wood_initial_prod'] = uw_initial_production
        forest_input[self.study_name + self.forest_name +
                     '.unmanaged_wood_initial_surface'] = 1.25 * 0.08
        forest_input[self.study_name + self.forest_name +
                     '.unmanaged_wood_invest_before_year_start'] = invest_before_year_start
        forest_input[self.study_name +
                     '.unmanaged_wood_investment'] = mw_invest_df
        forest_input[self.study_name + self.forest_name +
                     '.transport_cost'] = transport_df
        forest_input[self.study_name + self.forest_name +
                     '.margin'] = margin_df
        forest_input[self.study_name + self.forest_name +
                     '.initial_unsused_forest_surface'] = initial_unsused_forest_surface

        setup_data_list.append(forest_input)

        deforestation_surface_ctrl = np.linspace(10.0, 5.0, self.nb_poles)
        forest_investment_array_mix = np.linspace(5.0, 8.0, self.nb_poles)

        design_space_ctrl_dict = {}
        design_space_ctrl_dict['deforested_surface_ctrl'] = deforestation_surface_ctrl
        design_space_ctrl_dict['forest_investment_array_mix'] = forest_investment_array_mix

        design_space_ctrl = pd.DataFrame(design_space_ctrl_dict)
        self.design_space_ctrl = design_space_ctrl
        self.dspace = self.setup_design_space_ctrl_new()
        return setup_data_list

    def setup_initial_design_variable(self):

        init_design_var_df = pd.DataFrame(
            columns=['deforested_surface', 'forest_investment'], index=arange(self.year_start, self.year_end + 1, self.time_step))

        init_design_var_df['deforested_surface'] = self.deforestation_surface['deforested_surface']
        init_design_var_df['forest_investment'] = self.forest_invest_df['forest_investment']

        return init_design_var_df

    def setup_design_space(self):
        #-- energy optimization inputs
        # Design Space
        dim_a = len(
            self.deforestation_surface['deforested_surface'].values)
        lbnd1 = [0.0] * dim_a
        ubnd1 = [100.0] * dim_a

        # Design variables:
        self.update_dspace_dict_with(
            'deforested_surface_array', self.deforestation_surface['deforested_surface'].values, lbnd1, ubnd1)
        self.update_dspace_dict_with(
            'forest_investment_array', self.forest_invest_df['forest_investment'].values, lbnd1, ubnd1)

    def setup_design_space_ctrl_new(self):
        # Design Space
        # header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables:
        update_dspace_dict_with(ddict, 'deforested_surface_ctrl',
                                list(self.design_space_ctrl['deforested_surface_ctrl'].values), [0.0] * self.nb_poles, [100.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True, True])

        update_dspace_dict_with(ddict, 'forest_investment_array_mix',
                                list(self.design_space_ctrl['forest_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True, True])
        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    # ppf = PostProcessingFactory()
    # for disc in uc_cls.execution_engine.root_process.sos_disciplines:
    #     filters = ppf.get_post_processing_filters_by_discipline(
    #         disc)
    #     graph_list = ppf.get_post_processing_by_discipline(
    #         disc, filters, as_json=False)

    #     for graph in graph_list:
    #         graph.to_plotly().show()
