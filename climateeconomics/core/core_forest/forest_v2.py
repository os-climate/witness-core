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

import numpy as np
import pandas as pd
from copy import deepcopy
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry


class Forest():
    """
    Forest model class 
    basic for now, to evolve 

    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    LIMIT_DEFORESTATION_SURFACE = 'limit_deforestation_surface'
    DEFORESTATION_SURFACE = 'deforestation_surface'
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'
    REFORESTATION_INVESTMENT = 'forest_investment'
    REFORESTATION_COST_PER_HA = 'reforestation_cost_per_ha'
    WOOD_TECHNO_DICT = 'wood_techno_dict'
    MW_INITIAL_PROD = 'managed_wood_initial_prod'
    MW_INITIAL_SURFACE = 'managed_wood_initial_surface'
    MW_INVEST_BEFORE_YEAR_START = 'managed_wood_invest_before_year_start'
    MW_INVESTMENT = 'managed_wood_investment'
    UW_INITIAL_PROD = 'unmanaged_wood_initial_prod'
    UW_INITIAL_SURFACE = 'unmanaged_wood_initial_surface'
    UW_INVEST_BEFORE_YEAR_START = 'unmanaged_wood_invest_before_year_start'
    UW_INVESTMENT = 'unmanaged_wood_investment'
    TRANSPORT_COST = 'transport_cost'
    MARGIN = 'margin'
    UNUSED_FOREST = 'initial_unsused_forest_surface'

    FOREST_SURFACE_DF = 'forest_surface_df'
    FOREST_DETAIL_SURFACE_DF = 'forest_surface_detail_df'
    CO2_EMITTED_FOREST_DF = 'CO2_emitted_forest_df'
    CO2_EMITTED_DETAIL_DF = 'CO2_emissions_detail_df'
    MW_DF = 'managed_wood_df'
    UW_DF = 'unmanaged_wood_df'
    BIOMASS_DRY_DETAIL_DF = 'biomass_dry_detail_df'
    BIOMASS_DRY_DF = 'biomass_dry_df'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        """
        """
        self.year_start = self.param[self.YEAR_START]
        self.year_end = self.param[self.YEAR_END]
        self.time_step = self.param[self.TIME_STEP]
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        self.deforestation_surface = self.param[self.DEFORESTATION_SURFACE]
        self.CO2_per_ha = self.param[self.CO2_PER_HA]
        # initial CO2 emissions
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        # forest data
        self.forest_investment = self.param[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = self.param[self.REFORESTATION_COST_PER_HA]
        self.techno_wood_info = self.param[self.WOOD_TECHNO_DICT]
        self.managed_wood_inital_prod = self.param[self.MW_INITIAL_PROD]
        self.managed_wood_initial_surface = self.param[self.MW_INITIAL_SURFACE]
        self.managed_wood_invest_before_year_start = self.param[
            self.MW_INVEST_BEFORE_YEAR_START]
        self.managed_wood_investment = self.param[self.MW_INVESTMENT]
        self.unmanaged_wood_inital_prod = self.param[self.UW_INITIAL_PROD]
        self.unmanaged_wood_initial_surface = self.param[self.UW_INITIAL_SURFACE]
        self.unmanaged_wood_invest_before_year_start = self.param[
            self.UW_INVEST_BEFORE_YEAR_START]
        self.unmanaged_wood_investment = self.param[self.UW_INVESTMENT]
        self.transport = self.param[self.TRANSPORT_COST]
        self.margin = self.param[self.MARGIN]
        self.initial_unsused_forest_surface = self.param[self.UNUSED_FOREST]
        self.scaling_factor_techno_consumption = self.param['scaling_factor_techno_consumption']
        self.scaling_factor_techno_production = self.param['scaling_factor_techno_production']

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.forest_surface_df = pd.DataFrame({'years': self.years})
        self.CO2_emitted_df = pd.DataFrame({'years': self.years})
        self.managed_wood_df = pd.DataFrame({'years': self.years})
        self.unmanaged_wood_df = pd.DataFrame({'years': self.years})
        self.biomass_dry_df = pd.DataFrame({'years': self.years})

        # output dataframes:
        self.techno_production = pd.DataFrame({'years': self.years})
        self.techno_prices = pd.DataFrame({'years': self.years})
        self.techno_consumption = pd.DataFrame({'years': self.years})
        self.techno_consumption_woratio = pd.DataFrame({'years': self.years})
        self.land_use_required = pd.DataFrame({'years': self.years})
        self.CO2_emissions = pd.DataFrame({'years': self.years})
        self.lost_capital = pd.DataFrame({'years': self.years})

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.biomass_dry_calorific_value = BiomassDry.data_energy_dict[
            'calorific_value']  # kwh/kg
        # calorific value to be taken from factorised info
        self.deforestation_surface = in_dict[self.DEFORESTATION_SURFACE]
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.forest_investment = in_dict[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = in_dict[self.REFORESTATION_COST_PER_HA]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        self.years = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        self.managed_wood_investment = in_dict[self.MW_INVESTMENT]
        self.unmanaged_wood_investment = in_dict[self.UW_INVESTMENT]

        # compute data of each contribution
        self.compute_reforestation_deforestation()
        self.compute_managed_wood_production()
        self.compute_unmanaged_wood_production()
        # sum up global surface data
        self.sumup_global_surface_data()
        # check deforestation limit
        self.check_deforestation_limit()
        # compute cpaital and lost capital
        self.compute_lost_capital()
        # sum up global CO2 data
        self.compute_global_CO2_production()

        # compute biomass dry production
        self.compute_biomass_dry_production()

        # compute outputs:
        self.land_use_required['Forest (Gha)'] = self.forest_surface_df['global_forest_surface']
        # techno production in TWh
        self.techno_production[f'{BiomassDry.name} ({BiomassDry.unit})'] = self.biomass_dry_df[
            'biomass_dry_for_energy (Mt)'] * self.biomass_dry_calorific_value
        # price in $/MWh
        self.techno_prices['Forest'] = self.biomass_dry_df['price_per_MWh']

        if 'CO2_taxes_factory' in self.biomass_dry_df:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh'] - \
                self.biomass_dry_df['CO2_taxes_factory']
        else:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh']

        # emissions are not computed here because the global emission balance
        # is directly passed to carbon emission model
        self.CO2_emissions['Forest'] = np.zeros(len(self.years))

        # no consumption
        self.techno_consumption['biomass_dry (TWh)'] = np.zeros(
            len(self.years))
        self.techno_consumption_woratio['biomass_dry (TWh)'] = np.zeros(
            len(self.years))

    def compute_managed_wood_production(self):
        """
        compute data concerning managed wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """
        construction_delay = self.techno_wood_info['construction_delay']
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        residue_density_percentage = self.techno_wood_info['residue_density_percentage']
        residue_percentage_for_energy = self.techno_wood_info['residue_percentage_for_energy']
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']
        # ADD TEST FOR $  or euro unit OF PRICE #############
        mw_cost = self.techno_wood_info['managed_wood_price_per_ha']
        # managed wood from past invest. invest in G$ - surface in Gha.
        mw_from_past_invest = self.managed_wood_invest_before_year_start[
            'investment'] / mw_cost
        # managed wood from actual invest
        mw_from_invest = self.managed_wood_investment['investment'] / mw_cost
        # concat all managed wood form invest
        mw_added = pd.concat([mw_from_past_invest, mw_from_invest]).values
        # remove value that exceed year_end
        for i in range(0, construction_delay):
            mw_added = np.delete(mw_added, len(mw_added) - 1)

        # Surface part
        self.managed_wood_df['delta_surface'] = mw_added
        self.managed_wood_df['cumulative_surface'] = np.cumsum(
            self.managed_wood_df['delta_surface']) + self.managed_wood_initial_surface

        # Biomass production part
        # Gha * m3/ha * kg/m3 => Mt
        # recycle part is from the 2nd hand wood that will be recycled from the
        # first investment
        self.managed_wood_df['delta_biomass_production (Mt)'] = self.managed_wood_df['delta_surface'] * density_per_ha * mean_density / \
            years_between_harvest / (1 - recycle_part)
        self.managed_wood_df['biomass_production (Mt)'] = np.cumsum(
            self.managed_wood_df['delta_biomass_production (Mt)']) + self.managed_wood_inital_prod / self.biomass_dry_calorific_value
        self.managed_wood_df['residues_production (Mt)'] = self.managed_wood_df['biomass_production (Mt)'] * \
            residue_density_percentage
        self.managed_wood_df['residues_production_for_energy (Mt)'] = self.managed_wood_df['residues_production (Mt)'] * \
            residue_percentage_for_energy
        self.managed_wood_df['residues_production_for_industry (Mt)'] = self.managed_wood_df['residues_production (Mt)'] * \
            (1 - residue_percentage_for_energy)

        self.managed_wood_df['wood_production (Mt)'] = self.managed_wood_df['biomass_production (Mt)'] * \
            (1 - residue_density_percentage)
        self.managed_wood_df['wood_production_for_energy (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
            wood_percentage_for_energy
        self.managed_wood_df['wood_production_for_industry (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
            (1 - wood_percentage_for_energy)

        # CO2 part
        self.managed_wood_df['delta_CO2_emitted'] = - \
            self.managed_wood_df['delta_surface'] * self.CO2_per_ha / 1000
        # CO2 emitted is delta cumulate
        self.managed_wood_df['CO2_emitted'] = - \
            (self.managed_wood_df['cumulative_surface'] -
             self.managed_wood_initial_surface) * self.CO2_per_ha / 1000

    def compute_unmanaged_wood_production(self):
        """
        compute data concerning unmanaged wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """

        construction_delay = self.techno_wood_info['construction_delay']
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        residue_density_percentage = self.techno_wood_info['residue_density_percentage']
        residue_percentage_for_energy = self.techno_wood_info['residue_percentage_for_energy']
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']
        # ADD TEST FOR $  or euro unit OF PRICE #############
        uw_cost = self.techno_wood_info['unmanaged_wood_price_per_ha']
        # unmanaged wood from past invest. invest in G$ - surface in Gha.
        uw_from_past_invest = self.unmanaged_wood_invest_before_year_start[
            'investment'] / uw_cost
        # unmanaged wood from actual invest
        uw_from_invest = self.unmanaged_wood_investment['investment'] / uw_cost
        # concat all unmanaged wood form invest
        uw_added = pd.concat([uw_from_past_invest, uw_from_invest]).values
        # remove value that exceed year_end
        for i in range(0, construction_delay):
            uw_added = np.delete(uw_added, len(uw_added) - 1)

        # Surface part
        self.unmanaged_wood_df['delta_surface'] = uw_added
        self.unmanaged_wood_df['cumulative_surface'] = np.cumsum(
            uw_added) + self.unmanaged_wood_initial_surface

        # Biomass production part
        # recycle part is from the 2nd hand wood that will be recycled from the
        # first investment
        self.unmanaged_wood_df['delta_biomass_production (Mt)'] = self.unmanaged_wood_df['delta_surface'] * density_per_ha * mean_density / \
            years_between_harvest / (1 - recycle_part)
        self.unmanaged_wood_df['biomass_production (Mt)'] = np.cumsum(
            self.unmanaged_wood_df['delta_biomass_production (Mt)']) + self.unmanaged_wood_inital_prod / self.biomass_dry_calorific_value
        self.unmanaged_wood_df['residues_production (Mt)'] = self.unmanaged_wood_df['biomass_production (Mt)'] * \
            residue_density_percentage
        self.unmanaged_wood_df['residues_production_for_energy (Mt)'] = self.unmanaged_wood_df['residues_production (Mt)'] * \
            residue_percentage_for_energy
        self.unmanaged_wood_df['residues_production_for_industry (Mt)'] = self.unmanaged_wood_df['residues_production (Mt)'] * \
            (1 - residue_percentage_for_energy)

        self.unmanaged_wood_df['wood_production (Mt)'] = self.unmanaged_wood_df['biomass_production (Mt)'] * \
            (1 - residue_density_percentage)
        self.unmanaged_wood_df['wood_production_for_energy (Mt)'] = self.unmanaged_wood_df['wood_production (Mt)'] * \
            wood_percentage_for_energy
        self.unmanaged_wood_df['wood_production_for_industry (Mt)'] = self.unmanaged_wood_df['wood_production (Mt)'] * \
            (1 - wood_percentage_for_energy)

        # CO2 part: no absorption of CO2 because it is unmanaged

    def compute_reforestation_deforestation(self):
        """
        compute land use and due to reforestation et deforestation activities
        CO2 is not computed here because surface limit need to be taken into account before.
        """
        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.forest_surface_df['delta_deforestation_surface'] = - \
            self.deforestation_surface['deforested_surface'].values / 1000

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.forest_surface_df['delta_reforestation_surface'] = self.forest_investment['forest_investment'].values / self.cost_per_ha

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])
        self.forest_surface_df['reforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_reforestation_surface'])

    def sumup_global_surface_data(self):
        """
        managed wood and unmanaged wood impact forest_surface_df
        """
        self.forest_surface_df['delta_global_forest_surface'] = self.forest_surface_df['delta_reforestation_surface'] + self.forest_surface_df['delta_deforestation_surface'] +\
            self.unmanaged_wood_df['delta_surface'] + \
            self.managed_wood_df['delta_surface']
        self.forest_surface_df['global_forest_surface'] = self.forest_surface_df['reforestation_surface'] + self.forest_surface_df['deforestation_surface'] + \
            self.unmanaged_wood_df['cumulative_surface'] + \
            self.managed_wood_df['cumulative_surface'] + \
            self.initial_unsused_forest_surface

    def check_deforestation_limit(self):
        """
        take into acount deforestation limit.
        If limit is not crossed, nothing happen
        If limit is crossed, deforestation_surface is limited and delta_deforestation is set to 0.
        """

        # check limit of deforestation
        for i in range(0, len(self.years)):
            if self.forest_surface_df.loc[i, 'global_forest_surface'] < -self.limit_deforestation_surface / 1000:
                self.forest_surface_df.loc[i,
                                           'delta_global_forest_surface'] = 0
                self.forest_surface_df.loc[i, 'delta_deforestation_surface'] = - \
                    self.forest_surface_df.loc[i,
                                               'delta_global_forest_surface']
                self.forest_surface_df.loc[i,
                                           'global_forest_surface'] = -self.limit_deforestation_surface / 1000
                self.forest_surface_df.loc[i, 'deforestation_surface'] = - self.forest_surface_df.loc[i, 'reforestation_surface'] - \
                    self.managed_wood_df.loc[i, 'cumulative_surface'] - self.unmanaged_wood_df.loc[i, 'cumulative_surface'] - \
                    self.limit_deforestation_surface / 1000

    def compute_global_CO2_production(self):
        """
        compute the global CO2 production in Gt
        """
        # in Gt of CO2
        self.CO2_emitted_df['delta_CO2_emitted'] = -self.forest_surface_df['delta_global_forest_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['delta_CO2_deforestation'] = -self.forest_surface_df['delta_deforestation_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['delta_CO2_reforestation'] = -self.forest_surface_df['delta_reforestation_surface'] * \
            self.CO2_per_ha / 1000

        self.CO2_emitted_df['CO2_deforestation'] = -self.forest_surface_df['deforestation_surface'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['CO2_reforestation'] = -self.forest_surface_df['reforestation_surface'] * \
            self.CO2_per_ha / 1000
        # global sum up
        self.CO2_emitted_df['global_CO2_emitted'] = -self.forest_surface_df['deforestation_surface'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['global_CO2_captured'] = -self.forest_surface_df['reforestation_surface'] * \
            self.CO2_per_ha / 1000 + \
            self.managed_wood_df['CO2_emitted']
        self.CO2_emitted_df['global_CO2_emission_balance'] = self.CO2_emitted_df['global_CO2_emitted'] + \
            self.CO2_emitted_df['global_CO2_captured']

    def compute_biomass_dry_production(self):
        """
        compute total biomass dry prod
        """

        self.biomass_dry_df['biomass_dry_for_energy (Mt)'] = self.unmanaged_wood_df['residues_production_for_energy (Mt)'] + \
            self.unmanaged_wood_df['wood_production_for_energy (Mt)'] + \
            self.managed_wood_df['wood_production_for_energy (Mt)'] + \
            self.managed_wood_df['residues_production_for_energy (Mt)']

        self.managed_wood_part = self.managed_wood_df['biomass_production (Mt)'] / (
            self.managed_wood_df['biomass_production (Mt)'] + self.unmanaged_wood_df['biomass_production (Mt)'])
        self.unmanaged_wood_part = self.unmanaged_wood_df['biomass_production (Mt)'] / (
            self.managed_wood_df['biomass_production (Mt)'] + self.unmanaged_wood_df['biomass_production (Mt)'])

        self.compute_price('managed_wood')
        self.compute_price('unmanaged_wood')

        self.biomass_dry_df['price_per_ton'] = self.biomass_dry_df['managed_wood_price_per_ton'] * \
            self.managed_wood_part + \
            self.biomass_dry_df['unmanaged_wood_price_per_ton'] * \
            self.unmanaged_wood_part

        self.biomass_dry_df['managed_wood_price_per_MWh'] = self.biomass_dry_df['managed_wood_price_per_ton'] / \
            self.biomass_dry_calorific_value
        self.biomass_dry_df['unmanaged_wood_price_per_MWh'] = self.biomass_dry_df['unmanaged_wood_price_per_ton'] / \
            self.biomass_dry_calorific_value
        self.biomass_dry_df['price_per_MWh'] = self.biomass_dry_df['price_per_ton'] / \
            self.biomass_dry_calorific_value

    def compute_price(self, techno_name):
        """
        compute price as in techno_type
        """

        # Maximize with smooth exponential
#         price_df['invest'] = compute_func_with_exp_min(
#             investment, self.min_value_invest)
        density_per_ha = self.techno_wood_info['density_per_ha']  # m3/ha
        mean_density = self.techno_wood_info['density']  # kg/m3

        self.crf = self.compute_crf()

        self.biomass_dry_df[f'{techno_name}_transport ($/t)'] = self.transport['transport']

        # Factory cost including CAPEX OPEX
        # $/ha * ha/m3 * m3/kg * 1000 = $/t
        self.biomass_dry_df[f'{techno_name}_capex ($/t)'] = self.techno_wood_info[f'{techno_name}_price_per_ha'] * \
            (self.crf + 0.045) / density_per_ha / mean_density * 1000

        self.biomass_dry_df[f'{techno_name}_price_per_ton'] = (
            self.biomass_dry_df[f'{techno_name}_capex ($/t)'] +
            self.biomass_dry_df[f'{techno_name}_transport ($/t)']) * self.margin['margin'] / 100.0

    def compute_crf(self):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        wacc = self.techno_wood_info['WACC']
        crf = (wacc * (1.0 + wacc) ** 100) / \
              ((1.0 + wacc) ** 100 - 1.0)

        return crf

    def compute_lost_capital(self):
        """
        Compute the loss of capital due to reforestation and deforestation activities that have opposite effect but cost money.
        To deforest and to reforest only for surface expanse result as a lost of capital.

        lost_capital = min(reforest_surface, deforest_surface) * cost_per_ha
        cost_per_ha is in $/ha
        reforest_surface and deforest_surface are in Gha
        lost_capital is in G$ 
        """
        self.lost_capital['lost_capital_G$'] = 0
        self.lost_capital['capital_G$'] = 0
        # abs() needed because deforestation surface is negative

        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) < self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                self.lost_capital.loc[element, 'lost_capital_G$'] = abs(self.forest_surface_df.loc[element,
                                                                                                   'delta_deforestation_surface']) * self.cost_per_ha
            else:
                self.lost_capital.loc[element, 'lost_capital_G$'] = self.forest_surface_df.loc[element,
                                                                                               'delta_reforestation_surface'] * self.cost_per_ha

        self.lost_capital['capital_G$'] = self.forest_surface_df['delta_reforestation_surface'] * self.cost_per_ha

    # Gradients
    def d_deforestation_surface_d_deforestation_surface(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = np.identity(number_of_values)
        for i in range(0, number_of_values):
            # derivate = -1/1000 for unit conversion if limit is not broken
            if self.forest_surface_df['global_forest_surface'].values[i] != -self.limit_deforestation_surface / 1000:
                d_deforestation_surface_d_forests[i][i] = - 1 / 1000
            # if limit is broken, grad is null
            else:
                d_deforestation_surface_d_forests[i][i] = 0

        return d_deforestation_surface_d_forests

    def d_forestation_surface_d_invest(self, ):
        """
        Compute gradient of reforestation surface by invest (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_forestation_surface_d_invest = np.identity(number_of_values)
        for i in range(0, number_of_values):
            # surface = invest / cost_per_ha if limit is not borken
            if self.forest_surface_df['global_forest_surface'].values[i] != -self.limit_deforestation_surface / 1000:
                d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
            #surface = constant is limit is broken
            else:
                d_forestation_surface_d_invest[i][i] = 0

        return d_forestation_surface_d_invest

    def d_wood_techno_surface_d_invest(self, price_per_ha):
        """
        Compute gradient of managed wood surface by invest
        Same function for managed wood and unmanaged wood. Only the price_per_ha change.
        construction delay impact becasue there is a shift of investment impact of construction_delay year.
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_wood_surface_d_invest = np.identity(number_of_values) * 0
        construction_delay = self.techno_wood_info['construction_delay']
        for i in range(construction_delay, number_of_values):
            d_wood_surface_d_invest[i][i -
                                       construction_delay] = 1 / price_per_ha

        return d_wood_surface_d_invest

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if i > 0:
                d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_CO2_emitted(self, d_deforestation_surface):
        """
        Compute gradient of non_captured_CO2 by deforestation surface
        :param: d_deforestation_surface, derivative of deforestation surface
        CO2_emitted = surface * constant --> d_surface is reused.
        """

        d_CO2_emitted = - d_deforestation_surface * self.CO2_per_ha / 1000

        return d_CO2_emitted

    def d_biomass_prod_d_invest(self, d_surf_d_invest, wood_or_residues_percentage, percentage_for_energy, biomass_part):
        """
        Compute derivate of biomaass production by investment. Biomass production is : mw_residu / un_residu / mw_wood / uw_wood
        prod = surface * density_per_ha * density * wood_or_residues_percentage * percentage_for_energy / years_between_harvest / (1 - recycle_part)
        --> only surface is dependant of invest, the other parameters does not depends of invest.
        d_surf_d_invest is alread computed and known.
        # recycle part is from the 2nd hand wood that will be recycled from the first investment
        """
        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']

        ddelta_prod_dinvest = d_surf_d_invest * density_per_ha * density * \
            wood_or_residues_percentage * percentage_for_energy / \
            years_between_harvest / \
            (1 - recycle_part)

        return ddelta_prod_dinvest

    def d_biomass_price_d_invest_mw(self, price_per_ha):
        """
        compute derivate of biomass price by invest in managed wood
        price = mw_price * mw_part + uw_price * uw_part
        mw_part and uw_part are independant of invest
        mw_part = mw_prod / (mw_prod + uw_prod) with mw_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and uw_part = (1-mw_part)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        construction_delay = self.techno_wood_info['construction_delay']
        d_wood_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        res = np.zeros((number_of_values, number_of_values))
        for i in range(0, number_of_values):
            d_wood_surface_d_invest[i][i] = 1 / price_per_ha
        deriv_2 = self.d_cum(d_wood_surface_d_invest)
        d_surf_d_invest = deriv_2

        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        # compute d_prod_dinvest. total production is taken into account :
        # energy + non energy
        dprod_dinvest = d_surf_d_invest * density_per_ha * density / \
            years_between_harvest / \
            (1 - recycle_part)

        mw_prod = self.managed_wood_df['biomass_production (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'].values + \
            self.unmanaged_wood_df['biomass_production (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_mwpart_d_mw_invest = (dprod_dinvest * biomass_prod - mw_prod *
                                dprod_dinvest) / biomass_prod**2 / self.biomass_dry_calorific_value

        derivate = self.biomass_dry_df['managed_wood_price_per_ton'].values * d_mwpart_d_mw_invest - \
            self.biomass_dry_df['unmanaged_wood_price_per_ton'].values * \
            d_mwpart_d_mw_invest
        # shift needed due to construction delay
        for i in range(construction_delay, number_of_values):
            for j in range(construction_delay, i + 1):
                res[i, j - construction_delay] = derivate[i, i]

        return res

    def d_biomass_price_d_invest_uw(self, price_per_ha):
        """
        compute derivate of biomass price by invest in unmanaged wood
        price = mw_price * mw_part + uw_price * uw_part
        uw_part and uw_part are independant of invest
        uw_part = uw_prod / (mw_prod + uw_prod) with uw_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and mw_part = (1-uw_part)
        """

        number_of_values = (self.year_end - self.year_start + 1)
        construction_delay = self.techno_wood_info['construction_delay']
        d_wood_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        res = np.zeros((number_of_values, number_of_values))
        for i in range(0, number_of_values):
            d_wood_surface_d_invest[i][i] = 1 / price_per_ha
        deriv_2 = self.d_cum(d_wood_surface_d_invest)
        d_surf_d_invest = deriv_2

        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        # compute d_prod_dinvest. total production is taken into account :
        # energy + non energy
        dprod_dinvest = d_surf_d_invest * density_per_ha * density / \
            years_between_harvest / \
            (1 - recycle_part)

        uw_prod = self.unmanaged_wood_df['biomass_production (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'] .values + \
            self.unmanaged_wood_df['biomass_production (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_uwpart_d_uw_invest = (
            dprod_dinvest * biomass_prod - uw_prod * dprod_dinvest) / biomass_prod**2 / self.biomass_dry_calorific_value

        derivate = self.biomass_dry_df['unmanaged_wood_price_per_ton'].values * d_uwpart_d_uw_invest - \
            self.biomass_dry_df['managed_wood_price_per_ton'].values * \
            d_uwpart_d_uw_invest
        for i in range(construction_delay, number_of_values):
            for j in range(construction_delay, i + 1):
                res[i, j - construction_delay] = derivate[i, i]

        return res

    def d_capital_total_d_invest(self,):
        """
        Compute derivate of capital total of reforestation regarding reforestation_investment
        """
        dcapital_d_invest = np.identity(len(self.years))

        return dcapital_d_invest

    def d_lostcapitald_invest(self, d_delta_reforestation_dinvest):
        """"
        compute derivate of lost capital regarding reforestation_investment
        if deforestation_surf < reforestation surf : no dependancies --> derivate is null
        if deforestation_sur > reforestation_surf : derivate is d_delta_reforestation_dinvest * cost_per_ha
        """
        result = d_delta_reforestation_dinvest
        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) < self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                result[element, element] = 0
            else:
                result[element, element] = result[element,
                                                  element] * self.cost_per_ha
        return result

    def d_lostcapitald_deforestation(self, d_delta_deforestation_d_deforestation):
        """"
        compute derivate of lost capital regarding reforestation_investment
        if deforestation_surf < reforestation surf : d_delta_deforestation_d_deforestation * cost_per_ha
        if deforestation_sur > reforestation_surf : no dependencies
        """
        result = d_delta_deforestation_d_deforestation
        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) > self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                result[element, element] = 0
            else:
                result[element, element] = result[element,
                                                  element] * self.cost_per_ha

        return result
