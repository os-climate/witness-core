'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2024/06/24 Copyright 2023 Capgemini

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
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_optimization_plugins.tools.cst_manager.constraint_manager import (
    compute_func_with_exp_min,
)

from climateeconomics.glossarycore import GlossaryCore


class Forest():
    """
    Forest model class 
    basic for now, to evolve 

    """
    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    TIME_STEP = GlossaryCore.TimeStep
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'
    REFORESTATION_INVESTMENT = 'forest_investment'
    REFORESTATION_COST_PER_HA = 'reforestation_cost_per_ha'
    WOOD_TECHNO_DICT = 'wood_techno_dict'
    MW_INITIAL_PROD = 'managed_wood_initial_prod'
    MW_INITIAL_SURFACE = 'managed_wood_initial_surface'
    MW_INVEST_BEFORE_YEAR_START = 'managed_wood_invest_before_year_start'
    MW_INVESTMENT = 'managed_wood_investment'
    DEFORESTATION_INVESTMENT = 'deforestation_investment'
    DEFORESTATION_COST_PER_HA = 'deforestation_cost_per_ha'

    TRANSPORT_COST = 'transport_cost'
    MARGIN = 'margin'
    UNMANAGED_FOREST = 'initial_unmanaged_forest_surface'
    PROTECTED_FOREST = 'protected_forest_surface'

    FOREST_SURFACE_DF = 'forest_surface_df'
    FOREST_DETAIL_SURFACE_DF = 'forest_surface_detail_df'
    CO2_EMITTED_FOREST_DF = GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)
    CO2_EMITTED_DETAIL_DF = GlossaryCore.CO2EmissionsDetailDfValue
    MW_DF = 'managed_wood_df'
    # UW_DF = 'unmanaged_wood_df'
    BIOMASS_DRY_DETAIL_DF = 'biomass_dry_detail_df'
    BIOMASS_DRY_DF = 'biomass_dry_df'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.product_energy_unit = 'TWh'
        self.mass_unit = 'Mt'
        self.set_data()
        self.create_dataframe()
        self.counter = 0

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
        self.CO2_per_ha = self.param[self.CO2_PER_HA]
        # initial CO2 emissions
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        # forest data
        self.forest_investment = self.param[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = self.param[self.REFORESTATION_COST_PER_HA]
        self.deforest_invest = self.param[self.DEFORESTATION_INVESTMENT]
        self.deforest_cost_per_ha = self.param[self.DEFORESTATION_COST_PER_HA]
        self.techno_wood_info = self.param[self.WOOD_TECHNO_DICT]
        self.managed_wood_initial_surface = self.param[self.MW_INITIAL_SURFACE]
        self.managed_wood_invest_before_year_start = self.param[
            self.MW_INVEST_BEFORE_YEAR_START]
        self.managed_wood_investment = self.param[self.MW_INVESTMENT]
        self.transport = self.param[self.TRANSPORT_COST]
        self.margin = self.param[self.MARGIN]
        self.initial_unmanaged_forest_surface = self.param[self.UNMANAGED_FOREST]
        self.protected_forest_surface = self.param[self.PROTECTED_FOREST]
        self.scaling_factor_techno_consumption = self.param['scaling_factor_techno_consumption']
        self.scaling_factor_techno_production = self.param['scaling_factor_techno_production']

        self.wood_density = self.techno_wood_info['wood_density']
        self.residues_density = self.techno_wood_info['residues_density']

        self.residues_density_percentage = self.techno_wood_info['residues_density_percentage']

        self.mean_density = self.wood_density * (1. - self.residues_density_percentage) + \
                            self.residues_density * (1. - self.residues_density_percentage)

        self.actual_yield = self.techno_wood_info['actual_yield']
        self.managed_yield = self.techno_wood_info['managed_yield']
        self.unmanaged_yield = self.techno_wood_info['unmanaged_yield']
        self.wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.forest_surface_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CO2_emitted_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.managed_wood_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.mw_from_invests = pd.DataFrame({GlossaryCore.Years: self.years})
        self.biomass_dry_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.ratio = pd.DataFrame({GlossaryCore.Years: self.years})

        # output dataframes:
        self.techno_production = pd.DataFrame({GlossaryCore.Years: self.years})
        self.techno_prices = pd.DataFrame({GlossaryCore.Years: self.years})
        self.techno_consumption = pd.DataFrame({GlossaryCore.Years: self.years})
        self.techno_consumption_woratio = pd.DataFrame({GlossaryCore.Years: self.years})
        self.land_use_required = pd.DataFrame({GlossaryCore.Years: self.years})
        self.CO2_emissions = pd.DataFrame({GlossaryCore.Years: self.years})
        self.forest_lost_capital = pd.DataFrame({GlossaryCore.Years: self.years})

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.biomass_dry_calorific_value = BiomassDry.data_energy_dict[
            'calorific_value']  # kwh/kg
        # kwh/kg
        self.biomass_dry_high_calorific_value = BiomassDry.data_energy_dict[
            'high_calorific_value']
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.forest_investment = in_dict[self.REFORESTATION_INVESTMENT]
        self.deforest_invest = in_dict[self.DEFORESTATION_INVESTMENT]
        self.cost_per_ha = in_dict[self.REFORESTATION_COST_PER_HA]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        self.years = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        self.managed_wood_investment = in_dict[self.MW_INVESTMENT]

        self.forest_surface_df['unmanaged_forest'] = self.initial_unmanaged_forest_surface

        self.forest_lost_capital['reforestation'] = 0
        self.forest_lost_capital['managed_wood'] = 0
        self.forest_lost_capital['deforestation'] = 0

        # compute data of each contribution
        self.compute_managed_wood_surface()
        self.compute_reforestation_deforestation_surface()
        self.compute_deforestation_biomass()
        self.compute_managed_wood_production()
        # sum up global surface data
        self.sumup_global_surface_data()
        # compute capital and lost capital
        # sum up global CO2 data
        self.compute_global_CO2_production()

        # compute biomass dry production
        self.compute_biomass_dry_production()

        # compute outputs:

        # compute land_use for energy
        self.land_use_required['Forest (Gha)'] = self.managed_wood_df['cumulative_surface']

        # compute forest constrain evolution: reforestation + deforestation
        self.forest_surface_df['forest_constraint_evolution'] = self.forest_surface_df['reforestation_surface'] + \
                                                                self.forest_surface_df['deforestation_surface']

        # techno production in TWh
        self.techno_production[f'{BiomassDry.name} ({BiomassDry.unit})'] = (self.managed_wood_df[
                                                                                'wood_production_for_energy (Mt)'] + \
                                                                            self.biomass_dry_df[
                                                                                'deforestation_for_energy']) * self.biomass_dry_calorific_value + \
                                                                           self.managed_wood_df[
                                                                               'residues_production_for_energy (Mt)'] * \
                                                                           self.techno_wood_info[
                                                                               'residue_calorific_value']
        # price in $/MWh
        self.techno_prices['Forest'] = self.biomass_dry_df['price_per_MWh']

        if 'CO2_taxes_factory' in self.biomass_dry_df:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh'] - \
                                                   self.biomass_dry_df['CO2_taxes_factory']
        else:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh']

        # CO2 emissions
        self.compute_carbon_emissions()

        # CO2 consumed
        self.techno_consumption[f'{GlossaryEnergy.carbon_capture} ({self.mass_unit})'] = -self.techno_wood_info['CO2_from_production'] / \
                                                                    self.biomass_dry_high_calorific_value * \
                                                                    self.techno_production[
                                                                        f'{BiomassDry.name} ({BiomassDry.unit})']

        self.techno_consumption_woratio[f'{GlossaryEnergy.carbon_capture} ({self.mass_unit})'] = -self.techno_wood_info[
            'CO2_from_production'] / \
                                                                            self.biomass_dry_high_calorific_value * \
                                                                            self.techno_production[
                                                                                f'{BiomassDry.name} ({BiomassDry.unit})']

    def compute_managed_wood_surface(self):
        """
        compute managed wood delta and cumulative surface from investments
        Will be recalculate with deforestation impact in compute_reforestation_deforestation_surface method
        """
        construction_delay = self.techno_wood_info[GlossaryCore.ConstructionDelay]
        mw_cost = self.techno_wood_info['managed_wood_price_per_ha']
        # managed wood from past invest. invest in G$ - surface in Gha.
        mw_from_past_invest = self.managed_wood_invest_before_year_start[
                                  GlossaryCore.InvestmentsValue] / mw_cost
        # managed wood from actual invest
        mw_from_invest = self.managed_wood_investment[GlossaryCore.InvestmentsValue] / mw_cost
        # concat all managed wood form invest
        mw_added = pd.concat([mw_from_past_invest, mw_from_invest]).values

        # remove value that exceed year_end
        for i in range(0, construction_delay):
            mw_added = np.delete(mw_added, len(mw_added) - 1)
        self.mw_from_invests['mw_surface'] = mw_added
        self.managed_wood_df['delta_surface'] = mw_added
        cumulative_mw = np.cumsum(mw_added)
        self.managed_wood_df['cumulative_surface'] = cumulative_mw + \
                                                     self.managed_wood_initial_surface

    def compute_managed_wood_production(self):
        """
        compute data concerning managed wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """

        # Biomass production part
        # Gha * m3/ha
        cubic_meter_production = (self.managed_wood_initial_surface * self.actual_yield +
                                  (self.managed_wood_df['cumulative_surface'] - self.managed_wood_initial_surface)
                                  * self.managed_yield)
        self.managed_wood_df['delta_wood_production (Mt)'] = self.managed_wood_df[
                                                                 'delta_surface'] * self.managed_yield * (
                                                                     1 - self.residues_density_percentage) * self.wood_density
        # Gm3* kg/m3 => Mt
        self.managed_wood_df['wood_production (Mt)'] = cubic_meter_production * (
                1 - self.residues_density_percentage) * self.wood_density

        self.managed_wood_df['wood_production_for_energy (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
                                                                  self.wood_percentage_for_energy
        self.managed_wood_df['wood_production_for_industry (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
                                                                    (1 - self.wood_percentage_for_energy)

        self.managed_wood_df['delta_residues_production (Mt)'] = self.managed_wood_df[
                                                                     'delta_surface'] * self.managed_yield * self.residues_density_percentage * self.residues_density
        self.managed_wood_df[
            'residues_production (Mt)'] = cubic_meter_production * self.residues_density_percentage * self.residues_density
        self.managed_wood_df['residues_production_for_energy (Mt)'] = self.managed_wood_df['residues_production (Mt)'] * \
                                                                      self.wood_percentage_for_energy
        self.managed_wood_df['residues_production_for_industry (Mt)'] = self.managed_wood_df[
                                                                            'residues_production (Mt)'] * \
                                                                        (1 - self.wood_percentage_for_energy)

        self.managed_wood_df['delta_biomass_production (Mt)'] = self.managed_wood_df['delta_wood_production (Mt)'] + \
                                                                self.managed_wood_df['delta_residues_production (Mt)']
        self.managed_wood_df['biomass_production (Mt)'] = self.managed_wood_df['wood_production (Mt)'] + \
                                                          self.managed_wood_df['residues_production (Mt)']

        # CO2 part
        self.managed_wood_df['delta_CO2_emitted'] = - \
                                                        self.managed_wood_df['delta_surface'] * self.CO2_per_ha / 1000
        # CO2 emitted is delta cumulate
        self.managed_wood_df['CO2_emitted'] = - \
                                                  (self.managed_wood_df['cumulative_surface'] -
                                                   self.managed_wood_initial_surface) * self.CO2_per_ha / 1000

    def compute_reforestation_deforestation_surface(self):
        """
        compute land use and due to reforestation et deforestation activities
        CO2 is not computed here because surface limit need to be taken into account before.
        """

        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.forest_surface_df['delta_deforestation_surface'] = - \
                                                                    self.deforest_invest[
                                                                        GlossaryCore.InvestmentsValue].values / \
                                                                self.deforest_cost_per_ha

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.forest_surface_df['delta_reforestation_surface'] = self.forest_investment[
                                                                    'forest_investment'].values / self.cost_per_ha

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])
        self.forest_surface_df['reforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_reforestation_surface'])

        deforested_unmanaged_surface = 0
        for i in range(0, len(self.years)):
            # recompute unmanaged forest cumulated each year
            if i == 0:
                self.forest_surface_df.loc[i, 'unmanaged_forest'] = self.initial_unmanaged_forest_surface + \
                                                                    self.forest_surface_df.loc[
                                                                        i, 'delta_reforestation_surface'] + \
                                                                    self.forest_surface_df.loc[i,
                                                                    'delta_deforestation_surface']
            else:
                self.forest_surface_df.loc[i, 'unmanaged_forest'] = self.forest_surface_df.loc[
                                                                        i - 1, 'unmanaged_forest'] + \
                                                                    self.forest_surface_df.loc[
                                                                        i, 'delta_reforestation_surface'] + \
                                                                    self.forest_surface_df.loc[i,
                                                                    'delta_deforestation_surface']
            # if unmanaged forest are empty, managed forest are removed
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 0:
                # remove managed wood
                self.managed_wood_df.loc[i,
                'delta_surface'] += self.forest_surface_df.loc[i, 'unmanaged_forest']
                # compute reforestation lost capital
                # in this loop all unmanaged forest + reforested forest has been deforested
                # if i == 0, lost capital is the initial unmanaged + reforested surface
                # else it is previous year unmanaged surface + reforested
                # surface
                if i == 0:
                    deforested_unmanaged_surface = self.initial_unmanaged_forest_surface + \
                                                   self.forest_surface_df.loc[i,
                                                   'delta_reforestation_surface']
                else:
                    deforested_unmanaged_surface = self.forest_surface_df.loc[i - 1,
                    'unmanaged_forest'] + self.forest_surface_df.loc[i, 'delta_reforestation_surface']
                self.forest_lost_capital.loc[i,
                'reforestation'] = deforested_unmanaged_surface * self.cost_per_ha

                # lost capital of managed wood is what is deforested into
                # managed forest
                self.forest_lost_capital.loc[i, 'managed_wood'] = -  self.forest_surface_df.loc[i,
                'unmanaged_forest'] * self.techno_wood_info['managed_wood_price_per_ha']
                # set unmanaged forest to 0
                self.forest_surface_df.loc[i, 'unmanaged_forest'] = 0
            else:
                # reforestation lost capital equals deforestation
                self.forest_lost_capital.loc[i, 'reforestation'] = - \
                                                                       self.forest_surface_df.loc[i,
                                                                       'delta_deforestation_surface'] * self.cost_per_ha
            # recompute managed forest cumulated each year
            if i > 0:
                self.managed_wood_df.loc[i, 'cumulative_surface'] = self.managed_wood_df.loc[
                                                                        i - 1, 'cumulative_surface'] + \
                                                                    self.managed_wood_df.loc[i, 'delta_surface']

            # if managed forest are empty, all is removed
            if self.managed_wood_df.loc[i, 'cumulative_surface'] <= 0:
                # the cumulative surface is the excedent surface deforested
                # leading to lost capital
                self.forest_lost_capital.loc[i, 'deforestation'] = - \
                                                                       self.managed_wood_df.loc[
                                                                           i, 'cumulative_surface'] * \
                                                                   self.deforest_cost_per_ha

                # lost capital of managed wood is what is left of managed wood
                # + what have been invested in the i year
                deforested_managed_surface = self.forest_surface_df.loc[
                                                 i, 'delta_deforestation_surface'] + deforested_unmanaged_surface - \
                                             self.managed_wood_df.loc[i, 'cumulative_surface']
                self.forest_lost_capital.loc[i, 'managed_wood'] = - \
                                                                      deforested_managed_surface * \
                                                                  self.techno_wood_info['managed_wood_price_per_ha']

                # delta is all the managed wood available
                self.managed_wood_df.loc[i, 'delta_surface'] = - \
                    self.managed_wood_df.loc[i - 1, 'cumulative_surface']
                self.managed_wood_df.loc[i, 'cumulative_surface'] = 0

                # set a limit to deforestation at the forest that have been reforested because there is no other
                # real_deforested surface = -delta_reforestation_surface + delta_mw_surface
                # lost_capital = (delta_deforestation_surface - real_deforested) * deforestation_cost_per_ha
                self.forest_surface_df.loc[i, 'delta_deforestation_surface'] = - self.forest_surface_df.loc[i,
                'delta_reforestation_surface'] + self.managed_wood_df.loc[i, 'delta_surface']

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])
        self.managed_wood_df['cumulative_surface'] = compute_func_with_exp_min(
            self.managed_wood_df['cumulative_surface'].values, 1e-15)
        self.forest_surface_df['unmanaged_forest'] = compute_func_with_exp_min(
            self.forest_surface_df['unmanaged_forest'].values, 1e-15)

    def compute_deforestation_biomass(self):
        """
        compute biomass produce by deforestation. It is a one time production.
        We use actual yield because deforestation is done on actual forest not managed ones
        """

        self.biomass_dry_df['deforestation (Mt)'] = -self.forest_surface_df[
            'delta_deforestation_surface'] * self.unmanaged_yield * self.wood_density
        self.biomass_dry_df['deforestation_for_energy'] = self.biomass_dry_df['deforestation (Mt)'] * \
                                                          self.wood_percentage_for_energy
        self.biomass_dry_df['deforestation_for_industry'] = self.biomass_dry_df['deforestation (Mt)'] - \
                                                            self.biomass_dry_df['deforestation_for_energy']
        self.biomass_dry_df[
            'deforestation_price_per_ton'] = self.unmanaged_yield * self.wood_density / self.deforest_cost_per_ha
        self.biomass_dry_df['deforestation_price_per_MWh'] = self.biomass_dry_df['deforestation_price_per_ton'] / \
                                                             self.biomass_dry_calorific_value

    def sumup_global_surface_data(self):
        """
        managed wood and unmanaged wood impact forest_surface_df
        """
        self.forest_surface_df['delta_global_forest_surface'] = self.forest_surface_df['delta_reforestation_surface'] + \
                                                                self.forest_surface_df['delta_deforestation_surface']
        self.forest_surface_df['global_forest_surface'] = self.managed_wood_df['cumulative_surface'] + \
                                                          self.forest_surface_df['unmanaged_forest'] + \
                                                          self.protected_forest_surface
        self.forest_surface_df['protected_forest_surface'] = self.protected_forest_surface

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

        # remove CO2 managed surface from global emission because CO2_per_ha
        # from managed forest = 0
        self.CO2_emitted_df['CO2_deforestation'] = - self.forest_surface_df['deforestation_surface'] * \
                                                   self.CO2_per_ha / 1000
        self.CO2_emitted_df['CO2_reforestation'] = -self.forest_surface_df['reforestation_surface'] * \
                                                   self.CO2_per_ha / 1000
        self.CO2_emitted_df['initial_CO2_land_use_change'] = self.initial_emissions
        # global sum up
        self.CO2_emitted_df['global_CO2_emitted'] = self.CO2_emitted_df['CO2_deforestation'] + \
                                                    self.CO2_emitted_df['initial_CO2_land_use_change']
        self.CO2_emitted_df['global_CO2_captured'] = self.CO2_emitted_df['CO2_reforestation']
        self.CO2_emitted_df['emitted_CO2_evol_cumulative'] = self.CO2_emitted_df['global_CO2_emitted'] + \
                                                             self.CO2_emitted_df['global_CO2_captured']

    def compute_biomass_dry_production(self):
        """
        compute total biomass dry prod
        """

        self.biomass_dry_df['biomass_dry_for_energy (Mt)'] = self.managed_wood_df['wood_production_for_energy (Mt)'] + \
                                                             self.managed_wood_df[
                                                                 'residues_production_for_energy (Mt)'] + \
                                                             self.biomass_dry_df['deforestation_for_energy']

        self.compute_price()

        self.managed_wood_part = self.managed_wood_df['biomass_production (Mt)'] / (
                self.managed_wood_df['biomass_production (Mt)'] + self.biomass_dry_df['deforestation (Mt)'])
        self.deforestation_part = self.biomass_dry_df['deforestation (Mt)'] / (
                self.managed_wood_df['biomass_production (Mt)'] + self.biomass_dry_df['deforestation (Mt)'])

        self.biomass_dry_df['price_per_ton'] = self.biomass_dry_df[
                                                   'managed_wood_price_per_ton'] * self.managed_wood_part + \
                                               self.biomass_dry_df['deforestation_price_per_ton'] * \
                                               self.deforestation_part

        self.biomass_dry_df['managed_wood_price_per_MWh'] = self.biomass_dry_df['managed_wood_price_per_ton'] / \
                                                            self.biomass_dry_calorific_value
        self.biomass_dry_df['price_per_MWh'] = self.biomass_dry_df['price_per_ton'] / \
                                               self.biomass_dry_calorific_value

    def compute_price(self):
        """
        compute price as in techno_type
        """

        self.crf = self.compute_crf()

        self.biomass_dry_df['managed_wood_transport ($/t)'] = self.transport['transport'].values

        # Factory cost including CAPEX OPEX
        # $/ha * ha/m3 * m3/kg * 1000 = $/t
        self.biomass_dry_df['managed_wood_capex ($/t)'] = self.techno_wood_info['managed_wood_price_per_ha'] * \
                                                           (
                                                                   self.crf + 0.045) / self.managed_yield / self.mean_density * 1000

        self.biomass_dry_df['managed_wood_price_per_ton'] = (
                                                                     self.biomass_dry_df['managed_wood_capex ($/t)'].values +
                                                                     self.biomass_dry_df['managed_wood_transport ($/t)']).values * \
                                                             self.margin['margin'].values / 100.0

    def compute_crf(self):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        wacc = self.techno_wood_info['WACC']
        crf = (wacc * (1.0 + wacc) ** 100) / \
              ((1.0 + wacc) ** 100 - 1.0)

        return crf

    def compute_carbon_emissions(self):
        '''
        Compute the carbon emissions from the technology taking into account 
        CO2 from production + CO2 from primary resources 
        '''
        if 'CO2_from_production' not in self.techno_wood_info:
            self.CO2_emissions['production'] = self.get_theoretical_co2_prod(
                unit='kg/kWh')
        elif self.techno_wood_info['CO2_from_production'] == 0.0:
            self.CO2_emissions['production'] = 0.0
        else:
            if self.techno_wood_info['CO2_from_production_unit'] == 'kg/kg':
                self.CO2_emissions['production'] = self.techno_wood_info['CO2_from_production'] / \
                                                   self.biomass_dry_high_calorific_value
            elif self.techno_wood_info['CO2_from_production_unit'] == 'kg/kWh':
                self.CO2_emissions['production'] = self.techno_wood_info['CO2_from_production']

        # Add carbon emission from input energies (resources or other
        # energies)

        co2_emissions_frominput_energies = self.compute_CO2_emissions_from_input_resources(
        )

        # Add CO2 from production + C02 from input energies
        self.CO2_emissions['Forest'] = self.CO2_emissions['production'] + \
                                       co2_emissions_frominput_energies

    def get_theoretical_co2_prod(self, unit='kg/kWh'):
        ''' 
        Get the theoretical CO2 production for a given technology,
        '''
        return 0.0

    def compute_CO2_emissions_from_input_resources(self):
        '''
        Need to take into account  CO2 from electricity/fuel production
        '''
        return 0.0

    # Gradients
    def compute_d_deforestation_surface_d_invest(self):
        """

        Compute gradient of deforestation surface by deforestation_invest (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = - \
                                                np.identity(number_of_values) / self.deforest_cost_per_ha

        return d_deforestation_surface_d_forests

    def compute_d_reforestation_surface_d_invest(self):
        """

        Compute gradient of reforestation surface by invest (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_forestation_surface_d_invest = np.identity(
            number_of_values) / self.cost_per_ha

        return d_forestation_surface_d_invest

    def compute_d_mw_surface_d_invest(self):
        """
        compute gradient of managed_wood surface vs managed_wood_investment
        """
        number_of_values = (self.year_end - self.year_start + 1)
        result = np.identity(number_of_values) * 0.0
        construction_delay = self.techno_wood_info[GlossaryCore.ConstructionDelay]
        for i in range(construction_delay, number_of_values):
            result[i, i - construction_delay] = 1 / \
                                                self.techno_wood_info['managed_wood_price_per_ha']
        return result

    def compute_d_limit_surfaces_d_deforestation_invest(self, d_deforestation_surface_d_invest):
        """
        Compute gradient of delta managed wood surface, delta deforestation surface, unmanaged wood cumulated surface,
        mw lost capital, deforestation lost capital and reforestation lost capital vs deforestation invest
        """
        number_of_values = (self.year_end - self.year_start + 1)

        d_delta_mw_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_delta_deforestation_surface_d_invest = d_deforestation_surface_d_invest
        d_cum_umw_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_deforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_reforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_mw_d_invest = np.zeros(
            (number_of_values, number_of_values))

        for i in range(0, len(self.years)):
            if i == 0:
                d_cum_umw_surface_d_invest[i] = d_delta_deforestation_surface_d_invest[i]
            else:
                d_cum_umw_surface_d_invest[i] = d_cum_umw_surface_d_invest[i -
                                                                           1] + d_delta_deforestation_surface_d_invest[
                                                    i]
            # if unmanaged forest are empty, managed forest are removed
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 1e-10:
                # remove managed wood
                d_delta_mw_surface_d_invest[i] += d_cum_umw_surface_d_invest[i]

                if i == 0:
                    d_lc_reforestation_d_invest[i] = np.zeros(number_of_values)
                else:
                    d_lc_reforestation_d_invest[i] = d_cum_umw_surface_d_invest[i -
                                                                                1] * self.cost_per_ha

                d_lc_mw_d_invest[i] = - d_cum_umw_surface_d_invest[i] * \
                                      self.techno_wood_info['managed_wood_price_per_ha']
                # set unmanaged forest to 0
                d_cum_umw_surface_d_invest[i] = np.zeros(number_of_values)

            else:
                d_lc_reforestation_d_invest[i] = - \
                                                     d_deforestation_surface_d_invest[i] * self.cost_per_ha

            # if managed forest are empty, all is removed
            if self.managed_wood_df.loc[i, 'cumulative_surface'] <= 1e-10:
                sum = self.d_cum(d_delta_mw_surface_d_invest)
                # delta is all the managed wood available
                d_lc_deforestation_d_invest[i] = - \
                                                     sum[i] * self.deforest_cost_per_ha
                d_lc_mw_d_invest[i] = - (
                        d_deforestation_surface_d_invest[i] + d_lc_reforestation_d_invest[i] / self.cost_per_ha -
                        sum[i]) * \
                                      self.techno_wood_info['managed_wood_price_per_ha']

                #                 d_delta_mw_surface_d_invest[i] = - \
                #                     compute_dfunc_with_exp_min(
                #                         sum[i - 1], 1e-15).reshape(number_of_values)
                d_delta_mw_surface_d_invest[i] = - \
                    sum[i - 1]
                d_delta_deforestation_surface_d_invest[i] = d_delta_mw_surface_d_invest[i]

        return d_cum_umw_surface_d_invest, d_delta_mw_surface_d_invest, d_delta_deforestation_surface_d_invest, d_lc_deforestation_d_invest, d_lc_reforestation_d_invest, d_lc_mw_d_invest

    def compute_d_limit_surfaces_d_reforestation_invest(self, d_reforestation_surface_d_invest):
        """
        Compute gradient of delta managed wood surface, delta deforestation surface, unmanaged wood cumulated surface,
        mw lost capital, deforestation lost capital and reforestation lost capital vs reforestation invest
        """
        number_of_values = (self.year_end - self.year_start + 1)

        d_delta_mw_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_delta_deforestation_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_delta_reforestation_surface_d_invest = d_reforestation_surface_d_invest
        d_cum_umw_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_deforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_reforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_mw_d_invest = np.zeros(
            (number_of_values, number_of_values))

        for i in range(0, len(self.years)):
            if i == 0:
                d_cum_umw_surface_d_invest[i] = d_delta_reforestation_surface_d_invest[i] + \
                                                d_delta_deforestation_surface_d_invest[i]
            else:
                d_cum_umw_surface_d_invest[i] = d_cum_umw_surface_d_invest[i - 1] + \
                                                d_delta_reforestation_surface_d_invest[i] + \
                                                d_delta_deforestation_surface_d_invest[i]
            # if unmanaged forest are empty, managed forest are removed
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 1e-10:
                # remove managed wood
                d_delta_mw_surface_d_invest[i] += d_cum_umw_surface_d_invest[i]

                if i == 0:
                    d_lc_reforestation_d_invest[i] = d_reforestation_surface_d_invest[i] * \
                                                     self.cost_per_ha
                else:
                    d_lc_reforestation_d_invest[i] = (
                                                             d_cum_umw_surface_d_invest[i - 1] +
                                                             d_reforestation_surface_d_invest[i]) * self.cost_per_ha
                d_lc_mw_d_invest[i] = -  d_cum_umw_surface_d_invest[i] * \
                                      self.techno_wood_info['managed_wood_price_per_ha']
                # set unmanaged forest to 0
                d_cum_umw_surface_d_invest[i] = np.zeros(number_of_values)

            else:
                d_lc_reforestation_d_invest[i] = np.zeros(number_of_values)
            # if managed forest are empty, all is removed
            if self.managed_wood_df.loc[i, 'cumulative_surface'] <= 1e-10:
                sum = self.d_cum(d_delta_mw_surface_d_invest)
                # delta is all the managed wood available
                d_lc_deforestation_d_invest[i] = - \
                                                     sum[i] * self.deforest_cost_per_ha
                d_lc_mw_d_invest[i] = - (d_lc_reforestation_d_invest[i] / self.cost_per_ha -
                                         sum[i]) * self.techno_wood_info['managed_wood_price_per_ha']

                d_delta_mw_surface_d_invest[i] = - \
                    sum[i - 1]
                d_delta_deforestation_surface_d_invest[i] = - \
                                                                d_reforestation_surface_d_invest[i] + \
                                                            d_delta_mw_surface_d_invest[i]

        return d_cum_umw_surface_d_invest, d_delta_mw_surface_d_invest, d_delta_deforestation_surface_d_invest, d_lc_deforestation_d_invest, d_lc_reforestation_d_invest, d_lc_mw_d_invest

    def compute_d_limit_surfaces_d_mw_invest(self, d_mw_surface_d_mw_invest):
        """
        Compute gradient of delta managed wood surface, delta deforestation surface, unmanaged wood cumulated surface,
        mw lost capital, deforestation lost capital and reforestation lost capital vs mw invest
        """
        number_of_values = (self.year_end - self.year_start + 1)

        d_delta_mw_surface_d_invest = d_mw_surface_d_mw_invest
        d_delta_deforestation_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_cum_umw_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_deforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_reforestation_d_invest = np.zeros(
            (number_of_values, number_of_values))
        d_lc_mw_d_invest = np.zeros(
            (number_of_values, number_of_values))

        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 1e-10:
                if self.managed_wood_df.loc[i, 'cumulative_surface'] <= 1e-10:
                    sum = self.d_cum(d_delta_mw_surface_d_invest)
                    d_lc_deforestation_d_invest[i] = - \
                                                         sum[i] * self.deforest_cost_per_ha

                    # delta is all the managed wood available
                    d_delta_mw_surface_d_invest[i] = - sum[i - 1]
                    d_delta_deforestation_surface_d_invest[i] = d_delta_mw_surface_d_invest[i]
                    d_lc_mw_d_invest[i] = sum[i] * \
                                          self.techno_wood_info['managed_wood_price_per_ha']

        return d_cum_umw_surface_d_invest, d_delta_mw_surface_d_invest, d_delta_deforestation_surface_d_invest, d_lc_deforestation_d_invest, d_lc_reforestation_d_invest, d_lc_mw_d_invest

    def compute_d_CO2_land_emission(self, d_forest_delta_surface):
        """
        Compute gradient of CO2_land_emission by surface
        :param: d_forest_delta_surface, derivative of forest constraint surface
        d_CO2_emission = surface * constant --> d_surface is reused.
        """

        d_CO2_emission = - d_forest_delta_surface * self.CO2_per_ha / 1000

        return d_CO2_emission

    def compute_d_techno_prod_d_invest(self, d_delta_mw_d_invest, d_delta_deforestation_d_invest):
        """
        Compute gradient of techno prod by invest
        :param: d_delta_mw_d_invest, derivative of managed wood surface vs invest
        :param: d_delta_deforestation_d_invest, derivative of deforestation surface vs invest
        """

        # compute gradient of managed wood prod for energy
        d_mw_prod_wood_for_nrj = self.d_cum(
            d_delta_mw_d_invest * self.wood_density * self.managed_yield * (1 - self.residues_density_percentage))
        d_mw_prod_residue_for_nrj = self.d_cum(
            d_delta_mw_d_invest * self.residues_density * self.managed_yield * self.residues_density_percentage)

        # compute gradient of deforestation production for nrj
        d_deforestation_prod_for_nrj = -d_delta_deforestation_d_invest * self.wood_density * self.unmanaged_yield

        d_techno_prod_d_invest = (
                                             d_mw_prod_wood_for_nrj + d_deforestation_prod_for_nrj) * self.biomass_dry_calorific_value + \
                                 d_mw_prod_residue_for_nrj * self.techno_wood_info['residue_calorific_value']
        d_techno_prod_d_invest = d_techno_prod_d_invest * self.wood_percentage_for_energy
        return d_techno_prod_d_invest

    def compute_d_techno_conso_d_invest(self, d_techno_prod_d_invest):
        """
        Compute gradient of techno consumption by invest
        :param: d_techno_prod_d_invest, derivative of techno_prod vs invest
        """
        d_techno_conso_d_invest = -self.techno_wood_info['CO2_from_production'] / \
                                  self.biomass_dry_high_calorific_value * d_techno_prod_d_invest

        return d_techno_conso_d_invest

    def compute_d_techno_price_d_invest(self, d_delta_mw_d_invest, d_delta_deforestation_d_invest):
        """
        Compute gradient of techno price by invest
        :param: d_delta_mw_d_invest, derivative of managed wood surface vs invest
        :param: d_delta_deforestation_d_invest, derivative of deforestation surface vs invest
        """

        d_mw_prod_wood_for_nrj = self.d_cum(
            d_delta_mw_d_invest * self.wood_density * self.managed_yield) * (1 - self.residues_density_percentage)
        d_mw_prod_residue_for_nrj = self.d_cum(
            d_delta_mw_d_invest * self.residues_density * self.managed_yield) * self.residues_density_percentage

        # compute gradient of deforestation production for nrj
        d_deforestation_prod = -d_delta_deforestation_d_invest * self.wood_density * self.unmanaged_yield

        # compute gradient of managed wood prod
        d_mw_prod = d_mw_prod_wood_for_nrj + d_mw_prod_residue_for_nrj

        # derivative of mw_prod /(mw_prod + deforestation_prod)
        # we get the transpose of the matrix to compute the right indexes
        v = self.managed_wood_df['biomass_production (Mt)'].values + \
            self.biomass_dry_df['deforestation (Mt)'].values
        v_prime = (d_mw_prod + d_deforestation_prod).T
        v_square = v * v
        u = self.managed_wood_df['biomass_production (Mt)'].values
        u_prime = d_mw_prod.T
        d_mw_price_per_ton = self.biomass_dry_df['managed_wood_price_per_ton'].values * (
                u_prime * v - v_prime * u) / v_square

        # derivative of deforestation_prod /(mw_prod + deforestation_prod)
        u = self.biomass_dry_df['deforestation (Mt)'].values
        u_prime = d_deforestation_prod.T
        d_deforestation_price_per_ton = self.biomass_dry_df['deforestation_price_per_ton'].values * (
                u_prime * v - v_prime * u) / v_square

        d_price_per_ton = d_mw_price_per_ton.T + d_deforestation_price_per_ton.T
        d_price_per_mwh = d_price_per_ton / self.biomass_dry_calorific_value

        return d_price_per_mwh

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
